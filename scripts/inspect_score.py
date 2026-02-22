import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_preds(path):
    with open(path, "r") as f:
        preds = json.load(f)
    if not isinstance(preds, list):
        raise ValueError("Expected coco_predictions_segm.json to be a list of predictions.")
    return preds


def try_import_pycocotools():
    try:
        from pycocotools import mask as mask_utils  # type: ignore
        return mask_utils
    except Exception:
        return None


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True, help="Path to coco_predictions_segm.json")
    ap.add_argument("--num_images", type=int, default=3, help="How many image_ids (sorted) to plot")
    ap.add_argument("--bins", type=int, default=20, help="Histogram bins")
    ap.add_argument("--check_decode", action="store_true", help="Decode a few masks to check area/bbox units")
    ap.add_argument("--decode_per_image", type=int, default=2, help="How many masks to decode per image_id")
    args = ap.parse_args()

    preds = load_preds(args.pred_json)

    # 1) Confidence is stored in each prediction as the scalar field "score"
    scores_by_img = defaultdict(list)
    preds_by_img = defaultdict(list)
    for p in preds:
        img_id = p.get("image_id")
        score = p.get("score", None)
        if img_id is None:
            continue
        preds_by_img[img_id].append(p)
        if score is not None:
            scores_by_img[img_id].append(float(score))

    img_ids = sorted(preds_by_img.keys())
    print(f"Loaded {len(preds)} predictions across {len(img_ids)} images.")
    if len(scores_by_img) == 0:
        print("No 'score' fields found. (Unexpected for COCO-style predictions.)")
        return

    all_scores = [s for v in scores_by_img.values() for s in v]
    print(f"Score summary: n={len(all_scores)}, min={min(all_scores):.5f}, mean={sum(all_scores)/len(all_scores):.5f}, max={max(all_scores):.5f}")

    # 2) Plot histogram of mask scores for first N images
    chosen = [i for i in img_ids if len(scores_by_img.get(i, [])) > 0][: args.num_images]
    print("Plotting per-image score histograms for image_ids:", chosen)

    for img_id in chosen:
        scores = scores_by_img[img_id]
        print(f"image_id={img_id}: {len(scores)} masks, score range=[{min(scores):.4f}, {max(scores):.4f}]")
        plt.figure()
        plt.hist(scores, bins=args.bins)
        plt.title(f"Mask-level confidence (score) histogram | image_id={img_id}")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    # 3) Optional: decode a few masks and sanity-check bbox/area units
    if args.check_decode:
        mask_utils = try_import_pycocotools()
        if mask_utils is None:
            print("\npycocotools not found. Install it to decode RLE masks:")
            print("  pip install pycocotools")
            return

        print("\nDecoding a few masks per image to sanity-check stored bbox/area...")
        for img_id in chosen:
            plist = preds_by_img[img_id][: args.decode_per_image]
            for k, p in enumerate(plist):
                seg = p.get("segmentation", None)
                if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
                    print(f"image_id={img_id} pred#{k}: missing/invalid segmentation.")
                    continue

                H, W = seg["size"]
                rle = {"size": [H, W], "counts": seg["counts"]}
                m = mask_utils.decode(rle)  # (H,W,1) or (H,W)
                if m.ndim == 3:
                    m = m[:, :, 0]
                m = (m > 0).astype(np.uint8)

                area_px = int(m.sum())
                bbox_px = bbox_from_mask(m)

                stored_bbox = p.get("bbox", None)
                stored_area = p.get("area", None)
                score = p.get("score", None)

                # Detect if bbox seems normalized
                bbox_note = ""
                if isinstance(stored_bbox, (list, tuple)) and len(stored_bbox) == 4:
                    mx = max(stored_bbox)
                    if mx <= 1.5:  # likely normalized [0,1]
                        x, y, w, h = stored_bbox
                        bbox_note = f"(stored bbox looks normalized; ~px=({x*W:.1f},{y*H:.1f},{w*W:.1f},{h*H:.1f}))"

                area_note = ""
                if isinstance(stored_area, (int, float)) and stored_area <= 1.5:
                    area_note = f"(stored area looks fractional; ~px={stored_area*H*W:.1f})"

                print(
                    f"image_id={img_id} pred#{k} score={score:.5f} | "
                    f"decoded_area_px={area_px} {area_note} | "
                    f"decoded_bbox_px={bbox_px} | stored_bbox={stored_bbox} {bbox_note}"
                )


if __name__ == "__main__":
    main()
