#!/usr/bin/env python3
"""
Create side-by-side visualization panels from COCO prediction JSON files.

Each output image contains four columns:
1) raw image
2) bbox-only overlay
3) segm-only overlay
4) bbox + segm overlay
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pprint import pprint
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side COCO prediction visualizations."
    )
    parser.add_argument("--ann-file", required=True, help="COCO annotations JSON path.")
    parser.add_argument(
        "--img-root",
        required=True,
        help="Root directory containing image files referenced by ann-file.",
    )
    parser.add_argument(
        "--bbox-pred",
        required=False,
        help="COCO bbox prediction JSON path.",
    )
    parser.add_argument(
        "--segm-pred",
        required=True,
        help="COCO segm prediction JSON path.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.05,
        help="Score threshold for drawing predictions.",
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=20,
        help="Maximum predictions to draw per image per type.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of image ids to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used for deterministic colors.",
    )
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def group_predictions_by_image(preds: Iterable[Dict]) -> Dict[int, List[Dict]]:
    out = defaultdict(list)
    for pred in preds:
        out[int(pred["image_id"])].append(pred)
    return out


def filter_and_sort(preds: List[Dict], score_thr: float, max_per_image: int) -> List[Dict]:
    kept = [p for p in preds if float(p.get("score", 0.0)) >= score_thr]
    kept.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return kept[:max_per_image]


def make_colors(count: int, seed: int) -> List[tuple]:
    rng = random.Random(seed)
    colors = []
    for _ in range(count):
        colors.append(
            (
                rng.randint(40, 255),
                rng.randint(40, 255),
                rng.randint(40, 255),
            )
        )
    return colors


def draw_bboxes(base_img: Image.Image, bbox_preds: List[Dict], colors: List[tuple]) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    for idx, pred in enumerate(bbox_preds):
        x, y, w, h = pred["bbox"]
        c = colors[idx % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=c, width=2)
        draw.text((x + 2, y + 2), f"{float(pred['score']):.2f}", fill=c)
    return img


def overlay_masks(base_img: Image.Image, segm_preds: List[Dict], colors: List[tuple]) -> Image.Image:
    arr = np.asarray(base_img).astype(np.float32)
    alpha = 0.35
    for idx, pred in enumerate(segm_preds):
        seg = pred.get("segmentation")
        if seg is None:
            continue
        mask = mask_utils.decode(seg)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        c = np.array(colors[idx % len(colors)], dtype=np.float32)
        arr[mask > 0] = arr[mask > 0] * (1.0 - alpha) + c * alpha
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def make_panel(
    raw_img: Image.Image,
    bbox_img: Image.Image,
    segm_img: Image.Image,
    both_img: Image.Image,
) -> Image.Image:
    titles = ["raw", "bbox", "segm", "bbox+segm"]
    cols = [raw_img, bbox_img, segm_img, both_img]
    w, h = raw_img.size
    margin = 8
    title_h = 26
    panel_w = 4 * w + 5 * margin
    panel_h = h + title_h + 2 * margin
    panel = Image.new("RGB", (panel_w, panel_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for i, (title, col_img) in enumerate(zip(titles, cols)):
        x = margin + i * (w + margin)
        y = margin + title_h
        panel.paste(col_img, (x, y))
        draw.text((x, margin), title, fill=(240, 240, 240), font=font)
    return panel


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ann = load_json(args.ann_file)
    bbox_preds = load_json(args.bbox_pred) if args.bbox_pred else []
    segm_preds = load_json(args.segm_pred)

    id_to_img = {int(x["id"]): x for x in ann["images"]}
    bbox_by_img = group_predictions_by_image(bbox_preds) if bbox_preds else defaultdict(list)
    segm_by_img = group_predictions_by_image(segm_preds)

    if args.image_ids:
        image_ids = [int(x) for x in args.image_ids if int(x) in id_to_img]
    else:
        image_ids = sorted(id_to_img.keys())

    summary = {
        "output_dir": args.out_dir,
        "score_threshold": args.score_thr,
        "max_per_image": args.max_per_image,
        "images": [],
    }

    for image_id in image_ids:
        img_meta = id_to_img[image_id]
        img_path = os.path.join(args.img_root, img_meta["file_name"])
        raw = Image.open(img_path).convert("RGB")

        b = filter_and_sort(
            bbox_by_img.get(image_id, []), args.score_thr, args.max_per_image
        )
        s = filter_and_sort(
            segm_by_img.get(image_id, []), args.score_thr, args.max_per_image
        )

        color_count = max(len(b), len(s), 1)
        colors = make_colors(color_count, seed=args.seed + image_id)

        bbox_img = draw_bboxes(raw, b, colors)
        segm_img = overlay_masks(raw, s, colors)
        both_img = draw_bboxes(segm_img, b, colors)

        panel = make_panel(raw, bbox_img, segm_img, both_img)
        out_path = os.path.join(args.out_dir, f"panel_img_{image_id}.png")
        panel.save(out_path)
        summary["images"].append(
            {
                "image_id": image_id,
                "file_name": img_meta["file_name"],
                "bbox_drawn": len(b),
                "segm_drawn": len(s),
                "panel_path": out_path,
            }
        )

    pprint(summary, sort_dicts=False)


if __name__ == "__main__":
    main()
