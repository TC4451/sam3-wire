#!/usr/bin/env python3
"""
Convert color-coded PNG masks into a single COCO-style JSON (RLE) using pycocotools.

Expected layout (example):
  data_root/
    plane_wires_2_highlight_organized/
      rgb/
      seg/
      index.jsonl
    plane_wires_4_highlight_organized/
      rgb/
      seg/
    flying_wires_8_organized/
      rgb/
      seg/

Run:
  pip install pycocotools pillow numpy
  # Convert from rgb/seg folder layout:
  python convert_mask_to_coco.py /path/to/data_root /path/to/output.json
  # Or sanitize an existing COCO JSON (auto-detected):
  python convert_mask_to_coco.py /path/to/input_coco.json /path/to/output_coco.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# ----------------------------
# Helpers
# ----------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(root: Path, exts=IMAGE_EXTS) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def load_mask_rgb(mask_path: Path) -> np.ndarray:
    """Return HxWx3 uint8 RGB array (alpha dropped if present)."""
    im = Image.open(mask_path)
    if im.mode not in ("RGB", "RGBA"):
        # If palette/LA/L/etc, normalize
        im = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def unique_colors(mask_rgb: np.ndarray) -> np.ndarray:
    """Return unique colors as Nx3 uint8 array."""
    flat = mask_rgb.reshape(-1, 3)
    return np.unique(flat, axis=0)


def color_to_binary(mask_rgb: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return HxW uint8 mask (0/1) for pixels matching given RGB color."""
    m = np.all(mask_rgb == color[None, None, :], axis=-1)
    return m.astype(np.uint8)


def encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Encode binary mask to COCO RLE (JSON-serializable)."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("ascii")  # bytes -> str for JSON
    return rle


def decode_rle_counts_if_needed(rle: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure compressed RLE counts is str for JSON serialization."""
    if isinstance(rle.get("counts"), bytes):
        rle = dict(rle)
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def normalize_rle_to_image_size(
    rle: Dict[str, Any], image_h: int, image_w: int
) -> Tuple[Dict[str, Any] | None, str]:
    """
    Normalize one RLE to match image (H, W).

    Returns:
        (normalized_rle_or_none, status)
    status is one of:
        ok, fixed_by_reencode, fixed_by_transpose, invalid_size, decode_failed, shape_mismatch
    """
    rle = decode_rle_counts_if_needed(rle)
    size = rle.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None, "invalid_size"

    try:
        rle_h, rle_w = int(size[0]), int(size[1])
    except Exception:
        return None, "invalid_size"

    if (rle_h, rle_w) == (image_h, image_w):
        return rle, "ok"

    try:
        decoded = mask_utils.decode(rle)
    except Exception:
        return None, "decode_failed"

    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    decoded = (decoded > 0).astype(np.uint8)

    if decoded.shape == (image_h, image_w):
        return encode_rle(decoded), "fixed_by_reencode"

    if decoded.shape == (image_w, image_h):
        decoded_t = np.ascontiguousarray(decoded.T)
        return encode_rle(decoded_t), "fixed_by_transpose"

    return None, "shape_mismatch"

def encode_rle_uncompressed(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Encode binary mask to *uncompressed* COCO RLE (counts as list[int])."""
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array")

    h, w = binary_mask.shape
    # Ensure 0/1 uint8
    m = (binary_mask > 0).astype(np.uint8)

    # Flatten in Fortran order (COCO expects column-major)
    pixels = m.flatten(order="F")

    counts: List[int] = []
    run_val = 0  # start with zeros
    run_len = 0

    for p in pixels:
        if p == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = int(p)
            run_len = 1
    counts.append(run_len)

    # If mask starts with 1, COCO wants counts to start with 0-run
    if pixels.size > 0 and pixels[0] == 1:
        counts = [0] + counts

    return {"size": [h, w], "counts": counts}


def sanitize_coco_annotation_rles(
    coco: Dict[str, Any], drop_unfixable: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Fix malformed annotation RLE sizes against image metadata.

    If drop_unfixable=True, invalid annotations are removed. Otherwise they are kept unchanged.
    """
    image_hw: Dict[int, Tuple[int, int]] = {
        int(img["id"]): (int(img["height"]), int(img["width"]))
        for img in coco.get("images", [])
        if "id" in img and "height" in img and "width" in img
    }

    stats = {
        "annotations_total": 0,
        "rle_annotations_checked": 0,
        "fixed_by_reencode": 0,
        "fixed_by_transpose": 0,
        "dropped_unfixable": 0,
        "missing_image_metadata": 0,
    }
    sanitized: List[Dict[str, Any]] = []

    for ann in coco.get("annotations", []):
        stats["annotations_total"] += 1
        segm = ann.get("segmentation")

        # Only RLE dicts can carry a malformed size field.
        if not (isinstance(segm, dict) and "counts" in segm and "size" in segm):
            sanitized.append(ann)
            continue

        img_id = int(ann.get("image_id", -1))
        hw = image_hw.get(img_id)
        if hw is None:
            stats["missing_image_metadata"] += 1
            if drop_unfixable:
                stats["dropped_unfixable"] += 1
            else:
                sanitized.append(ann)
            continue

        image_h, image_w = hw
        stats["rle_annotations_checked"] += 1
        normalized_rle, status = normalize_rle_to_image_size(segm, image_h, image_w)

        if normalized_rle is None:
            if drop_unfixable:
                stats["dropped_unfixable"] += 1
            else:
                sanitized.append(ann)
            continue

        out_ann = dict(ann)
        out_ann["segmentation"] = normalized_rle
        if status == "fixed_by_reencode":
            stats["fixed_by_reencode"] += 1
        elif status == "fixed_by_transpose":
            stats["fixed_by_transpose"] += 1

        try:
            out_ann["area"] = float(mask_utils.area(normalized_rle))
        except Exception:
            pass
        try:
            bbox = mask_utils.toBbox(normalized_rle).tolist()
            out_ann["bbox"] = [float(x) for x in bbox]
        except Exception:
            pass

        sanitized.append(out_ann)

    return sanitized, stats


def sanitize_coco_json(
    in_json: Path, out_json: Path, drop_unfixable: bool = True
) -> Path:
    coco = json.loads(in_json.read_text())
    sanitized_annotations, stats = sanitize_coco_annotation_rles(
        coco, drop_unfixable=drop_unfixable
    )
    coco["annotations"] = sanitized_annotations

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, indent=2))

    fixed_total = stats["fixed_by_reencode"] + stats["fixed_by_transpose"]
    print(f"Wrote sanitized COCO JSON: {out_json}")
    print(
        "Sanitizer summary | "
        f"annotations={stats['annotations_total']} "
        f"rle_checked={stats['rle_annotations_checked']} "
        f"fixed={fixed_total} "
        f"(reencode={stats['fixed_by_reencode']}, transpose={stats['fixed_by_transpose']}) "
        f"dropped={stats['dropped_unfixable']} "
        f"missing_image_meta={stats['missing_image_metadata']}"
    )
    return out_json

# ----------------------------
# Main conversion
# ----------------------------
def find_dataset_roots(root: Path) -> List[Path]:
    """Find subfolders that contain rgb/seg pairs."""
    roots: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name == "__MACOSX":
            continue
        rgb_dir = p / "rgb"
        if not rgb_dir.exists():
            alt = p / "rbg"
            rgb_dir = alt if alt.exists() else rgb_dir
        seg_dir = p / "seg"
        if rgb_dir.exists() and seg_dir.exists():
            roots.append(p)
    return sorted(roots)


def image_mask_pair(img_path: Path, rgb_dir: Path, seg_dir: Path) -> Path:
    rel = img_path.relative_to(rgb_dir)
    name = rel.name
    mask_name = name.replace("rgb_", "seg_", 1)
    mask_path = seg_dir / rel.with_name(mask_name)
    if mask_path.exists():
        return mask_path
    fallback = seg_dir / rel.with_suffix(".png")
    if fallback.exists():
        return fallback
    fallback2 = seg_dir / Path(rel).with_suffix(".png").name
    if fallback2.exists():
        return fallback2
    raise FileNotFoundError(f"Missing mask for image {img_path}. Expected: {mask_path}")


def _process_one(
    idx: int,
    img_path: Path,
    mask_path: Path,
    file_name: str,
    background_rgb: Tuple[int, int, int],
    min_pixels: int,
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    with Image.open(img_path) as im:
        width, height = im.size

    image_rec = {"id": -1, "file_name": file_name, "width": int(width), "height": int(height)}

    mask_rgb = load_mask_rgb(mask_path)
    colors = unique_colors(mask_rgb)

    bg = np.array(background_rgb, dtype=np.uint8)
    anns: List[Dict[str, Any]] = []
    for color in colors:
        if np.array_equal(color, bg):
            continue

        binary = color_to_binary(mask_rgb, color)
        pix = int(binary.sum())
        if pix < min_pixels:
            continue

        rle = encode_rle(binary)
        area = float(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()

        anns.append(
            {
                "id": -1,
                "image_id": -1,
                "category_id": 1,
                "segmentation": rle,
                "area": area,
                "bbox": [float(x) for x in bbox],
                "iscrowd": 0,
                "color_rgb": [int(color[0]), int(color[1]), int(color[2])],
            }
        )

    return idx, image_rec, anns


def build_single_coco_from_root(
    data_root: Path,
    out_json: Path,
    background_rgb: Tuple[int, int, int] = (0, 0, 0),
    min_pixels: int = 1,
) -> Path:
    coco: Dict[str, Any] = {
        "info": {"description": f"{data_root.name} color-mask dataset converted to COCO RLE"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }

    next_image_id = 1
    next_ann_id = 1

    dataset_roots = find_dataset_roots(data_root)
    if not dataset_roots:
        raise FileNotFoundError(
            f"No dataset roots found under {data_root}. Expected subfolders with rgb/seg."
        )

    jobs: List[Tuple[int, Path, Path, str]] = []
    job_idx = 0

    for scene_root in dataset_roots:
        rgb_dir = scene_root / "rgb"
        if not rgb_dir.exists():
            alt = scene_root / "rbg"
            rgb_dir = alt if alt.exists() else rgb_dir
        seg_dir = scene_root / "seg"

        images = list_images(rgb_dir)
        if not images:
            print(f"Skip (no images): {rgb_dir}")
            continue

        for img_path in images:
            rel_img = img_path.relative_to(data_root).as_posix()
            mask_path = image_mask_pair(img_path, rgb_dir, seg_dir)
            jobs.append((job_idx, img_path, mask_path, rel_img))
            job_idx += 1

    workers = os.cpu_count() or 1
    results: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]] = []

    print(f"Found {len(jobs)} images. Starting conversion with {workers} workers...")
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _process_one,
                idx,
                img_path,
                mask_path,
                rel_img,
                background_rgb,
                min_pixels,
            )
            for (idx, img_path, mask_path, rel_img) in jobs
        ]
        total = len(futures)
        report_every = max(1, total // 50)  # ~2% increments
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done == total or done % report_every == 0:
                elapsed = max(1e-6, time.time() - start)
                rate = done / elapsed
                remaining = total - done
                eta = remaining / max(rate, 1e-6)
                pct = (100.0 * done) / total if total else 100.0
                print(
                    f"Processed {done}/{total} images "
                    f"({pct:.1f}%) | {rate:.1f} img/s | ETA {eta:.1f}s"
                )
    elapsed = time.time() - start
    if futures:
        print(f"Finished processing {len(futures)} images in {elapsed:.1f}s")

    results.sort(key=lambda x: x[0])
    for _idx, image_rec, anns in results:
        image_id = next_image_id
        next_image_id += 1
        image_rec["id"] = image_id
        coco["images"].append(image_rec)

        for ann in anns:
            ann["id"] = next_ann_id
            ann["image_id"] = image_id
            coco["annotations"].append(ann)
            next_ann_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, indent=2))
    print(f"Wrote: {out_json}")
    print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")
    return out_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert color masks to COCO JSON, or sanitize an existing COCO JSON "
            "by fixing/dropping invalid RLE sizes. Mode is auto-detected from input path."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Data root directory (convert mode) or input COCO JSON file (sanitize mode).",
    )
    parser.add_argument("output_json", type=Path, help="Output JSON path.")
    parser.add_argument(
        "--keep-unfixable",
        action="store_true",
        help="Keep malformed annotations that cannot be normalized (default: drop).",
    )
    args = parser.parse_args()

    if args.input_path.is_file():
        sanitize_coco_json(
            args.input_path,
            args.output_json,
            drop_unfixable=not args.keep_unfixable,
        )
    elif args.input_path.is_dir():
        build_single_coco_from_root(args.input_path, args.output_json)
    else:
        raise FileNotFoundError(
            f"input_path does not exist or is neither file nor directory: {args.input_path}"
        )













