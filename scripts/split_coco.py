#!/usr/bin/env python3
"""
Split a single COCO annotation file into train/valid/test sets by image ID.

This script only reads the COCO JSON. It does not read, copy, or validate image
files on disk.

Outputs are written next to the input JSON:
  train.json
  valid.json
  test.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


SplitCounts = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split COCO annotations into train/valid/test by image ID."
    )
    parser.add_argument("input_json", type=Path, help="Path to source COCO JSON.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Valid split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic image shuffling (default: 123).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def validate_coco(coco: Mapping[str, Any]) -> None:
    if "images" not in coco or "annotations" not in coco:
        raise ValueError("COCO JSON must contain `images` and `annotations`.")
    if not isinstance(coco["images"], list):
        raise ValueError("`images` must be a list.")
    if not isinstance(coco["annotations"], list):
        raise ValueError("`annotations` must be a list.")

    missing_image_ids = [i for i, im in enumerate(coco["images"]) if "id" not in im]
    if missing_image_ids:
        raise ValueError(f"{len(missing_image_ids)} images are missing `id`.")

    image_ids = [im["id"] for im in coco["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Image IDs are not unique in `images`.")


def normalize_ratios(train: float, valid: float, test: float) -> Tuple[float, float, float]:
    ratios = (train, valid, test)
    if any(r < 0 for r in ratios):
        raise ValueError("Ratios must be non-negative.")
    s = sum(ratios)
    if s <= 0:
        raise ValueError("At least one ratio must be > 0.")
    return (train / s, valid / s, test / s)


def compute_split_counts(n_items: int, ratios: Tuple[float, float, float]) -> SplitCounts:
    # Largest-remainder allocation: deterministic and sums exactly to n_items.
    raw = [r * n_items for r in ratios]
    counts = [int(math.floor(x)) for x in raw]
    remaining = n_items - sum(counts)
    frac = sorted(
        [(raw[i] - counts[i], i) for i in range(3)],
        key=lambda x: x[0],
        reverse=True,
    )
    for k in range(remaining):
        counts[frac[k][1]] += 1
    return counts[0], counts[1], counts[2]


def split_image_ids(image_ids: Sequence[int], counts: SplitCounts, seed: int) -> Dict[str, List[int]]:
    ids = list(image_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n_train, n_valid, _ = counts
    train_ids = ids[:n_train]
    valid_ids = ids[n_train : n_train + n_valid]
    test_ids = ids[n_train + n_valid :]
    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


def gather_split_records(
    coco: Mapping[str, Any], split_image_ids_dict: Mapping[str, List[int]]
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    images_by_id = {im["id"]: im for im in coco["images"]}

    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for split_name, ids in split_image_ids_dict.items():
        id_set = set(ids)
        split_images = [dict(images_by_id[i]) for i in ids]
        split_annotations = [
            dict(ann) for ann in coco["annotations"] if ann.get("image_id") in id_set
        ]
        out[split_name] = {"images": split_images, "annotations": split_annotations}
    return out


def build_split_coco(
    coco: Mapping[str, Any],
    split_images: List[Dict[str, Any]],
    split_annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    split_obj: Dict[str, Any] = {}
    for k, v in coco.items():
        if k in ("images", "annotations"):
            continue
        split_obj[k] = v
    split_obj["images"] = split_images
    split_obj["annotations"] = split_annotations
    return split_obj


def print_stats(split_data: Mapping[str, Mapping[str, List[Dict[str, Any]]]]) -> None:
    for split_name in ("train", "valid", "test"):
        data = split_data[split_name]
        image_count = len(data["images"])
        ann_count = len(data["annotations"])
        print(f"{split_name:>5}: images={image_count:6d} annotations={ann_count:8d}")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = args.input_json.parent

    coco = load_json(args.input_json)
    validate_coco(coco)

    ratios = normalize_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    image_ids = [im["id"] for im in coco["images"]]
    counts = compute_split_counts(len(image_ids), ratios)
    splits = split_image_ids(image_ids, counts, args.seed)
    split_data = gather_split_records(coco, splits)

    print(f"Input: {args.input_json}")
    print(f"Total images: {len(coco['images'])}")
    print(f"Total annotations: {len(coco['annotations'])}")
    print(
        "Ratios (normalized): "
        f"train={ratios[0]:.6f}, valid={ratios[1]:.6f}, test={ratios[2]:.6f}"
    )
    print_stats(split_data)

    for split_name in ("train", "valid", "test"):
        split_json = build_split_coco(
            coco=coco,
            split_images=split_data[split_name]["images"],
            split_annotations=split_data[split_name]["annotations"],
        )
        out_path = output_dir / f"{split_name}.json"
        write_json(out_path, split_json)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
