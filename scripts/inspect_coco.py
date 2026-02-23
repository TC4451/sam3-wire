#!/usr/bin/env python3
"""
Thorough COCO Ground-Truth validator.

Checks:
- JSON structure: images/annotations/categories
- image id uniqueness, width/height positive
- annotation fields: image_id exists, category_id exists
- bbox validity: finite, w/h>0, non-negative, optional in-bounds checks
- segmentation validity:
  - polygons: even length, >=3 points, finite, optional in-bounds
  - RLE: decodes, counts type, size matches image, area finite
- area consistency: (optional) compare ann["area"] vs decoded
- iscrowd correctness (0/1)
- keypoints validity (if present)
- duplicates: identical bbox+seg+cat on same image
- empty images: images with no anns (not always wrong, but reported)

Outputs:
- report to stdout
- optional JSON with filtered/auto-fixed anns
"""

import argparse
import copy
import json
import math
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from pycocotools import mask as maskUtils
except Exception as e:
    maskUtils = None


# -----------------------------
# Utilities
# -----------------------------
def is_finite_num(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def finite_array(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def bbox_xywh_to_xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)


def bbox_area(b):
    return float(b[2]) * float(b[3])


def polygon_area(poly_xy: np.ndarray) -> float:
    """
    poly_xy: (N,2) closed or open polygon; we handle open.
    Shoelace formula. Returns absolute area.
    """
    if poly_xy.shape[0] < 3:
        return 0.0
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    # wrap-around
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def try_decode_segmentation(seg, H, W):
    """
    Returns (rle, decoded_area, err_str)
    """
    if maskUtils is None:
        return None, None, "pycocotools not installed"
    try:
        if isinstance(seg, list):
            # polygons -> RLEs -> merge
            rles = maskUtils.frPyObjects(seg, H, W)
            rle = maskUtils.merge(rles)
        elif isinstance(seg, dict):
            rle = seg
        else:
            return None, None, f"unknown seg type: {type(seg)}"
        area = float(maskUtils.area(rle))
        return rle, area, None
    except Exception as e:
        return None, None, str(e)


@dataclass
class Issue:
    kind: str
    ann_id: Optional[int]
    img_id: Optional[int]
    detail: str


class CocoValidator:
    def __init__(
        self,
        coco: Dict[str, Any],
        *,
        strict: bool,
        check_in_bounds: bool,
        tolerate_empty_seg: bool,
        max_examples_per_issue: int,
        compare_area: bool,
        area_rel_tol: float,
        auto_fix: bool,
        fix_mode: str,
    ):
        self.coco = coco
        self.strict = strict
        self.check_in_bounds = check_in_bounds
        self.tolerate_empty_seg = tolerate_empty_seg
        self.max_examples_per_issue = max_examples_per_issue
        self.compare_area = compare_area
        self.area_rel_tol = area_rel_tol
        self.auto_fix = auto_fix
        self.fix_mode = fix_mode  # "filter" or "clamp"
        self.issues: List[Issue] = []
        self.issue_counts = Counter()

        self.images_by_id: Dict[int, Dict[str, Any]] = {}
        self.cats_by_id: Dict[int, Dict[str, Any]] = {}
        self.ann_by_id: Dict[int, Dict[str, Any]] = {}

        # For duplicates
        self.dup_hashes = defaultdict(list)

        # Stats
        self.images_with_no_anns = []
        self.img_ann_count = Counter()

        # Fixed outputs
        self.fixed = copy.deepcopy(coco)
        self.fixed_annotations = []
        self.filtered_out_ann_ids = set()

    def add_issue(self, kind, ann_id, img_id, detail):
        self.issue_counts[kind] += 1
        # keep limited examples
        if sum(1 for i in self.issues if i.kind == kind) < self.max_examples_per_issue:
            self.issues.append(Issue(kind, ann_id, img_id, detail))

    def build_indices(self):
        # Images
        imgs = self.coco.get("images", None)
        if not isinstance(imgs, list):
            raise ValueError("COCO missing 'images' list")

        for img in imgs:
            if "id" not in img:
                self.add_issue("image_missing_id", None, None, f"image keys={list(img.keys())}")
                continue
            img_id = img["id"]
            if img_id in self.images_by_id:
                self.add_issue("image_duplicate_id", None, img_id, "duplicate image id")
            self.images_by_id[img_id] = img

        # Categories
        cats = self.coco.get("categories", [])
        if not isinstance(cats, list):
            self.add_issue("categories_not_list", None, None, "categories is not a list")
            cats = []
        for cat in cats:
            if "id" not in cat:
                self.add_issue("category_missing_id", None, None, f"cat keys={list(cat.keys())}")
                continue
            cid = cat["id"]
            if cid in self.cats_by_id:
                self.add_issue("category_duplicate_id", None, None, f"duplicate category id {cid}")
            self.cats_by_id[cid] = cat

        # Annotations
        anns = self.coco.get("annotations", None)
        if not isinstance(anns, list):
            raise ValueError("COCO missing 'annotations' list")

        for ann in anns:
            if "id" not in ann:
                # COCO technically allows no id, but most tooling expects it
                self.add_issue("ann_missing_id", None, ann.get("image_id", None), f"ann keys={list(ann.keys())}")
                continue
            aid = ann["id"]
            if aid in self.ann_by_id:
                self.add_issue("ann_duplicate_id", aid, ann.get("image_id", None), "duplicate ann id")
            self.ann_by_id[aid] = ann

    def check_images(self):
        for img_id, img in self.images_by_id.items():
            W = img.get("width", None)
            H = img.get("height", None)
            if W is None or H is None:
                self.add_issue("image_missing_size", None, img_id, f"width/height missing: {img}")
                continue
            if not is_finite_num(W) or not is_finite_num(H):
                self.add_issue("image_nonfinite_size", None, img_id, f"W,H={W},{H}")
                continue
            if int(W) <= 0 or int(H) <= 0:
                self.add_issue("image_nonpositive_size", None, img_id, f"W,H={W},{H}")

    def _bbox_checks(self, ann, img):
        aid = ann["id"]
        img_id = ann.get("image_id", None)

        bbox = ann.get("bbox", None)
        if bbox is None:
            self.add_issue("ann_missing_bbox", aid, img_id, "missing bbox")
            return None  # cannot fix
        if not (isinstance(bbox, list) or isinstance(bbox, tuple)) or len(bbox) != 4:
            self.add_issue("ann_bad_bbox_format", aid, img_id, f"bbox={bbox}")
            return None

        x, y, w, h = bbox
        arr = np.array([x, y, w, h], dtype=np.float64)
        if not finite_array(arr):
            self.add_issue("ann_bbox_nonfinite", aid, img_id, f"bbox={bbox}")
            return bbox

        if w <= 0 or h <= 0:
            self.add_issue("ann_bbox_nonpositive_wh", aid, img_id, f"bbox={bbox}")
            return bbox

        if self.check_in_bounds and img is not None:
            W, H = img.get("width", None), img.get("height", None)
            if W is not None and H is not None:
                x1, y1, x2, y2 = bbox_xywh_to_xyxy(bbox)
                # allow small tolerance
                tol = 1.0
                if x1 < -tol or y1 < -tol or x2 > W + tol or y2 > H + tol:
                    self.add_issue(
                        "ann_bbox_out_of_bounds",
                        aid,
                        img_id,
                        f"bbox={bbox} img_size=({W},{H})",
                    )
        return bbox

    def _seg_checks(self, ann, img):
        """
        Returns (seg_ok, decoded_area or None)
        """
        aid = ann["id"]
        img_id = ann.get("image_id", None)

        seg = ann.get("segmentation", None)
        if seg is None:
            # Some tasks are box-only; but SAM3 uses segmentation. Report it.
            self.add_issue("ann_missing_segmentation", aid, img_id, "missing segmentation")
            return False, None

        # image size needed for decoding RLE
        if img is None:
            self.add_issue("ann_image_missing_for_seg", aid, img_id, "cannot locate image for segmentation checks")
            return False, None

        W, H = img.get("width", None), img.get("height", None)
        if W is None or H is None:
            self.add_issue("ann_image_missing_size_for_seg", aid, img_id, "image missing width/height")
            return False, None

        # polygon case
        if isinstance(seg, list):
            if len(seg) == 0:
                if self.tolerate_empty_seg:
                    self.add_issue("ann_empty_segmentation", aid, img_id, "empty polygon list (tolerated)")
                    return True, 0.0
                self.add_issue("ann_empty_segmentation", aid, img_id, "empty polygon list")
                return False, 0.0

            decoded_area = 0.0
            for pi, poly in enumerate(seg):
                if not isinstance(poly, list):
                    self.add_issue("ann_polygon_not_list", aid, img_id, f"poly[{pi}] type={type(poly)}")
                    return False, None
                if len(poly) < 6:
                    self.add_issue("ann_polygon_too_short", aid, img_id, f"poly[{pi}] len={len(poly)}")
                    return False, None
                if len(poly) % 2 != 0:
                    self.add_issue("ann_polygon_odd_len", aid, img_id, f"poly[{pi}] len={len(poly)}")
                    return False, None

                arr = np.array(poly, dtype=np.float64)
                if not finite_array(arr):
                    self.add_issue("ann_polygon_nonfinite", aid, img_id, f"poly[{pi}] has nonfinite")
                    return False, None

                pts = arr.reshape(-1, 2)
                # area sanity for polygons
                a = polygon_area(pts)
                if a <= 0:
                    self.add_issue("ann_polygon_zero_area", aid, img_id, f"poly[{pi}] area={a}")
                    # not necessarily fatal if decode works, but suspicious

                if self.check_in_bounds:
                    x = pts[:, 0]
                    y = pts[:, 1]
                    tol = 1.0
                    if (x < -tol).any() or (y < -tol).any() or (x > W + tol).any() or (y > H + tol).any():
                        self.add_issue("ann_polygon_out_of_bounds", aid, img_id, f"poly[{pi}] oob img=({W},{H})")

            # optional: decode polygons to RLE and compute area
            rle, area, err = try_decode_segmentation(seg, H, W)
            if err is not None:
                self.add_issue("ann_polygon_decode_error", aid, img_id, err)
                return False, None
            if area is None or (not is_finite_num(area)):
                self.add_issue("ann_polygon_decoded_area_nonfinite", aid, img_id, f"area={area}")
                return False, None
            decoded_area = float(area)
            if decoded_area <= 0 and not self.tolerate_empty_seg:
                self.add_issue("ann_seg_decoded_area_zero", aid, img_id, f"decoded_area={decoded_area}")
                return False, decoded_area
            return True, decoded_area

        # RLE case
        if isinstance(seg, dict):
            # Basic fields
            if "size" in seg:
                size = seg["size"]
                if not (isinstance(size, list) and len(size) == 2):
                    self.add_issue("ann_rle_bad_size_field", aid, img_id, f"size={size}")
                else:
                    if int(size[0]) != int(H) or int(size[1]) != int(W):
                        self.add_issue("ann_rle_size_mismatch", aid, img_id, f"rle.size={size} img=({H},{W})")

            rle, area, err = try_decode_segmentation(seg, H, W)
            if err is not None:
                self.add_issue("ann_rle_decode_error", aid, img_id, err)
                return False, None
            if area is None or (not is_finite_num(area)):
                self.add_issue("ann_rle_area_nonfinite", aid, img_id, f"area={area}")
                return False, None
            area = float(area)
            if area <= 0 and not self.tolerate_empty_seg:
                self.add_issue("ann_seg_decoded_area_zero", aid, img_id, f"decoded_area={area}")
                return False, area
            return True, area

        self.add_issue("ann_unknown_segmentation_type", aid, img_id, f"type={type(seg)}")
        return False, None

    def _misc_ann_checks(self, ann, img):
        aid = ann["id"]
        img_id = ann.get("image_id", None)

        # image exists?
        if img_id not in self.images_by_id:
            self.add_issue("ann_image_id_missing", aid, img_id, "image_id not found in images")
            return

        # category exists?
        cid = ann.get("category_id", None)
        if cid is None:
            self.add_issue("ann_missing_category_id", aid, img_id, "missing category_id")
        elif cid not in self.cats_by_id:
            self.add_issue("ann_unknown_category_id", aid, img_id, f"category_id={cid} not in categories")

        # iscrowd should be 0 or 1 if present
        if "iscrowd" in ann:
            v = ann["iscrowd"]
            if v not in (0, 1):
                self.add_issue("ann_bad_iscrowd", aid, img_id, f"iscrowd={v}")

        # area sanity if present
        if "area" in ann and ann["area"] is not None:
            if not is_finite_num(ann["area"]):
                self.add_issue("ann_area_nonfinite", aid, img_id, f"area={ann['area']}")
            elif float(ann["area"]) < 0:
                self.add_issue("ann_area_negative", aid, img_id, f"area={ann['area']}")

        # keypoints sanity if present
        if "keypoints" in ann:
            kps = ann["keypoints"]
            if not isinstance(kps, list) or len(kps) % 3 != 0:
                self.add_issue("ann_bad_keypoints_format", aid, img_id, f"len={len(kps) if isinstance(kps, list) else type(kps)}")
            else:
                arr = np.array(kps, dtype=np.float64)
                if not finite_array(arr):
                    self.add_issue("ann_keypoints_nonfinite", aid, img_id, "keypoints contain nonfinite")
                # visibility is every 3rd
                vis = arr[2::3]
                if not np.isin(vis, [0, 1, 2]).all():
                    self.add_issue("ann_keypoints_bad_visibility", aid, img_id, "vis not in {0,1,2}")

    def _dup_check_hash(self, ann):
        # A simple duplicate hash: (image_id, category_id, bbox rounded, seg signature)
        img_id = ann.get("image_id", None)
        cid = ann.get("category_id", None)
        bbox = ann.get("bbox", None)
        seg = ann.get("segmentation", None)

        def round_list(x, nd=2):
            return tuple(round(float(v), nd) for v in x)

        bbox_key = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(is_finite_num(v) for v in bbox):
            bbox_key = round_list(bbox, 2)

        # segmentation signature (not perfect but catches many dupes)
        seg_key = None
        if isinstance(seg, list):
            # polygons: number of polys + rounded first few coords
            flat = []
            for poly in seg[:2]:
                flat.extend(poly[:10])
            seg_key = ("poly", len(seg), tuple(round(float(v), 2) for v in flat))
        elif isinstance(seg, dict):
            # RLE: size + first chars/ints of counts
            counts = seg.get("counts", None)
            if isinstance(counts, str):
                sig = counts[:32]
            elif isinstance(counts, list):
                sig = tuple(counts[:16])
            else:
                sig = str(type(counts))
            seg_key = ("rle", tuple(seg.get("size", [])), sig)
        else:
            seg_key = ("none",)

        return (img_id, cid, bbox_key, seg_key)

    def validate(self):
        self.build_indices()
        self.check_images()

        anns_in = self.coco["annotations"]
        for ann in anns_in:
            if "id" not in ann:
                continue
            aid = ann["id"]
            img_id = ann.get("image_id", None)
            img = self.images_by_id.get(img_id, None)

            # misc checks
            self._misc_ann_checks(ann, img)

            # bbox checks
            bbox = self._bbox_checks(ann, img)

            # seg checks
            seg_ok, decoded_area = self._seg_checks(ann, img)

            # compare ann["area"] vs decoded
            if self.compare_area and decoded_area is not None and "area" in ann and is_finite_num(ann["area"]):
                a0 = float(ann["area"])
                a1 = float(decoded_area)
                # ignore if both are ~0 and we tolerate empty seg
                if not (self.tolerate_empty_seg and a0 <= 0 and a1 <= 0):
                    denom = max(1.0, abs(a1))
                    rel = abs(a0 - a1) / denom
                    if rel > self.area_rel_tol:
                        self.add_issue(
                            "ann_area_mismatch",
                            aid,
                            img_id,
                            f"ann.area={a0} decoded_area={a1} rel_err={rel:.3f}",
                        )

            # duplicate check hash
            self.dup_hashes[self._dup_check_hash(ann)].append(aid)

            # decide keep/filter/fix
            keep = True
            if self.strict:
                # strict mode: any fatal issues => filter
                fatal_kinds = {
                    "ann_missing_bbox",
                    "ann_bad_bbox_format",
                    "ann_bbox_nonfinite",
                    "ann_bbox_nonpositive_wh",
                    "ann_missing_segmentation",
                    "ann_unknown_segmentation_type",
                    "ann_rle_decode_error",
                    "ann_polygon_decode_error",
                    "ann_seg_decoded_area_zero",
                    "ann_image_id_missing",
                    "ann_unknown_category_id",
                }
                # If this ann has any of these issues, drop it.
                # We detect by re-checking counts is hard, so do local logic:
                # simplest: recompute fatal from current checks:
                if bbox is None:
                    keep = False
                else:
                    x, y, w, h = bbox
                    if not finite_array(np.array([x, y, w, h], dtype=np.float64)) or w <= 0 or h <= 0:
                        keep = False
                if not seg_ok and not self.tolerate_empty_seg:
                    keep = False
                if img_id not in self.images_by_id:
                    keep = False
                cid = ann.get("category_id", None)
                if (cid is None) or (cid not in self.cats_by_id):
                    keep = False

            # auto-fix behavior
            if self.auto_fix and img is not None and bbox is not None:
                W, H = img.get("width", None), img.get("height", None)
                if W is not None and H is not None:
                    x, y, w, h = bbox
                    if finite_array(np.array([x, y, w, h], dtype=np.float64)):
                        if self.fix_mode == "clamp":
                            # Clamp bbox into image bounds, preserve positive size with eps
                            eps = 1e-6
                            x1, y1, x2, y2 = bbox_xywh_to_xyxy([x, y, w, h])
                            x1 = clamp(x1, 0.0, float(W))
                            y1 = clamp(y1, 0.0, float(H))
                            x2 = clamp(x2, 0.0, float(W))
                            y2 = clamp(y2, 0.0, float(H))
                            w2 = max(eps, x2 - x1)
                            h2 = max(eps, y2 - y1)
                            ann = copy.deepcopy(ann)
                            ann["bbox"] = [float(x1), float(y1), float(w2), float(h2)]

            if keep:
                self.fixed_annotations.append(ann)
            else:
                self.filtered_out_ann_ids.add(aid)

        # image-level stats: images with no annotations (after filtering)
        for ann in self.fixed_annotations:
            self.img_ann_count[ann["image_id"]] += 1
        for img_id in self.images_by_id.keys():
            if self.img_ann_count[img_id] == 0:
                self.images_with_no_anns.append(img_id)

        # duplicate report
        for k, aids in self.dup_hashes.items():
            if len(aids) > 1:
                self.add_issue("ann_possible_duplicates", aids[0], k[0], f"duplicate-like anns: {aids}")

        # build fixed output
        self.fixed["annotations"] = self.fixed_annotations

    def print_report(self):
        n_imgs = len(self.coco.get("images", []))
        n_anns = len(self.coco.get("annotations", []))
        n_cats = len(self.coco.get("categories", []))
        n_keep = len(self.fixed_annotations)
        n_drop = len(self.filtered_out_ann_ids)

        print("=" * 80)
        print("COCO VALIDATION REPORT")
        print("=" * 80)
        print(f"images: {n_imgs}")
        print(f"annotations: {n_anns}  (kept: {n_keep}, dropped: {n_drop})")
        print(f"categories: {n_cats}")
        if len(self.images_with_no_anns) > 0:
            print(f"images with no annotations (after filter): {len(self.images_with_no_anns)} (example: {self.images_with_no_anns[:10]})")
        print("-" * 80)

        if len(self.issue_counts) == 0:
            print("No issues found ✅")
            return

        print("Issue counts:")
        for kind, cnt in self.issue_counts.most_common():
            print(f"  {kind:32s} : {cnt}")

        print("-" * 80)
        print("Examples (limited):")
        for iss in self.issues:
            print(f"- {iss.kind:28s} ann_id={iss.ann_id} img_id={iss.img_id} :: {iss.detail}")

        print("=" * 80)

    def save_fixed_json(self, out_path: str):
        with open(out_path, "w") as f:
            json.dump(self.fixed, f)
        print(f"Saved fixed COCO to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_json", type=str, help="path to COCO annotations json")
    ap.add_argument("--strict", action="store_true",
                    help="Drop annotations that have fatal issues (recommended for training stability).")
    ap.add_argument("--check-in-bounds", action="store_true",
                    help="Check bbox/polygon points against image bounds (tolerates small +/-1px).")
    ap.add_argument("--tolerate-empty-seg", action="store_true",
                    help="Treat empty/zero-area decoded seg as warning instead of fatal.")
    ap.add_argument("--compare-area", action="store_true",
                    help="Compare ann['area'] vs decoded area from segmentation.")
    ap.add_argument("--area-rel-tol", type=float, default=0.25,
                    help="Relative tolerance for area mismatch (default 0.25).")
    ap.add_argument("--max-examples", type=int, default=20,
                    help="Max example lines printed per issue type.")
    ap.add_argument("--auto-fix", action="store_true",
                    help="Write a fixed JSON (filter/clamp only affects output json, not original).")
    ap.add_argument("--fix-mode", type=str, default="filter", choices=["filter", "clamp"],
                    help="If --auto-fix: 'filter' drops strict-bad anns; 'clamp' also clamps bboxes into image bounds.")
    ap.add_argument("--out", type=str, default=None,
                    help="If set, write fixed json to this path (requires --auto-fix).")
    args = ap.parse_args()

    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    v = CocoValidator(
        coco,
        strict=args.strict,
        check_in_bounds=args.check_in_bounds,
        tolerate_empty_seg=args.tolerate_empty_seg,
        max_examples_per_issue=args.max_examples,
        compare_area=args.compare_area,
        area_rel_tol=args.area_rel_tol,
        auto_fix=args.auto_fix,
        fix_mode=args.fix_mode,
    )
    v.validate()
    v.print_report()

    if args.auto_fix and args.out:
        v.save_fixed_json(args.out)


if __name__ == "__main__":
    main()