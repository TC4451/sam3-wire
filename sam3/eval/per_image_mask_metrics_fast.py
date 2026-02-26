# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

import csv
import json
import logging
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from sam3.train.masks_ops import compute_F_measure, compute_boundary, rle_encode
from sam3.train.utils.distributed import is_main_process
from scipy.optimize import linear_sum_assignment

_BOUNDARY_SHAPE_MISMATCH_WARNED = False


def _decode_mask_rle(rle: Dict[str, Any]) -> torch.Tensor:
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.from_numpy(mask.astype(np.bool_))


def _kernel_size(h: int, w: int, rel_tol: float) -> int:
    bound_pix = max(1, int(math.ceil(rel_tol * math.sqrt(h * h + w * w))))
    return 2 * bound_pix + 1


def _binary_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    k = torch.ones((1, 1, kernel_size, kernel_size), dtype=x.dtype)
    y = F.conv2d(x, k, padding=kernel_size // 2)
    return (y[0, 0] > 0).to(dtype=torch.bool)


def _ap_from_pr(precision: List[float], recall: List[float]) -> Optional[float]:
    if len(precision) == 0 or len(recall) == 0:
        return None

    pairs = sorted(zip(recall, precision), key=lambda x: x[0])
    rec = np.array([p[0] for p in pairs], dtype=np.float64)
    pre = np.array([p[1] for p in pairs], dtype=np.float64)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], pre, [0.0]))

    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changed = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1])
    return float(ap)


def _fmt_conf(thresh: float) -> str:
    return f"{thresh:.2f}".replace(".", "p")


def _fmt_tol(rel_tol: float) -> str:
    # 0.001 -> 0p1pct
    return f"{(100.0 * rel_tol):.1f}".replace(".", "p") + "pct"


def _run_matching(ious: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ious.size == 0 or ious.shape[0] == 0 or ious.shape[1] == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)

    matched_dt, matched_gt = linear_sum_assignment(-ious)
    scores = ious[matched_dt, matched_gt]
    return matched_dt.astype(np.int64), matched_gt.astype(np.int64), scores


def _compute_counts_from_match_scores(
    match_scores: np.ndarray,
    num_dt: int,
    num_gt: int,
    iou_thrs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if match_scores.size == 0:
        tps = np.zeros(iou_thrs.shape[0], dtype=np.int64)
    else:
        tps = np.sum(
            match_scores[:, None] >= iou_thrs[None, :],
            axis=0,
            dtype=np.int64,
        )
    fps = np.array([num_dt - tp for tp in tps], dtype=np.int64)
    fns = np.array([num_gt - tp for tp in tps], dtype=np.int64)

    precision = tps / (tps + fps + 1e-4)
    recall = tps / (tps + fns + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)
    return tps, fps, fns, f1


def _precompute_boundary_cache(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache = []
    for item in items:
        boundary = compute_boundary(_decode_mask_rle(item["segmentation"]))
        cache.append(
            {
                "boundary": boundary,
                "boundary_area": int(boundary.sum().item()),
                "dilated_by_tol": {},
            }
        )
    return cache


def _ensure_dilated_boundary(entry: Dict[str, Any], rel_tol: float) -> Dict[str, Any]:
    cached = entry["dilated_by_tol"].get(rel_tol)
    if cached is not None:
        return cached

    h, w = entry["boundary"].shape[-2:]
    kernel = _kernel_size(h, w, rel_tol)
    dilated = _binary_dilate(entry["boundary"], kernel)
    payload = {"mask": dilated}
    entry["dilated_by_tol"][rel_tol] = payload
    return payload


def _compute_boundary_f_measure(
    gt_boundary: torch.Tensor,
    gt_boundary_area: int,
    gt_dilated_boundary: torch.Tensor,
    dt_boundary: torch.Tensor,
    dt_boundary_area: int,
    dt_dilated_boundary: torch.Tensor,
) -> float:
    global _BOUNDARY_SHAPE_MISMATCH_WARNED
    n_dt = dt_boundary_area
    n_gt = gt_boundary_area

    if n_dt == 0 and n_gt > 0:
        precision = 1.0
        recall = 0.0
    elif n_dt > 0 and n_gt == 0:
        precision = 0.0
        recall = 1.0
    elif n_dt == 0 and n_gt == 0:
        precision = 1.0
        recall = 1.0
    else:
        # Some prediction masks may come with inconsistent spatial sizes.
        # Fast boolean intersection requires exact shape alignment.
        if (
            dt_boundary.shape != gt_dilated_boundary.shape
            or gt_boundary.shape != dt_dilated_boundary.shape
        ):
            if not _BOUNDARY_SHAPE_MISMATCH_WARNED:
                logging.warning(
                    "PerImageMaskMetricsFastEvaluator: boundary shape mismatch "
                    "detected (dt=%s gt_dilated=%s gt=%s dt_dilated=%s). "
                    "Falling back to RLE boundary F for these pairs.",
                    tuple(dt_boundary.shape),
                    tuple(gt_dilated_boundary.shape),
                    tuple(gt_boundary.shape),
                    tuple(dt_dilated_boundary.shape),
                )
                _BOUNDARY_SHAPE_MISMATCH_WARNED = True
            try:
                return float(
                    compute_F_measure(
                        gt_boundary_rle=rle_encode(gt_boundary.unsqueeze(0))[0],
                        gt_dilated_boundary_rle=rle_encode(
                            gt_dilated_boundary.unsqueeze(0)
                        )[0],
                        dt_boundary_rle=rle_encode(dt_boundary.unsqueeze(0))[0],
                        dt_dilated_boundary_rle=rle_encode(
                            dt_dilated_boundary.unsqueeze(0)
                        )[0],
                    )
                )
            except Exception:
                # Keep evaluation robust even for malformed masks.
                return 0.0

        dt_match = int(torch.logical_and(dt_boundary, gt_dilated_boundary).sum().item())
        gt_match = int(torch.logical_and(gt_boundary, dt_dilated_boundary).sum().item())
        precision = dt_match / float(n_dt)
        recall = gt_match / float(n_gt)

    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _init_worker_torch_threads(torch_num_threads_per_worker: Optional[int]) -> None:
    if torch_num_threads_per_worker is None:
        return

    try:
        torch.set_num_threads(int(torch_num_threads_per_worker))
    except Exception:
        logging.warning(
            "PerImageMaskMetricsFastEvaluator: failed to set torch num threads to %s",
            str(torch_num_threads_per_worker),
        )

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _compute_one_image(
    payload: Dict[str, Any],
    confidence_thresholds: List[float],
    boundary_rel_tolerances: List[float],
    primary_boundary_rel_tol: float,
    score_threshold: float,
    iou_thrs: List[float],
) -> Dict[str, Any]:
    iou_thrs_np = np.asarray(iou_thrs, dtype=np.float64)
    idx_50 = np.where(np.isclose(iou_thrs_np, 0.5))[0]
    idx_50 = int(idx_50[0]) if len(idx_50) > 0 else 0

    row: Dict[str, Any] = {
        "image_id": int(payload["image_id"]),
        "file_name": payload.get("file_name"),
    }

    gt_all = payload.get("gt", [])
    gt = [g for g in gt_all if not g.get("ignore", False)]
    dt_all = list(payload.get("dt", []))
    dt_all.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    num_gt = len(gt)
    dt_scores = np.asarray([float(d.get("score", 0.0)) for d in dt_all], dtype=np.float64)

    if len(dt_all) > 0 and num_gt > 0:
        dt_segm = [d["segmentation"] for d in dt_all]
        gt_segm = [g["segmentation"] for g in gt]
        iscrowd = [int(bool(g.get("iscrowd", 0))) for g in gt]
        ious_all = mask_utils.iou(dt_segm, gt_segm, iscrowd)
        if ious_all.size == 0:
            ious_all = np.zeros((len(dt_all), num_gt), dtype=np.float64)
    else:
        ious_all = np.zeros((len(dt_all), num_gt), dtype=np.float64)

    # Confidence sweep metrics (f1_conf_* + per-image AP@0.5)
    per_image_precision: List[float] = []
    per_image_recall: List[float] = []
    match_scores_cache: Dict[int, np.ndarray] = {}
    count_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for conf in confidence_thresholds:
        conf_key = f"f1_conf_{_fmt_conf(conf)}"

        if len(dt_all) == 0:
            num_dt = 0
        else:
            num_dt = int(np.count_nonzero(dt_scores >= conf))

        if num_gt == 0 and num_dt == 0:
            row[conf_key] = None
            continue

        match_scores = match_scores_cache.get(num_dt)
        if match_scores is None:
            ious = (
                ious_all[:num_dt, :]
                if num_dt > 0
                else np.zeros((0, num_gt), dtype=np.float64)
            )
            _, _, match_scores = _run_matching(ious)
            match_scores_cache[num_dt] = match_scores

        cached_counts = count_cache.get(num_dt)
        if cached_counts is None:
            cached_counts = _compute_counts_from_match_scores(
                match_scores=match_scores,
                num_dt=num_dt,
                num_gt=num_gt,
                iou_thrs=iou_thrs_np,
            )
            count_cache[num_dt] = cached_counts
        tps, fps, fns, f1 = cached_counts
        row[conf_key] = float(np.mean(f1)) if f1.size > 0 else None

        if num_gt > 0:
            tp_50 = float(tps[idx_50])
            fp_50 = float(fps[idx_50])
            fn_50 = float(fns[idx_50])
            per_image_precision.append(tp_50 / (tp_50 + fp_50 + 1e-9))
            per_image_recall.append(tp_50 / (tp_50 + fn_50 + 1e-9))

    row["ap"] = _ap_from_pr(per_image_precision, per_image_recall) if num_gt > 0 else None

    # Boundary/J/J&F sweep at fixed score threshold
    if len(dt_all) == 0:
        num_dt_boundary = 0
    else:
        num_dt_boundary = int(np.count_nonzero(dt_scores >= score_threshold))

    dt_boundary = dt_all[:num_dt_boundary]
    ious_boundary = (
        ious_all[:num_dt_boundary, :]
        if num_dt_boundary > 0
        else np.zeros((0, num_gt), dtype=np.float64)
    )
    matched_dt, matched_gt, match_scores = _run_matching(ious_boundary)

    for rel_tol in boundary_rel_tolerances:
        tol_key = f"boundary_f_tol_{_fmt_tol(rel_tol)}"
        j_tol_key = f"jaccard_j_tol_{_fmt_tol(rel_tol)}"
        jf_tol_key = f"j_and_f_tol_{_fmt_tol(rel_tol)}"
        row[tol_key] = None
        row[j_tol_key] = None
        row[jf_tol_key] = None

    row["jaccard_j"] = None
    row["j_and_f"] = None

    if len(match_scores) > 0:
        j_score = float(np.mean(match_scores))

        gt_cache = _precompute_boundary_cache(gt)
        dt_cache = _precompute_boundary_cache(dt_boundary)

        matched_gt_ids = {int(i) for i in matched_gt.tolist()}
        matched_dt_ids = {int(i) for i in matched_dt.tolist()}
        for rel_tol in boundary_rel_tolerances:
            for idx in matched_gt_ids:
                _ensure_dilated_boundary(gt_cache[idx], rel_tol)
            for idx in matched_dt_ids:
                _ensure_dilated_boundary(dt_cache[idx], rel_tol)

        for rel_tol in boundary_rel_tolerances:
            tol_key = f"boundary_f_tol_{_fmt_tol(rel_tol)}"
            j_tol_key = f"jaccard_j_tol_{_fmt_tol(rel_tol)}"
            jf_tol_key = f"j_and_f_tol_{_fmt_tol(rel_tol)}"

            f_sum = 0.0
            for dt_id, gt_id in zip(matched_dt.tolist(), matched_gt.tolist()):
                gt_entry = gt_cache[gt_id]
                dt_entry = dt_cache[dt_id]
                gt_dilated = _ensure_dilated_boundary(gt_entry, rel_tol)
                dt_dilated = _ensure_dilated_boundary(dt_entry, rel_tol)
                f_sum += _compute_boundary_f_measure(
                    gt_boundary=gt_entry["boundary"],
                    gt_boundary_area=gt_entry["boundary_area"],
                    gt_dilated_boundary=gt_dilated["mask"],
                    dt_boundary=dt_entry["boundary"],
                    dt_boundary_area=dt_entry["boundary_area"],
                    dt_dilated_boundary=dt_dilated["mask"],
                )

            f_score = float(f_sum / (len(match_scores) + 1e-9))
            jf_score = float((j_score + f_score) * 0.5)

            row[tol_key] = f_score
            row[j_tol_key] = j_score
            row[jf_tol_key] = jf_score

            if np.isclose(rel_tol, primary_boundary_rel_tol):
                row["jaccard_j"] = j_score
                row["j_and_f"] = jf_score

    return row


class PerImageMaskMetricsFastEvaluator:
    """
    Faster drop-in evaluator that preserves the same output schema as
    PerImageMaskMetricsEvaluator while using:
      - per-image multiprocessing
      - cached per-image IoU matrix across confidence thresholds
      - cached per-image boundaries and dilations across tolerance sweep
    """

    def __init__(
        self,
        gt_path: str,
        iou_type: str = "segm",
        threshold: float = 0.5,
        confidence_thresholds: Optional[List[float]] = None,
        boundary_rel_tolerances: Optional[List[float]] = None,
        boundary_rel_tol: Optional[float] = None,
        output_json: Optional[str] = None,
        output_csv: Optional[str] = None,
        num_workers: Optional[int] = None,
        chunk_size: int = 8,
        multiprocessing_context: Optional[str] = None,
        torch_num_threads_per_worker: Optional[int] = 1,
        progress_log_every_n: int = 25,
    ) -> None:
        if iou_type != "segm":
            raise ValueError(
                f"PerImageMaskMetricsFastEvaluator only supports iou_type='segm', got {iou_type}"
            )
        self.gt_path = gt_path
        self.iou_type = iou_type
        self.threshold = threshold
        self.confidence_thresholds = (
            confidence_thresholds
            if confidence_thresholds is not None
            else self._range_with_step(0.4, 0.95, 0.05)
        )
        if boundary_rel_tolerances is not None:
            self.boundary_rel_tolerances = boundary_rel_tolerances
        elif boundary_rel_tol is not None:
            self.boundary_rel_tolerances = [float(boundary_rel_tol)]
        else:
            self.boundary_rel_tolerances = self._range_with_step(0.001, 0.005, 0.001)

        self.primary_boundary_rel_tol = (
            0.003
            if any(np.isclose(t, 0.003) for t in self.boundary_rel_tolerances)
            else float(self.boundary_rel_tolerances[0])
        )

        self.output_json = output_json
        self.output_csv = output_csv

        if len(self.confidence_thresholds) == 0:
            raise ValueError("confidence_thresholds must be non-empty")
        if len(self.boundary_rel_tolerances) == 0:
            raise ValueError("boundary_rel_tolerances must be non-empty")

        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            self.num_workers = max(1, min(8, cpu_count))
        else:
            self.num_workers = max(1, int(num_workers))

        self.chunk_size = max(1, int(chunk_size))
        self.multiprocessing_context = multiprocessing_context
        self.torch_num_threads_per_worker = (
            None
            if torch_num_threads_per_worker is None
            else max(1, int(torch_num_threads_per_worker))
        )
        self.progress_log_every_n = max(0, int(progress_log_every_n))

        if self.multiprocessing_context is not None:
            supported_contexts = set(mp.get_all_start_methods())
            if self.multiprocessing_context not in supported_contexts:
                raise ValueError(
                    "Unsupported multiprocessing_context="
                    f"{self.multiprocessing_context!r}. "
                    f"Supported: {sorted(supported_contexts)}"
                )

    @staticmethod
    def _range_with_step(start: float, end: float, step: float) -> List[float]:
        values = []
        cur = start
        while cur <= end + 1e-12:
            values.append(round(cur, 10))
            cur += step
        return values

    @staticmethod
    def _decode_counts_if_needed(rle: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(rle.get("counts"), bytes):
            rle = dict(rle)
            rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    @classmethod
    def _normalize_rle_to_image_size(
        cls, rle: Dict[str, Any], img_h: int, img_w: int
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize one RLE to expected image size (img_h, img_w).

        Returns normalized RLE, or None if the mask is malformed/unfixable.
        """
        rle = cls._decode_counts_if_needed(rle)
        size = rle.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            return None

        try:
            rle_h, rle_w = int(size[0]), int(size[1])
        except Exception:
            return None

        if (rle_h, rle_w) == (img_h, img_w):
            return rle

        try:
            decoded = mask_utils.decode(rle)
        except Exception:
            return None

        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        decoded = decoded.astype(np.bool_)

        if decoded.shape == (img_h, img_w):
            fixed = rle_encode(torch.from_numpy(decoded).unsqueeze(0))[0]
            return cls._decode_counts_if_needed(fixed)

        if decoded.shape == (img_w, img_h):
            decoded_t = np.ascontiguousarray(decoded.T)
            fixed = rle_encode(torch.from_numpy(decoded_t).unsqueeze(0))[0]
            return cls._decode_counts_if_needed(fixed)

        return None

    @classmethod
    def _coerce_pred_segm_to_rle(
        cls, segm: Any, img_h: int, img_w: int
    ) -> Optional[Dict[str, Any]]:
        if segm is None:
            return None

        if isinstance(segm, dict):
            if "counts" in segm and "size" in segm:
                return cls._decode_counts_if_needed(segm)

            # Uncompressed RLE-like dict
            if isinstance(segm.get("counts"), list):
                rle = mask_utils.frPyObjects(segm, img_h, img_w)
                if isinstance(rle, list):
                    rle = mask_utils.merge(rle)
                return cls._decode_counts_if_needed(rle)

        if isinstance(segm, list):
            # Polygon list
            rles = mask_utils.frPyObjects(segm, img_h, img_w)
            if isinstance(rles, list):
                rle = mask_utils.merge(rles)
            else:
                rle = rles
            return cls._decode_counts_if_needed(rle)

        return None

    @staticmethod
    def _load_predictions(dumped_file: str) -> List[Dict[str, Any]]:
        with open(dumped_file, "r") as f:
            return json.load(f)

    def _collect_rows_with_progress(
        self, rows_iter, total: int, mode_name: str
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if total <= 0:
            return rows

        start_t = time.time()
        next_log = self.progress_log_every_n if self.progress_log_every_n > 0 else None
        for done, row in enumerate(rows_iter, start=1):
            rows.append(row)
            should_log = done == total
            if (
                next_log is not None
                and done >= next_log
            ):
                should_log = True
                while next_log is not None and done >= next_log:
                    next_log += self.progress_log_every_n

            if should_log:
                elapsed = max(1e-6, time.time() - start_t)
                rate = done / elapsed
                eta = (total - done) / max(rate, 1e-6)
                logging.info(
                    "PerImageMaskMetricsFastEvaluator: %s progress %d/%d (%.1f%%) "
                    "rate=%.2f img/s eta=%.1fs",
                    mode_name,
                    done,
                    total,
                    (100.0 * done) / float(total),
                    rate,
                    eta,
                )
        return rows

    def _build_payloads(
        self, coco_gt: COCO, preds: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        img_ids = [int(i) for i in coco_gt.getImgIds()]

        gt_fixed_rle_size = 0
        gt_skipped_invalid_rle = 0
        dt_fixed_rle_size = 0
        dt_skipped_invalid_rle = 0

        gt_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in img_ids}
        for ann in coco_gt.dataset.get("annotations", []):
            img_id = int(ann["image_id"])
            if img_id not in gt_by_img:
                continue

            img_info = coco_gt.imgs.get(img_id, {})
            img_h = img_info.get("height", None)
            img_w = img_info.get("width", None)
            if img_h is None or img_w is None:
                gt_skipped_invalid_rle += 1
                continue

            rle = coco_gt.annToRLE(ann)
            rle = self._decode_counts_if_needed(rle)
            orig_size = tuple(rle.get("size", []))
            rle = self._normalize_rle_to_image_size(
                rle, img_h=int(img_h), img_w=int(img_w)
            )
            if rle is None:
                gt_skipped_invalid_rle += 1
                continue
            if tuple(rle.get("size", [])) != orig_size:
                gt_fixed_rle_size += 1

            iscrowd = bool(ann.get("iscrowd", 0))
            ignore = bool(ann.get("ignore", 0)) or iscrowd

            gt_by_img[img_id].append(
                {
                    "segmentation": rle,
                    "ignore": ignore,
                    "iscrowd": iscrowd,
                }
            )

        dt_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in img_ids}
        for pred in preds:
            if "image_id" not in pred:
                continue
            img_id = int(pred["image_id"])
            if img_id not in dt_by_img:
                continue

            img_info = coco_gt.imgs.get(img_id, {})
            img_h = img_info.get("height", None)
            img_w = img_info.get("width", None)
            if img_h is None or img_w is None:
                continue

            rle = self._coerce_pred_segm_to_rle(
                pred.get("segmentation"), img_h=img_h, img_w=img_w
            )
            if rle is None:
                dt_skipped_invalid_rle += 1
                continue
            orig_size = tuple(rle.get("size", []))
            rle = self._normalize_rle_to_image_size(
                rle, img_h=int(img_h), img_w=int(img_w)
            )
            if rle is None:
                dt_skipped_invalid_rle += 1
                continue
            if tuple(rle.get("size", [])) != orig_size:
                dt_fixed_rle_size += 1

            dt_by_img[img_id].append(
                {
                    "segmentation": rle,
                    "score": float(pred.get("score", 0.0)),
                }
            )

        if gt_fixed_rle_size > 0 or dt_fixed_rle_size > 0:
            logging.warning(
                "PerImageMaskMetricsFastEvaluator: normalized RLE size mismatch "
                "for gt=%d pred=%d masks.",
                gt_fixed_rle_size,
                dt_fixed_rle_size,
            )
        if gt_skipped_invalid_rle > 0 or dt_skipped_invalid_rle > 0:
            logging.warning(
                "PerImageMaskMetricsFastEvaluator: skipped invalid/unfixable RLE "
                "for gt=%d pred=%d masks.",
                gt_skipped_invalid_rle,
                dt_skipped_invalid_rle,
            )

        payloads = []
        for img_id in sorted(img_ids):
            payloads.append(
                {
                    "image_id": img_id,
                    "file_name": coco_gt.imgs.get(img_id, {}).get("file_name"),
                    "gt": gt_by_img.get(img_id, []),
                    "dt": dt_by_img.get(img_id, []),
                }
            )

        return payloads

    def evaluate(self, dumped_file: str) -> Dict[str, float]:
        if not is_main_process():
            return {}

        coco_gt = COCO(self.gt_path)
        preds = self._load_predictions(dumped_file)
        payloads = self._build_payloads(coco_gt, preds)

        worker_args = {
            "confidence_thresholds": [float(c) for c in self.confidence_thresholds],
            "boundary_rel_tolerances": [float(t) for t in self.boundary_rel_tolerances],
            "primary_boundary_rel_tol": float(self.primary_boundary_rel_tol),
            "score_threshold": float(self.threshold),
            "iou_thrs": [float(t) for t in np.arange(0.5, 0.95 + 1e-9, 0.05)],
        }

        logging.info(
            "PerImageMaskMetricsFastEvaluator: images=%d workers=%d chunk_size=%d "
            "mp_context=%s torch_threads_per_worker=%s progress_log_every_n=%d",
            len(payloads),
            self.num_workers,
            self.chunk_size,
            self.multiprocessing_context if self.multiprocessing_context is not None else "auto",
            (
                str(self.torch_num_threads_per_worker)
                if self.torch_num_threads_per_worker is not None
                else "default"
            ),
            self.progress_log_every_n,
        )

        if self.num_workers <= 1 or len(payloads) <= 1:
            serial_fn = partial(_compute_one_image, **worker_args)
            computed_rows = self._collect_rows_with_progress(
                (serial_fn(p) for p in payloads),
                total=len(payloads),
                mode_name="single-process",
            )
        else:
            computed_rows = None
            worker_fn = partial(_compute_one_image, **worker_args)

            if self.multiprocessing_context is not None:
                contexts_to_try = [self.multiprocessing_context]
            else:
                contexts_to_try = [
                    c
                    for c in ("fork", "forkserver", "spawn")
                    if c in set(mp.get_all_start_methods())
                ]

            for ctx_name in contexts_to_try:
                try:
                    mp_ctx = mp.get_context(ctx_name)
                    with ProcessPoolExecutor(
                        max_workers=self.num_workers,
                        mp_context=mp_ctx,
                        initializer=_init_worker_torch_threads,
                        initargs=(self.torch_num_threads_per_worker,),
                    ) as ex:
                        computed_rows = self._collect_rows_with_progress(
                            ex.map(worker_fn, payloads, chunksize=self.chunk_size),
                            total=len(payloads),
                            mode_name=f"parallel[{ctx_name}]",
                        )
                    break
                except Exception as e:
                    logging.warning(
                        "PerImageMaskMetricsFastEvaluator: parallel execution with "
                        "context '%s' failed (%s).",
                        ctx_name,
                        str(e),
                    )

            if computed_rows is None:
                logging.warning(
                    "PerImageMaskMetricsFastEvaluator: all parallel contexts failed. "
                    "Falling back to single-process execution."
                )
                serial_fn = partial(_compute_one_image, **worker_args)
                computed_rows = self._collect_rows_with_progress(
                    (serial_fn(p) for p in payloads),
                    total=len(payloads),
                    mode_name="single-process-fallback",
                )

        rows = [r for r in sorted(computed_rows, key=lambda x: int(x["image_id"]))]
        json_path, csv_path = self._dump_rows(rows, dumped_file)

        logging.info(
            "Per-image fast mask metrics written to json=%s csv=%s",
            str(json_path) if json_path is not None else "<disabled>",
            str(csv_path) if csv_path is not None else "<disabled>",
        )

        return self._summarize(rows)

    def _dump_rows(
        self, rows: List[Dict[str, Any]], dumped_file: str
    ) -> Tuple[Optional[Path], Optional[Path]]:
        base = Path(dumped_file)
        json_path = (
            Path(self.output_json)
            if self.output_json is not None
            else base.with_name(f"{base.stem}_per_image_metrics.json")
        )
        csv_path = (
            Path(self.output_csv)
            if self.output_csv is not None
            else base.with_name(f"{base.stem}_per_image_metrics.csv")
        )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with json_path.open("w") as f:
            json.dump(rows, f, indent=2)

        base_fields = ["image_id", "file_name", "ap", "jaccard_j", "j_and_f"]
        conf_fields = [
            f"f1_conf_{_fmt_conf(conf)}" for conf in self.confidence_thresholds
        ]
        boundary_tol_fields = [
            f"boundary_f_tol_{_fmt_tol(tol)}" for tol in self.boundary_rel_tolerances
        ]
        j_tol_fields = [
            f"jaccard_j_tol_{_fmt_tol(tol)}" for tol in self.boundary_rel_tolerances
        ]
        jf_tol_fields = [
            f"j_and_f_tol_{_fmt_tol(tol)}" for tol in self.boundary_rel_tolerances
        ]

        fieldnames = (
            base_fields
            + conf_fields
            + boundary_tol_fields
            + j_tol_fields
            + jf_tol_fields
        )
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return json_path, csv_path

    def _summarize(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        def mean_or_neg1(values: List[Optional[float]]) -> float:
            vals = [float(v) for v in values if v is not None]
            return float(np.mean(vals)) if len(vals) > 0 else -1.0

        summary = {
            "per_image_eval_num_images": float(len(rows)),
            "per_image_eval_ap_mean": mean_or_neg1([r.get("ap") for r in rows]),
            "per_image_eval_jaccard_j_mean": mean_or_neg1(
                [r.get("jaccard_j") for r in rows]
            ),
            "per_image_eval_j_and_f_mean": mean_or_neg1(
                [r.get("j_and_f") for r in rows]
            ),
        }

        for conf in self.confidence_thresholds:
            key = f"f1_conf_{_fmt_conf(conf)}"
            summary[f"per_image_eval_{key}_mean"] = mean_or_neg1(
                [r.get(key) for r in rows]
            )

        for tol in self.boundary_rel_tolerances:
            boundary_key = f"boundary_f_tol_{_fmt_tol(tol)}"
            j_key = f"jaccard_j_tol_{_fmt_tol(tol)}"
            jf_key = f"j_and_f_tol_{_fmt_tol(tol)}"
            summary[f"per_image_eval_{boundary_key}_mean"] = mean_or_neg1(
                [r.get(boundary_key) for r in rows]
            )
            summary[f"per_image_eval_{j_key}_mean"] = mean_or_neg1(
                [r.get(j_key) for r in rows]
            )
            summary[f"per_image_eval_{jf_key}_mean"] = mean_or_neg1(
                [r.get(jf_key) for r in rows]
            )

        return summary
