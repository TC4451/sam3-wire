# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as mask_utils
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from sam3.eval.demo_eval import DemoEval
from sam3.train.masks_ops import compute_boundary, rle_encode
from sam3.train.utils.distributed import is_main_process


class PerImageMaskMetricsEvaluator:
    """
    Offline evaluator that computes per-image mask metrics and writes them to disk.

    Expected input:
      - Ground truth: COCO JSON with segmentation as polygons or masks.
      - Predictions: COCO results JSON produced by PredictionDumper (segm mode).

    Per-image outputs include:
      - ap: AP@IoU=0.50 from a confidence sweep
      - f1_conf_*: F1 (mean across IoU thresholds [0.50:0.95]) at each confidence
      - boundary_f_tol_*: boundary F at each relative tolerance
      - jaccard_j: Jaccard index (IoU) on matched instances
      - j_and_f: (J + F) / 2 on matched instances
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
    ) -> None:
        if iou_type != "segm":
            raise ValueError(
                f"PerImageMaskMetricsEvaluator only supports iou_type='segm', got {iou_type}"
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
        # Prefer reporting scalar J/F aliases at 0.3% tolerance if available.
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

    def evaluate(self, dumped_file: str) -> Dict[str, float]:
        if not is_main_process():
            return {}

        coco_gt = COCO(self.gt_path)
        preds = self._load_predictions(dumped_file)

        rows = {
            int(img_id): {
                "image_id": int(img_id),
                "file_name": coco_gt.imgs.get(int(img_id), {}).get("file_name"),
            }
            for img_id in coco_gt.getImgIds()
        }

        self._populate_confidence_sweep_metrics(rows, coco_gt, preds)
        self._populate_boundary_sweep_metrics(rows, preds)

        rows = [rows[k] for k in sorted(rows.keys())]
        json_path, csv_path = self._dump_rows(rows, dumped_file)

        logging.info(
            "Per-image mask metrics written to json=%s csv=%s",
            str(json_path) if json_path is not None else "<disabled>",
            str(csv_path) if csv_path is not None else "<disabled>",
        )

        return self._summarize(rows)

    @staticmethod
    def _load_predictions(dumped_file: str) -> List[Dict[str, Any]]:
        with open(dumped_file, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_results_from_preds(coco_gt: COCO, preds: List[Dict[str, Any]]) -> COCO:
        if len(preds) == 0:
            coco_dt = COCO()
            coco_dt.dataset["images"] = list(coco_gt.dataset.get("images", []))
            coco_dt.dataset["categories"] = list(coco_gt.dataset.get("categories", []))
            coco_dt.dataset["annotations"] = []
            coco_dt.createIndex()
            return coco_dt

        return coco_gt.loadRes(preds)

    @staticmethod
    def _range_with_step(start: float, end: float, step: float) -> List[float]:
        values = []
        cur = start
        while cur <= end + 1e-12:
            values.append(round(cur, 10))
            cur += step
        return values

    def _ensure_boundaries(self, coco_api: COCO, rel_tol: float) -> None:
        anns = coco_api.dataset.get("annotations", [])
        if len(anns) == 0:
            return

        for ann in anns:
            rle = coco_api.annToRLE(ann)
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("utf-8")
            ann["segmentation"] = rle

            if "boundary" in ann and "dilated_boundary" in ann:
                continue

            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = torch.from_numpy(mask.astype(np.bool_))

            boundary = compute_boundary(mask)
            kernel_size = self._kernel_size(mask.shape[-2], mask.shape[-1], rel_tol)
            dilated_boundary = self._binary_dilate(boundary, kernel_size)

            ann["boundary"] = rle_encode(boundary.unsqueeze(0))[0]
            ann["dilated_boundary"] = rle_encode(dilated_boundary.unsqueeze(0))[0]

        coco_api.createIndex()

    @staticmethod
    def _kernel_size(h: int, w: int, rel_tol: float) -> int:
        bound_pix = max(1, int(math.ceil(rel_tol * math.sqrt(h * h + w * w))))
        return 2 * bound_pix + 1

    @staticmethod
    def _binary_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        x = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        k = torch.ones((1, 1, kernel_size, kernel_size), dtype=x.dtype)
        y = F.conv2d(x, k, padding=kernel_size // 2)
        return (y[0, 0] > 0).to(dtype=torch.bool)

    @staticmethod
    def _value_or_none(val: Any) -> Optional[float]:
        if val is None:
            return None
        val = float(val)
        if val < 0:
            return None
        return val

    @staticmethod
    def _fmt_conf(thresh: float) -> str:
        return f"{thresh:.2f}".replace(".", "p")

    @staticmethod
    def _fmt_tol(rel_tol: float) -> str:
        # 0.001 -> 0p1pct
        return f"{(100.0 * rel_tol):.1f}".replace(".", "p") + "pct"

    @staticmethod
    def _ap_from_pr(precision: List[float], recall: List[float]) -> Optional[float]:
        if len(precision) == 0 or len(recall) == 0:
            return None

        # Sort by recall so we can integrate precision envelope.
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

    def _populate_confidence_sweep_metrics(
        self, rows: Dict[int, Dict[str, Any]], coco_gt: COCO, preds: List[Dict[str, Any]]
    ) -> None:
        # AP is computed at IoU=0.50 from the PR curve induced by confidence sweep.
        per_image_precision = {img_id: [] for img_id in rows}
        per_image_recall = {img_id: [] for img_id in rows}
        per_image_has_gt = {img_id: False for img_id in rows}

        for conf in self.confidence_thresholds:
            filtered_preds = [p for p in preds if float(p.get("score", 0.0)) >= conf]
            coco_dt = self._load_results_from_preds(coco_gt, filtered_preds)

            demo_eval = DemoEval(
                coco_gt=coco_gt,
                coco_dt=coco_dt,
                iouType=self.iou_type,
                threshold=0.0,  # already thresholded above
                compute_JnF=False,
            )
            demo_eval.params.useCats = False
            demo_eval.params.imgIds = coco_gt.getImgIds()
            demo_eval.evaluate()

            iou_thrs = demo_eval.params.iouThrs
            idx_50 = np.where(np.isclose(iou_thrs, 0.5))[0]
            idx_50 = int(idx_50[0]) if len(idx_50) > 0 else 0
            conf_key = f"f1_conf_{self._fmt_conf(conf)}"

            for res in demo_eval.evalImgs:
                if res is None:
                    continue
                img_id = int(res["image_id"])
                local_f1s = res.get("local_F1s")
                f1_val = None
                if local_f1s is not None:
                    local_f1s = np.asarray(local_f1s, dtype=np.float64)
                    if local_f1s.size > 0:
                        f1_val = float(np.mean(local_f1s))
                rows[img_id][conf_key] = f1_val

                if "TPs" in res and "FPs" in res and "FNs" in res:
                    tp = float(res["TPs"][idx_50])
                    fp = float(res["FPs"][idx_50])
                    fn = float(res["FNs"][idx_50])

                    if tp + fn > 0:
                        per_image_has_gt[img_id] = True
                        precision = tp / (tp + fp + 1e-9)
                        recall = tp / (tp + fn + 1e-9)
                        per_image_precision[img_id].append(precision)
                        per_image_recall[img_id].append(recall)

        for img_id in rows:
            if per_image_has_gt[img_id]:
                rows[img_id]["ap"] = self._ap_from_pr(
                    per_image_precision[img_id], per_image_recall[img_id]
                )
            else:
                rows[img_id]["ap"] = None

    def _populate_boundary_sweep_metrics(
        self, rows: Dict[int, Dict[str, Any]], preds: List[Dict[str, Any]]
    ) -> None:
        # Boundary F is computed at fixed confidence threshold `self.threshold`.
        for rel_tol in self.boundary_rel_tolerances:
            coco_gt = COCO(self.gt_path)
            coco_dt = self._load_results_from_preds(coco_gt, preds)

            self._ensure_boundaries(coco_gt, rel_tol)
            self._ensure_boundaries(coco_dt, rel_tol)

            demo_eval = DemoEval(
                coco_gt=coco_gt,
                coco_dt=coco_dt,
                iouType=self.iou_type,
                threshold=self.threshold,
                compute_JnF=True,
            )
            demo_eval.params.useCats = False
            demo_eval.params.imgIds = coco_gt.getImgIds()
            demo_eval.evaluate()

            tol_key = f"boundary_f_tol_{self._fmt_tol(rel_tol)}"
            j_tol_key = f"jaccard_j_tol_{self._fmt_tol(rel_tol)}"
            jf_tol_key = f"j_and_f_tol_{self._fmt_tol(rel_tol)}"
            for res in demo_eval.evalImgs:
                if res is None:
                    continue
                img_id = int(res["image_id"])
                f_val = self._value_or_none(res.get("F"))
                j_val = self._value_or_none(res.get("J"))
                jf_val = self._value_or_none(res.get("J&F"))
                rows[img_id][tol_key] = f_val
                rows[img_id][j_tol_key] = j_val
                rows[img_id][jf_tol_key] = jf_val

                if np.isclose(rel_tol, self.primary_boundary_rel_tol):
                    rows[img_id]["jaccard_j"] = j_val
                    rows[img_id]["j_and_f"] = jf_val

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
            f"f1_conf_{self._fmt_conf(conf)}" for conf in self.confidence_thresholds
        ]
        boundary_tol_fields = [
            f"boundary_f_tol_{self._fmt_tol(tol)}"
            for tol in self.boundary_rel_tolerances
        ]
        j_tol_fields = [
            f"jaccard_j_tol_{self._fmt_tol(tol)}"
            for tol in self.boundary_rel_tolerances
        ]
        jf_tol_fields = [
            f"j_and_f_tol_{self._fmt_tol(tol)}"
            for tol in self.boundary_rel_tolerances
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
            key = f"f1_conf_{self._fmt_conf(conf)}"
            summary[f"per_image_eval_{key}_mean"] = mean_or_neg1(
                [r.get(key) for r in rows]
            )

        for tol in self.boundary_rel_tolerances:
            boundary_key = f"boundary_f_tol_{self._fmt_tol(tol)}"
            j_key = f"jaccard_j_tol_{self._fmt_tol(tol)}"
            jf_key = f"j_and_f_tol_{self._fmt_tol(tol)}"
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
