# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""
Streaming prediction dumper for eval-time progress metrics.

This keeps the standard prediction dump + offline evaluators behavior, while also
reporting lightweight partial metrics periodically via `compute()` so they show up
in `ProgressMeter.display(...)` during validation.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from sam3.eval.coco_eval_offline import COCOevalCustom
from sam3.eval.coco_writer import PredictionDumper
from sam3.eval.demo_eval import DemoEval
from sam3.train.utils.distributed import all_gather, is_main_process


class StreamingPredictionDumper(PredictionDumper):
    def __init__(
        self,
        dump_dir: str,
        postprocessor,
        maxdets: int,
        iou_type: str,
        gather_pred_via_filesys: bool = False,
        merge_predictions: bool = False,
        pred_file_evaluators: Optional[Any] = None,
        stream_gt_path: Optional[str] = None,
        stream_eval_threshold: float = 0.5,
        stream_eval_every_n_updates: int = 25,
        stream_eval_min_images: int = 32,
        stream_eval_compute_coco: bool = True,
        stream_eval_compute_demo: bool = True,
        stream_global_gather: bool = False,
    ) -> None:
        self.stream_gt_path = stream_gt_path
        self.stream_eval_threshold = float(stream_eval_threshold)
        self.stream_eval_every_n_updates = max(1, int(stream_eval_every_n_updates))
        self.stream_eval_min_images = max(1, int(stream_eval_min_images))
        self.stream_eval_compute_coco = bool(stream_eval_compute_coco)
        self.stream_eval_compute_demo = bool(stream_eval_compute_demo)
        self.stream_global_gather = bool(stream_global_gather)

        self._stream_update_count = 0
        self._stream_last_eval_update = -1
        self._stream_cached_metrics: Dict[str, float] = {}
        self._stream_coco_gt: Optional[COCO] = None
        self._stream_total_images: Optional[int] = None
        self._stream_warned = False

        super().__init__(
            dump_dir=dump_dir,
            postprocessor=postprocessor,
            maxdets=maxdets,
            iou_type=iou_type,
            gather_pred_via_filesys=gather_pred_via_filesys,
            merge_predictions=merge_predictions,
            pred_file_evaluators=pred_file_evaluators,
        )

    def reset(self):
        super().reset()
        self._stream_update_count = 0
        self._stream_last_eval_update = -1
        self._stream_cached_metrics = {}

    def update(self, *args, **kwargs):
        self._stream_update_count += 1
        return super().update(*args, **kwargs)

    def compute(self):
        should_eval = self._should_recompute_stream_metrics()
        gathered_preds = None
        if should_eval and self.stream_global_gather:
            gathered_preds = self._gather_predictions_for_eval()

        if not is_main_process():
            return {}

        out = {
            "stream_updates": float(self._stream_update_count),
            "stream_local_preds": float(len(self.dump)),
        }

        if should_eval:
            preds = gathered_preds if gathered_preds is not None else list(self.dump)
            try:
                self._stream_cached_metrics = self._compute_stream_metrics(preds)
            except Exception as e:
                if not self._stream_warned:
                    logging.warning(
                        "StreamingPredictionDumper: partial metric computation failed (%s).",
                        str(e),
                    )
                    self._stream_warned = True
            self._stream_last_eval_update = self._stream_update_count

        out.update(self._stream_cached_metrics)
        return out

    def _should_recompute_stream_metrics(self) -> bool:
        if self._stream_update_count <= 0:
            return False
        if self._stream_last_eval_update < 0:
            return True
        return (
            self._stream_update_count - self._stream_last_eval_update
            >= self.stream_eval_every_n_updates
        )

    def _gather_predictions_for_eval(self) -> List[Dict[str, Any]]:
        gathered = all_gather(self.dump, force_cpu=True)
        merged: List[Dict[str, Any]] = []
        for rank_preds in gathered:
            merged.extend(rank_preds)
        return merged

    def _ensure_stream_gt(self) -> None:
        if self._stream_coco_gt is not None or self.stream_gt_path is None:
            return
        self._stream_coco_gt = COCO(self.stream_gt_path)
        self._stream_total_images = len(self._stream_coco_gt.getImgIds())

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

    def _compute_stream_metrics(self, preds: List[Dict[str, Any]]) -> Dict[str, float]:
        seen_img_ids = sorted({int(p["image_id"]) for p in preds if "image_id" in p})
        num_seen = len(seen_img_ids)

        out: Dict[str, float] = {
            "stream_seen_images": float(num_seen),
            "stream_seen_fraction": -1.0,
        }

        self._ensure_stream_gt()
        if self._stream_total_images is not None and self._stream_total_images > 0:
            out["stream_total_images"] = float(self._stream_total_images)
            out["stream_seen_fraction"] = float(num_seen / self._stream_total_images)
        else:
            out["stream_total_images"] = -1.0

        if self._stream_coco_gt is None:
            return out
        if num_seen < self.stream_eval_min_images:
            return out

        coco_dt = self._load_results_from_preds(self._stream_coco_gt, preds)

        if self.stream_eval_compute_coco:
            coco_eval = COCOevalCustom(self._stream_coco_gt, coco_dt, iouType=self.iou_type)
            coco_eval.params.imgIds = seen_img_ids
            coco_eval.params.useCats = False
            coco_eval.evaluate()
            coco_eval.accumulate()
            if len(coco_eval.stats) >= 2:
                out["stream_partial_coco_ap"] = float(coco_eval.stats[0])
                out["stream_partial_coco_ap50"] = float(coco_eval.stats[1])

        if self.stream_eval_compute_demo:
            demo_eval = DemoEval(
                coco_gt=self._stream_coco_gt,
                coco_dt=coco_dt,
                iouType=self.iou_type,
                threshold=self.stream_eval_threshold,
                compute_JnF=False,
            )
            demo_eval.params.useCats = False
            demo_eval.params.imgIds = seen_img_ids
            demo_eval.evaluate()
            demo_eval.accumulate()

            iou_thrs = np.asarray(demo_eval.params.iouThrs, dtype=np.float64)
            idx_50 = np.where(np.isclose(iou_thrs, 0.5))[0]
            idx_50 = int(idx_50[0]) if len(idx_50) > 0 else 0

            f1_micro = np.asarray(
                demo_eval.eval.get("positive_micro_F1", []), dtype=np.float64
            )
            f1_macro = np.asarray(
                demo_eval.eval.get("positive_macro_F1", []), dtype=np.float64
            )
            if f1_micro.size > 0:
                out["stream_partial_demo_f1_iou50"] = float(f1_micro[idx_50])
                out["stream_partial_demo_f1_mean"] = float(np.mean(f1_micro))
            if f1_macro.size > 0:
                out["stream_partial_demo_f1_macro"] = float(np.mean(f1_macro))
            if "IL_MCC" in demo_eval.eval:
                out["stream_partial_demo_il_mcc"] = float(demo_eval.eval["IL_MCC"])

        return out
