# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Run per-image AP/F1/boundary-F mask metrics from COCO prediction/GT files."""

import argparse
import json

from sam3.eval.per_image_mask_metrics import PerImageMaskMetricsEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="Path to prediction JSON (COCO results format).",
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to GT JSON (COCO format, polygons or masks).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold used for boundary-F tolerance sweep.",
    )
    parser.add_argument(
        "--conf_min",
        type=float,
        default=0.4,
        help="Minimum confidence for F1/AP sweep.",
    )
    parser.add_argument(
        "--conf_max",
        type=float,
        default=0.95,
        help="Maximum confidence for F1/AP sweep.",
    )
    parser.add_argument(
        "--conf_step",
        type=float,
        default=0.05,
        help="Step size for confidence sweep.",
    )
    parser.add_argument(
        "--boundary_tol_min_pct",
        type=float,
        default=0.1,
        help="Minimum boundary tolerance in percent of image diagonal.",
    )
    parser.add_argument(
        "--boundary_tol_max_pct",
        type=float,
        default=0.5,
        help="Maximum boundary tolerance in percent of image diagonal.",
    )
    parser.add_argument(
        "--boundary_tol_step_pct",
        type=float,
        default=0.1,
        help="Step for boundary tolerance in percent of image diagonal.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output path for per-image JSON table.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional output path for per-image CSV table.",
    )

    args = parser.parse_args()
    confidence_thresholds = []
    cur = args.conf_min
    while cur <= args.conf_max + 1e-12:
        confidence_thresholds.append(round(cur, 10))
        cur += args.conf_step

    boundary_rel_tolerances = []
    cur = args.boundary_tol_min_pct
    while cur <= args.boundary_tol_max_pct + 1e-12:
        boundary_rel_tolerances.append(round(cur / 100.0, 10))
        cur += args.boundary_tol_step_pct

    evaluator = PerImageMaskMetricsEvaluator(
        gt_path=args.gt_file,
        iou_type="segm",
        threshold=args.threshold,
        confidence_thresholds=confidence_thresholds,
        boundary_rel_tolerances=boundary_rel_tolerances,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    summary = evaluator.evaluate(args.pred_file)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
