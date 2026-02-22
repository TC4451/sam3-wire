#!/usr/bin/env python3
"""Standalone synthetic-wire evaluation runner for SLURM usage.

This script mirrors the custom eval shell flow:
1) resolve dataset/COCO/image roots
2) generate runtime eval config from synthetic_wire_eval.yaml
3) run sam3/train/train.py in eval mode (local launcher)
4) write bbox-free summary json from val_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA_ROOT = (
    "/n/holylabs/ydu_lab/Lab/zilin/wire_segmentation/"
    "wire_segmentation_data_unzipped"
)


def _step_msg(step: int, total: int, msg: str, start_ts: float) -> None:
    elapsed = int(time.time() - start_ts)
    print(f"[{step}/{total}] {msg} (elapsed: {elapsed}s)", flush=True)


def _extract_first_int(raw: str) -> Optional[int]:
    match = re.search(r"\d+", raw)
    if not match:
        return None
    return int(match.group(0))


def _detect_num_gpus(explicit: Optional[int]) -> int:
    if explicit is not None and explicit > 0:
        return explicit

    env_candidates = [
        os.environ.get("NUM_GPUS"),
        os.environ.get("SLURM_GPUS_ON_NODE"),
        os.environ.get("SLURM_GPUS_PER_NODE"),
        os.environ.get("SLURM_GPUS"),
    ]
    for raw in env_candidates:
        if not raw:
            continue
        parsed = _extract_first_int(raw)
        if parsed is not None and parsed > 0:
            return parsed
    return 1


def _resolve_coco_json(data_root: Path) -> Path:
    p1 = data_root / "coco" / "annotations.json"
    p2 = data_root / "coco_all.json"
    if p1.is_file():
        return p1
    if p2.is_file():
        return p2
    return p1


def _infer_img_root(coco_json: Path, data_root: Path) -> Path:
    with coco_json.open("r") as f:
        coco = json.load(f)
    images = coco.get("images", [])
    first_name = images[0].get("file_name", "") if images else ""
    if "/" in first_name:
        return data_root
    rgb_root = data_root / "rgb"
    return rgb_root if rgb_root.is_dir() else data_root


def _rewrite_runtime_config(
    base_config_file: Path,
    runtime_config_file: Path,
    experiment_dir: Path,
    coco_json: Path,
    img_root: Path,
) -> None:
    with base_config_file.open("r") as f:
        lines = f.readlines()

    rewritten: List[str] = []
    for line in lines:
        if line.startswith("  experiment_log_dir: "):
            rewritten.append(f"  experiment_log_dir: {experiment_dir}\n")
        elif line.startswith("  coco_gt: "):
            rewritten.append(f"  coco_gt: {coco_json}\n")
        elif line.startswith("  img_path: "):
            rewritten.append(f"  img_path: {img_root}\n")
        else:
            rewritten.append(line)

    runtime_config_file.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_file.open("w") as f:
        f.writelines(rewritten)


def _generate_coco_if_needed(python_bin: str, repo_root: Path, data_root: Path, coco_json: Path) -> None:
    coco_json.parent.mkdir(parents=True, exist_ok=True)
    convert_script = repo_root / "scripts" / "convert_mask_to_coco.py"
    cmd = [python_bin, str(convert_script), str(data_root), str(coco_json)]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _sanity_dump(coco_json: Path, runtime_config_file: Path) -> None:
    with coco_json.open("r") as f:
        coco = json.load(f)
    images = len(coco.get("images", []))
    anns = len(coco.get("annotations", []))
    cats = [c.get("name", "<unnamed>") for c in coco.get("categories", [])]

    cfg_text = runtime_config_file.read_text()
    has_cuda = "accelerator: cuda" in cfg_text
    has_nccl = "backend: nccl" in cfg_text
    has_cable_prompt = '"name": "cable"' in cfg_text

    print(
        f"Sanity: images={images} annotations={anns} categories={cats[:5]}",
        flush=True,
    )
    print(
        f"Sanity: accelerator_cuda={has_cuda} backend_nccl={has_nccl} cable_prompt={has_cable_prompt}",
        flush=True,
    )
    print(f"Sanity: python_executable={sys.executable}", flush=True)


def _run_eval(
    python_bin: str,
    repo_root: Path,
    runtime_config_path: str,
    num_gpus: int,
) -> None:
    train_script = repo_root / "sam3" / "train" / "train.py"
    cmd = [
        python_bin,
        str(train_script),
        "-c",
        runtime_config_path,
        "--use-cluster",
        "0",
        "--num-gpus",
        str(num_gpus),
        "--num-nodes",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _postprocess_summary(val_stats_path: Path, summary_json_path: Path, per_image_json_path: Path) -> None:
    last_record: Optional[Dict] = None
    with val_stats_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if line:
                last_record = json.loads(line)

    if last_record is None:
        raise RuntimeError(f"No JSON records found in {val_stats_path}")

    filtered_record = {k: v for k, v in last_record.items() if "/bbox/" not in k}

    boundary_keys = sorted(
        k
        for k in filtered_record
        if "per_image_eval_boundary_f_tol_" in k and k.endswith("_mean")
    )
    boundary_values = [filtered_record[k] for k in boundary_keys]

    preferred_boundary_key = (
        "Meters_train/val_synthetic_wire/segm/"
        "per_image_eval_boundary_f_tol_0p3pct_mean"
    )
    if preferred_boundary_key in filtered_record:
        filtered_record["boundary_f_score"] = filtered_record[preferred_boundary_key]
    elif boundary_values:
        filtered_record["boundary_f_score"] = statistics.fmean(boundary_values)
    elif per_image_json_path.exists():
        rows = json.loads(per_image_json_path.read_text())
        tol_cols = (
            [k for k in rows[0].keys() if k.startswith("boundary_f_tol_")] if rows else []
        )
        vals: List[float] = []
        for row in rows:
            row_vals = [row.get(k) for k in tol_cols if row.get(k) is not None]
            if row_vals:
                vals.extend(row_vals)
        if vals:
            filtered_record["boundary_f_score"] = statistics.fmean(vals)

    j_key = "Meters_train/val_synthetic_wire/segm/per_image_eval_jaccard_j_mean"
    jf_key = "Meters_train/val_synthetic_wire/segm/per_image_eval_j_and_f_mean"
    if j_key in filtered_record:
        filtered_record["jaccard_j_score"] = filtered_record[j_key]
    if jf_key in filtered_record:
        filtered_record["j_and_f_score"] = filtered_record[jf_key]
    elif per_image_json_path.exists():
        rows = json.loads(per_image_json_path.read_text())
        j_vals = [row.get("jaccard_j") for row in rows if row.get("jaccard_j") is not None]
        jf_vals = [row.get("j_and_f") for row in rows if row.get("j_and_f") is not None]
        if j_vals:
            filtered_record["jaccard_j_score"] = statistics.fmean(j_vals)
        if jf_vals:
            filtered_record["j_and_f_score"] = statistics.fmean(jf_vals)

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(filtered_record, indent=2))
    print(f"Wrote bbox-free summary JSON to: {summary_json_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    start_ts = time.time()
    total_steps = 8

    script_path = Path(__file__).resolve()
    inferred_repo_root = script_path.parents[2]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else inferred_repo_root
    data_root = Path(args.data_root).resolve()
    python_bin = args.python_bin
    num_gpus = _detect_num_gpus(args.num_gpus)

    coco_json = _resolve_coco_json(data_root)
    base_config_file = repo_root / "sam3" / "train" / "configs" / "custom" / "synthetic_wire_eval.yaml"
    runtime_config_file = repo_root / "sam3" / "train" / "configs" / "custom" / "_runtime_synthetic_wire_eval.yaml"
    runtime_config_path = "configs/custom/_runtime_synthetic_wire_eval.yaml"

    dataset_tag = data_root.name
    if dataset_tag == "synthetic_wire":
        experiment_dir = repo_root / "outputs" / "synthetic_wire_eval"
    else:
        experiment_dir = repo_root / "outputs" / f"synthetic_wire_eval_{dataset_tag}"

    val_stats_path = experiment_dir / "logs" / "val_stats.json"
    summary_json_path = experiment_dir / "dumps" / "synthetic_wire" / "segm_eval_summary.json"
    per_image_json_path = experiment_dir / "dumps" / "synthetic_wire" / "per_image_metrics_segm.json"

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    _step_msg(1, total_steps, f"Using python: {python_bin}", start_ts)
    _step_msg(2, total_steps, f"Validated dataset root: {data_root}", start_ts)

    if not base_config_file.is_file():
        raise FileNotFoundError(f"Base config does not exist: {base_config_file}")
    _step_msg(3, total_steps, f"Validated base config: {base_config_file}", start_ts)

    if not coco_json.is_file():
        _step_msg(4, total_steps, f"COCO annotation not found. Generating: {coco_json}", start_ts)
        _generate_coco_if_needed(python_bin, repo_root, data_root, coco_json)
    else:
        _step_msg(4, total_steps, f"Using existing COCO annotation: {coco_json}", start_ts)

    img_root = _infer_img_root(coco_json, data_root)
    _rewrite_runtime_config(base_config_file, runtime_config_file, experiment_dir, coco_json, img_root)
    _step_msg(5, total_steps, f"Prepared runtime config: {runtime_config_path}", start_ts)

    _step_msg(6, total_steps, "Sanity check: dataset/config summary", start_ts)
    _sanity_dump(coco_json, runtime_config_file)

    # fresh outputs for this run
    for p in [val_stats_path, summary_json_path, per_image_json_path]:
        if p.exists():
            p.unlink()

    _step_msg(
        7,
        total_steps,
        f"Running evaluation (single SLURM job, local launcher, num_gpus={num_gpus})",
        start_ts,
    )
    _run_eval(python_bin, repo_root, runtime_config_path, num_gpus)

    if not val_stats_path.is_file():
        raise FileNotFoundError(f"Expected val stats file was not created: {val_stats_path}")

    _postprocess_summary(val_stats_path, summary_json_path, per_image_json_path)
    _step_msg(8, total_steps, "Done", start_ts)
    print(f"Per-image metrics JSON: {per_image_json_path}", flush=True)
    print(f"Summary metrics JSON: {summary_json_path}", flush=True)


if __name__ == "__main__":
    main()
