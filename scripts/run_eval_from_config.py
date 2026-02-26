#!/usr/bin/env python3
"""Run SAM3 eval from a config yaml while overriding output directory.

This script:
1) resolves a base config yaml path
2) writes a runtime config with `paths.experiment_log_dir` overridden
3) forces `trainer.mode=val`
4) launches `sam3/train/train.py` with the runtime config
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:
    raise ImportError(
        "Missing dependency `omegaconf`. Run this script from the SAM3 env "
        "(for example: /home/daizi/miniconda3/envs/sam3/bin/python)."
    ) from exc


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


def _resolve_config_file(repo_root: Path, config_yaml: str) -> Path:
    raw = Path(config_yaml)

    candidates = [
        raw,
        (repo_root / raw),
        (repo_root / "sam3" / "train" / raw),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve config yaml. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def _build_runtime_config(
    base_config_file: Path,
    runtime_config_file: Path,
    output_dir: Path,
) -> None:
    cfg = OmegaConf.load(str(base_config_file))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Unexpected config type for {base_config_file}: {type(cfg)}")

    if "paths" not in cfg:
        cfg.paths = {}
    cfg.paths.experiment_log_dir = str(output_dir)

    if "launcher" not in cfg:
        cfg.launcher = {}
    cfg.launcher.experiment_log_dir = str(output_dir)

    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.mode = "val"

    runtime_config_file.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_file.open("w") as f:
        # Keep this as a global package so Hydra composes it like existing custom configs.
        f.write("# @package _global_\n")
        f.write(OmegaConf.to_yaml(cfg))


def _runtime_config_name(runtime_config_file: Path, repo_root: Path) -> str:
    train_cfg_root = (repo_root / "sam3" / "train").resolve()
    runtime_file = runtime_config_file.resolve()
    try:
        rel = runtime_file.relative_to(train_cfg_root)
    except ValueError as exc:
        raise ValueError(
            f"Runtime config must be under {train_cfg_root}; got {runtime_file}"
        ) from exc
    return rel.as_posix()


def _run_eval(
    repo_root: Path,
    runtime_config_name: str,
    use_cluster: int,
    num_gpus: int,
    num_nodes: int,
) -> None:
    train_script = repo_root / "sam3" / "train" / "train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "-c",
        runtime_config_name,
        "--use-cluster",
        str(use_cluster),
        "--num-gpus",
        str(num_gpus),
        "--num-nodes",
        str(num_nodes),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        required=True,
        help=(
            "Base config yaml path. Can be absolute, repo-relative, or "
            "train-config-relative (e.g. configs/custom/real_wire_eval.yaml)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to use for this eval run.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root path. Defaults to auto-detect from this script location.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPUs per node. If omitted, auto-detected from env; fallback=1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes to pass to train.py.",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=0,
        choices=[0, 1],
        help="Pass-through for train.py --use-cluster.",
    )
    parser.add_argument(
        "--keep-runtime-config",
        action="store_true",
        help="Keep generated runtime config file instead of deleting it.",
    )
    args = parser.parse_args()

    start_ts = time.time()
    total_steps = 5

    script_path = Path(__file__).resolve()
    inferred_repo_root = script_path.parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else inferred_repo_root
    output_dir = Path(args.output_dir).resolve()
    base_config_file = _resolve_config_file(repo_root, args.config_yaml)
    num_gpus = _detect_num_gpus(args.num_gpus)

    runtime_name = (
        f"_runtime_eval_{base_config_file.stem}_{int(time.time())}_{os.getpid()}.yaml"
    )
    runtime_config_file = (
        repo_root / "sam3" / "train" / "configs" / "custom" / runtime_name
    )

    _step_msg(1, total_steps, f"Repo root: {repo_root}", start_ts)
    _step_msg(2, total_steps, f"Base config: {base_config_file}", start_ts)

    output_dir.mkdir(parents=True, exist_ok=True)
    _build_runtime_config(base_config_file, runtime_config_file, output_dir)
    runtime_config_name = _runtime_config_name(runtime_config_file, repo_root)
    _step_msg(3, total_steps, f"Runtime config: {runtime_config_name}", start_ts)

    _step_msg(
        4,
        total_steps,
        (
            "Launching eval "
            f"(use_cluster={args.use_cluster}, num_nodes={args.num_nodes}, num_gpus={num_gpus})"
        ),
        start_ts,
    )
    try:
        _run_eval(
            repo_root=repo_root,
            runtime_config_name=runtime_config_name,
            use_cluster=args.use_cluster,
            num_gpus=num_gpus,
            num_nodes=args.num_nodes,
        )
    finally:
        if runtime_config_file.exists() and not args.keep_runtime_config:
            runtime_config_file.unlink()

    _step_msg(5, total_steps, "Done", start_ts)
    print(f"Outputs written under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
