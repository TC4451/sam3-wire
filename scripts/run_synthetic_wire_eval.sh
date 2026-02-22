#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-/n/holylabs/ydu_lab/Lab/zilin/wire_segmentation/wire_segmentation_data_unzipped}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sam3}"
AUTO_ACTIVATE_CONDA="${AUTO_ACTIVATE_CONDA:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SLURM_GPU_RAW="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-1}}}}"
NUM_GPUS="$(echo "${SLURM_GPU_RAW}" | grep -oE '[0-9]+' | head -n 1 || true)"
if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS=1
fi

SCRIPT_START_TS="$(date +%s)"
TOTAL_STEPS=8

print_step() {
  local step="$1"
  local msg="$2"
  local now_ts elapsed
  now_ts="$(date +%s)"
  elapsed="$((now_ts - SCRIPT_START_TS))"
  echo "[${step}/${TOTAL_STEPS}] ${msg} (elapsed: ${elapsed}s)"
}

activate_conda_env() {
  if [[ "${AUTO_ACTIVATE_CONDA}" != "1" ]]; then
    return
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
    return
  fi

  local conda_sh=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base
    conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      conda_sh="${conda_base}/etc/profile.d/conda.sh"
    fi
  fi

  if [[ -z "${conda_sh}" && -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  if [[ -z "${conda_sh}" && -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi

  if [[ -z "${conda_sh}" && -x "$(command -v conda || true)" ]]; then
    local base_from_conda
    base_from_conda="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base_from_conda}" && -f "${base_from_conda}/etc/profile.d/conda.sh" ]]; then
      conda_sh="${base_from_conda}/etc/profile.d/conda.sh"
    fi
  fi

  if [[ -z "${conda_sh}" ]]; then
    echo "Could not find conda.sh to activate env '${CONDA_ENV_NAME}'." >&2
    echo "Set AUTO_ACTIVATE_CONDA=0 if your sbatch script already activates conda." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${CONDA_ENV_NAME}"
}

activate_conda_env

if [[ "${PYTHON_BIN}" != /* ]]; then
  RESOLVED_PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
  if [[ -z "${RESOLVED_PYTHON_BIN}" ]]; then
    echo "Could not resolve python binary: ${PYTHON_BIN}" >&2
    exit 1
  fi
  PYTHON_BIN="${RESOLVED_PYTHON_BIN}"
fi

if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" != "1" ]]; then
  echo "Warning: this script runs single-node only; ignoring SLURM_NNODES=${SLURM_NNODES} and using num_nodes=1." >&2
fi

if [[ -f "${DATA_ROOT}/coco/annotations.json" ]]; then
  COCO_JSON="${DATA_ROOT}/coco/annotations.json"
elif [[ -f "${DATA_ROOT}/coco_all.json" ]]; then
  COCO_JSON="${DATA_ROOT}/coco_all.json"
else
  COCO_JSON="${DATA_ROOT}/coco/annotations.json"
fi
BASE_CONFIG_FILE="${REPO_ROOT}/sam3/train/configs/custom/synthetic_wire_eval.yaml"
RUNTIME_CONFIG_FILE="${REPO_ROOT}/sam3/train/configs/custom/_runtime_synthetic_wire_eval.yaml"
RUNTIME_CONFIG_PATH="configs/custom/_runtime_synthetic_wire_eval.yaml"
DATASET_TAG="$(basename "${DATA_ROOT}")"
if [[ "${DATASET_TAG}" == "synthetic_wire" ]]; then
  EXPERIMENT_DIR="${REPO_ROOT}/outputs/synthetic_wire_eval"
else
  EXPERIMENT_DIR="${REPO_ROOT}/outputs/synthetic_wire_eval_${DATASET_TAG}"
fi
VAL_STATS_PATH="${EXPERIMENT_DIR}/logs/val_stats.json"
SUMMARY_JSON_PATH="${EXPERIMENT_DIR}/dumps/synthetic_wire/segm_eval_summary.json"
PER_IMAGE_JSON_PATH="${EXPERIMENT_DIR}/dumps/synthetic_wire/per_image_metrics_segm.json"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

print_step 1 "Activated env '${CONDA_DEFAULT_ENV:-unknown}', using python: ${PYTHON_BIN}"
print_step 2 "Validated dataset root: ${DATA_ROOT}"

if [[ ! -f "${BASE_CONFIG_FILE}" ]]; then
  echo "Base config does not exist: ${BASE_CONFIG_FILE}" >&2
  exit 1
fi

print_step 3 "Validated base config: ${BASE_CONFIG_FILE}"

if [[ ! -f "${COCO_JSON}" ]]; then
  print_step 4 "COCO annotation not found. Generating: ${COCO_JSON}"
  mkdir -p "$(dirname "${COCO_JSON}")"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_mask_to_coco.py" "${DATA_ROOT}" "${COCO_JSON}"
else
  print_step 4 "Using existing COCO annotation: ${COCO_JSON}"
fi

IMG_ROOT="$("${PYTHON_BIN}" - "${COCO_JSON}" "${DATA_ROOT}" <<'PY'
import json
import os
import sys

coco_json = sys.argv[1]
data_root = sys.argv[2]

with open(coco_json, "r") as f:
    coco = json.load(f)

images = coco.get("images", [])
first_name = images[0].get("file_name", "") if images else ""
if "/" in first_name:
    print(data_root)
else:
    rgb_root = os.path.join(data_root, "rgb")
    print(rgb_root if os.path.isdir(rgb_root) else data_root)
PY
)"

cleanup_runtime_config() {
  if [[ -f "${RUNTIME_CONFIG_FILE}" ]]; then
    rm -f "${RUNTIME_CONFIG_FILE}"
  fi
}
trap cleanup_runtime_config EXIT

sed \
  -e "s|^  experiment_log_dir: .*|  experiment_log_dir: ${EXPERIMENT_DIR}|" \
  -e "s|^  coco_gt: .*|  coco_gt: ${COCO_JSON}|" \
  -e "s|^  img_path: .*|  img_path: ${IMG_ROOT}|" \
  "${BASE_CONFIG_FILE}" > "${RUNTIME_CONFIG_FILE}"

print_step 5 "Prepared runtime config: ${RUNTIME_CONFIG_PATH}"
print_step 6 "Sanity check: dataset/config summary"
"${PYTHON_BIN}" - "${COCO_JSON}" "${RUNTIME_CONFIG_FILE}" <<'PY'
import json
import sys

coco_json = sys.argv[1]
cfg_path = sys.argv[2]

with open(coco_json, "r") as f:
    coco = json.load(f)

images = len(coco.get("images", []))
anns = len(coco.get("annotations", []))
cats = [c.get("name", "<unnamed>") for c in coco.get("categories", [])]

with open(cfg_path, "r") as f:
    cfg = f.read()

has_cuda = "accelerator: cuda" in cfg
has_nccl = "backend: nccl" in cfg
has_cable_prompt = '"name": "cable"' in cfg

print(f"Sanity: images={images} annotations={anns} categories={cats[:5]}")
print(f"Sanity: accelerator_cuda={has_cuda} backend_nccl={has_nccl} cable_prompt={has_cable_prompt}")
print(f"Sanity: python_executable={sys.executable}")
PY

TRAIN_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/sam3/train/train.py"
  -c "${RUNTIME_CONFIG_PATH}"
  --use-cluster 0
  --num-gpus "${NUM_GPUS}"
  --num-nodes 1
)

rm -f "${VAL_STATS_PATH}" "${SUMMARY_JSON_PATH}" "${PER_IMAGE_JSON_PATH}"

print_step 7 "Running evaluation (single SLURM job, local launcher, num_gpus=${NUM_GPUS})"
"${TRAIN_CMD[@]}"

if [[ ! -f "${VAL_STATS_PATH}" ]]; then
  echo "Expected val stats file was not created: ${VAL_STATS_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${VAL_STATS_PATH}" "${SUMMARY_JSON_PATH}" "${PER_IMAGE_JSON_PATH}" <<'PY'
import json
import pathlib
import statistics
import sys

val_stats_path = pathlib.Path(sys.argv[1])
summary_json_path = pathlib.Path(sys.argv[2])
per_image_json_path = pathlib.Path(sys.argv[3])

last_record = None
for line in val_stats_path.read_text().splitlines():
    line = line.strip()
    if line:
        last_record = json.loads(line)

if last_record is None:
    raise RuntimeError(f"No JSON records found in {val_stats_path}")

filtered_record = {k: v for k, v in last_record.items() if "/bbox/" not in k}

boundary_keys = sorted(
    [
        k
        for k in filtered_record
        if "per_image_eval_boundary_f_tol_" in k and k.endswith("_mean")
    ]
)
boundary_values = [filtered_record[k] for k in boundary_keys]

if "Meters_train/val_synthetic_wire/segm/per_image_eval_boundary_f_tol_0p3pct_mean" in filtered_record:
    filtered_record["boundary_f_score"] = filtered_record[
        "Meters_train/val_synthetic_wire/segm/per_image_eval_boundary_f_tol_0p3pct_mean"
    ]
elif boundary_values:
    filtered_record["boundary_f_score"] = statistics.fmean(boundary_values)
elif per_image_json_path.exists():
    rows = json.loads(per_image_json_path.read_text())
    tol_cols = [k for k in rows[0].keys() if k.startswith("boundary_f_tol_")] if rows else []
    vals = []
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

print(f"Wrote bbox-free summary JSON to: {summary_json_path}")
PY

print_step 8 "Done"
echo "Per-image metrics JSON: ${EXPERIMENT_DIR}/dumps/synthetic_wire/per_image_metrics_segm.json"
echo "Summary metrics JSON: ${SUMMARY_JSON_PATH}"
