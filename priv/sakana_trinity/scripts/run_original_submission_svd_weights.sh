#!/usr/bin/env bash
set -euo pipefail

# Runs the unmodified supplemental decomposer from docs/priv/trinity_code_submission
# and writes all outputs under tmp/. The submission tree is read-only input.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
OUT_ROOT="${OUT_ROOT:-tmp/sakana_parity/original_submission_svd}"
THREADS="${THREADS:-4}"
XLA_TARGET="${XLA_TARGET:-cuda12}"
UV_PY=(
  uv run
  --python 3.11
  --with fire==0.7.0
  --with torch==2.7.1
  --with transformers==4.55.2
  --with accelerate==1.6.0
  --with numpy
  --with safetensors
  python
)

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

mkdir -p "$OUT_ROOT"

echo "[original-svd] model=$MODEL_NAME"
echo "[original-svd] out_root=$OUT_ROOT"
echo "[original-svd] threads=$THREADS"
echo "[original-svd] python runner=uv python 3.11"

"${UV_PY[@]}" docs/priv/trinity_code_submission/decompose_model.py \
  --model_name="$MODEL_NAME" \
  --output_dir="$PWD/$OUT_ROOT" \
  2>&1 | tee "$OUT_ROOT/decompose_model.log"

SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
SVD_PATH="$PWD/$OUT_ROOT/$SAFE_MODEL_NAME/svd_weights.pt"

"${UV_PY[@]}" - "$SVD_PATH" <<'PY'
from pathlib import Path
import hashlib
import sys
import torch

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"svd_weights.pt was not created: {path}")

digest = hashlib.sha256(path.read_bytes()).hexdigest()
weights = torch.load(path, map_location="cpu")
s_keys = [key for key in weights if key.endswith(".S")]
print(f"[original-svd] svd_weights={path}")
print(f"[original-svd] sha256={digest}")
print(f"[original-svd] bytes={path.stat().st_size}")
print(f"[original-svd] tensor_count={len(weights)}")
print(f"[original-svd] singular_tensor_count={len(s_keys)}")
print(f"[original-svd] first_keys={list(weights)[:8]}")
PY

"${UV_PY[@]}" priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights "$SVD_PATH" \
  --all-selected-tensors \
  --out "$OUT_ROOT/python_sample_trace.json" \
  --write-components-dir "$OUT_ROOT/python_components" \
  2>&1 | tee "$OUT_ROOT/python_all_selected.log"

echo "[original-svd] elixir replay filter=model.layers.26.*"
echo "[original-svd] full embedding/lm_head replay is intentionally skipped here; use a chunked gate for those tensors"

XLA_TARGET="$XLA_TARGET" mix trinity.sakana.parity_sample \
  --semantic-only \
  --device-semantic-only \
  --preferred-layout-only \
  --source-from-python-stage \
  --all-selected-tensors \
  --selected-source-filter 'model.layers.26.' \
  --components-dir "$OUT_ROOT/python_components" \
  --python-report "$OUT_ROOT/python_sample_trace.json" \
  --stage-dir "$OUT_ROOT/elixir_stages" \
  --out "$OUT_ROOT/elixir_sample_trace.json" \
  2>&1 | tee "$OUT_ROOT/elixir_all_selected.log"

"${UV_PY[@]}" priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  "$OUT_ROOT/python_sample_trace.json" \
  "$OUT_ROOT/elixir_sample_trace.json" \
  2>&1 | tee "$OUT_ROOT/compare.log"

echo "[original-svd] complete"
echo "[original-svd] provide back: $OUT_ROOT/decompose_model.log $OUT_ROOT/python_all_selected.log $OUT_ROOT/elixir_all_selected.log $OUT_ROOT/compare.log"
