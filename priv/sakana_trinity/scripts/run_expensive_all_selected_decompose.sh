#!/usr/bin/env bash
set -euo pipefail

# Explicitly recomputes all selected SVD components from the current base Qwen
# model through the parity debug harness, then runs Elixir all-selected replay.

OUT_ROOT="${OUT_ROOT:-tmp/sakana_parity/expensive_all_selected_decompose}"
THREADS="${THREADS:-4}"
XLA_TARGET="${XLA_TARGET:-cuda12}"
UV_PY=(
  uv run
  --python 3.11
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

echo "[expensive-all-selected] out_root=$OUT_ROOT"
echo "[expensive-all-selected] threads=$THREADS"
echo "[expensive-all-selected] python runner=uv python 3.11"
echo "[expensive-all-selected] this intentionally recomputes selected SVDs from Qwen"

"${UV_PY[@]}" priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --all-selected-tensors \
  --decompose-all-selected-if-missing \
  --out "$OUT_ROOT/python_sample_trace.json" \
  --write-components-dir "$OUT_ROOT/python_components" \
  2>&1 | tee "$OUT_ROOT/python_decompose_all_selected.log"

echo "[expensive-all-selected] elixir replay filter=model.layers.26.*"
echo "[expensive-all-selected] full embedding/lm_head replay is intentionally skipped here; use a chunked gate for those tensors"

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

echo "[expensive-all-selected] complete"
echo "[expensive-all-selected] provide back: $OUT_ROOT/python_decompose_all_selected.log $OUT_ROOT/elixir_all_selected.log $OUT_ROOT/compare.log"
