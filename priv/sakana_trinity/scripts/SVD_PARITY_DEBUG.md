# Sakana SVD hash parity debug flow

Run from the repository root.

## 1. Emit the Python-side checkpoint report

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

This writes:

- `tmp/sakana_parity/python_sample_trace.json`
- `tmp/sakana_parity/python_components/trinity_svf_components.safetensors`
- `tmp/sakana_parity/python_components/trinity_svf_scale_offsets.safetensors`
- `tmp/sakana_parity/python_components/trinity_svf_debug_manifest.json`
- `tmp/sakana_parity/python_components/trinity_svf_stage_debug.safetensors`

The report separates three concepts:

- **stored reference hash**: the historical value in `sakana_python_reference_manifest.json`;
- **current Python recomputation hash**: the value produced by the current Python/PyTorch environment;
- **Python safetensors readback hash**: the value produced after reading back the
  exact component files that Elixir consumes.

If the script prints `reference_hash_reproducible: False`, do **not** expect Elixir
or freshly recomputed Python SVD components to match the stored `600be6...` hash.
That means the stored hash is provenance-sensitive to the original SVD component
basis.

The readback variant is the decisive export check. If
`python_safetensors_readback_torch_v_final_bf16` matches the recomputed Python
variant, the component files are not the cause of an Elixir mismatch.

The stage tensor bundle is the side-by-side correctness contract. It contains
source tensor bytes, offsets, scaled singular values, normalization,
reconstruction tensors, and final `bf16` bytes from Python safetensors readback.
Use it to isolate the first stage where Elixir stops byte-matching Python.

## 1a. Strict historical reproduction, when original SVD weights are available

If you have the original Python `svd_weights.pt`, run:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights path/to/svd_weights.pt \
  --strict-reference-hash \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Only use strict stored-reference assertions after the Python report itself says
`reference_hash_reproducible: True`.

## 2. Emit the Elixir-side checkpoint report

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

`--semantic-only` skips native `Nx.LinAlg.svd/2` diagnostics and avoids the
long CUDA SVD compilation path while debugging Python-component parity. Omit it
only when you specifically need native Nx SVD diagnostics.

The Elixir tracer snapshots intermediate tensors to `Nx.BinaryBackend` before
reconstruction so EXLA donated buffers cannot crash the report. Semantic
variants include both the final `bf16` tensor summary and a
`final_f32_before_bf16` summary so formula/accumulation differences can be
separated from final byte-hash rounding.

With `--stage-dir`, the Elixir tracer writes:

- `tmp/sakana_parity/elixir_stages/trinity_svf_elixir_stage_host_binary_torch_v.safetensors`

It also embeds stage checks in the Elixir JSON report when the Python report
points at a Python stage bundle. The host `torch_v` semantic path is the
functional-parity target because it consumes the exact Python `U/S/V` components
and avoids native SVD basis differences.

For semantic Python components, the tracer now emits both host/BinaryBackend and
device/EXLA variants. Use the host/BinaryBackend semantic variant for strict
functional parity with the current Python report; use the device variant to
inspect runtime CUDA numerical drift. CUDA/EXLA may use different fp32 GEMM
semantics than PyTorch CPU, so a device variant can have a larger zero-offset
error and a different `bf16` hash even when the formula and V layout are right.

Native variants are expected to differ when the SVD basis differs. Semantic
Python-component variants isolate formula, V/Vh layout, orientation, framework
GEMM accumulation behavior, final `bf16` cast, raw-byte hashing, and compute
backend. Exact `bf16` hashes can still differ across PyTorch and Nx/EXLA when a
large fp32 matmul accumulates differently; use zero-offset error and pre-bf16
summaries to decide whether the formula is correct before treating a hash
mismatch as a porting bug.

## 3. Compare both reports

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

For the rigorous functional gate:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

This prints:

- exact-vs-numeric stage status for every compared stage;
- the first practical byte-match failure surface;
- whether all required stages pass their declared tolerances;
- top differing flat indices and values for large stage tensors.

Current interpretation rules:

- `stage.source_f32`, `stage.offsets_f32`, and `stage.scaled_s` should
  byte-match.
- `stage.normalization`, `stage.zero_source_f32`,
  `stage.adapted_source_f32`, and `stage.final_f32` must pass numeric
  tolerances.
- `stage.final_bf16` byte equality is aspirational. A final `bf16` mismatch is
  not a functional failure when all required f32 stages pass.

For opt-in exact checks:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-reference \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

or:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

## 4. Run the focused test with diagnostics enabled

```bash
TRINITY_SVD_PARITY_OUT=tmp/sakana_parity/elixir_from_test.json \
TRINITY_PYTHON_PARITY_REPORT=tmp/sakana_parity/python_sample_trace.json \
TRINITY_PYTHON_COMPONENTS_DIR=tmp/sakana_parity/python_components \
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only slow_qwen_svd --trace
```

Default behavior verifies shapes, offsets, zero-offset reconstruction sanity, and
Python-component V-layout handling without requiring a non-reproducible stored
hash. Strict byte-level checks are explicit:

- `TRINITY_STRICT_REFERENCE_HASH=1` requires a semantic component variant to
  match the stored manifest hash. Use only after Python itself reproduces it.
- `TRINITY_STRICT_CURRENT_PYTHON_HASH=1` requires a semantic component variant
  to match the current Python baseline hash.
- `TRINITY_STRICT_NATIVE_SVD_HASH=1` requires native Nx SVD to match the stored
  Python hash. This is expected to fail when native SVD produces a different but
  valid singular-vector basis.
