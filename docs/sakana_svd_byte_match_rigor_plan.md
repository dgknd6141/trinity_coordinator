# Sakana SVD Byte-Match And Functional-Parity Rigor Plan

## Goal

This lane has two different standards.

Required standard: prove the Elixir port is mathematically and functionally
correct against Python-exported Sakana SVD components.

Aspirational standard: make the final Elixir `bf16` tensor bytes match Python's
final `bf16` tensor bytes.

If byte equality is not achieved, the report must isolate the first stage where
byte equality fails and show that every required mathematical stage still passes
explicit tolerances. The process below is designed to make that judgment
inspectable from source code, generated reports, and stage tensor bundles.

## Current Verdict

The current Python hash is deterministic in the checked environment for this
sample. Python in-memory recomputation and Python safetensors readback both
produce:

```text
5aaa24c15898794dec09dccae650e35549c33cc24815e70ac6641cc3b466b725
```

That proves the component export/readback path is not the remaining mismatch.

The historical stored reference hash remains provenance-sensitive:

```text
600be6ab0f5a34325b9857182ccb5fce5971549a0ce8588cdacc992eda54014c
```

Do not use that hash as a default oracle unless the original `svd_weights.pt` or
equivalent original component provenance is supplied and Python itself reports
`reference_hash_reproducible: True`.

The latest semantic Elixir `torch_v` path does not byte-match Python:

```text
bf089ea0607c93ae69f92bf7b9fcf71dc2a2b53d231cfe307b8cd6f4ef6a85ae
```

Stage checks show why. Source tensor, offsets, and scaled singular values
byte-match. The first relevant non-byte-identical numerical stage is the large
matrix multiply/reconstruction, where Nx and PyTorch use different accumulation
behavior. The observed f32 gap is small enough for functional parity under the
declared tolerance:

```text
stage.zero_source_f32 max_abs ~= 3.12e-4, mean_abs ~= 6.39e-6
stage.adapted_source_f32 max_abs ~= 3.65e-4, mean_abs ~= 6.41e-6
stage.final_bf16 max_abs = 0.00390625, not required for functional parity
```

## Python Versus Elixir Architecture

Python side:

- loads the Qwen source tensor from Transformers;
- extracts the sample SVF offset slice from the Sakana router vector;
- computes or loads `U`, `S`, and legacy `torch.svd` `V`;
- reconstructs with `U @ diag(S * (1 + offsets)) @ V.T`;
- applies Sakana normalization `sum(S) / sum(S * (1 + offsets))`;
- casts the final oriented tensor to `torch.bfloat16`;
- writes the component safetensors and the Python stage tensor bundle.

Elixir side:

- loads the same Qwen source tensor through Bumblebee/Nx;
- reads Python-exported `U`, `S`, `V`, and offsets from safetensors;
- uses `v_layout: :torch_v`, which means `Nx.transpose(V)` for legacy
  `torch.svd` compatibility;
- reconstructs with `Nx.multiply(U, scaled_s)` followed by `Nx.dot(..., V.T)`;
- applies the same Sakana normalization formula;
- orients the final tensor to the manifest shape;
- casts to `:bf16`;
- writes Elixir stage tensors and compares them against the Python stage bundle.

Native Elixir `Nx.LinAlg.svd/2` variants remain diagnostics only. They are not a
byte oracle for PyTorch adapted hashes because singular-vector bases are not
unique, and nonzero per-singular-value scaling makes the adapted tensor depend
on that basis.

## Stage Contract

The Python report writes:

```text
tmp/sakana_parity/python_components/trinity_svf_stage_debug.safetensors
```

The Elixir report writes, when `--stage-dir` is supplied:

```text
tmp/sakana_parity/elixir_stages/trinity_svf_elixir_stage_host_binary_torch_v.safetensors
```

Stage tensors:

| Stage | Producer | Meaning | Byte expectation | Functional role |
| --- | --- | --- | --- | --- |
| `stage.source_f32` | Python + Elixir | Source tensor in Sakana/PyTorch orientation | exact | required |
| `stage.offsets_f32` | Python + Elixir | Selected SVF offset span | exact | required |
| `stage.scaled_s` | Python + Elixir | `S * (1 + offsets)` | exact/tight | required |
| `stage.normalization` | Python + Elixir | `sum(S) / sum(scaled_s)` | tight | required |
| `stage.u_scaled` | Python | `U * reshape(scaled_s, {1, k})` | diagnostic | optional Elixir artifact |
| `stage.matmul_pre_norm` | Python | Pre-normalization matmul | diagnostic | optional Elixir artifact |
| `stage.zero_source_f32` | Python + Elixir | Zero-offset source reconstruction | numeric | required |
| `stage.adapted_source_f32` | Python + Elixir | Adapted source-orientation f32 tensor | numeric | required |
| `stage.final_f32` | Python + Elixir | Final-orientation f32 tensor | numeric | required |
| `stage.final_bf16` | Python + Elixir | Final `bf16` bytes | aspirational | not required |

Elixir intentionally avoids recomputing extra large diagnostic matmuls just to
emit `stage.u_scaled` and `stage.matmul_pre_norm`; Python writes those for top
diff inspection. The required Elixir stages are enough to prove source/component
identity, scalar formula identity, and final f32 functional equivalence.

## Tolerance Policy

Exact stages:

- `stage.source_f32`: max abs `0`, mean abs `0`.
- `stage.offsets_f32`: max abs `0`, mean abs `0`.

Scalar/vector stages:

- `stage.scaled_s`: max abs `1e-6`, mean abs `1e-8`.
- `stage.normalization`: max abs `1e-6`, mean abs `1e-6`.

Large reconstruction stages:

- `stage.zero_source_f32`: max abs `1e-3`, mean abs `1e-5`.
- `stage.adapted_source_f32`: max abs `1e-3`, mean abs `1e-5`.
- `stage.final_f32`: max abs `1e-3`, mean abs `1e-5`.

Aspirational byte stage:

- `stage.final_bf16`: reported with the same numeric tolerance for visibility,
  but not required for functional parity. It is the final byte-match target.

Rationale: `source_f32`, offsets, and scaled singular values are not reductions
and should match exactly or almost exactly. Reconstruction is a large matrix
multiply reduction. PyTorch and Nx/EXLA can use different kernel families,
thread reductions, and accumulation order, so exact byte equality after final
`bf16` rounding is not guaranteed even when the formula is correct.

## Canonical Workflow

Run from the repository root.

1. Generate Python report, component bundle, and Python stage tensors:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

2. Generate Elixir semantic report and Elixir stage tensors:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

3. Compare hashes, stage checks, and top tensor differences:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

Optional exact final hash checks:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

Expected current interpretation:

- `--strict-stage-tolerances` should pass.
- `--strict-current-python` is expected to fail until final `bf16` byte parity is
  achieved.
- Wrong V layouts (`nx`/`vh`) should show large zero-offset error and should not
  be used as correctness targets.

## Human Inspection Checklist

- [x] Verify Python current recomputation and Python safetensors readback hashes
  match each other.
- [x] Verify Python reports `reference_hash_reproducible: False` before treating
  the historical hash as non-default.
- [x] Verify Elixir `torch_v` is the only plausible semantic V layout.
- [x] Verify host and EXLA semantic `torch_v` variants agree.
- [x] Verify `stage.source_f32`, `stage.offsets_f32`, and `stage.scaled_s`
  byte-match.
- [x] Verify `stage.normalization` is within scalar tolerance.
- [x] Verify `stage.zero_source_f32`, `stage.adapted_source_f32`, and
  `stage.final_f32` pass required tolerances.
- [x] Verify `stage.final_bf16` mismatch is explicitly reported as
  aspirational, with top differing indices printed.
- [x] Keep native Nx SVD variants out of the semantic debugging loop unless the
  question is specifically about Nx's SVD basis.
- [x] Run the fast regression suite after any parity code change.
- [x] Run Credo and Dialyzer before committing.

## Decision Rules

Byte match achieved:

- Some Elixir semantic variant equals the current Python baseline hash; and
- required stage checks pass.

Functional parity achieved, byte match not achieved:

- no Elixir semantic variant equals the current Python baseline hash; and
- all required stage checks pass; and
- the first non-byte-identical stage is a documented backend numeric stage,
  currently the reconstruction matmul/rounding path.

Functional parity not achieved:

- any required stage check fails; or
- exact source/offset/scaled singular-value identity fails; or
- the wrong V layout is the only low-error path; or
- Python safetensors readback diverges from Python in-memory recomputation.

## Implementation Checklist

- [x] Python debug script writes `trinity_svf_stage_debug.safetensors`.
- [x] Python report records stage file path, stage schema, stage keys, and
  interpretation.
- [x] Elixir Mix task accepts `--stage-dir`.
- [x] Elixir report writes comparable host `torch_v` stage safetensors.
- [x] Elixir report compares required stage tensors against Python stage tensors
  when the Python report supplies a stage file.
- [x] Comparator prints stage pass/fail, byte-match status, numeric error, and
  top differing tensor indices.
- [x] Comparator supports `--strict-stage-tolerances` separately from
  `--strict-current-python`.
- [x] README and parity docs explain the rigorous workflow and interpretation.
- [x] Elixir JSON reports preserve booleans/nulls so strict comparator results
  are reliable.
- [ ] Continue iterative byte-match investigation only after required functional
  parity checks stay green.
