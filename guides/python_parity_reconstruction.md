# Recreating The Python Parity Process

The current foundation is to recreate Python's Sakana/Qwen artifact application
process using the same base Qwen model and the same exported SVD components.

## Python Source Of Truth

The Python debug script is:

```text
priv/sakana_trinity/scripts/debug_sakana_parity_sample.py
```

It loads:

- `Qwen/Qwen3-0.6B` through Transformers;
- the Sakana router vector;
- the historical reference manifest;
- optional original `svd_weights.pt`;
- otherwise current-environment SVD components recomputed from the current base
  Qwen tensor.

The current default baseline is the Python safetensors readback variant, not the
historical stored hash.

## Why The Historical Hash Is Not The Default Oracle

The manifest records:

```text
600be6ab0f5a34325b9857182ccb5fce5971549a0ce8588cdacc992eda54014c
```

Current Python recomputation does not reproduce that hash in the checked
environment. Python reports:

```text
reference_hash_reproducible: False
```

That means the historical hash is provenance-bound. It may depend on the
original `svd_weights.pt`, PyTorch version, model-loading path, or saved SVD
basis.

Use the historical hash only when:

1. the original SVD components are available; and
2. Python itself reproduces the stored hash.

## Current Python Baseline

Current Python in-memory SVD and Python safetensors readback both produce:

```text
5aaa24c15898794dec09dccae650e35549c33cc24815e70ac6641cc3b466b725
```

That is the current Python baseline for same-run parity.

The fact that readback matches in-memory Python rules out component export and
readback as the cause of the current Elixir hash mismatch.

## Python Reconstruction Formula

For legacy `torch.svd`, Python returns `U`, `S`, and `V`.

The reconstruction formula is:

```python
scaled_s = S * (1.0 + offsets)
normalization = S.sum() / scaled_s.sum()
adapted = (U * scaled_s.reshape(1, -1)) @ V.T
adapted = adapted * normalization
final = orient_to_manifest_shape(adapted).to(torch.bfloat16)
```

The Elixir semantic `torch_v` layout must match that `V.T` behavior.

## Python Outputs

Run:

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

The stage bundle is the baseline for Elixir's stage checks.

For full selected-tensor component metadata, use:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights path/to/svd_weights.pt \
  --all-selected-tensors \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

`--all-selected-tensors` requires `--svd-weights` by default. To intentionally
recompute every selected SVD from the base model, add
`--decompose-all-selected-if-missing`. That opt-in exists because the selected
set includes embedding and LM-head matrices and can otherwise create an
accidental long-running decomposition job.

All-selected mode writes the legacy sample stage bundle and, additionally:

```text
tmp/sakana_parity/python_components/trinity_svf_all_selected_stage_debug.safetensors
```

The all-selected stage bundle keys are namespaced per source tensor:

```text
tensor.<safe_source_name>.source_f32
tensor.<safe_source_name>.offsets_f32
tensor.<safe_source_name>.scaled_s
tensor.<safe_source_name>.normalization
tensor.<safe_source_name>.u_scaled
tensor.<safe_source_name>.matmul_pre_norm
tensor.<safe_source_name>.zero_source_f32
tensor.<safe_source_name>.adapted_source_f32
tensor.<safe_source_name>.final_f32
tensor.<safe_source_name>.final_bf16
```

For non-sample selected tensors, `final_f32` and `final_bf16` are
source-oriented. Target-orientation validation belongs to canonical artifact
import and adapted profile loading, where the Bumblebee parameter path is known.
This keeps the all-selected parity gate focused on Python component semantics:
same source tensor, same offsets, same `U/S/V`, same formula, and declared
numeric tolerances.

## Elixir Semantic Reconstruction

Run:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --device-semantic-only \
  --preferred-layout-only \
  --source-from-python-stage \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

Important options:

- `--semantic-only`: skip native Nx SVD diagnostics.
- `--components-dir`: read Python-exported `U/S/V` and offsets.
- `--python-report`: read Python baseline metadata and Python stage file path.
- `--stage-dir`: write Elixir stage tensors.
- `--source-from-python-stage`: reuse Python's serialized `stage.source_f32`
  instead of loading Qwen only to retrieve the sample source tensor.
- `--preferred-layout-only`: run the manifest-preferred layout, currently
  `torch_v`, instead of repeating known-wrong `nx` and `vh` diagnostics.
- `--device-semantic-only`: run semantic reconstruction on EXLA CUDA and avoid a
  large Nx BinaryBackend CPU matmul.
- `--all-selected-tensors`: replay every selected tensor from the Python
  component metadata and compare against the all-selected Python stage bundle.
  Use it only with an all-selected Python report.
- `--selected-source-filter`: restrict semantic replay to source or Elixir
  names containing a fixed string. Use `model.layers.26.` for the current bounded
  service-critical all-selected gate; embedding and LM-head need a chunked
  large-tensor gate before they are enforced in Elixir.

Use the slower layout-diagnostic command only when investigating orientation:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

The semantic `torch_v` variant is the active functional-parity target. The
recommended fast command emits the device `torch_v` variant and still writes the
same required stage checks.

All-selected replay command:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --device-semantic-only \
  --preferred-layout-only \
  --source-from-python-stage \
  --all-selected-tensors \
  --selected-source-filter 'model.layers.26.' \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

This command requires `stage_debug.all_selected_stage_tensor_file` in the Python
report. Missing all-selected stage tensors are treated as a setup error, not as
a tolerated fallback, because the full gate must prove each selected source
tensor explicitly.

## Comparison

Run:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

This command fails if required mathematical stages fail. It does not fail solely
because the final `bf16` hash differs.

For byte equality, use:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

That is expected to fail in the current state.
