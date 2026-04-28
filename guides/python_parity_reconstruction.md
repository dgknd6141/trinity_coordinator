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

## Elixir Semantic Reconstruction

Run:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
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

The semantic host `torch_v` variant is the active functional-parity target.

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
