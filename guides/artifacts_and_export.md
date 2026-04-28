# Sakana Artifacts And Export

This guide explains the artifact path used by the active buildout.

## Artifact Sources

Local artifact directory:

```text
priv/sakana_trinity/
```

Important files:

```text
priv/sakana_trinity/artifacts/sakana_model_iter_60.npy
priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors
priv/sakana_trinity/reference/sakana_python_reference_manifest.json
priv/sakana_trinity/reference/sakana_decompose_model.original.py
```

The `.npy` file is the original vector artifact. The safetensors file is the
runtime-friendly conversion.

## Router Vector Layout

The inspected vector has `19_456` values:

```text
0..9215       SVF scale offsets
9216..19455   router head weights
```

The head region reshapes to:

```text
{10, 1024}
```

The `10` outputs represent `L + 3`: agent logits plus the three TRINITY roles.

## Selected Tensor Set

The current Sakana/Qwen lane focuses on the layer-26 tensor set. The selected
tensors consume exactly `9216` singular-value offsets.

The parity sample uses:

```text
source_name: model.layers.26.mlp.gate_proj.weight
elixir_name: decoder.blocks.26.ffn.gate.kernel
offset_start: 5120
offset_end: 6144
```

This sample is large enough to expose real matmul/rounding behavior while still
being practical for repeated diagnostics.

## Semantic Component Export

The semantic Python component files are:

```text
trinity_svf_components.safetensors
trinity_svf_scale_offsets.safetensors
trinity_svf_debug_manifest.json
```

For parity diagnostics, `debug_sakana_parity_sample.py` writes a sample-specific
component bundle under `tmp/sakana_parity/python_components`.

For broader export, use:

```bash
python3 priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py \
  --svd-weights path/to/svd_weights.pt
```

If original SVD weights are unavailable, the exporter can decompose from the
base model:

```bash
python3 priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py \
  --decompose-if-missing
```

That path is heavier and may not reproduce the historical stored hash.

## Elixir Artifact Export

The Elixir export task materializes adapted artifacts:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.export_adapted
```

Useful smoke command:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.export_adapted --only-index 1 --force
```

Resume:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.export_adapted --resume --only-index 1
```

The exporter writes manifests, per-tensor checkpoints, router-head artifacts,
and event logs. The runtime profile should only consume a complete manifest.

## Resume And Integrity Rules

Resume should validate:

- source vector path;
- source vector hash;
- selected tensor list;
- singular-value counts;
- router-head shape;
- output tensor shapes;
- output tensor types;
- checkpoint hashes.

If identity changed, rebuild with `--force` rather than trusting old
checkpoints.

## Historical Reproduction

Strict historical hash reproduction requires original provenance.

Use:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights path/to/original/svd_weights.pt \
  --strict-reference-hash \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Only after Python reports `reference_hash_reproducible: True` should the
historical hash be treated as an exact target.
