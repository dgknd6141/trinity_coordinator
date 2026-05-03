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

It can also write component metadata for the entire selected tensor set:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights path/to/svd_weights.pt \
  --all-selected-tensors \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Without `--svd-weights`, this all-selected debug mode fails fast unless
`--decompose-all-selected-if-missing` is explicitly supplied. That protects the
normal parity loop from accidentally decomposing the large embedding and LM-head
matrices.

All-selected debug mode also writes:

```text
trinity_svf_all_selected_stage_debug.safetensors
```

That file is not the canonical runtime artifact. It is a diagnostic bundle for
the all-selected parity gate. Its final stage tensors are source-oriented for
every selected tensor; canonical target-orientation validation happens later
when `mix trinity.sakana.import_python` materializes the runtime artifact layout
and checks the Bumblebee parameter names, shapes, and checkpoint hashes.
The current Elixir replay should be bounded with
`--selected-source-filter 'model.layers.26.'`; embedding and LM-head stage
checks need a chunked large-tensor gate before they are practical as a
monolithic EXLA replay.

For broader export, use:

```bash
uv run --python 3.11 \
  --with torch==2.7.1 \
  --with transformers==4.55.2 \
  --with accelerate==1.6.0 \
  --with numpy \
  --with safetensors \
  python priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py \
    --svd-weights path/to/svd_weights.pt \
    --output-dir tmp/sakana_parity/python_semantic_export
```

If original SVD weights are unavailable, the exporter can decompose from the
base model:

```bash
python3 priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py \
  --decompose-if-missing
```

That path is heavier and may not reproduce the historical stored hash.

Import the full Python semantic export into canonical Elixir artifacts:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.import_python \
  --source-dir tmp/sakana_parity/python_semantic_export \
  --manifest trinity_sakana_export_manifest.json \
  --reference priv/sakana_trinity/reference/sakana_python_reference_manifest.json \
  --out tmp/sakana_parity/adapted_artifacts_from_python \
  --force
```

The current canonical import writes checkpoint-directory artifacts instead of a
single giant adapted tensor file. This keeps embedding and LM-head materializing
bounded to one tensor at a time while still validating per-checkpoint hashes on
load. The latest Phase 2 gate produced:

```text
status=complete
artifact_layout=checkpoint_directory
selected_tensor_count=9
selected_singular_value_count=9216
router_head_shape=[10, 1024]
target_verified_count=9
```

Orientation is semantic, not only shape-driven. PyTorch stores Qwen linear
weights in source layout, while Bumblebee dense kernels use target layout. Most
selected layer tensors reveal this through reversed rectangular shapes, but
`k_proj` and `v_proj` are square `{1024, 1024}` matrices. The importer therefore
transposes Qwen `model.layers.*.weight` tensors whose Elixir target is a
`.kernel` path even when the source and target shapes are identical. This rule
was validated by fixed-transcript router trace parity; without it, token and
head hashes still matched but hidden/logit parity and role argmax diverged.

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
