# Sakana TRINITY Artifacts

This directory holds local reference artifacts and conversion tools for using
the Sakana TRINITY supplementary outputs with the Elixir/Nx coordinator runtime.

## Layout

- `reference/sakana_decompose_model.original.py`
  - Unmodified reference copy of Sakana's `decompose_model.py`.
- `reference/sakana_es_log.json`
  - Reference copy of the inspected Sakana ES run config.
- `artifacts/sakana_model_iter_60.npy`
  - Reference copy of Sakana's trained ES/router vector.
- `artifacts/trinity_router_es_vector.safetensors`
  - Raw safetensors conversion of the `.npy` vector under tensor key
    `trinity_router_es_vector`.
- `scripts/convert_router_vector_to_safetensors.py`
  - Dumb `.npy` to safetensors converter. It does not split the vector.
- `scripts/export_sakana_trinity_safetensors.py`
  - Semantic export script. It follows Sakana's SVD/head parameter ordering and
    emits Elixir-friendly safetensors plus a JSON manifest.

## Raw Vector Conversion

```sh
python3 priv/sakana_trinity/scripts/convert_router_vector_to_safetensors.py --float32
```

This writes:

`priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors`

The tensor key is:

`trinity_router_es_vector`

## Semantic Export

The semantic exporter needs SVD weights. If `svd_weights.pt` is already
available, pass it explicitly:

```sh
python3 priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py \
  --svd-weights path/to/svd_weights.pt
```

If it is not available, the exporter can generate it from the base Qwen model:

```sh
python3 priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py \
  --decompose-if-missing
```

That path loads `Qwen/Qwen3-0.6B` with Transformers and runs SVD over selected
weight matrices, so it is heavier than raw conversion.

The semantic export writes:

- `trinity_svf_components.safetensors`
- `trinity_svf_scale_offsets.safetensors`
- `trinity_router_head.safetensors`
- `trinity_sakana_export_manifest.json`

## Parity Debugging

Use `scripts/debug_sakana_parity_sample.py` before asserting byte-level parity.
It now reconstructs both from in-memory Python SVD components and from the exact
safetensors files written for Elixir. Matching Python recomputation/readback
hashes prove the component export did not alter `U/S/V` or scale offsets.

For Elixir-side semantic checks, pass `--semantic-only` to
`mix trinity.sakana.parity_sample` while debugging Python-component parity. That
skips native Nx SVD diagnostics and avoids the expensive CUDA SVD compilation
path. Native SVD diagnostics are useful only when investigating Nx's SVD basis;
they are not expected to byte-match PyTorch adapted hashes under nonzero SVF
offsets.
