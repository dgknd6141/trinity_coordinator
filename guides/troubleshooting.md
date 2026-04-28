# Troubleshooting

This guide covers common failure modes in the current Qwen/Sakana parity lane.

## The Elixir Parity Task Is Slow

Use semantic-only mode:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

Without `--semantic-only`, the task runs native Nx SVD diagnostics. Native SVD
can trigger expensive XLA compilation and ptxas register-spill warnings. That is
not needed for semantic Python-component debugging.

## Python Does Not Match The Historical Hash

If Python prints:

```text
reference_hash_reproducible: False
```

then do not require Elixir to match the stored historical hash. Use the current
Python readback baseline instead.

To pursue historical reproduction, provide the original `svd_weights.pt`:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --svd-weights path/to/original/svd_weights.pt \
  --strict-reference-hash \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

## Wrong V Layout

Symptoms:

- `torch_v` has low zero-offset error;
- `nx` or `vh` has very large zero-offset error, around `1.6` for the current
  sample.

Interpretation:

- legacy `torch.svd` `V` requires `V.T` during reconstruction;
- the Elixir semantic layout should be `v_layout: :torch_v`.

Do not try to make the wrong layout match.

## Source Or Final Shape Mismatch

The sample has different source and final orientations:

```text
source_shape: [3072, 1024]
sample_reconstructed_shape: [1024, 3072]
```

If shape checks fail, inspect whether the code is comparing Python source
orientation against Bumblebee target orientation. The parity code should orient
tensors explicitly before hashing or comparing.

## Stage Checks Fail At Source Or Offsets

If these fail:

```text
stage.source_f32
stage.offsets_f32
stage.scaled_s
```

then it is probably not a backend numeric issue. Check:

- source tensor key;
- Qwen model dtype and loading path;
- offset span;
- router vector file and hash;
- safetensors key names;
- source orientation.

## Stage Checks Fail At Reconstruction

If exact input stages pass but reconstruction exceeds tolerance, check:

- `v_layout`;
- normalization formula;
- dtype cast timing;
- whether offsets were cast to singular-value dtype;
- final/source orientation;
- backend used for host and device variants.

Do not immediately loosen tolerances. First inspect top diffs from the
comparator output.

## Final Hash Differs But Stage Tolerances Pass

This is the current known state.

Interpretation:

- the mathematical port is functionally correct under declared tolerances;
- exact final `bf16` bytes still differ;
- byte-match work should focus on reconstruction accumulation and rounding.

Run:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  --top-diffs 10 \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

Use the top-diff output to inspect the largest f32 and `bf16` differences.

## EXLA Donated Buffer Errors

Errors like:

```text
Buffer has been deleted or donated.
```

usually mean diagnostic code tried to read an EXLA tensor after a compiled
operation consumed it. The parity tracer snapshots diagnostic tensors to
`Nx.BinaryBackend` to avoid this. If this recurs, make sure any safetensors
readback used for diagnostics is immediately transferred to `Nx.BinaryBackend`.

## Missing CUDA

Check:

```bash
XLA_TARGET=cuda12 mix run -e 'IO.inspect(EXLA.Client.get_supported_platforms())'
```

If CUDA is missing, verify:

- NVIDIA driver;
- `nvidia-smi`;
- `XLA_TARGET=cuda12`;
- EXLA dependency target;
- environment isolation, especially shells launched without CUDA env vars.
