# Sakana SVD Parity Debug Checklist

## Ground Rules

- [x] Re-read the Python parity/export/reference source from scratch before changing behavior.
- [x] Re-read the Elixir parity, importer, SVD, and Mix task source from scratch before changing behavior.
- [x] Treat the stored historical hash `600be6ab0f5a34325b9857182ccb5fce5971549a0ce8588cdacc992eda54014c` as a strict target only when the original SVD provenance is available.
- [x] Treat current Python recomputation and Python safetensors readback as separate targets until proven identical.
- [x] Debug zero-offset reconstruction before final adapted bf16 hashes.
- [x] Do not try to force native Nx SVD adapted hashes to byte-match PyTorch SVD adapted hashes.
- [x] Avoid unnecessary native Nx SVD recompilation while debugging semantic Python-component parity.

## Source Audit Findings

- [x] Python reference decomposition uses `torch.svd(weight)` and stores `U`, `S`, and legacy `V`.
- [x] Python reconstruction formula is `U @ diag(S * (1 + offsets)) @ V.T`, then multiply by `sum(S) / sum(S * (1 + offsets))`.
- [x] Elixir `v_layout: :torch_v` means `Nx.transpose(v)` before the final dot, matching legacy `torch.svd` `V`.
- [x] Elixir parity with `--components-dir` used to compute native Nx SVD variants first, causing wasted compilation during semantic debugging.
- [x] The Python debug run supplied in the handoff uses `--model-torch-dtype float32`.
- [x] The Elixir `:qwen_coordinator` profile loads Qwen via Bumblebee with `type: :bf16`.
- [x] Python safetensors readback matches Python in-memory reconstruction.
- [x] Python readback from exported components does not match Elixir semantic reconstruction by exact final hash.

## Confirmed Runtime Findings

- Python safetensors readback matches Python in-memory reconstruction:
  - `python_recomputed_torch_svd_v_final_bf16 = 5aaa24c15898794dec09dccae650e35549c33cc24815e70ac6641cc3b466b725`
  - `python_safetensors_readback_torch_v_final_bf16 = 5aaa24c15898794dec09dccae650e35549c33cc24815e70ac6641cc3b466b725`
  - readback zero-offset error remains `2.3543834686279297e-06`
- The Python and Elixir source tensor summaries line up by hash/orientation; the remaining `3.1197e-4` Elixir zero-offset error is not explained by component export/readback.
- Elixir `--semantic-only` skips native SVD variants and completes without the long native SVD ptxas compile sequence.
- Latest Elixir semantic `torch_v` produced `74dc61d765c95e80ca7298b6e97f29a4fd76e2ae4bfb348b2abbffcbc5e0dff8`, so the remaining exact-hash gap is not caused by Python component export/readback.
- EXLA precision probes (`:default`, `:high`, `:highest`, and f64+`:highest`) did not reproduce the Python `5aaa24...` hash. Treat the remaining exact-hash difference as framework GEMM accumulation/rounding until a contrary stage-level diff proves a formula bug.
- Stage diagnostics now isolate the current byte mismatch:
  - `stage.source_f32`, `stage.offsets_f32`, and `stage.scaled_s` byte-match Python.
  - `stage.normalization` differs by about `5.96e-8`, within scalar tolerance.
  - `stage.zero_source_f32`, `stage.adapted_source_f32`, and `stage.final_f32` pass required reconstruction tolerances.
  - `stage.final_bf16` does not byte-match Python and remains an aspirational byte target, not the required functional gate.

## Completed Implementation

- [x] Added Python debug readback diagnostics:
  - [x] write component bundle as today;
  - [x] read back `trinity_svf_components.safetensors`;
  - [x] read back `trinity_svf_scale_offsets.safetensors`;
  - [x] reconstruct with zero offsets and real offsets;
  - [x] emit component before-write and after-readback summaries;
  - [x] emit `python_safetensors_readback_torch_v_final_bf16`.
- [x] Added source-comparison diagnostics:
  - [x] report Python source `sha256_as_f32` and `sha256_as_bf16`;
  - [x] report Python zero-error against both float32 source and bf16-rounded source;
  - [x] report Elixir source summary and pre-bf16 final summaries for stage comparison.
- [x] Added Elixir semantic-only/native-skip path:
  - [x] add `native?: false` option in `ParityTrace.sample_report!/1`;
  - [x] add `--semantic-only` and `--skip-native-svd` to `mix trinity.sakana.parity_sample`;
  - [x] ensure `native_elixir_svd_variants` is an empty list without invoking `Nx.LinAlg.svd/2`.
- [x] Added rigorous stage-level functional checks:
  - [x] Python writes `trinity_svf_stage_debug.safetensors`;
  - [x] Elixir accepts `--stage-dir` and writes host `torch_v` stage tensors;
  - [x] Elixir reads Python stage tensors onto `Nx.BinaryBackend` to avoid EXLA donation;
  - [x] comparator supports `--strict-stage-tolerances`;
  - [x] comparator prints top differing tensor indices and values.
- [x] Added fast Mix task option parsing tests.
- [x] Updated `README.md`, `priv/sakana_trinity/README.md`, and `priv/sakana_trinity/scripts/SVD_PARITY_DEBUG.md`.
- [x] Added `docs/sakana_svd_byte_match_rigor_plan.md`.

## Verification

- [x] `python3 -m py_compile priv/sakana_trinity/scripts/*.py`
- [x] `XLA_TARGET=cuda12 mix test`
- [x] `mix credo --strict`
- [x] `mix dialyzer`
- [x] Runtime Python parity report with readback diagnostics
- [x] Runtime Elixir parity report with `--semantic-only`
- [x] Report comparison confirming the remaining exact-hash gap is after Python component readback

## Iterative Debugging Continuation

- [x] After completing the checklist above, proceed to iteratively debug this issue, picking up where the other agent left off.
- [ ] Continue byte-match investigation only after `--strict-stage-tolerances` stays green.
