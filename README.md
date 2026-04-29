# TrinityCoordinator

<p align="center">
  <img src="assets/trinity_coordinator.svg" alt="TRINITY Coordinator" width="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-0f172a?style=for-the-badge" />
  </a>
  <a href="https://github.com/nshkrdotcom/trinity_coordinator">
    <img alt="GitHub" src="https://img.shields.io/badge/github-nshkrdotcom%2Ftrinity__coordinator-111827?style=for-the-badge&logo=github" />
  </a>
</p>

`trinity_coordinator` is an Elixir/Nx implementation of a TRINITY-style local
small-model router. The active project direction is to recreate the Python
Sakana/Qwen artifact application process in Elixir, prove it with rigorous
stage-level parity checks, and use the resulting Qwen coordinator to route real
LLM provider calls.

The current focus is parity and service foundation, not reproducing the original
training experiment.

The original paper sources and the full supplemental Python submission have
been audited. The supplemental checkpoint metadata is now treated as the
compatibility source of truth for the imported artifact path: Qwen3-0.6B,
layer-26 SVF, seven agents, five turns, no-generation penultimate hidden-state
extraction, and raw role order `solver`, `thinker`, `verifier`.

## Current Direction

The active lane is:

1. Load the same base `Qwen/Qwen3-0.6B` model.
2. Consume the Sakana router vector and SVD/SVF components.
3. Reconstruct adapted Qwen tensors in Elixir/Nx.
4. Prove the Elixir path against Python with stage-level checks and explicit
   tolerances.
5. Materialize reusable adapted artifacts.
6. Run the adapted small local coordinator in front of real provider-backed
   LLM calls.

The earlier experiment-reproduction lane, including sep-CMA-ES training
reproduction work, is shelved. It remains in the repository for now, but the
planned direction is to remove or archive it once the parity and service path is
fully stable.

## Current Status

Working now:

- Qwen3-0.6B loads through Bumblebee on EXLA CUDA.
- The Sakana router vector is converted to safetensors.
- The vector split is understood:
  - first `9216` values: SVF scale offsets;
  - final `10240` values: router-head weights reshaped to `{10, 1024}`.
- The Elixir SVD/SVF code reconstructs adapted tensors.
- Python and Elixir parity scripts emit detailed JSON reports.
- Python emits a stage tensor bundle from safetensors readback.
- Elixir emits comparable semantic `torch_v` stage tensors with `--stage-dir`.
- The fast semantic loop can reuse Python's `stage.source_f32`, skip wrong
  layouts, and run the required reconstruction check through EXLA without
  loading Qwen for every debug run.
- `--strict-stage-tolerances` is the required functional correctness gate.
- Full Python semantic export imports into canonical checkpoint-directory
  Elixir artifacts with 9 target-verified tensors, 9,216 singular offsets, and
  router head shape `{10, 1024}`.
- The adapted coordinator smoke loads those canonical artifacts, patches Qwen,
  and routes a fixed transcript on CUDA with hidden `{1, 1024}`, logits
  `{1, 10}`, agent logits `{7}`, and role logits `{3}`.
- Fixed-transcript router trace parity passes for exact transcript, token ids,
  router-head hash, and argmax agent/role ids. Hidden/logit vectors are compared
  with declared alignment thresholds because Python currently runs this trace on
  CPU while Elixir runs Qwen through EXLA CUDA.
- The adapted runtime loop can route through fake providers with persisted JSONL
  traces. The safe smoke path dispatches Worker first, Verifier second, and
  terminates on verifier `ACCEPT`.
- Thinker suggestions, verifier-before-worker failure, max-turn latest-worker
  termination, and provider failure tracing are covered by focused tests.

Current parity result:

- Original-submission `svd_weights.pt` generation succeeds and produces
  current Python safetensors readback hash
  `b4cab13f8a82ccaf49603356e658bc9b77f65b08a69678a7d053a2e4b3197c43`.
- Historical stored hash
  `600be6ab0f5a34325b9857182ccb5fce5971549a0ce8588cdacc992eda54014c`
  remains non-reproducible from that regenerated `.pt`.
- The bounded layer-26 all-selected replay checks 7 tensors, 70 stages, and 63
  required stages with `failed_required=0`.
- Source tensors, offsets, scaled singular values, and `u_scaled` byte-match;
  required f32 reconstruction stages pass explicit tolerances.
- Final `bf16` byte matching remains aspirational and is reported separately.
- Canonical import validation passes with `status=complete`,
  `artifact_layout=checkpoint_directory`, `selected_tensor_count=9`,
  `selected_singular_value_count=9216`, `loaded_tensor_count=9`, and
  `target_verified_count=9`.
- Adapted coordinator validation passes against
  `tmp/sakana_parity/adapted_artifacts_from_python`; the observed fixed-route
  smoke selected `agent_id=4`, `role_id=0`, public role `Worker`.
- Router trace parity passes with exact token ids and head hash, exact
  `agent_id=4`/`role_id=0`, hidden cosine `0.99449`, and logits cosine
  `0.99743`.

Recent non-matching Elixir final hashes have included
`bf089ea0607c93ae69f92bf7b9fcf71dc2a2b53d231cfe307b8cd6f4ef6a85ae` and
`74dc61d765c95e80ca7298b6e97f29a4fd76e2ae4bfb348b2abbffcbc5e0dff8`.
The stage report, not the final Elixir hash alone, is the correctness verdict.

## Start Here

Read the guides in this order:

1. [Onboarding](guides/onboarding.md)
2. [Current Direction And Planning](guides/current_direction.md)
3. [System Architecture](guides/system_architecture.md)
4. [Recreating The Python Parity Process](guides/python_parity_reconstruction.md)
5. [Stage Checks And Tolerances](guides/stage_checks_and_tolerances.md)
6. [Sakana Artifacts And Export](guides/artifacts_and_export.md)
7. [SVD Generation Runbook](guides/svd_generation_runbook.md)
8. [Service Buildout Plan](guides/service_buildout.md)
9. [Provider Service Hardening](guides/provider_service_hardening.md)
10. [Operations And Quality Gates](guides/operations_qc.md)
11. [Troubleshooting](guides/troubleshooting.md)

Runnable reviewer examples are in [Examples](examples/README.md).

Additional technical reference notes are included in HexDocs under
`Reference Notes`.

For implementation handoff work, the private checklist
`docs/priv/20260428/06_next_phase_execution_checklist.md` is the current
next-phase task board.

## Requirements

- Elixir `~> 1.18`.
- NVIDIA driver visible to `nvidia-smi`.
- `XLA_TARGET=cuda12`.
- Internet access for first-time Hugging Face model downloads.
- Python with PyTorch, Transformers, and safetensors for parity scripts.

Resolved core dependency lane:

- `nx 0.10.0`
- `exla 0.10.0`
- `axon 0.7.0`
- `bumblebee` pinned to `elixir-nx/bumblebee`
  `0fd8114cf5429af9236f100f3350986e9d823c02`

## Quick Verification

Run the fast Elixir suite:

```bash
XLA_TARGET=cuda12 mix test
```

Run static checks:

```bash
mix format --check-formatted
mix credo --strict
mix dialyzer
```

Build docs:

```bash
mix docs
```

## Current Parity Workflow

Generate Python report, components, and stage tensors:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Generate Elixir semantic report and stage tensors:

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

Those extra flags are the current recommended debug loop: skip native Nx SVD,
skip the large host CPU matmul, skip known-wrong V-layout diagnostics, and reuse
Python's stage source tensor instead of loading the full Qwen profile just to
recover the sample source.

Run the required functional parity gate:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

Use final byte equality only as an explicit opt-in target:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

That exact-byte gate is expected to fail in the current state while functional
stage parity passes.

## Adapted Coordinator Smoke

After generating and importing the full Python semantic export, validate the
live adapted Qwen coordinator directly:

```bash
XLA_TARGET=cuda12 mix trinity.hitl.adapted \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
```

This proves the runtime shape contract:

```text
adapted Qwen vector shape: {1, 1024}
adapted route logits shape: {1, 10}
adapted agent logits shape: {7}
adapted role logits shape: {3}
```

The live CUDA smoke proves the operational shape contract. The router trace
below adds the side-by-side Python/Elixir semantic check for tokenization,
hidden extraction, router-head weights, logits, and selected agent/role ids.

## Router Trace Parity

Generate the Python trace from the canonical artifact directory. On the current
RTX 5060 Ti host, PyTorch 2.7.1 does not ship CUDA kernels for `sm_120`, so this
trace is run on CPU:

```bash
uv run --python 3.11 \
  --with torch==2.7.1 \
  --with transformers==4.55.2 \
  --with accelerate==1.6.0 \
  --with safetensors \
  python priv/sakana_trinity/scripts/debug_sakana_router_trace.py \
    --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
    --device cpu \
    --model-torch-dtype bfloat16 \
    --out tmp/sakana_parity/python_router_trace_bf16_cpu.json
```

Compare from Elixir:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.router_trace \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --python-report tmp/sakana_parity/python_router_trace_bf16_cpu.json \
  --out tmp/sakana_parity/elixir_router_trace.json
```

Required exact checks: transcript hash, token ids, router-head hash,
hidden/logit shapes, and argmax agent/role ids. Hidden/logit numeric payloads
must pass declared cosine and relative-L2 alignment thresholds; max/mean
absolute errors remain diagnostics.

For the opt-in all-selected tensor gate, generate Python components and the
source-oriented all-selected stage bundle with original SVD components:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights path/to/svd_weights.pt \
  --all-selected-tensors \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Then replay a bounded layer-26 slice from those Python components:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --device-semantic-only \
  --preferred-layout-only \
  --source-from-python-stage \
  --all-selected-tensors \
  --selected-source-regex 'model\.layers\.26\.' \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

This path is deliberately explicit because it can materialize very large stage
tensors for the embedding and LM-head matrices. Keep embedding/LM-head replay
out of the monolithic EXLA command until the chunked large-tensor gate is in
place. Without `--svd-weights`, the Python script still requires
`--decompose-all-selected-if-missing`.

## Runtime Shape

The intended service path is:

1. Format and tokenize the transcript.
2. Run the adapted local Qwen coordinator on CUDA.
3. Extract the penultimate-token hidden state.
4. Route through the imported Sakana head.
5. Select agent and TRINITY role.
6. Inject the selected role prompt.
7. Dispatch to a configured LLM provider.
8. Persist trace metadata for audit and debugging.

Provider dispatch now enters the shared `:inference` boundary through
`TrinityCoordinator.AgentPool.Inference` for hosted, GeminiEx, and Agent Session
Manager specs. Live calls are still explicitly gated; tests verify routing and
provider-boundary behavior without pretending that external LLM calls happened.

Run the adapted mock-provider loop:

```bash
XLA_TARGET=cuda12 mix trinity.hitl.mock_loop \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --trace-out tmp/trinity_mock_trace.jsonl
```

Run the safe route demo:

```bash
XLA_TARGET=cuda12 mix trinity.route.demo \
  --mock \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --trace-out tmp/trinity_route_demo.jsonl
```

Live provider mode is explicitly gated:

```bash
TRINITY_ENABLE_PROVIDER_DEMO=1 XLA_TARGET=cuda12 mix trinity.route.demo \
  --profile qwen_sakana_adapted \
  --provider-pool configured \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --trace-out tmp/trinity_route_demo.jsonl
```

Without `--mock`, `--allow-live`, or `TRINITY_ENABLE_PROVIDER_DEMO=1`, live
provider demo mode fails before dispatch.

## Examples

The `examples/` directory contains runnable, no-provider reviewer diagnostics.

Local coordinator routing:

```bash
XLA_TARGET=cuda12 mix run examples/local_coordinator_route.exs -- \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --prompt "Select a TRINITY role for this reasoning task."
```

Mock orchestration trace:

```bash
XLA_TARGET=cuda12 mix run examples/mock_orchestration_trace.exs -- \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --prompt "Select a TRINITY role for this reasoning task." \
  --trace-out tmp/examples/mock_orchestration_trace.jsonl
```

These examples print the prompt, artifact identity, hidden/vector/logit shapes,
selected agent/role ids, mock provider turns, and trace summaries.

## Quality Standard

Before committing changes that affect parity or runtime behavior, run:

```bash
mix format --check-formatted
python3 -m py_compile priv/sakana_trinity/scripts/*.py
XLA_TARGET=cuda12 mix test
mix credo --strict
mix dialyzer
mix docs
```

When parity code changes, also run:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

## Project Files

- [License](LICENSE)
- [Changelog](CHANGELOG.md)
- [Onboarding](guides/onboarding.md)
- [Current Direction](guides/current_direction.md)
- [Operations And Quality Gates](guides/operations_qc.md)

## Attribution

This repository is a research implementation inspired by *TRINITY: An Evolved
LLM Coordinator*.[1] The paper motivates the hidden-state router, the
Thinker/Worker/Verifier role split, the lightweight coordination head, and the
preference for compact local coordination.

This package does not claim to reproduce the paper's reported scores. The active
focus is a robust, inspectable Elixir implementation of the Qwen/Sakana
coordinator path.

## References

[1] Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, and
Yujin Tang. *TRINITY: An Evolved LLM Coordinator*. arXiv:2512.04695, 2026.
https://arxiv.org/abs/2512.04695

## License

This project is released under the MIT License.
