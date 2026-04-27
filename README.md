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

`trinity_coordinator` is an Elixir implementation of the core TRINITY routing
mechanism: use a compact language model as a state encoder, extract the
second-to-last token hidden state for the current transcript, and pass that
vector through a tiny Axon coordination head that selects both an agent and a
role.

The core path is real `Bumblebee` + `Nx` + `EXLA` + `Axon`:

- real tokenizer and SLM loading through `Bumblebee`,
- real SLM forward pass through `Axon.predict/3`,
- real hidden-state extraction from model outputs,
- real second-to-last token vector slicing,
- real Axon dense coordination head inference,
- real supervised head training on extracted hidden-state vectors,
- CUDA-backed verification on hosts where EXLA exposes a CUDA platform.

Provider LLM execution is intentionally separated from router verification.
`AgentPool` uses a real OpenAI-compatible HTTP adapter when credentials are
present, while tests can verify the router path without pretending to call an
external model.

## Contents

- [Research Context](#research-context)
- [Active Direction](#active-direction)
- [Current Status](#current-status)
- [Requirements](#requirements)
- [GPU Setup](#gpu-setup)
- [Demonstration](#demonstration)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Modules](#modules)
- [Testing](#testing)
- [Development Checks](#development-checks)
- [Roadmap](#roadmap)
- [Credits](#credits)
- [References](#references)
- [License](#license)

## Research Context

TRINITY proposes a lightweight coordinator for multi-agent reasoning. At each
turn, a compact SLM reads the transcript and exposes a contextual hidden-state
vector. A small coordination head maps that vector to `L + 3` logits: `L` logits
select an LLM from the pool, and three logits select the role:
`Thinker`, `Worker`, or `Verifier`.

The local paper sources emphasize a few engineering constraints that this repo
now follows:

- The router uses the penultimate-token hidden state, not generated text.
- The head is a lightweight linear projection by default.
- Role routing matters: Thinker plans, Worker executes, Verifier emits
  `ACCEPT` or `REVISE`.
- The paper's full training story includes label-free terminal reward
  optimization. That remains important, but this repo is no longer trying to
  reproduce that experiment first. The active buildout is the usable coordinator
  system: load the Qwen-class coordinator in Elixir, import Sakana-provided
  artifacts, route through the adapted coordinator/head path, and make the whole
  path observable and testable on GPU.

## Active Direction

The current direction is artifact-first TRINITY coordinator bring-up in Elixir.
The goal is to build a usable system from the concrete resources we have, not to
start by reproducing Sakana's full training run.

Active path:

- Run `Qwen/Qwen3-0.6B` directly through `Bumblebee.Text.Qwen3` on
  `{EXLA.Backend, client: :cuda}`.
- Extract real Qwen hidden-state vectors through the same
  `Extractor`/`CoordinationHead` contract used by the tiny test profile.
- Run the artifact exporter to materialize router-head + adapted-tensor artifacts once.
- Convert Sakana's router ES vector artifact into safetensors so the Elixir
  runtime can load it without Python at application runtime.
- Split that vector into the exact inspected layout:
  - first `9216` values: SVF scale offsets for the selected Qwen tensor set
  - final `10240` values: router head weights reshaped to `{10, 1024}`
- Load the router head weights into the existing Axon routing head and route a
  real Qwen hidden vector through it.
- Implement the SVD/SVF mechanics in Elixir/Nx:
  - deterministic tensor selection
  - singular-value counting
  - Sakana normalization
  - adapted tensor reconstruction
  - params-container reinsertion using preserved path segments
- Keep artifact export and resume behavior as the explicit opt-in GPU lane, not a
  default `mix test` path.

Deferred path:

- Reproducing the actual Sakana training process end to end is deferred.
- The sep-CMA-ES implementation remains in the repo as a foundation and as a
  later route to experiment reproduction.
- Full reproduction work still needs task datasets, terminal reward plumbing,
  repeated trajectory evaluation, budget controls, and comparison against paper
  metrics.
- The immediate work is not to regenerate Sakana's weights. The immediate work
  is to correctly consume and apply the available artifacts and make the
  resulting coordinator operational.

This means the next milestones are integration and parity milestones, not
training-reproduction milestones:

- prove the imported Qwen/Sakana head path on CUDA,
- apply SVF-adapted tensors into the Bumblebee params tree,
- compare Elixir/Nx outputs against the Python reference path where practical,
- expose this coordinator path through demo/orchestrator commands,
- persist traces showing profile, backend, vector shape, logits shape, selected
  agent, and selected role.

## Current Status

Implemented and tested:

- `Runtime`: checks EXLA-supported platforms and provides scoped CUDA backend
  helpers without leaking global Nx backend state across tests.
- `Extractor`: formats transcripts, loads a real SLM/tokenizer, runs a real
  forward pass, extracts the final hidden-state tensor, and slices the
  second-to-last token vector.
- `CoordinationHead`: builds the dense Axon routing head, returns logits and
  route choices, builds training labels, and trains the head with real Axon and
  Polaris.
- `Orchestrator`: requires an SLM context, routes with the real extracted
  vector, injects the selected role, and dispatches to the provider boundary.
- `mix trinity.demo`: prints a complete, step-by-step GPU-backed demonstration.
- `sep-CMA-ES trainer`: deterministic codec/recombination loop, terminal-reward
  objective, and explicit provider-gated trajectory mode. This is now deferred
  reproduction infrastructure rather than the active mainline.
- `SLMProfile.qwen_coordinator/0`: loads `Qwen/Qwen3-0.6B` through the pinned
  upstream Bumblebee Qwen3 implementation on CUDA with `bf16` params.
- `SLMProfile.qwen_sakana_adapted/0`: loads `Qwen/Qwen3-0.6B` and applies
  generated adapted tensors/router head from
  `priv/sakana_trinity/adapted_qwen3_0_6b_layer26`.
- `TrinityCoordinator.Sakana.SVD`: loads the Sakana safetensors vector, splits
  the `9216 + 10240` layout, implements SVD/SVF reconstruction mechanics, loads
  the imported head into Axon, and supports adapted tensor reinsertion.
- `TrinityCoordinator.Sakana.Artifact`: validates artifact manifests and loads
  persisted artifacts back into the Qwen params tree and routing head.
- `Trace.JSONL` and `Trace.Hash`: serialize tensors by preserving original
  backend labels while hashing host-transferred tensor values, avoiding EXLA
  donated-buffer flakes.

The integration tests use `hf-internal-testing/tiny-random-gpt2` because it is
small enough for repeatable CI/local verification. It proves the mechanics with
a 32-dimensional hidden state. The paper-scale target is a Qwen-class SLM with
1024-dimensional hidden states; the same extractor/head API is dimension-driven
and builds the head from the observed vector width.

The Qwen/Sakana path is also tested directly:

- Qwen loads on CUDA and exposes a real `{1, 1024}` hidden vector.
- The selected layer-26 Qwen tensor set consumes exactly `9216` SVF offsets.
- The imported router head reshapes to `{10, 1024}` and routes through the real
  Axon head.
- A real Qwen hidden vector routes through the imported Sakana head on CUDA.
- The artifact export gate now exists as an explicit long-running smoke test and
  writes reusable checkpoints/artifacts for runtime loading.

## Requirements

- Elixir `~> 1.18`
- A working `mix`/OTP toolchain
- NVIDIA driver visible to `nvidia-smi` for CUDA verification
- Internet access for first-time Hugging Face model download
- `XLA_TARGET=cuda12` for the current Bumblebee-compatible dependency lane

Current dependency lane:

- `bumblebee` pinned to upstream `elixir-nx/bumblebee`
  `0fd8114cf5429af9236f100f3350986e9d823c02`
- `axon ~> 0.7`
- `nx ~> 0.9`
- `exla ~> 0.9`

Exact resolved dependency versions for this host (`mix.lock`):

- `bumblebee` git ref `0fd8114cf5429af9236f100f3350986e9d823c02`
- `axon 0.7.0`
- `nx 0.10.0`
- `exla 0.10.0`
- `xla 0.9.1`

`qwen_cuda_ready` outcome on this host:

- The pinned Bumblebee commit exposes `Bumblebee.Text.Qwen3`.
- `SLMProfile.qwen_coordinator/0` loads `Qwen/Qwen3-0.6B` through
  `Bumblebee.Text.Qwen3` with params on `{EXLA.Backend, client: :cuda}` and
  `type: :bf16`.
- `XLA_TARGET=cuda12 mix test --only qwen --trace` verifies a real
  `{1, 1024}` Qwen hidden-state vector on `EXLA.Backend<cuda:...>`.

That stack uses the CUDA12 EXLA target on this host.

The Qwen profile gate applies to profile selection:

- `qwen_coordinator` is expected to load successfully on GPU-backed EXLA.
- do not treat CPU-only Qwen runs as passing the production profile gate.
- do not claim full Sakana parity until the SVF-adapted Qwen params and router
  outputs are compared against the Python reference path.

## GPU Setup

Run project commands with:

```bash
XLA_TARGET=cuda12 mix test
```

To confirm CUDA is visible from Elixir:

```bash
XLA_TARGET=cuda12 mix run -e 'IO.inspect(EXLA.Client.get_supported_platforms())'
```

Expected shape:

```elixir
%{host: _, cuda: _}
```

The project config registers an EXLA CUDA client and disables large upfront GPU
memory preallocation:

```elixir
config :exla,
  clients: [
    cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.35],
    host: [platform: :host]
  ],
  preferred_clients: [:cuda, :host]
```

## Demonstration

Run the full real-router demo:

```bash
XLA_TARGET=cuda12 mix trinity.demo
```

The command prints:

- the active `XLA_TARGET`,
- EXLA-supported platforms,
- the transcript passed to the tokenizer,
- tokenizer input shapes,
- final hidden-state tensor shape,
- second-to-last token vector shape and backend,
- real SLM feature batch shape and backend,
- real Axon training status,
- routing logits,
- selected agent id and role id.

This is the fastest way to verify that the local stack is actually using
`EXLA.Backend<cuda:0>` for the core tensors.

## Quick Start

Load a small SLM, extract the hidden-state vector, train a tiny head, and route:

```elixir
TrinityCoordinator.Runtime.with_cuda_backend!(fn ->
  {:ok, {model_info, tokenizer}} =
    TrinityCoordinator.Extractor.load_slm_model(
      {:hf, "hf-internal-testing/tiny-random-gpt2"},
      Bumblebee.Text.Gpt2,
      :base
    )

  messages = [%{role: "user", content: "Find a strategy for a short proof."}]

  {:ok, metadata} =
    TrinityCoordinator.Extractor.extract_penultimate_hidden_state_with_metadata(
      model_info,
      tokenizer,
      messages
    )

  model = TrinityCoordinator.CoordinationHead.build_model(32, 3, 3)
  {init_fn, _predict_fn} = Axon.build(model)
  params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())

  route = TrinityCoordinator.CoordinationHead.route(model, params, metadata.vector, 3, 3)
  {route.agent_id, route.role_id}
end)
```

For supervised head training:

```elixir
{:ok, features} =
  TrinityCoordinator.Extractor.extract_batch_penultimate_hidden_states(
    model_info,
    tokenizer,
    [
      [%{role: "user", content: "Solve a symbolic algebra problem."}],
      [%{role: "user", content: "Write and debug a small function."}]
    ]
  )

labels = TrinityCoordinator.CoordinationHead.build_labels([0, 1], [1, 1], 3, 3)

trained_state =
  TrinityCoordinator.CoordinationHead.train_supervised(
    model,
    features,
    labels,
    num_agents: 3,
    num_roles: 3,
    epochs: 20,
    compiler: EXLA
  )
```

## Architecture

1. `StateManager` stores the transcript as role/content maps.
2. `Extractor.format_messages/1` renders a deterministic transcript string.
3. `Bumblebee.apply_tokenizer/2` tokenizes the transcript.
4. `Axon.predict/3` runs the SLM forward pass.
5. `Extractor` finds the final hidden-state tensor and slices token index
   `sequence_length - 2` into a `{batch, hidden_dim}` vector.
6. `CoordinationHead` applies one dense layer to produce
   `num_agents + num_roles` logits.
7. `CoordinationHead.route/5` splits logits into agent and role segments and
   chooses each with `argmax`.
8. `RoleInjector` prepends the selected role prompt.
9. `AgentPool` dispatches to a real provider adapter.
10. `Orchestrator` loops until `Verifier` returns `ACCEPT` or max turns is
    reached.

## Modules

### `TrinityCoordinator.Runtime`

CUDA visibility and scoped backend setup helpers. Runtime code uses
process-local backend selection instead of global Nx mutation.

### `TrinityCoordinator.Extractor`

SLM loading, transcript formatting, tokenization, model forward pass, hidden
state extraction, and batch feature extraction.

### `TrinityCoordinator.CoordinationHead`

Axon model construction, logits, route selection, label construction, and real
supervised training.

### `TrinityCoordinator.Orchestrator`

Multi-turn policy driver. It requires a real SLM context and passes provider
calls through `AgentPool`.

### `TrinityCoordinator.AgentPool`

Provider boundary for selected agents. The built-in adapter is
OpenAI-compatible and expects `OPENAI_API_KEY` or `openai_api_key: ...`.

### `TrinityCoordinator.ProviderPool`

Provider pool management with normalized runtime specs, validation, and explicit
pool selection. Use a named pool to control which providers are available.

### `TrinityCoordinator.SLMProfile`

Named coordinator model profiles. The production-intent profile is
`:qwen_coordinator`, which loads `Qwen/Qwen3-0.6B` through
`Bumblebee.Text.Qwen3` on CUDA.

The runtime-adapted profile is `:qwen_sakana_adapted`, which applies
`priv/sakana_trinity/adapted_qwen3_0_6b_layer26` artifacts to the Qwen params
and routing head.

### `TrinityCoordinator.Sakana.SVD`

Sakana artifact import and SVD/SVF mechanics:

- loads `priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors`,
- reads tensor `trinity_router_es_vector`,
- splits SVF scale offsets and router head weights,
- selects Qwen tensors with Sakana-compatible matrix rules,
- counts singular values deterministically,
- reconstructs adapted tensors with Sakana's normalization formula,
- reinserts adapted tensors into nested Bumblebee params containers,
- loads imported head weights into the Axon routing head.

### `TrinityCoordinator.Sakana.Artifact`

Runtime artifact loader and patcher for Sakana-adapted Qwen artifacts:

- validate manifest identity and completeness,
- load checkpointed or merged artifacts,
- patch adapted tensors into Qwen params,
- load and patch the Sakana router head.

### Export Mix Task

Run this once to materialize the canonical artifact set:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.export_adapted
```

Useful options:

- `--out`: output directory override (default `priv/sakana_trinity/adapted_qwen3_0_6b_layer26`)
- `--source-vector`: source safetensors path
- `--tensor-name`: source tensor key
- `--resume`: reuse verified checkpoints
- `--force`: rebuild output directory
- `--only-index`: export a single tensor for smoke testing
- `--skip-existing`: skip verified checkpoints during export

Resumption policy:

- `--resume` recomputes any checkpoint that is missing or fails integrity checks
  (`status`, tensor shape, tensor type, and checksum must all match).
- `--force` removes and recreates the output directory, then performs a clean run.
- For repeated smoke checks, use `--only-index 1`.
- The command also writes `export.log.jsonl` in the output directory. It includes
  `export_started`, per-tensor `tensor_export_*`, merge, and completion/failure
  events with an `event_time_utc` timestamp, so partial runs remain auditable.

Runtime and recovery notes:

- Canonical profile path expects a complete manifest:
  `manifest["status"] == "complete"` and `manifest["export_complete"] == true`.
- If an export fails, rerun with `--resume` to continue from verified
  checkpoints. `--resume` always validates manifest identity and requires it to
  match the same source vector, profile, and tensor selection.
- If identity changed or any required source file changed (e.g. vector path),
  `--resume` aborts with an explicit resume-blocking error.
- If recovery is unreliable, use `--force` to remove prior state and rebuild from
  scratch.

To inspect failure state manually:

- open `manifest.json` for machine-readable state, especially `status`,
  `selected_tensors[].status`, and any `error` fields;
- open `export.log.jsonl` for the latest event sequence.

Canonical smoke commands:

- `XLA_TARGET=cuda12 mix trinity.sakana.export_adapted --only-index 1 --force`
- `XLA_TARGET=cuda12 mix trinity.sakana.export_adapted --resume --only-index 1`

Canonical full export profile command:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.export_adapted
```

Use this to materialize all `9` selected tensors and merge
`adapted_tensors.safetensors` (status becomes `complete` when all checkpoints are
written and merged).

## Testing

Fast tests:

```bash
XLA_TARGET=cuda12 mix test
```

This excludes `:integration`, `:expensive_qwen_svd`, and `:slow_qwen_svd` by
default.

Integration tests load the tiny Hugging Face model and assert CUDA-backed
tensors where applicable:

```bash
XLA_TARGET=cuda12 mix test --only integration
```

Qwen/Sakana focused gates:

```bash
XLA_TARGET=cuda12 mix test --only qwen --trace
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --only qwen --exclude expensive_qwen_svd --trace
```

The first canonical export run can take multiple minutes with CUDA warmup. After a
successful canonical export, repeated runs with the checkpoint cache should be
faster.

Full opt-in SVF reconstruction/import gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --only expensive_qwen_svd --trace
```

That gate performs an exporter smoke path (`only-index=1`) and verifies
`manifest.json`, router head artifact, and checkpoint persistence. It is
intentionally excluded from plain `mix test`.

Canonical runtime profile test:

```bash
XLA_TARGET=cuda12 mix test --only qwen_sakana_adapted --trace
```

Provider calls are not silently simulated in core tests. Without credentials,
provider-boundary tests assert the real adapter returns
`:missing_openai_api_key`.

### Provider pool example

```elixir
Application.put_env(:trinity_coordinator, :provider_pools,
  default: [
    [id: 0, name: :openai_fast, provider: :openai, model: "gpt-4o-mini"],
    [id: 1, name: :local, provider: :openai_compatible, model: "llama", base_url: "http://127.0.0.1:11434/v1"]
  ])

TrinityCoordinator.Orchestrator.run_loop(
  pid,
  model,
  params,
  max_turns: 4,
  slm_context: {model_info, tokenizer},
  provider_pool: :default,
  agent_pool_opts: [openai_api_key: "<api-key>"]
)
```

## Development Checks

Before committing:

```bash
XLA_TARGET=cuda12 mix format
XLA_TARGET=cuda12 mix test
XLA_TARGET=cuda12 mix test --only integration
XLA_TARGET=cuda12 mix credo --strict
XLA_TARGET=cuda12 mix dialyzer
XLA_TARGET=cuda12 mix docs
```

## Roadmap

- Finish the active Qwen/Sakana artifact lane: complete and validate canonical
  artifact export, apply adapted tensors in the runtime Qwen coordinator path,
  and compare Elixir/Nx outputs against the
  Python reference where practical. See
  [Elixir-Native SVD Decomposition For TRINITY](docs/elixir_svd_decomposition.md).
- Expose the imported Qwen/Sakana coordinator through demo/orchestrator
  commands, including profile selection, backend metadata, vector shape, logits
  shape, selected agent, and selected role. See
  [Production Qwen SLM Profile](docs/production_qwen_slm_profile.md).
- Preserve the sep-CMA-ES training implementation as deferred reproduction
  infrastructure. Full paper-style label-free training is not the current
  mainline; resume it after artifact import/parity is stable. See
  [sep-CMA-ES Training For Terminal Rewards](docs/sep_cma_es_training.md).
- Add block-diagonal and sparse head variants from the appendix for parameter
  efficiency and ablation work. See
  [Coordination Head Variants](docs/coordination_head_variants.md).
- Add trace persistence for every routed turn: transcript hash, hidden-state
  shape/backend, logits, selected agent, selected role, provider response, and
  verifier result. See [Trace Persistence](docs/trace_persistence.md).
- Add configurable provider pools instead of the current static OpenAI-compatible
  mapping. See
  [Configurable Provider Pools](docs/configurable_provider_pools.md).
- Add benchmark harnesses for task-family separability, routing accuracy, and
  turn-budget behavior. See
  [Benchmark Harnesses](docs/benchmark_harnesses.md).
- Add real multi-turn provider smoke tests gated by explicit credentials and
  budget controls. See [Provider Smoke Tests](docs/provider_smoke_tests.md).

## Credits

This repository is a research implementation inspired by *TRINITY: An Evolved
LLM Coordinator*.[1] The paper motivates the hidden-state router, the
Thinker/Worker/Verifier role split, the lightweight coordination head, and the
preference for sep-CMA-ES under tight terminal-reward budgets.

This package does not claim to reproduce the paper's reported scores. It focuses
first on a robust, inspectable Elixir implementation of the core router
mechanics.

## References

[1] Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, and
Yujin Tang. *TRINITY: An Evolved LLM Coordinator*. arXiv:2512.04695, 2026.
https://arxiv.org/abs/2512.04695

## License

This project is released under the MIT License.
