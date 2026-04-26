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
- The full paper favors sep-CMA-ES for terminal reward optimization, while the
  appendix also describes a supervised frozen-SLM path. This repo currently
  implements the supervised head-training path and leaves sep-CMA-ES as a
  roadmap item.

## Current Status

Implemented and tested:

- `Runtime`: checks EXLA-supported platforms and sets CUDA as the Nx backend.
- `Extractor`: formats transcripts, loads a real SLM/tokenizer, runs a real
  forward pass, extracts the final hidden-state tensor, and slices the
  second-to-last token vector.
- `CoordinationHead`: builds the dense Axon routing head, returns logits and
  route choices, builds training labels, and trains the head with real Axon and
  Polaris.
- `Orchestrator`: requires an SLM context, routes with the real extracted
  vector, injects the selected role, and dispatches to the provider boundary.
- `mix trinity.demo`: prints a complete, step-by-step GPU-backed demonstration.

The integration tests use `hf-internal-testing/tiny-random-gpt2` because it is
small enough for repeatable CI/local verification. It proves the mechanics with
a 32-dimensional hidden state. The paper-scale target is a Qwen-class SLM with
1024-dimensional hidden states; the same extractor/head API is dimension-driven
and builds the head from the observed vector width.

## Requirements

- Elixir `~> 1.18`
- A working `mix`/OTP toolchain
- NVIDIA driver visible to `nvidia-smi` for CUDA verification
- Internet access for first-time Hugging Face model download
- `XLA_TARGET=cuda12` for the current Bumblebee-compatible dependency lane

Current dependency lane:

- `bumblebee ~> 0.6`
- `axon ~> 0.7`
- `nx ~> 0.9`
- `exla ~> 0.9`

That stack uses the CUDA12 EXLA target on this host. A newer CUDA13 lane is
available for newer Nx/EXLA versions, but current Bumblebee constraints keep
this project on CUDA12 until the dependency set can be upgraded together.

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
TrinityCoordinator.Runtime.put_cuda_backend!()

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

CUDA visibility and backend setup helpers.

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

## Testing

Fast tests:

```bash
XLA_TARGET=cuda12 mix test
```

Integration tests load the tiny Hugging Face model and assert CUDA-backed
tensors where applicable:

```bash
XLA_TARGET=cuda12 mix test --only integration
```

Provider calls are not silently simulated in core tests. Without credentials,
provider-boundary tests assert the real adapter returns
`:missing_openai_api_key`.

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

- Add a production SLM profile for a Qwen-class coordinator model once the
  Bumblebee/Nx dependency lane supports it cleanly on this host. See
  [Production Qwen SLM Profile](docs/production_qwen_slm_profile.md).
- Implement sep-CMA-ES training for terminal binary rewards, matching the
  paper's label-free optimization path. See
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
