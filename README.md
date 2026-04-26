# TrinityCoordinator

<p align="center">
  <img src="assets/trinity_coordinator.svg" alt="TRINITY Coordinator" width="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-0f172a?style=for-the-badge" />
  </a>
  <a href="https://github.com/nshkrdotcom/trinity_coordinator">
    <img alt="GitHub" src="https://img.shields.io/badge/github-nshkrdotcom%2Ftrinity__coordinator-111827?style=for-the-badge&logo=github" />
  </a>
</p>

TRINITY is a compact multi-agent routing loop implemented in Elixir. This project
implements the TRINITY idea from the papers and notes in this workspace:

- run a **small language model (SLM)** with no text generation,
- extract the **second-to-last token hidden state** for the current conversation,
- apply a tiny learned **coordination head** in Axon,
- route the call to an LLM agent and role (**Thinker / Worker / Verifier**),
- loop until the Verifier emits `ACCEPT` or a max-turns cutoff is reached.

The package includes a full TDD implementation with mocked external execution and
an optional real hidden-state path using `Bumblebee`, `Nx`, and `Axon`.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Router Architecture](#router-architecture)
- [Modules](#modules)
- [Testing](#testing)
- [Configuration and Runtime Notes](#configuration-and-runtime-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

### Routing Core

- **StateManager**: transcript state storage with BEAM-friendly process state.
- **Extractor**: tensor utilities and hidden-state extraction from real SLM outputs.
- **CoordinationHead**: tiny Axon linear head returning `(agent_id, role_id)`.
- **RoleInjector**: injects role-specific system prompts.
- **AgentPool**: pluggable execution layer for LLM calls (mock implementation by
  default; replace with provider-backed calls for production).
- **Orchestrator**: recursive orchestration loop with configurable max turns.

### Quality gates

- **ExUnit tests**: unit and integration tests.
- **Credo strict** and **Dialyzer** configured and clean in the local environment.
- **Hex-ready metadata** for documentation and package publication.

## Requirements

- Elixir `~> 1.18`
- OTP compatible with your current `mix` toolchain
- Internet access for integration tests that load a tiny model from Hugging Face
  (`hf-internal-testing/tiny-random-gpt2`)
- Optional: EXLA toolchain for improved numeric throughput

## Installation

If you plan to use this package from Hex:

```elixir
def deps do
  [
    {:trinity_coordinator, "~> 0.1.0"}
  ]
end
```

Development dependencies are kept in `dev` scope for quality tooling:
`credo`, `dialyxir`, and `ex_doc`.

## Quick Start

Start with a small in-memory state and route through a mocked orchestrator:

```elixir
{:ok, pid} = TrinityCoordinator.StateManager.start_link([
  %{role: "user", content: "Prove the loop reaches ACCEPT."}
])

model = TrinityCoordinator.CoordinationHead.build_model(1024, 7, 3)
{init_fn, _predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())

TrinityCoordinator.Orchestrator.run_loop(pid, params, 5)
```

To exercise the real extraction path with a tiny local model:

```elixir
{:ok, {model_info, tokenizer}} =
  TrinityCoordinator.Extractor.load_slm_model(
    {:hf, "hf-internal-testing/tiny-random-gpt2"},
    Bumblebee.Text.Gpt2,
    :base
  )

{:ok, vector} =
  TrinityCoordinator.Extractor.extract_penultimate_hidden_state_from_texts(
    model_info,
    tokenizer,
    [%{"role" => "user", "content" => "Hello world"}]
  )
```

The returned `vector` shape for the tiny model is `{1, 32}`.

## Router Architecture

The coordinator implements two operational modes:

### 1) Mock mode (default)

`Orchestrator.run_loop/3` uses a fixed placeholder tensor
`Nx.broadcast(0.5, {1, 1024})` to avoid heavy model startup during fast test
passes and local smoke checks.

### 2) Real-model mode

`Orchestrator.run_loop/4` accepts an `{model_info, tokenizer}` tuple so the SLM
hidden-state path is active:

- transcript formatting -> tokenizer -> Axon forward pass,
- extraction of final-layer tensor,
- slicing of penultimate token,
- router forward pass,
- role injection + call dispatch.

## Modules

### TrinityCoordinator.Extractor

Utilities for extracting hidden states and the second-to-last token vector.

### TrinityCoordinator.CoordinationHead

Linear routing head (single dense output) that maps concatenated state to:
`num_agents + num_roles` logits, then returns `argmax` pair for agent and role.

### TrinityCoordinator.RoleInjector

Converts role choice into a deterministic system prompt:
Thinker, Worker, Verifier.

### TrinityCoordinator.AgentPool

Default mocked adapter that returns deterministic text responses; replace with
provider integration as needed.

### TrinityCoordinator.Orchestrator

Recursive policy driver that applies `StateManager`, routing, prompting, and exit
conditions (`ACCEPT` or `max_turns`).

## Testing

```bash
mix test
mix test --only integration
```

Integration tests are networked and can take longer due to model download and
initialization. For repeated local runs use:

```bash
EXLA_CPU_ONLY=true mix test --only integration
```

Static checks:

```bash
mix credo --strict
mix dialyzer
```

## Configuration and Runtime Notes

- `StateManager` stores messages as `%{role: "...", content: "..."}` maps compatible
  with chat template formatting.
- Roles are hard-coded as `Thinker`, `Worker`, `Verifier` with injectable prompts.
- Max turn count defaults to `5` but can be configured per invocation.
- For production routing, replace `AgentPool.call_agent/2` with provider clients
  (OpenAI/Anthropic/Gemini/local adapters) while preserving return shape.

## Roadmap

- Replace mocked `AgentPool` with concrete adapters and provider credentials.
- Persist experiment traces for routing metrics and acceptance analytics.
- Add `:train`/`fit` path for `CoordinationHead` with reward signals.
- Add CLI entrypoint for running complete TRINITY sessions directly from command line.

## Contributing

Contributions should maintain passing tests, `mix credo --strict`, and
`mix dialyzer` before merge. This repo is research-oriented and welcomes
targeted improvements to router quality, docs, and integration reliability.

## License

This project is released under the MIT License.
