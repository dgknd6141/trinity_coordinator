# Configurable Provider Pools

This guide describes how to replace the current static OpenAI-compatible agent
map with a configurable provider-pool system.

The router outputs an agent id. Today that id maps to a static, in-code set of
OpenAI-compatible specs. Production use needs named pools, provider-specific
configuration, validation, credentials from runtime config, and clear separation
between provider metadata and secrets.

## Implementation Discipline

Use TDD/RGR for every provider-pool behavior: red validation test, minimal green
parser or integration change, then refactor. Maintain a live checklist and
revise it whenever config shape, provider adapter behavior, secret handling, or
orchestrator integration changes.

After context compaction, re-read this guide, inspect the active checklist, run
`git status --short`, and rerun the smallest affected provider-pool test before
editing. Complete the milestone quality gate before advancing. Before merge,
run the final quality gate, commit only QA-passing changes, and push every repo
touched by the work.

## Target Contract

Provider pools should be configured without code changes:

```elixir
config :trinity_coordinator, :provider_pools,
  default: [
    %{id: 0, name: :fast_openai, provider: :openai, model: "gpt-4o-mini"},
    %{id: 1, name: :reasoning_openai, provider: :openai, model: "gpt-5.4"},
    %{id: 2, name: :local_llama, provider: :openai_compatible, model: "llama", base_url: "http://127.0.0.1:11434/v1"}
  ]
```

At runtime:

```elixir
pool = TrinityCoordinator.ProviderPool.fetch!(:default)
AgentPool.call_agent(0, messages, provider_pool: pool)
```

The coordination head must be built with `num_agents == ProviderPool.size(pool)`.

## Design Constraints

- Provider specs are metadata, not secrets.
- API keys come from environment, runtime config, or explicit opts.
- Unknown provider ids fail before network dispatch.
- Unknown agent ids fail before network dispatch.
- Provider adapters implement a shared behaviour.
- Tests must not perform real provider calls unless explicitly credential-gated.
- Provider-pool validation should be fast and deterministic.

## Proposed Modules

- `ProviderPool.Spec`: normalized agent/provider spec.
- `ProviderPool`: load, validate, fetch, and size pools.
- `ProviderPool.Config`: config parsing helpers.
- `AgentPool.Adapter`: existing behaviour, extended if needed.
- `AgentPool.OpenAI`: current adapter.
- `AgentPool.OpenAICompatible`: adapter for custom base URLs.

## Provider Spec Shape

```elixir
%TrinityCoordinator.ProviderPool.Spec{
  id: 0,
  name: :fast_openai,
  provider: :openai,
  model: "gpt-4o-mini",
  base_url: nil,
  timeout_ms: 30_000,
  max_tokens: 200,
  temperature: 0.2,
  metadata: %{}
}
```

Validation rules:

- ids are contiguous integers starting at 0 unless explicitly configured
  otherwise,
- names are unique atoms,
- providers are known,
- model is a non-empty string,
- base URL is required for `:openai_compatible`,
- timeouts and token limits are positive integers.

## TDD/RGR Checklist

Maintain and revise this checklist during implementation.

- [ ] Red: config parser rejects duplicate ids.
- [ ] Green: implement provider spec normalization.
- [ ] Red: config parser rejects unknown providers.
- [ ] Green: add provider validation.
- [ ] Red: pool size matches highest contiguous id count.
- [ ] Green: implement `ProviderPool.size/1`.
- [ ] Red: `AgentPool.call_agent/3` accepts a provider pool.
- [ ] Green: route selected agent through the supplied pool.
- [ ] Red: orchestrator derives default `num_agents` from provider pool.
- [ ] Green: integrate `ProviderPool` with `Orchestrator`.
- [ ] Red: OpenAI-compatible adapter respects configured base URL.
- [ ] Green: implement adapter without leaking credentials.
- [ ] Red: README/demo show pool selection.
- [ ] Green: add docs and demo option.

## Implementation Plan

### Milestone 1: Spec Normalization

Add `ProviderPool.Spec` and tests for valid/invalid specs.

Quality gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/provider_pool
```

### Milestone 2: AgentPool Integration

Change `AgentPool.call_agent/3` to accept:

```elixir
provider_pool: pool
```

Keep the existing static default temporarily as the default pool.

Quality gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/agent_pool_test.exs
```

### Milestone 3: Orchestrator Integration

The orchestrator should derive `num_agents` from the pool unless explicitly
overridden:

```elixir
pool = ProviderPool.fetch!(:default)
num_agents = ProviderPool.size(pool)
```

Quality gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/orchestrator_test.exs
```

### Milestone 4: Runtime Configuration

Load pools from `config/runtime.exs` or application env. Document examples for:

- OpenAI,
- OpenAI-compatible local endpoints,
- disabled provider entries.

Quality gate:

```bash
XLA_TARGET=cuda12 mix docs
```

## Demo Requirements

Extend the demo:

```bash
XLA_TARGET=cuda12 mix trinity.demo --pool default
```

The demo should print:

- pool name,
- agent count,
- provider names and model ids,
- selected agent id,
- selected provider/model.

Do not call provider APIs in the default demo.

## Compaction Handoff

After compaction, re-read this guide, check the active provider-pool checklist,
run `git status --short`, inspect `AgentPool` and `Orchestrator`, then run the
smallest affected provider-pool test before editing.

## Final Quality Gate

```bash
XLA_TARGET=cuda12 mix format --check-formatted
XLA_TARGET=cuda12 mix test
XLA_TARGET=cuda12 mix test --only integration
XLA_TARGET=cuda12 mix credo --strict
XLA_TARGET=cuda12 mix dialyzer
XLA_TARGET=cuda12 mix docs
```

Commit and push all QA-passing repos touched by the work.
