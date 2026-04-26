# Provider Smoke Tests

This guide describes how to add real multi-turn provider smoke tests gated by
explicit credentials and budget controls.

The default test suite must never spend provider budget or require credentials.
Provider smoke tests are opt-in checks that prove the full orchestration loop can
call real LLM providers, inject roles, observe verifier output, and stop within
budget.

## Implementation Discipline

Use TDD/RGR for every smoke-test behavior: red guard/budget/trace assertion,
minimal green implementation, then refactor. Maintain a live checklist and
revise it whenever provider requirements, budget controls, trace schema, or test
tags change.

After context compaction, re-read this guide, inspect the active checklist, run
`git status --short`, and rerun the smallest affected non-provider guard test
before editing. Complete the milestone quality gate before advancing. Before
merge, run the final quality gate, commit only QA-passing changes, and push
every repo touched by the work.

## Target Contract

Provider tests run only when explicitly enabled:

```bash
TRINITY_ENABLE_PROVIDER_TESTS=1 \
TRINITY_PROVIDER_BUDGET_USD=1.00 \
OPENAI_API_KEY=... \
XLA_TARGET=cuda12 mix test --only provider_smoke --trace
```

If not enabled, tests skip with a clear message.

## Safety Requirements

- Require `TRINITY_ENABLE_PROVIDER_TESTS=1`.
- Require a positive `TRINITY_PROVIDER_BUDGET_USD`.
- Require provider credentials.
- Use low token limits.
- Use a tiny task set.
- Persist a redacted trace.
- Fail closed if budget accounting cannot be initialized.
- Never print API keys or authorization headers.

## Proposed Test Tags

- `:provider_smoke`
- `:openai`
- `:budgeted`
- `:multi_turn`

Keep these separate from `:integration`. Normal integration tests prove real
SLM/router mechanics without provider spend.

## Proposed Modules

- `ProviderTest.Budget`: local budget accounting helper.
- `ProviderTest.Credentials`: environment validation.
- `ProviderTest.Tasks`: tiny safe task fixtures.
- `ProviderTest.Assertions`: verifier and trace assertions.

These can live under `test/support` unless production code needs reusable budget
guards.

## TDD/RGR Checklist

Maintain and revise this checklist during implementation.

- [x] Red: provider tests skip when enable env var is absent.
- [x] Green: implement credential/budget guard.
- [x] Red: budget parser rejects missing, zero, and invalid values.
- [x] Green: implement budget parsing.
- [x] Red: provider call options enforce low token limits and timeouts.
- [x] Green: add smoke-test provider options.
- [x] Red: real smoke test requires redacted trace path.
- [x] Green: integrate trace persistence.
- [x] Red: multi-turn test asserts at least one role prompt is injected.
- [x] Green: run real orchestrator with a tiny task.
- [x] Red: verifier termination or max-turn outcome is recorded.
- [x] Green: assert trace contains final run outcome.
- [x] Update README and this guide.

Provider smoke tests must use the real `AgentPool` adapter and real provider
responses. Do not replace provider calls with local fakes in this suite.

## Smoke Task Design

Use tasks that are:

- short,
- non-sensitive,
- cheap,
- deterministic enough to validate,
- not benchmark claims.

Example:

```elixir
%{
  id: "smoke_arithmetic_001",
  messages: [%{role: "user", content: "What is 17 + 25? Answer briefly."}],
  expected_substring: "42",
  max_turns: 3
}
```

Do not include private user data in smoke fixtures.

## Budget Accounting

Minimum accounting:

- configured budget cap,
- provider model,
- max tokens,
- prompt token estimate if available,
- completion token count if provider returns usage,
- estimated cost,
- stop when cap is reached.

If usage data is unavailable, use a conservative configured per-call estimate.

## Trace Requirements

Each provider smoke run must emit a redacted trace with:

- run id,
- task id,
- provider,
- model,
- max tokens,
- selected agent,
- selected role,
- status code,
- response hash,
- verifier result,
- estimated cost.

Do not store full provider responses by default.

## Milestone Gates

Guard behavior:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/provider_smoke_guard_test.exs
```

Provider smoke:

```bash
TRINITY_ENABLE_PROVIDER_TESTS=1 \
TRINITY_PROVIDER_BUDGET_USD=1.00 \
OPENAI_API_KEY=... \
XLA_TARGET=cuda12 mix test --only provider_smoke --trace
```

Full non-provider gate:

```bash
XLA_TARGET=cuda12 mix format --check-formatted
XLA_TARGET=cuda12 mix test
XLA_TARGET=cuda12 mix test --only integration
XLA_TARGET=cuda12 mix credo --strict
XLA_TARGET=cuda12 mix dialyzer
XLA_TARGET=cuda12 mix docs
```

## Compaction Handoff

After compaction, re-read this guide, inspect the active checklist, confirm no
provider tests are running unexpectedly, run `git status --short`, and continue
from the next unchecked item.

## Commit Requirements

Commit and push only after the non-provider gate passes. Provider smoke results
should be summarized in the PR or commit notes, but raw provider traces should
not be committed unless they are redacted fixtures intentionally added for
tests.
