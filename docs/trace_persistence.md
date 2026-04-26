# Trace Persistence

This guide describes how to persist every routed TRINITY turn in a reproducible,
auditable format.

Trace persistence is the backbone for training, benchmarking, debugging, and
provider-budget safety. A routed turn should leave enough structured evidence to
answer: what transcript was routed, what hidden-state representation was used,
what logits were produced, which agent and role were selected, what provider was
called, and why the loop stopped or continued.

## Implementation Discipline

Use TDD/RGR for every trace behavior: first add a failing schema, redaction,
sink, or integration test; then implement the smallest passing behavior; then
refactor. Maintain a live checklist and revise it whenever event names, schema
fields, redaction rules, or storage backends change.

After context compaction, re-read this guide, inspect the active checklist, run
`git status --short`, and rerun the smallest affected trace test before editing.
Complete the milestone quality gate before advancing. Before merge, run the
final quality gate, commit only QA-passing changes, and push every repo touched
by the work.

## Target Contract

The orchestrator should accept trace options:

```elixir
Orchestrator.run_loop(pid, model, params,
  slm_context: {model_info, tokenizer},
  trace: [
    enabled: true,
    sink: {:jsonl, "tmp/trinity_runs/run.jsonl"},
    run_id: "run_20260425_001"
  ]
)
```

Each turn emits one structured event:

```elixir
%{
  event: :turn_completed,
  run_id: run_id,
  turn: 0,
  transcript_hash: "...",
  hidden_state_shape: {1, 23, 32},
  vector_shape: {1, 32},
  vector_backend: "EXLA.Backend<cuda:0, ...>",
  logits: [...],
  agent_logits: [...],
  role_logits: [...],
  selected_agent: 2,
  selected_role: "Verifier",
  provider: :openai,
  provider_model: "gpt-4o-mini",
  response_hash: "...",
  verifier_status: :revise,
  duration_ms: 1234
}
```

Persist hashes and metadata by default. Persist full transcripts/responses only
when explicitly enabled.

## Data Model

Add modules under `lib/trinity_coordinator/trace/`:

- `Trace.Event`: event struct and validation.
- `Trace.Sink`: behaviour for trace writers.
- `Trace.JSONL`: append-only JSONL sink.
- `Trace.Redactor`: secret and content redaction.
- `Trace.Hash`: stable SHA-256 helpers for transcripts, responses, vectors, and
  params.
- `Trace.Context`: run id, sink, redaction mode, and per-run metadata.

## Event Types

Minimum event set:

- `:run_started`
- `:turn_started`
- `:slm_extracted`
- `:route_selected`
- `:provider_called`
- `:turn_completed`
- `:run_completed`
- `:run_failed`

Training and benchmark work may add:

- `:candidate_sampled`
- `:candidate_evaluated`
- `:generation_completed`
- `:benchmark_case_completed`

## Redaction Rules

Default trace files must not store secrets or full provider content.

Default mode stores:

- hashes,
- shapes,
- backends,
- ids,
- status values,
- timing,
- small numeric logits.

Explicit debug mode may store full content:

```elixir
trace: [content: :full]
```

This mode must be documented as unsafe for sensitive tasks and should never be
enabled in automated provider tests by default.

## TDD/RGR Checklist

Maintain and revise this checklist as the trace design evolves.

- [ ] Red: validate a minimal `Trace.Event`.
- [ ] Green: implement event struct and required field checks.
- [ ] Red: assert stable transcript hash for equivalent message maps.
- [ ] Green: implement canonical message serialization and SHA-256 hashing.
- [ ] Red: assert response redaction removes API keys and authorization headers.
- [ ] Green: implement `Trace.Redactor`.
- [ ] Red: assert JSONL sink appends exactly one line per event.
- [ ] Green: implement `Trace.JSONL`.
- [ ] Red: assert orchestrator emits route metadata without full content.
- [ ] Green: integrate trace context into `Orchestrator`.
- [ ] Red: assert provider boundary emits provider/model/status metadata.
- [ ] Green: integrate `AgentPool` event emission.
- [ ] Red: add integration test with real SLM extraction and trace output.
- [ ] Green: verify trace includes CUDA backend and logits.
- [ ] Update README, demo output, and this guide.

## Integration Points

### Extractor

Expose metadata already produced by
`extract_penultimate_hidden_state_with_metadata/3`:

- transcript,
- input shapes,
- hidden-state shape,
- vector shape,
- vector backend.

### Coordination Head

Use `CoordinationHead.route/5` so traces can capture:

- full logits,
- agent logits,
- role logits,
- selected ids.

### Agent Pool

Expose provider metadata before dispatch:

- provider,
- model,
- request shape,
- timeout,
- status.

Do not log authorization headers.

### Orchestrator

Own run and turn ids. A failed turn should still emit `:run_failed` with reason.

## Storage Format

Use JSONL first:

- append-only,
- easy to inspect,
- easy to stream,
- robust for interrupted runs.

Each line should be valid JSON with a `schema_version`.

Example:

```json
{"schema_version":1,"event":"route_selected","run_id":"r1","turn":0,"selected_agent":2,"selected_role":"Verifier"}
```

Future storage backends can include SQLite or ETS, but JSONL should remain the
portable base format.

## Milestone Gates

Trace core:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/trace
```

Orchestrator trace integration:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/orchestrator_test.exs
XLA_TARGET=cuda12 mix test --only integration --trace
```

Full gate:

```bash
XLA_TARGET=cuda12 mix format --check-formatted
XLA_TARGET=cuda12 mix test
XLA_TARGET=cuda12 mix test --only integration
XLA_TARGET=cuda12 mix credo --strict
XLA_TARGET=cuda12 mix dialyzer
XLA_TARGET=cuda12 mix docs
```

## Compaction Handoff

After compaction:

1. Re-read this guide.
2. Open the active trace schema or checklist.
3. Run `git status --short`.
4. Inspect changed trace/orchestrator/provider files.
5. Run the smallest trace test.
6. Continue from the next unchecked item.

Update the checklist if event names, schema fields, or redaction rules change.

## Commit Requirements

Commit and push after the quality gate passes. Do not commit generated trace
files unless they are tiny fixtures intentionally stored under `test/fixtures`.
