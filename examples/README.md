# Examples

These examples are reviewer-facing smoke runs for the current safe runtime lane.
They use the adapted local Qwen coordinator and avoid live LLM calls unless a
separate provider-gated command is used.

Prerequisites:

- `XLA_TARGET=cuda12`;
- the canonical imported artifact directory at
  `tmp/sakana_parity/adapted_artifacts_from_python`;
- a CUDA device capable of loading Qwen3-0.6B through EXLA.

If the artifact directory is missing, run the Python semantic export and
`mix trinity.sakana.import_python` workflow documented in
`guides/artifacts_and_export.md`.

## Local Coordinator Route

This example proves the adapted local Qwen coordinator can load, tokenize a
prompt, extract the route hidden vector, and produce real router logits.
It performs no provider dispatch.

```bash
XLA_TARGET=cuda12 mix run examples/local_coordinator_route.exs -- \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --prompt "Select a TRINITY role for this reasoning task."
```

Expected evidence:

- artifact manifest hash and source vector hash;
- formatted transcript and token ids;
- hidden-state shape and selected hidden index;
- route vector backend and hash;
- full route logits, agent logits, role logits;
- selected agent id/name and selected role id/name for the current run.

## Mock Orchestration Trace

This example proves the adapted coordinator can drive the orchestrator through
role injection, the provider boundary, verifier termination, and JSONL trace
persistence while using deterministic mock responses.

```bash
XLA_TARGET=cuda12 mix run examples/mock_orchestration_trace.exs -- \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --prompt "Select a TRINITY role for this reasoning task." \
  --trace-out tmp/examples/mock_orchestration_trace.jsonl
```

Expected evidence:

- mock provider calls printed with role and selected agent id;
- trace summary containing `run_started`, `slm_extracted`, `route_selected`,
  `provider_called`, `turn_completed`, and `run_completed`;
- Worker selected before Verifier for the default prompt;
- final result accepted by the mock verifier;
- persisted JSONL trace at the supplied path.

## Live Providers

Live provider calls are not part of these examples. Hosted, GeminiEx, and Agent
Session Manager specs are routed through the shared `:inference` package by
`TrinityCoordinator.AgentPool.Inference`; use the gated route demo only after a
real provider pool is configured:

```bash
TRINITY_ENABLE_PROVIDER_DEMO=1 XLA_TARGET=cuda12 mix trinity.route.demo \
  --profile qwen_sakana_adapted \
  --provider-pool configured \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --trace-out tmp/trinity_route_demo_live.jsonl
```

Without `TRINITY_ENABLE_PROVIDER_DEMO=1` or `--allow-live`, live mode fails
before provider dispatch.
