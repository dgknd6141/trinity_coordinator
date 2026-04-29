# Provider Service Hardening

This guide is the next implementation plan after the local coordinator and mock
orchestration examples are green. It deliberately does not replace the current
safe examples; it describes the work needed to put real LLM providers behind the
adapted local coordinator.

## Current Boundary

Implemented and verified:

- adapted Qwen coordinator loads from canonical Python semantic artifacts;
- local routing emits hidden vector `{1, 1024}`, logits `{1, 10}`, agent logits
  `{7}`, and role logits `{3}`;
- orchestrator role order preserves Python `solver`, `thinker`, `verifier`
  semantics with public `Worker` for `solver`;
- thinker suggestions can override one subsequent route;
- verifier-before-worker fails before provider dispatch;
- max-turn exhaustion returns the latest Worker response when one exists;
- provider success and failure are persisted to JSONL trace with latency;
- hosted, GeminiEx, and Agent Session Manager provider specs enter the shared
  `:inference` package through `TrinityCoordinator.AgentPool.Inference`;
- `examples/` includes local routing and mock orchestration runs.

Not implemented in this checkpoint:

- a supervised long-lived service process;
- warm model reuse across external requests;
- production provider pool configuration;
- retry/backoff policy;
- budget accounting for live provider calls;
- trace retention/export policy;
- live-provider CI or release smoke automation.

## Implementation Checklist

Use red-green-refactor and commit/push only after the listed quality gates pass.

1. Provider pool configuration
   - Add a documented `configured` provider pool for the target deployment.
   - Validate contiguous agent ids `0..6`.
   - Use shared `:inference` adapters rather than app-local provider-specific
     transport code for new provider integrations.
   - Record provider name, model, endpoint, timeout, max tokens, and
     temperature in traces.
   - Keep credentials in environment variables only.

2. Supervised runtime process
   - Start one coordinator process that loads Qwen and the router head once.
   - Expose a function boundary such as
     `route_transcript(messages, opts) :: {:ok, result} | {:error, reason}`.
   - Keep provider dispatch behind `AgentPool` adapters.
   - Add graceful shutdown and model-load error reporting.

3. Request and trace contract
   - Accept role/content message lists.
   - Return selected agent, selected role, response text, verifier status, and
     trace path.
   - Persist transcript hash by default, not raw prompt content.
   - Allow opt-in full trace content only for local debugging.

4. Provider failure semantics
   - Classify credential, timeout, HTTP, rate-limit, invalid-response, and
     adapter errors.
   - Retry only retryable failures.
   - Ensure failed providers do not become fake passes.
   - Keep verifier acceptance as the only successful final condition except
     explicit max-turn latest-worker fallback.

5. Budget and live-smoke gates
   - Require `TRINITY_ENABLE_PROVIDER_DEMO=1` or equivalent for live calls.
   - Require a positive budget variable for release smoke tests.
   - Limit max tokens and max turns in smoke tests.
   - Skip live smokes when credentials or budget are absent.

6. Examples and operator UX
   - Keep `examples/local_coordinator_route.exs` as the no-provider diagnostic.
   - Keep `examples/mock_orchestration_trace.exs` as the safe orchestrator
     diagnostic.
   - Add a live provider example only after provider pool configuration is
     stable and credential-gated.

## Success Metrics

- `XLA_TARGET=cuda12 mix test` remains green.
- `mix credo --strict`, `mix dialyzer`, and `mix docs` remain green.
- Local coordinator example prints hidden/logit shape evidence.
- Mock orchestration example writes a complete JSONL trace and terminates.
- Live route demo fails closed without explicit live enablement.
- Live route demo succeeds only with configured credentials and budget.
- No default test or example performs live provider calls.

## Quality Gates

Minimum gates for provider-service commits:

```bash
mix format --check-formatted
git diff --check
python3 -m py_compile priv/sakana_trinity/scripts/*.py
XLA_TARGET=cuda12 mix test
mix credo --strict
mix dialyzer
mix docs
XLA_TARGET=cuda12 mix run examples/local_coordinator_route.exs -- \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
XLA_TARGET=cuda12 mix run examples/mock_orchestration_trace.exs -- \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --trace-out tmp/examples/mock_orchestration_trace.jsonl
```

Credential-gated live smoke, only after configuration:

```bash
TRINITY_ENABLE_PROVIDER_DEMO=1 XLA_TARGET=cuda12 mix trinity.route.demo \
  --profile qwen_sakana_adapted \
  --provider-pool gemini_cli_asm \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
  --trace-out tmp/trinity_route_demo_live.jsonl
```
