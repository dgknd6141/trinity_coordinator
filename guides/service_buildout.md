# Service Buildout Plan

The parity work is a foundation. The service goal is to run the final small
local coordinator model and integrate it with real LLM provider calls.

## Target Runtime

The target runtime loop is:

1. Receive a transcript.
2. Load or reuse `:qwen_sakana_adapted`.
3. Extract the penultimate hidden-state vector.
4. Route through the imported Sakana head.
5. Select agent and role.
6. Inject the role prompt.
7. Dispatch to a configured provider.
8. Persist a trace.
9. Continue until verifier acceptance or budget exhaustion.

## What Is Already In Place

- Core extractor/head/orchestrator contracts.
- Qwen base profile.
- Adapted Qwen profile hook.
- Router vector conversion.
- SVD/SVF reconstruction mechanics.
- Artifact loader and manifest validation.
- OpenAI-compatible provider boundary.
- Trace hashing and JSONL helpers.

## What Remains Before Service Use

Complete adapted artifact validation:

- run full export for all selected tensors;
- validate merged artifacts;
- load `:qwen_sakana_adapted` repeatedly from disk;
- verify route shape and backend metadata;
- add a canonical smoke command for adapted routing.

Strengthen provider integration:

- define provider pools for the intended deployment;
- replace mock-heavy checks with explicit credential-gated smoke tests;
- record provider model, endpoint, latency, and error class in traces;
- make provider failure behavior explicit.

Build service ergonomics:

- add a runnable service/demo command for the adapted coordinator;
- keep model loading warm across requests;
- expose route diagnostics without dumping sensitive prompt content by default;
- provide a minimal config story for CUDA, profile, provider pool, and budgets.

Persist traces:

- transcript hash;
- profile name;
- backend labels;
- hidden-state shape;
- route logits summary;
- selected agent and role;
- provider request metadata;
- verifier decision;
- error/retry metadata.

## Provider Boundary Status

The provider boundary exists, but the project should still be described as
provider-integration-in-progress. Tests can verify routing without pretending to
call real external models.

Production-like provider tests should be explicitly gated by credentials and
budget. A missing API key should produce a clear skipped/gated result, not a
fake pass.

## Operational Profile

The operational profile should prefer:

- one loaded local Qwen coordinator per runtime;
- CUDA-backed tensors for the SLM path;
- persisted adapted artifacts loaded from a complete manifest;
- real provider calls only through configured pools;
- JSONL traces for later audit.

## Success Criteria

The service foundation is complete when:

- `--strict-stage-tolerances` remains green for the semantic parity sample;
- canonical adapted artifact export succeeds and resumes reliably;
- `:qwen_sakana_adapted` loads and routes a real transcript on CUDA;
- a demo command runs the adapted coordinator through the provider boundary;
- provider calls are either real and traceable or explicitly gated;
- docs describe the exact operator commands and expected outputs.
