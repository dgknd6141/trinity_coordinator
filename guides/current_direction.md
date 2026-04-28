# Current Direction And Planning

The project direction has changed from experiment reproduction to parity-first
artifact consumption and service buildout.

## Direction

The immediate goal is:

1. Start from the same base Qwen model used by the Python process.
2. Recreate the Python Sakana SVD/SVF artifact application path in Elixir.
3. Prove that the Elixir path is mathematically correct with stage-level checks.
4. Materialize reusable adapted-model artifacts.
5. Run the adapted small local coordinator in front of real LLM provider calls.

The practical product is a local Qwen-backed routing coordinator that can select
an agent and role for each turn, then dispatch to external LLMs through a real
provider boundary.

## What Changed

The first implementation direction included work toward recreating the
underlying training experiment, including sep-CMA-ES scaffolding and terminal
reward machinery. That lane is now on the shelf.

It is not the active buildout because it has a much wider dependency surface:

- task datasets;
- terminal reward functions;
- repeated trajectory evaluation;
- provider-call budgeting;
- baseline comparison methodology;
- paper-metric reproduction;
- long-running live provider experiments.

That code was useful for understanding the paper, but it is not needed for the
current foundation. The current foundation is the artifact path: consume the
available Sakana outputs correctly and make the coordinator operational.

## Planned Cleanup

After the Qwen/Sakana parity and runtime artifact path is stable, remove or
archive the shelved experiment-reproduction code from the mainline.

Candidate areas to audit later:

- sep-CMA-ES trainer modules;
- old anti-agent or benchmark scaffolding that no longer describes the active
  project;
- provider-heavy experiment scripts;
- stale README/changelog claims from earlier research lanes;
- tests that validate abandoned experiment infrastructure rather than the
  service-facing coordinator.

Do not remove that code during parity work unless it blocks correctness. The
near-term priority is to finish the validated coordinator path without adding a
large cleanup diff on top.

## Near-Term Milestones

1. Keep `--strict-stage-tolerances` green for the semantic Python-component
   parity path.
2. Decide whether final Python `bf16` byte matching is worth pursuing further.
3. Complete and validate canonical adapted artifact export.
4. Load `:qwen_sakana_adapted` as the default service coordinator profile.
5. Run a complete local coordinator loop:
   - transcript in;
   - Qwen hidden state extracted;
   - Sakana router head selected;
   - role injected;
   - provider adapter called.
6. Replace provider mocks with explicit gated smoke tests against real
   OpenAI-compatible endpoints.
7. Add trace persistence around every route decision.
8. Remove or archive shelved experiment-reproduction code.

## Correctness Standard

Correctness is not a vibes-based judgment and not a single hash comparison.

The required standard is stage-level functional parity:

- exact identity for source tensors and serialized component inputs;
- tight scalar/vector tolerances for non-reduction arithmetic;
- explicit numerical tolerances for large matrix multiplication outputs;
- separate reporting for final `bf16` byte equality.

The aspirational standard is exact Python byte matching. If that fails while the
required f32 stages pass, the docs and reports must isolate where byte equality
first fails and explain the likely backend reason.

## Current Byte-Match Status

Current Python in-memory and Python safetensors readback both produce:

```text
5aaa24c15898794dec09dccae650e35549c33cc24815e70ac6641cc3b466b725
```

Elixir semantic `torch_v` currently produces:

```text
bf089ea0607c93ae69f92bf7b9fcf71dc2a2b53d231cfe307b8cd6f4ef6a85ae
```

The stage checks isolate the mismatch:

- source tensor, offsets, and scaled singular values byte-match;
- normalization is within scalar tolerance;
- f32 zero-offset and adapted reconstructions pass required tolerances;
- final `bf16` bytes differ after reconstruction/rounding.

This makes the current state functionally correct under the declared standard,
while exact final byte matching remains an open optimization/debug target.
