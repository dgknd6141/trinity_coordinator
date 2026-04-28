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

The full supplemental Python submission in `docs/priv/trinity_code_submission`
has now been audited as the executable specification for runtime semantics. It
confirms the imported checkpoint uses `Qwen/Qwen3-0.6B`, layer 26 SVF, seven
agents, five coordination turns, a biasless linear `{10, 1024}` head, no
generation for router hidden extraction, and role order `solver`, `thinker`,
`verifier`. The paper's `Worker` role maps to the Python code's `solver` role.

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

1. Extend sample parity to all selected tensors:
   - Python emits stage/debug data for every selected tensor.
   - Elixir replays every selected tensor from Python components.
   - The comparator fails if any required stage for any selected tensor fails.
2. Import the full Python semantic export bundle into canonical Elixir
   artifacts:
   - use `export_sakana_trinity_safetensors.py` for the full export bundle;
   - use `mix trinity.sakana.import_python` for canonical runtime artifacts;
   - validate manifest, checkpoint hashes, tensor shapes/types, and router head
     shape `{10, 1024}`.
3. Load `:qwen_sakana_adapted` as the service coordinator profile:
   - apply adapted tensors;
   - load router head;
   - prove hidden state `{1, 1024}`, logits `{1, 10}`, agent logits `{7}`, and
     role logits `{3}` on a fixed transcript.
4. Add fixed-transcript router trace parity:
   - compare tokenization, hidden extraction, head weights, logits, and argmax
     agent/role between Python and Elixir.
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

The private implementation handoff for these first five items is
`docs/priv/20260428/06_next_phase_execution_checklist.md`.

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

Elixir semantic `torch_v` currently produces non-matching final `bf16` hashes
in recent reports.

```text
observed examples:
bf089ea0607c93ae69f92bf7b9fcf71dc2a2b53d231cfe307b8cd6f4ef6a85ae
74dc61d765c95e80ca7298b6e97f29a4fd76e2ae4bfb348b2abbffcbc5e0dff8
```

The stage checks isolate the mismatch:

- source tensor, offsets, and scaled singular values byte-match;
- normalization is within scalar tolerance;
- f32 zero-offset and adapted reconstructions pass required tolerances;
- final `bf16` bytes differ after reconstruction/rounding.

This makes the current state functionally correct under the declared standard,
while exact final byte matching remains an open optimization/debug target.
The stage-bundle report, not the final Elixir hash alone, is the correctness
verdict.
