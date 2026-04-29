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
   - Python emits source-oriented stage/debug data for every selected tensor.
   - Elixir currently replays the bounded `model.layers.26.*` slice from Python
     components with `--all-selected-tensors --selected-source-regex`.
   - The comparator fails if any required stage for any replayed selected tensor
     fails.
   - Embedding and LM-head replay remain a chunked large-tensor follow-up before
     this gate can honestly cover every selected tensor end to end in Elixir.
   - Final `bf16` byte equality remains reported as an aspirational diagnostic,
     not the required gate.
2. Import the full Python semantic export bundle into canonical Elixir
   artifacts:
   - use `export_sakana_trinity_safetensors.py` for the full export bundle;
   - use `mix trinity.sakana.import_python` for canonical runtime artifacts;
   - validate manifest, checkpoint hashes, tensor shapes/types, and router head
     shape `{10, 1024}`.
   - Status: complete for
     `tmp/sakana_parity/adapted_artifacts_from_python`; the importer writes
     checkpoint-directory artifacts with 9 target-verified tensors and router
     head shape `{10, 1024}`. The importer now transposes Qwen layer kernels by
     semantic source path, which is required for square k/v projection matrices
     where shape alone cannot reveal orientation.
3. Load `:qwen_sakana_adapted` as the service coordinator profile:
   - apply adapted tensors;
   - load router head;
   - prove hidden state `{1, 1024}`, logits `{1, 10}`, agent logits `{7}`, and
     role logits `{3}` on a fixed transcript.
   - Status: complete against
     `tmp/sakana_parity/adapted_artifacts_from_python` with
     `mix trinity.hitl.adapted --artifact-dir ...`; the smoke proved the
     checkpoint-directory artifact patches Qwen, loads the `{10, 1024}` router
     head, and routes a fixed transcript on CUDA.
4. Add fixed-transcript router trace parity:
   - compare tokenization, hidden extraction, head weights, logits, and argmax
     agent/role between Python and Elixir.
   - Status: complete. Exact transcript/token/head/argmax parity passes for the
     fixed trace; hidden/logit tensors pass declared cosine and relative-L2
     alignment thresholds across Python CPU bf16 and Elixir EXLA CUDA.
5. Run a complete local coordinator loop:
   - transcript in;
   - Qwen hidden state extracted;
   - Sakana router head selected;
   - role injected;
   - provider adapter called through the shared `:inference` boundary.
   - Status: complete for the safe mock lane and implemented for hosted,
     GeminiEx, and Agent Session Manager specs through
     `TrinityCoordinator.AgentPool.Inference`; live provider smoke remains
     credential-gated.
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
b4cab13f8a82ccaf49603356e658bc9b77f65b08a69678a7d053a2e4b3197c43
```

This value comes from the regenerated original-submission `svd_weights.pt` path.
That regenerated `.pt` still does not reproduce the historical stored
`600be6...` manifest hash, so `600be6...` remains provenance-bound metadata.

Elixir semantic `torch_v` currently produces non-matching final `bf16` hashes.
For the latest bounded layer-26 all-selected run, the semantic final hashes were
per-tensor and did not match the single sample Python hash.

```text
latest gate:
selected_tensors_checked=7
total_checks=70
required_checks=63
failed_required=0
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

## Adapted Coordinator Status

The canonical import artifact produced in Phase 2 is now loadable by the live
adapted coordinator gate:

```bash
XLA_TARGET=cuda12 mix trinity.hitl.adapted \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
```

The latest run validated:

- `status=complete`, `artifact_layout=checkpoint_directory`;
- `selected_tensor_count=9`, `selected_singular_value_count=9216`;
- selected Qwen tensor patch differs from base on CUDA;
- extracted hidden vector shape `{1, 1024}`;
- route logits shape `{1, 10}`;
- agent logits shape `{7}`;
- role logits shape `{3}`;
- observed route `agent_id=4`, `role_id=0`, public role `Worker`.

This means the imported Python semantic bundle is operational as a local Qwen
router. The fixed-transcript Python/Elixir router trace now passes exact
transcript, token-id, head-hash, and argmax-id checks. Hidden and logit payloads
are not byte-identical across Python CPU and Elixir EXLA CUDA, but they pass the
declared alignment gate:

```text
hidden cosine >= 0.99, observed 0.99449
hidden relative L2 <= 0.12, observed 0.10493
logits cosine >= 0.99, observed 0.99743
logits relative L2 <= 0.10, observed 0.07303
```

The next implementation gap is the runtime service loop: route decisions need
to drive role injection, provider adapters, and persisted traces.
