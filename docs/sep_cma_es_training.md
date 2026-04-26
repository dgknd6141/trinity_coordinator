# sep-CMA-ES Training For Terminal Rewards

This guide describes how to implement the paper-aligned, label-free training
path for the coordination head: separable CMA-ES over the router parameters,
optimized against terminal task reward.

The current repository has a real supervised head-training path for extracted
hidden-state vectors. That path is useful for smoke tests and oracle-label
experiments, but it is not the main training method described by the TRINITY
paper. The target of this roadmap item is to train the coordinator from
end-to-end run outcomes, where each sampled head is evaluated by running the
orchestration loop and scoring the final trajectory.

## Research Context

The local paper sources describe the training objective as a budget-constrained
black-box optimization problem:

- The SLM is frozen as a representation function.
- The coordination head maps hidden state `h` to `L + 3` logits.
- A complete multi-turn run is an atomic environment evaluation.
- Terminal reward is binary or scalar, commonly `0`/`1`.
- sep-CMA-ES is preferred because the head objective is weakly coupled and close
  to block-separable under tight evaluation budgets.

The implementation must preserve those constraints. Do not reduce sep-CMA-ES to
ordinary supervised learning. Do not optimize directly against oracle labels
unless the work is explicitly in the supervised baseline path.

## Implementation Discipline

Use TDD/RGR for every change: write the failing test first, implement the
smallest passing change, then refactor. Maintain a live checklist in the PR or
working tree and revise it when the optimizer design, budget model, evaluator,
or trace schema changes.

If context compaction occurs, recontextualize by reading this guide, reading the
active checklist, running `git status --short`, and rerunning the smallest
relevant failing or recently added test before editing. Every milestone below
has a quality gate; run it before moving to the next milestone. Before merge,
run the final quality gate, commit only QA-passing changes, and push every repo
touched by the work.

## Target Contract

The production training API should look like this:

```elixir
config = %TrinityCoordinator.Training.SepCMAES.Config{
  population_size: 32,
  sigma: 0.05,
  generations: 60,
  replications: 16,
  num_agents: 7,
  num_roles: 3,
  seed: 42
}

{:ok, trained} =
  TrinityCoordinator.Training.SepCMAES.train(
    initial_state,
    config,
    evaluator
  )

trained.model_state
trained.metrics.best_reward
trained.trace.generations
```

The evaluator owns expensive environment execution:

```elixir
evaluator = fn candidate_model_state, candidate_metadata ->
  TrinityCoordinator.Training.Evaluator.evaluate_candidate(
    candidate_model_state,
    candidate_metadata,
    tasks: task_batch,
    slm_context: {model_info, tokenizer},
    provider_pool: provider_pool,
    reward_fn: reward_fn
  )
end
```

## Architecture

Add modules under `lib/trinity_coordinator/training/`:

- `SepCMAES.Config`: validated configuration.
- `SepCMAES.State`: mean vector, diagonal covariance/scales, generation index,
  RNG seed, and best candidate.
- `SepCMAES.Candidate`: candidate vector, decoded model state, reward samples,
  mean reward, and metadata.
- `SepCMAES.Codec`: flatten and unflatten Axon model state tensors.
- `SepCMAES`: sample/evaluate/recombine loop.
- `Evaluator`: boundary for running complete trajectories and terminal rewards.
- `Reward`: small helpers for binary accept/reject reward normalization.

Keep flattening deterministic. A model state flattened twice without mutation
must produce identical vector ordering and metadata.

## Algorithm Outline

1. Flatten the initial `Axon.ModelState` into a parameter vector `m`.
2. Initialize diagonal scales `sigma_i` from config.
3. For generation `t`, sample `lambda` candidates:
   `x_i = m + sigma * z_i`, where `z_i ~ N(0, I)`.
4. Decode each vector back into an `Axon.ModelState`.
5. Evaluate each candidate with `replications` terminal-reward runs.
6. Rank candidates by mean reward.
7. Recombine top candidates into a new mean vector.
8. Update diagonal scales using sep-CMA-ES rules.
9. Persist generation trace and best candidate.
10. Stop after max generations, reward threshold, budget exhaustion, or explicit
    cancellation.

## TDD/RGR Checklist

Maintain this checklist in the implementation PR and revise it whenever the
design changes.

- [ ] Red: add tests for deterministic flatten/unflatten round trips.
- [ ] Green: implement `SepCMAES.Codec` without changing route behavior.
- [ ] Refactor: simplify codec metadata and document ordering guarantees.
- [ ] Red: add tests for deterministic candidate sampling with a fixed seed.
- [ ] Green: implement candidate sampling from a diagonal distribution.
- [ ] Red: add tests for rank/recombine behavior with synthetic rewards.
- [ ] Green: implement recombination and best-candidate tracking.
- [ ] Red: add tests for replication aggregation and terminal reward handling.
- [ ] Green: implement `Evaluator` boundary and reward normalization.
- [ ] Red: add a small real integration test using the tiny SLM, real router
      forward pass, and a deterministic local reward function.
- [ ] Green: run at least one generation end to end with real `Axon`/`Nx` model
      states.
- [ ] Refactor: isolate expensive provider-backed evaluation behind explicit
      tags and credentials.
- [ ] Update README and this guide with any changed API or command names.

No core training tests should use fake model states or fake tensor operations.
It is acceptable to use deterministic local reward functions while testing the
optimizer mechanics; provider-backed reward evaluation belongs in a separate
credential-gated integration suite.

## Implementation Milestones

### Milestone 1: Model State Codec

Acceptance:

- Flattens all trainable tensors into one `Nx` vector.
- Records tensor path, shape, type, and slice range.
- Rebuilds the same `Axon.ModelState`.
- Preserves `CoordinationHead.route/5` output before and after round trip.

Quality gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/training/sep_cma_es_codec_test.exs
XLA_TARGET=cuda12 mix credo --strict
```

### Milestone 2: Candidate Sampling

Acceptance:

- Fixed seed produces stable candidate vectors.
- Different generation seeds produce different vectors.
- Candidate metadata includes generation, index, vector norm, and sigma summary.
- Sampling works on CUDA-backed tensors where practical.

Quality gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/training/sep_cma_es_sampling_test.exs
```

### Milestone 3: Recombination

Acceptance:

- Candidates rank by aggregated reward.
- Top candidates update the parent vector.
- Best-so-far candidate survives bad generations.
- Budget accounting is explicit.

Quality gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/training/sep_cma_es_recombine_test.exs
```

### Milestone 4: End-To-End Local Training

Acceptance:

- Uses real `CoordinationHead` model states.
- Uses real extracted hidden-state vectors.
- Runs at least one generation without provider credentials.
- Emits a training trace sufficient to reproduce the generation.

Quality gate:

```bash
XLA_TARGET=cuda12 mix test --only integration --trace
XLA_TARGET=cuda12 mix docs
```

### Milestone 5: Provider-Backed Training

Acceptance:

- Runs only when explicit provider credentials and budget env vars are set.
- Logs every provider call through trace persistence.
- Stops on budget exhaustion.
- Produces reward summaries without leaking secrets.

Quality gate:

```bash
TRINITY_ENABLE_PROVIDER_TESTS=1 \
TRINITY_PROVIDER_BUDGET_USD=1.00 \
XLA_TARGET=cuda12 mix test --only provider_training --trace
```

## Trace Requirements

Every training run should persist:

- run id,
- git commit,
- dependency versions,
- `XLA_TARGET`,
- EXLA supported platforms,
- profile name,
- candidate seed,
- generation number,
- candidate index,
- flattened vector hash,
- sigma summary,
- task ids,
- per-replication rewards,
- aggregate reward,
- selected model state hash,
- best-so-far reward.

This depends on the trace persistence roadmap item. If trace persistence is not
implemented yet, write a minimal JSONL training trace and migrate it later.

## Compaction Handoff

If context compaction occurs during this work, recontextualize before changing
code:

1. Read this guide.
2. Read the active implementation checklist in the PR or working tree.
3. Run `git status --short`.
4. Read changed files before editing.
5. Re-run the smallest failing test from the checklist.
6. Continue from the latest unchecked item, not from the beginning.

Update the checklist after every milestone and whenever a design assumption
changes.

## Final Quality Gate

Before merging:

```bash
XLA_TARGET=cuda12 mix format --check-formatted
XLA_TARGET=cuda12 mix test
XLA_TARGET=cuda12 mix test --only integration
XLA_TARGET=cuda12 mix credo --strict
XLA_TARGET=cuda12 mix dialyzer
XLA_TARGET=cuda12 mix docs
```

If provider-backed tests were touched:

```bash
TRINITY_ENABLE_PROVIDER_TESTS=1 \
TRINITY_PROVIDER_BUDGET_USD=1.00 \
XLA_TARGET=cuda12 mix test --only provider_training
```

Commit and push every QA-passing repo changed by the work. Do not commit secrets,
raw provider transcripts with sensitive content, or local cache artifacts.

## Non-Goals

- Do not implement Qwen support in this work.
- Do not replace the existing supervised training path.
- Do not make provider-backed training run by default.
- Do not claim paper-score reproduction.
