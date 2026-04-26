# Coordination Head Variants

This guide describes how to add block-diagonal and sparse coordination heads for
parameter-efficiency and ablation work.

The current implementation has one head: a dense Axon layer mapping a
second-to-last token hidden-state vector to `num_agents + num_roles` logits. The
paper appendix compares this linear head against low-rank, sparse, and
block-diagonal alternatives. The roadmap item here is to add the variants needed
for rigorous local ablations without weakening the current linear baseline.

## Implementation Discipline

Use TDD/RGR for each variant: red test, minimal green implementation, then
refactor. Maintain a live checklist and revise it whenever head options, shape
rules, parameter counts, or Axon implementation details change.

After context compaction, re-read this guide, inspect the active checklist, run
`git status --short`, and rerun the smallest affected head-variant test before
editing. Complete the milestone quality gate before advancing. Before merge,
run the final quality gate, commit only QA-passing changes, and push every repo
touched by the work.

## Target Contract

The public API should make the head type explicit:

```elixir
model =
  TrinityCoordinator.CoordinationHead.build_model(
    input_dim,
    num_agents,
    3,
    head: :block_diagonal,
    blocks: 10
  )
```

Supported target variants:

- `:linear`: current dense baseline.
- `:block_diagonal`: partitions hidden dimensions and output logits into
  independent blocks.
- `:sparse`: learns or applies feature selection/gating before projection.

All variants must return the same route shape:

```elixir
%{
  agent_id: non_neg_integer(),
  role_id: non_neg_integer(),
  logits: Nx.Tensor.t(),
  agent_logits: Nx.Tensor.t(),
  role_logits: Nx.Tensor.t()
}
```

## Design Constraints

- Do not change the `Extractor` contract.
- Do not change role semantics.
- Do not remove the dense linear baseline.
- Do not hand-roll tensor math outside `Nx`/`Axon` unless Axon cannot express
  the required operation.
- Preserve GPU compatibility.
- Parameter counts must be measurable and reported in tests or docs.

## Variant Details

### Linear

Current baseline:

```text
logits = h W + b
```

This remains the default because it is simple, robust, and paper-supported.

### Block-Diagonal

The block-diagonal head partitions the hidden vector and output logits so each
output block depends on only part of the hidden state. The high-independence
variant described in the appendix uses one block per logit when possible.

Implementation options:

1. Build block-specific dense layers over slices, then concatenate logits.
2. Use a masked dense matrix where non-block entries are fixed at zero.

Prefer option 1 unless Axon parameter management becomes awkward. It makes the
parameter count and block assignment obvious.

Required metadata:

- block count,
- hidden slice per block,
- output slice per block,
- total trainable parameters,
- whether hidden dimensions are evenly divisible by block count.

### Sparse

The sparse head gates hidden dimensions before projection.

Start with a deterministic top-k or mask-based implementation before adding
learned gates. The first useful API can be:

```elixir
build_model(input_dim, num_agents, num_roles,
  head: :sparse,
  sparse_k: 128
)
```

Later learned-sparse support may add trainable feature scores, temperature, and
hard/soft mask modes.

## TDD/RGR Checklist

Maintain this checklist during implementation and revise it as details change.

- [ ] Red: assert the current linear head still produces the same route shape.
- [ ] Green: introduce `head: :linear` option with no behavior change.
- [ ] Red: assert unknown head variants fail with a clear error.
- [ ] Green: add option validation.
- [ ] Red: assert block partition metadata for uneven input/output dimensions.
- [ ] Green: implement partition helper.
- [ ] Red: assert block-diagonal logits have expected shape and parameter count.
- [ ] Green: implement block-diagonal head.
- [ ] Red: assert sparse head validates `sparse_k`.
- [ ] Green: implement sparse deterministic top-k/mask head.
- [ ] Red: assert all variants work with `CoordinationHead.route/5`.
- [ ] Green: route through all variants using real Axon forward passes.
- [ ] Red: add integration test confirming CUDA backend for variant logits.
- [ ] Green: run variant routing on `EXLA.Backend<cuda:0>`.
- [ ] Update README and this guide with final API.

## Tests

Fast tests:

- option validation,
- partition calculation,
- parameter count calculation,
- output shapes,
- route bounds,
- error messages.

Integration tests:

- real Axon forward pass for each variant,
- real training for at least `:linear` and one variant,
- CUDA-backed logits for at least one variant.

Suggested test files:

```text
test/trinity_coordinator/coordination_head/variant_options_test.exs
test/trinity_coordinator/coordination_head/block_diagonal_test.exs
test/trinity_coordinator/coordination_head/sparse_test.exs
```

## Demo Requirements

Extend `mix trinity.demo` with:

```bash
XLA_TARGET=cuda12 mix trinity.demo --head linear
XLA_TARGET=cuda12 mix trinity.demo --head block-diagonal --blocks 10
XLA_TARGET=cuda12 mix trinity.demo --head sparse --sparse-k 128
```

The output must show:

- head variant,
- trainable parameter count,
- input dimension,
- output dimension,
- backend for logits,
- selected route.

## Milestone Gates

After variant option parsing:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/coordination_head_test.exs
```

After block-diagonal implementation:

```bash
XLA_TARGET=cuda12 mix test --only integration --trace
```

Before merge:

```bash
XLA_TARGET=cuda12 mix format --check-formatted
XLA_TARGET=cuda12 mix test
XLA_TARGET=cuda12 mix test --only integration
XLA_TARGET=cuda12 mix credo --strict
XLA_TARGET=cuda12 mix dialyzer
XLA_TARGET=cuda12 mix docs
```

## Compaction Handoff

After compaction, re-read this guide, inspect `CoordinationHead`, inspect the
active checklist, and run the smallest variant test before editing. Continue at
the next unchecked item.

## Commit Requirements

Commit and push only after the quality gate passes. If this work modifies
benchmark docs or generated parameter-count snapshots in another repo, QA and
push that repo too. Do not mix unrelated refactors into the head-variant commit.
