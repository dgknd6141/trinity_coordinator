# Benchmark Harnesses

This guide describes how to add benchmark harnesses for task-family
separability, routing accuracy, and turn-budget behavior.

The current project verifies mechanics. Benchmarks should measure whether the
hidden-state representation and coordination head are useful under controlled
task sets, without claiming reproduction of the paper's full results.

## Implementation Discipline

Use TDD/RGR for each benchmark component: red dataset/report/metric test,
minimal green implementation, then refactor. Maintain a live checklist and
revise it whenever dataset schema, metric definitions, report format, or command
options change.

After context compaction, re-read this guide, inspect the active checklist, run
`git status --short`, and rerun the smallest affected benchmark test before
editing. Complete the milestone quality gate before advancing. Before merge,
run the final quality gate, commit only QA-passing changes, and push every repo
touched by the work.

## Benchmark Goals

Add harnesses for:

- task-family separability in extracted hidden states,
- routing accuracy against labeled or oracle routes,
- turn-budget behavior and verifier termination,
- head-variant ablations,
- provider-pool behavior when credentialed tests are enabled.

## Non-Goals

- Do not claim paper-score reproduction.
- Do not require provider credentials for default benchmarks.
- Do not mix benchmark fixtures with secret provider transcripts.
- Do not make long-running benchmarks part of `mix test`.

## Target Commands

```bash
XLA_TARGET=cuda12 mix trinity.benchmark --suite separability
XLA_TARGET=cuda12 mix trinity.benchmark --suite routing
XLA_TARGET=cuda12 mix trinity.benchmark --suite turn-budget
```

Output should include:

- suite name,
- profile name,
- head variant,
- dataset id,
- sample count,
- metrics,
- output path,
- git commit,
- CUDA target and backend.

## Proposed Modules

- `Benchmark.Dataset`: fixture loading and validation.
- `Benchmark.FeatureExtractor`: batch hidden-state extraction.
- `Benchmark.Separability`: nearest-centroid/logistic/SVM-compatible exports.
- `Benchmark.Routing`: route-label accuracy and confusion matrices.
- `Benchmark.TurnBudget`: max-turn, accept-rate, and verifier behavior.
- `Benchmark.Report`: JSON and Markdown reports.
- `Mix.Tasks.Trinity.Benchmark`: command-line entrypoint.

## Dataset Shape

Use JSONL fixtures:

```json
{"id":"math_001","family":"math","messages":[{"role":"user","content":"..."}],"expected_agent":0,"expected_role":1}
```

Required fields:

- `id`,
- `family`,
- `messages`.

Optional fields:

- `expected_agent`,
- `expected_role`,
- `difficulty`,
- `source`,
- `metadata`.

## TDD/RGR Checklist

Maintain and revise this checklist during implementation.

- [ ] Red: dataset loader rejects malformed JSONL.
- [ ] Green: implement dataset schema validation.
- [ ] Red: feature extractor returns one vector per case.
- [ ] Green: use real `Extractor.extract_batch_penultimate_hidden_states/3`.
- [ ] Red: separability suite computes deterministic simple metrics.
- [ ] Green: implement centroid and export-ready report.
- [ ] Red: routing suite computes agent/role accuracy and confusion matrices.
- [ ] Green: implement route evaluation with real `CoordinationHead.route/5`.
- [ ] Red: turn-budget suite records accept/revise/max-turn outcomes.
- [ ] Green: integrate orchestrator in a credential-free local mode where
      provider calls are replaced by explicitly labeled fixture outcomes only
      outside core router tests.
- [ ] Red: benchmark command writes JSON report.
- [ ] Green: implement `mix trinity.benchmark`.
- [ ] Update README and this guide.

For core route and extraction metrics, use real Bumblebee/Axon/Nx/EXLA. Fixture
labels are acceptable as benchmark labels; they are not mocks of the core tensor
path.

## Metrics

Separability:

- within-family cosine distance,
- between-family cosine distance,
- simple nearest-centroid accuracy,
- optional export for external SVM/UMAP analysis.

Routing:

- agent accuracy,
- role accuracy,
- joint route accuracy,
- confusion matrices,
- entropy or margin of logits.

Turn budget:

- average turns,
- max-turn hit rate,
- verifier accept rate,
- revise rate,
- provider-call count,
- estimated cost when provider metadata is available.

## Milestone Gates

Dataset loader:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/benchmark/dataset_test.exs
```

Real extraction benchmark:

```bash
XLA_TARGET=cuda12 mix test --only integration --trace
```

Benchmark command:

```bash
XLA_TARGET=cuda12 mix trinity.benchmark --suite separability --limit 4 --out tmp/benchmark.json
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

## Report Requirements

Each report must include:

- schema version,
- git commit,
- dependency versions,
- CUDA target,
- supported EXLA platforms,
- model profile,
- head variant,
- provider pool if used,
- dataset hash,
- metrics,
- warnings and skipped cases.

## Compaction Handoff

After compaction, read this guide, inspect the benchmark checklist, run
`git status --short`, and run the smallest benchmark loader or report test before
editing. Update checklist items when metrics or dataset schemas change.

## Commit Requirements

Commit and push all QA-passing repos touched by benchmark work. Do not commit
large benchmark outputs. Small fixtures are acceptable under `test/fixtures`.
