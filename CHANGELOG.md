# Changelog

## 2026-04-28

- Reframed the public documentation around the active Qwen/Sakana parity and
  service-foundation direction.
- Added the `guides/` documentation set covering onboarding, direction,
  architecture, Python parity reconstruction, stage tolerances, artifacts,
  service buildout, operations, and troubleshooting.
- Updated ExDoc configuration so README, guides, reference notes, changelog, and
  license render as a structured HexDocs menu.
- Clarified that the old experiment-reproduction lane is shelved and planned
  for later removal or archival after the parity/service path is stable.
- Kept final Python byte matching as an aspirational target while documenting
  `--strict-stage-tolerances` as the required functional correctness gate.

## 2026-04-21

- P1: Added a repo-local Gemma server helper and optional SSoT JSON schema for
  local OpenAI-compatible experiments; Codex/Gemini remains the evidence path.
- P2: Made `distance: embedding` fail loudly on backend errors, retained
  degraded semantic-descriptor status in traces, and exposed centroid IDs under
  `run.centroids`.
- P3: Hardened SSoT anti-collapse coverage with boundary tests for chunk
  thresholds and monotonic verified mapping coverage.
- P4: Centralized the evidence gate in `AntiAgents.Statistics.evidence_hypothesis/3`
  so benchmark and calibration both require positive CI, sign-test support, and
  non-saturated calibration status.
- P5: Added `mix anti_agents.ablate` for offline descriptor ablation against
  reference traces with `provider_calls: 0`.
- P6: Ran the allowed one-field live smoke with Codex inference and
  `gemini_ex` embeddings; recorded explicit Gemini auth and descriptor
  provenance.
- P7: Prepared and dry-run validated the 156-call live calibration command;
  deferred execution to the human operator under the live-run budget policy.
- P8: Dry-run validated the evidence profile at 756 planned LLM calls and
  `<= 756` expected single-view embedding calls; documented human-run evidence
  and ablation commands.
- P9: Updated README, validation, and architecture docs to distinguish pilot,
  diagnostic smoke, deferred evidence, and descriptor-ablation claims.

## 2026-04-20

- W1: Added matched-budget hypothesis testing, bootstrap confidence intervals,
  12-field benchmark fixture, and `mix anti_agents.benchmark`.
- W2: Added archive-feedback rounds with per-round summaries and stagnation
  reporting.
- W3: Added pluggable distance backends for Jaccard, embedding, and judge paths;
  descriptor cells now include `semantic_cluster`.
- W4: Reframed public documentation around semantic-space frontier search and
  removed unsupported latent-space claims.
- W5: Added baseline artifact retry, permanent-loss accounting, and adjusted
  novelty reporting.
- W7: Added verification fixtures, trace schema, scoring-weight invariant tests,
  `mix verify`, and contributor verification docs.
- C1: Replaced the semantic-cluster stub with embedding-backed centroid
  assignment and traceable centroid IDs.
- C2: Changed the benchmark headline statistic to per-field mean-delta
  bootstrap plus one-sided sign test; retained pooled cells as diagnostics.
- C3: Added `--diagnostic` and blocked low-budget benchmark runs from being
  reported as evidence.
- C4: Added frontier temperature sweeps, matched-baseline temperature parity,
  and ignored seed/assembly heat warnings.
- C5: Added `priv/profiles/evidence.json` and benchmark `--profile` loading
  with recorded overrides.
- C6: Split matched-baseline retry/loss accounting from reachable-baseline
  accounting.
- C7: Changed archive feedback to positively target underfilled descriptor
  cells.
- C8: Added descriptor saturation metrics and benchmark calibration status.
- C9: Added `mix anti_agents.calibrate --stubbed` as the QC positive control.
- C10: Reserved README evidence reporting until an embedding-backed,
  expensive-gated evidence run is deliberately executed.
- C11: Added `gemini_ex` 0.13.0 as the production embedding provider for
  evidence runs, using `gemini-embedding-001` batch embeddings through the
  existing embedding-client seam.
