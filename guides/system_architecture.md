# System Architecture

This guide describes the active coordinator architecture, not the shelved
training-reproduction lane.

## Runtime Flow

The intended service path is:

1. A transcript is represented as role/content messages.
2. `Extractor` formats and tokenizes the transcript.
3. Qwen3-0.6B runs through Bumblebee/Axon on EXLA CUDA.
4. `Extractor` reads the final hidden-state tensor.
5. The penultimate-token vector is sliced from that tensor.
6. `CoordinationHead` applies the imported Sakana router head.
7. The logits are split into agent logits and role logits.
8. The selected role is injected into the prompt.
9. `AgentPool` dispatches the provider call.
10. The trace layer records what happened.

The supplemental Python runtime adds three compatibility details that matter
for the Elixir service:

- router hidden extraction uses the no-generation path with Qwen3 thinking
  disabled and hidden position `-2`;
- role logits use Python order `solver`, `thinker`, `verifier`, where `solver`
  is the paper's Worker role;
- the Python evaluation loop samples agent and role separately from softmax
  probabilities, while Elixir tests should also support deterministic argmax.

## Core Modules

`TrinityCoordinator.Runtime`

- Provides CUDA backend setup helpers.
- Keeps backend changes scoped where possible.
- Exposes backend labels for reports and traces.

`TrinityCoordinator.Extractor`

- Formats messages.
- Applies the tokenizer.
- Runs model forward passes.
- Extracts penultimate-token hidden-state vectors.

`TrinityCoordinator.CoordinationHead`

- Builds the dense Axon routing head.
- Loads trained/imported head weights.
- Splits logits into agent and role regions.
- Selects route IDs.

`TrinityCoordinator.SLMProfile`

- Defines named model profiles.
- `:qwen_coordinator` loads the base Qwen profile.
- `:qwen_sakana_adapted` loads Qwen and applies persisted adapted artifacts.

`TrinityCoordinator.Sakana.SVD`

- Loads the Sakana router vector.
- Splits scale offsets and router-head weights.
- Selects decomposable tensors.
- Runs SVD/SVF reconstruction.
- Reinserts adapted tensors into the Qwen params tree.

`TrinityCoordinator.Sakana.Artifact`

- Validates export manifests.
- Loads adapted tensors from checkpoint or merged artifacts.
- Loads router-head artifacts.
- Applies artifacts to runtime profiles.

`TrinityCoordinator.Sakana.ParityTrace`

- Builds Python-vs-Elixir parity reports.
- Supports semantic-only mode.
- Can reuse Python's `stage.source_f32` for sample semantic replay so Qwen does
  not need to be loaded on every debug run.
- Can restrict diagnostics to the preferred `torch_v` layout and/or the EXLA
  device semantic target.
- Reads Python-exported components.
- Compares stage tensors against Python stage bundles.

`TrinityCoordinator.AgentPool`

- Defines the provider boundary.
- Maps hosted, GeminiEx, and Agent Session Manager specs into shared
  `Inference.Client` requests through `TrinityCoordinator.AgentPool.Inference`.
- Allows tests to avoid pretending that provider calls happened.

## Artifact Data Flow

The Sakana artifact vector contains two regions:

```text
0..9215       SVF scale offsets
9216..19455   router head weights
```

The scale offsets are assigned to selected Qwen tensors in deterministic tensor
order. For the current layer-26 sample, the manifest records:

```text
source_name: model.layers.26.mlp.gate_proj.weight
elixir_name: decoder.blocks.26.ffn.gate.kernel
source_shape: [3072, 1024]
sample_reconstructed_shape: [1024, 3072]
offset_start: 5120
offset_end: 6144
```

The orientation difference is expected. Python and Bumblebee store this family
of weights in opposite orientations. The parity harness explicitly orients
source and final tensors to the manifest shapes before comparing.

## Semantic Versus Native SVD Paths

There are two different diagnostic paths.

Semantic Python-component path:

- consumes Python-exported `U`, `S`, `V`, and offsets;
- should be used for correctness checks;
- isolates formula, orientation, dtype, backend, and rounding behavior;
- avoids native SVD basis ambiguity.

Native Elixir SVD path:

- recomputes SVD with Nx;
- is useful only to inspect Nx's native decomposition behavior;
- can trigger long XLA/ptxas compile times;
- is not expected to byte-match PyTorch adapted tensors under nonzero offsets.

The active parity loop should use `--semantic-only --source-from-python-stage
--preferred-layout-only --device-semantic-only` unless the question is
specifically about native Nx SVD, wrong-layout diagnostics, or host/backend
comparison.

## Supplemental Runtime Contract

The checkpoint metadata under
`docs/priv/trinity_code_submission/logs/ckpt/es_log.json` is authoritative for
the imported vector:

```text
model: Qwen/Qwen3-0.6B
agents: 7
roles: 3
head shape: {10, 1024}
SVF layer: 26
max turns: 5
max tokens: 4096
temperature: 0.1
last_token_predict: false
```

The service implementation should preserve this compatibility mode before
introducing cleaner production aliases or policy changes.

## Why Exact Hashes Are Not The Only Contract

SVD bases are not unique. Even when two decompositions reconstruct the same
source tensor at zero offsets, nonzero singular-value offsets can produce
different adapted tensors if the singular vectors differ.

Large matrix multiplications are also reductions. PyTorch and Nx/EXLA can choose
different kernels and accumulation orders. That can leave small f32 differences
which round to different `bf16` bytes.

For that reason, the architecture uses a staged contract:

- exact input/component checks;
- tight scalar checks;
- explicit reconstruction tolerances;
- final hash checks as an opt-in byte target.
