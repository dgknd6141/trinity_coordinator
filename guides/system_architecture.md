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
- Reads Python-exported components.
- Compares stage tensors against Python stage bundles.

`TrinityCoordinator.AgentPool`

- Defines the provider boundary.
- Supports OpenAI-compatible runtime calls when configured.
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

The active parity loop should use `--semantic-only` unless the question is
specifically about native Nx SVD.

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
