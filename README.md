# TrinityCoordinator

<p align="center">
  <img src="assets/trinity_coordinator.svg" alt="TRINITY Coordinator" width="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-0f172a?style=for-the-badge" />
  </a>
  <a href="https://github.com/nshkrdotcom/trinity_coordinator">
    <img alt="GitHub" src="https://img.shields.io/badge/github-nshkrdotcom%2Ftrinity__coordinator-111827?style=for-the-badge&logo=github" />
  </a>
</p>

`trinity_coordinator` is an Elixir/Nx implementation of a TRINITY-style local
small-model router. The active project direction is to recreate the Python
Sakana/Qwen artifact application process in Elixir, prove it with rigorous
stage-level parity checks, and use the resulting Qwen coordinator to route real
LLM provider calls.

The current focus is parity and service foundation, not reproducing the original
training experiment.

## Current Direction

The active lane is:

1. Load the same base `Qwen/Qwen3-0.6B` model.
2. Consume the Sakana router vector and SVD/SVF components.
3. Reconstruct adapted Qwen tensors in Elixir/Nx.
4. Prove the Elixir path against Python with stage-level checks and explicit
   tolerances.
5. Materialize reusable adapted artifacts.
6. Run the adapted small local coordinator in front of real provider-backed
   LLM calls.

The earlier experiment-reproduction lane, including sep-CMA-ES training
reproduction work, is shelved. It remains in the repository for now, but the
planned direction is to remove or archive it once the parity and service path is
fully stable.

## Current Status

Working now:

- Qwen3-0.6B loads through Bumblebee on EXLA CUDA.
- The Sakana router vector is converted to safetensors.
- The vector split is understood:
  - first `9216` values: SVF scale offsets;
  - final `10240` values: router-head weights reshaped to `{10, 1024}`.
- The Elixir SVD/SVF code reconstructs adapted tensors.
- Python and Elixir parity scripts emit detailed JSON reports.
- Python emits a stage tensor bundle from safetensors readback.
- Elixir emits comparable semantic `torch_v` stage tensors with `--stage-dir`.
- `--strict-stage-tolerances` is the required functional correctness gate.

Current parity result:

- Python in-memory and Python safetensors readback both produce
  `5aaa24c15898794dec09dccae650e35549c33cc24815e70ac6641cc3b466b725`.
- Elixir semantic `torch_v` currently produces
  `bf089ea0607c93ae69f92bf7b9fcf71dc2a2b53d231cfe307b8cd6f4ef6a85ae`.
- Source tensors, offsets, and scaled singular values byte-match.
- Required f32 reconstruction stages pass explicit tolerances.
- Final `bf16` byte matching remains aspirational and is reported separately.

## Start Here

Read the guides in this order:

1. [Onboarding](guides/onboarding.md)
2. [Current Direction And Planning](guides/current_direction.md)
3. [System Architecture](guides/system_architecture.md)
4. [Recreating The Python Parity Process](guides/python_parity_reconstruction.md)
5. [Stage Checks And Tolerances](guides/stage_checks_and_tolerances.md)
6. [Sakana Artifacts And Export](guides/artifacts_and_export.md)
7. [Service Buildout Plan](guides/service_buildout.md)
8. [Operations And Quality Gates](guides/operations_qc.md)
9. [Troubleshooting](guides/troubleshooting.md)

Additional technical reference notes are included in HexDocs under
`Reference Notes`.

## Requirements

- Elixir `~> 1.18`.
- NVIDIA driver visible to `nvidia-smi`.
- `XLA_TARGET=cuda12`.
- Internet access for first-time Hugging Face model downloads.
- Python with PyTorch, Transformers, and safetensors for parity scripts.

Resolved core dependency lane:

- `nx 0.10.0`
- `exla 0.10.0`
- `axon 0.7.0`
- `bumblebee` pinned to `elixir-nx/bumblebee`
  `0fd8114cf5429af9236f100f3350986e9d823c02`

## Quick Verification

Run the fast Elixir suite:

```bash
XLA_TARGET=cuda12 mix test
```

Run static checks:

```bash
mix format --check-formatted
mix credo --strict
mix dialyzer
```

Build docs:

```bash
mix docs
```

## Current Parity Workflow

Generate Python report, components, and stage tensors:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Generate Elixir semantic report and stage tensors:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

Run the required functional parity gate:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

Use final byte equality only as an explicit opt-in target:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

That exact-byte gate is expected to fail in the current state while functional
stage parity passes.

## Runtime Shape

The intended service path is:

1. Format and tokenize the transcript.
2. Run the adapted local Qwen coordinator on CUDA.
3. Extract the penultimate-token hidden state.
4. Route through the imported Sakana head.
5. Select agent and TRINITY role.
6. Inject the selected role prompt.
7. Dispatch to a configured LLM provider.
8. Persist trace metadata for audit and debugging.

Provider LLM integration is still being hardened. Tests verify the router and
provider boundary without pretending that external LLM calls happened.

## Quality Standard

Before committing changes that affect parity or runtime behavior, run:

```bash
mix format --check-formatted
python3 -m py_compile priv/sakana_trinity/scripts/*.py
XLA_TARGET=cuda12 mix test
mix credo --strict
mix dialyzer
mix docs
```

When parity code changes, also run:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

## Project Files

- [License](LICENSE)
- [Changelog](CHANGELOG.md)
- [Onboarding](guides/onboarding.md)
- [Current Direction](guides/current_direction.md)
- [Operations And Quality Gates](guides/operations_qc.md)

## Attribution

This repository is a research implementation inspired by *TRINITY: An Evolved
LLM Coordinator*.[1] The paper motivates the hidden-state router, the
Thinker/Worker/Verifier role split, the lightweight coordination head, and the
preference for compact local coordination.

This package does not claim to reproduce the paper's reported scores. The active
focus is a robust, inspectable Elixir implementation of the Qwen/Sakana
coordinator path.

## References

[1] Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, and
Yujin Tang. *TRINITY: An Evolved LLM Coordinator*. arXiv:2512.04695, 2026.
https://arxiv.org/abs/2512.04695

## License

This project is released under the MIT License.
