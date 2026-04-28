# Stage Checks And Tolerances

This guide defines the correctness standard for Python-vs-Elixir Sakana SVD
parity.

## Principle

Functional correctness is required. Exact final byte matching is aspirational.

The system must prove:

- it is using the same source tensor;
- it is using the same offset slice;
- it is using the same serialized SVD components;
- it applies the same scalar formula;
- the reconstructed f32 tensors are numerically close under explicit tolerance;
- any final byte mismatch is isolated and reported.

## Stage Table

| Stage | Meaning | Required | Expected behavior |
| --- | --- | --- | --- |
| `stage.source_f32` | Source tensor in Python/Sakana orientation | yes | exact |
| `stage.offsets_f32` | SVF offset slice | yes | exact |
| `stage.scaled_s` | `S * (1 + offsets)` | yes | exact or tight |
| `stage.normalization` | `sum(S) / sum(scaled_s)` | yes | tight |
| `stage.u_scaled` | Python diagnostic `U * scaled_s` | no | diagnostic |
| `stage.matmul_pre_norm` | Python diagnostic pre-normalization matmul | no | diagnostic |
| `stage.zero_source_f32` | Zero-offset source reconstruction | yes | numeric |
| `stage.adapted_source_f32` | Adapted f32 source-orientation tensor | yes | numeric |
| `stage.final_f32` | Final-orientation adapted f32 tensor | yes | numeric |
| `stage.final_bf16` | Final `bf16` tensor bytes | no | aspirational |

## Current Tolerances

Exact identity stages:

```text
stage.source_f32: max_abs=0, mean_abs=0
stage.offsets_f32: max_abs=0, mean_abs=0
```

Scalar/vector stages:

```text
stage.scaled_s: max_abs=1e-6, mean_abs=1e-8
stage.normalization: max_abs=1e-6, mean_abs=1e-6
```

Large reconstruction stages:

```text
stage.zero_source_f32: max_abs=1e-3, mean_abs=1e-5
stage.adapted_source_f32: max_abs=1e-3, mean_abs=1e-5
stage.final_f32: max_abs=1e-3, mean_abs=1e-5
```

Final byte stage:

```text
stage.final_bf16: reported, not required
```

The final `bf16` stage still reports max and mean differences. It is not the
functional gate because small f32 reconstruction differences can round to
different `bf16` bytes.

## Current Observed Result

In the current environment:

- `stage.source_f32`: byte-match.
- `stage.offsets_f32`: byte-match.
- `stage.scaled_s`: byte-match.
- `stage.normalization`: max difference about `5.96e-8`.
- `stage.zero_source_f32`: max difference about `3.12e-4`, mean about
  `6.39e-6`.
- `stage.adapted_source_f32`: max difference about `3.65e-4`, mean about
  `6.40e-6`.
- `stage.final_bf16`: does not byte-match.

This is a functional pass and a byte-match miss.

## Why Matmul Is The First Practical Divergence

The reconstruction path contains a large matrix multiplication:

```text
(U * scaled_s) @ V.T
```

That operation is a reduction over 1024 singular-vector dimensions for every
output element. PyTorch and Nx/EXLA may choose different CPU/GPU kernels,
threading strategies, and accumulation orders. Floating-point addition is not
associative, so equivalent formulas can produce slightly different f32 outputs.

Those small f32 differences can matter after `bf16` rounding because `bf16` has
coarser spacing than `f32`.

## How To Interpret Comparator Output

`--strict-stage-tolerances` passes when all required stages pass. It still
prints non-byte-identical stages.

Example interpretation:

```text
stage.adapted_source_f32 required=True functional_passed=True byte_match=False
stage.final_bf16 required=False functional_passed=False byte_match=False
```

This means:

- the f32 adapted tensor is close enough for functional parity;
- the final bytes differ;
- final byte equality remains a separate target.

## When To Change Tolerances

Do not loosen tolerances just to make a report pass.

Tolerance changes require:

1. a documented reason;
2. a top-diff review;
3. evidence that exact input/component stages still match;
4. evidence that the divergence is in a reduction or backend-specific numeric
   stage;
5. a test or guide update describing the new standard.

If an exact stage fails, fix the implementation instead of changing tolerance.
