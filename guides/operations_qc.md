# Operations And Quality Gates

This guide lists the checks that should be run before changing the parity or
runtime lanes.

## Fast Regression Suite

```bash
XLA_TARGET=cuda12 mix test
```

Current expected result:

```text
1 doctest, 148 tests, 0 failures, 25 excluded
```

The excluded tests include slow Qwen/SVD gates and expensive artifact export
paths.

## Static Checks

```bash
mix format --check-formatted
mix credo --strict
mix dialyzer
```

Python script syntax:

```bash
python3 -m py_compile priv/sakana_trinity/scripts/*.py
```

Docs:

```bash
mix docs
```

Docs should build without undefined-reference warnings.

## Functional Parity Gate

Generate reports:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

Run strict functional comparison:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

This is the required parity gate.

## Byte-Match Gate

Use only when intentionally working on exact final bytes:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

This is expected to fail in the current state.

## Expensive Gates

Qwen-focused tests:

```bash
XLA_TARGET=cuda12 mix test --only qwen --trace
```

Slow SVD tests:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only slow_qwen_svd --trace
```

Expensive export gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only expensive_qwen_svd --trace
```

These are not part of every local commit loop.

## Expected XLA Warnings

EXLA/CUDA may print:

```text
ptxas warning : Registers are spilled to local memory
```

This does not mean VRAM overflow. It means a compiled GPU kernel used more
registers than available and some values were placed in CUDA local memory. It is
often a compile/runtime performance concern, not a parity failure by itself.

Use `--semantic-only` during semantic parity work to avoid the slow native Nx
SVD compilation path.

## Commit Checklist

- [ ] `mix format --check-formatted`
- [ ] `python3 -m py_compile priv/sakana_trinity/scripts/*.py`
- [ ] `XLA_TARGET=cuda12 mix test`
- [ ] `mix credo --strict`
- [ ] `mix dialyzer`
- [ ] `mix docs`
- [ ] parity runtime check when parity code changed:
  `compare_sakana_parity_reports.py --strict-stage-tolerances`
- [ ] README and guides updated when behavior or standards change.
