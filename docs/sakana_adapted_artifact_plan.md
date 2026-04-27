# Sakana Adapted Artifact Implementation Plan

This is the implementation-ready plan for turning Sakana's TRINITY artifacts
into durable Qwen/Bumblebee runtime artifacts.

The point of this plan is to correct the bad throwaway shape where a full Qwen
SVD/SVF run happens inside a test and discards the adapted tensors. The correct
long-term design is export once, save durable artifacts, load many times.

## Executive Summary

The working system should not run SVD during request-time routing.

The working system should not require a full SVD test that throws away useful
adapted tensors.

The working system should have two separate phases:

```text
export/build phase:
  Qwen base weights + Sakana vector -> adapted tensor artifact + router head artifact

runtime phase:
  Qwen base weights + adapted tensor artifact + router head artifact -> route transcript
```

For a fixed base model, fixed Bumblebee parameter mapping, fixed selected tensor
rule, and fixed Sakana vector, the adapted tensors are deterministic. They are
generated outputs, not source code.

## Current State

Already present in the repo:

```text
priv/sakana_trinity/README.md
priv/sakana_trinity/artifacts/sakana_model_iter_60.npy
priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors
priv/sakana_trinity/reference/sakana_decompose_model.original.py
priv/sakana_trinity/reference/sakana_es_log.json
priv/sakana_trinity/scripts/convert_router_vector_to_safetensors.py
priv/sakana_trinity/scripts/export_sakana_trinity_safetensors.py
```

Already implemented in Elixir:

- `TrinityCoordinator.SLMProfile.qwen_coordinator/0` loads
  `Qwen/Qwen3-0.6B` with `Bumblebee.Text.Qwen3` on CUDA.
- `TrinityCoordinator.Sakana.SVD.load_router_vector!/2` loads the raw Sakana ES
  vector from safetensors.
- `TrinityCoordinator.Sakana.SVD.split_router_vector/4` splits the vector into
  `9216` SVF offsets and `10240` head values.
- `TrinityCoordinator.Sakana.SVD.decomposable_tensor_entries/2` selects tensors.
- `TrinityCoordinator.Sakana.SVD.layer_index_filter([26])` matches the inspected
  Sakana layer-26 path.
- `TrinityCoordinator.Sakana.SVD.adapt_tensors/3` can decompose and reconstruct
  selected tensors with progress logging.
- `TrinityCoordinator.Sakana.SVD.put_tensor_entries/2` can reinsert adapted
  tensors into a nested params container.
- `TrinityCoordinator.Sakana.SVD.put_linear_head_weights/3` can load imported
  router head weights into the existing Axon routing head.

Known selected tensor set for the current mapping:

```text
decoder.blocks.26.ffn.gate.kernel                  {1024, 3072}
decoder.blocks.26.ffn.intermediate.kernel          {1024, 3072}
decoder.blocks.26.ffn.output.kernel                {3072, 1024}
decoder.blocks.26.self_attention.key.kernel        {1024, 1024}
decoder.blocks.26.self_attention.output.kernel     {2048, 1024}
decoder.blocks.26.self_attention.query.kernel      {1024, 2048}
decoder.blocks.26.self_attention.value.kernel      {1024, 1024}
embedder.token_embedding.kernel                     {151936, 1024}
language_modeling_head.output.kernel               {151936, 1024}
```

Total singular values consumed: `9216`.

Router head shape after split: `{10, 1024}`.

## Current Handoff State

The core export/runtime implementation is in place and most checkboxes are marked
completed. The newest runtime-facing change is in
`test/trinity_coordinator/sakana/svd_test.exs`, where the segment-based tensor
fetch helper now unpacks `Axon.ModelState` before traversing path segments for
qwen patched-model assertions.

Validated in this cycle:

- `XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --exclude qwen --exclude expensive_qwen_svd --trace`
- `XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/artifact_test.exs --trace`
- `XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --exclude qwen_sakana_adapted --exclude expensive_qwen_svd --trace`
- `XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --only qwen_sakana_adapted --trace`

Open questions carried forward:

- Exact Python parity checks are now backed by a checked-in reference manifest
  fixture and executable parity assertions in `svd_test.exs`.

## Non-Negotiable Design Rules

- Full Qwen SVD is an export/build operation, not runtime behavior.
- Full Qwen SVD must save useful artifacts as it progresses.
- Completed tensor work must be resumable and must not be recomputed by default.
- Runtime must load adapted artifacts and must not call `Nx.LinAlg.svd/2`.
- Generated adapted-weight artifacts must not be committed to git.
- Partial artifacts must never be accepted as canonical runtime artifacts.
- The artifact manifest is the source of truth for mapping saved tensors back to
  Bumblebee params paths.
- Python or another backend may be used for export if EXLA SVD is too slow, but
  runtime must remain Elixir/Bumblebee/Nx.

## Generated Artifact Git Policy

The generated artifact directory is ignored by git:

```gitignore
/priv/sakana_trinity/adapted_qwen3_0_6b_layer26/
```

Everything under that directory is generated by the exporter. A fresh clone
should regenerate it from checked-in source artifacts, exporter code, and the
base Qwen model download.

Do not commit:

- `adapted_tensors.safetensors`
- `router_head.safetensors` generated by the exporter
- checkpoint safetensors
- export logs
- generated manifest files

Commit only source code, small checked-in Sakana source artifacts that are
already part of the repo, and documentation.

## Target Artifact Layout

Default output directory:

```text
priv/sakana_trinity/adapted_qwen3_0_6b_layer26/
```

Final output files:

```text
manifest.json
router_head.safetensors
adapted_tensors.safetensors
export.log.jsonl
```

Checkpoint files:

```text
checkpoints/
  0001_decoder.blocks.26.ffn.gate.kernel.safetensors
  0002_decoder.blocks.26.ffn.intermediate.kernel.safetensors
  0003_decoder.blocks.26.ffn.output.kernel.safetensors
  0004_decoder.blocks.26.self_attention.key.kernel.safetensors
  0005_decoder.blocks.26.self_attention.output.kernel.safetensors
  0006_decoder.blocks.26.self_attention.query.kernel.safetensors
  0007_decoder.blocks.26.self_attention.value.kernel.safetensors
  0008_embedder.token_embedding.kernel.safetensors
  0009_language_modeling_head.output.kernel.safetensors
```

Atomic write temp files:

```text
*.tmp
```

Temp files must be ignored by virtue of the ignored output directory and must be
removed or overwritten safely on resume.

## Safetensors API Facts

The local Elixir dependency supports the following APIs:

```elixir
Safetensors.write!(path, %{"tensor_name" => tensor})
Safetensors.read!(path)
Safetensors.read!(path, lazy: true)
```

Important behavior from `deps/safetensors`:

- `Safetensors.write!/2` accepts a map of string tensor names to `Nx.Tensor`s.
- It writes tensors one by one, but it calls `Nx.to_binary/1` per tensor.
- Device tensors should be transferred to `Nx.BinaryBackend` before writing to
  avoid relying on device buffers during file serialization.
- `Safetensors.read!/2` can return lazy containers with `lazy: true`.
- Tensor names are strings. Dotted names are acceptable as string keys in the
  Elixir implementation, but the manifest still records the original path and
  artifact key explicitly.

Exporter write rule:

```elixir
host_tensor = Nx.backend_transfer(adapted_tensor, Nx.BinaryBackend)
Safetensors.write!(tmp_path, %{artifact_key => host_tensor})
File.rename!(tmp_path, final_path)
```

Never write a checkpoint directly to the final path. Always write temp file then
rename.

## Manifest Schema

Manifest file:

```text
priv/sakana_trinity/adapted_qwen3_0_6b_layer26/manifest.json
```

Required top-level fields:

```json
{
  "artifact_version": 1,
  "status": "partial",
  "created_at": "2026-04-26T00:00:00Z",
  "updated_at": "2026-04-26T00:00:00Z",
  "base_model_repo": "Qwen/Qwen3-0.6B",
  "bumblebee_module": "Bumblebee.Text.Qwen3",
  "architecture": "for_causal_language_modeling",
  "xla_target": "cuda12",
  "export_backend": "elixir_nx_exla_cuda",
  "source_vector_path": "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors",
  "source_vector_tensor": "trinity_router_es_vector",
  "source_vector_shape": [19456],
  "source_vector_sha256": "...",
  "scale_offset_count": 9216,
  "router_head_shape": [10, 1024],
  "router_head_artifact": "router_head.safetensors",
  "adapted_tensors_artifact": "adapted_tensors.safetensors",
  "selected_tensor_count": 9,
  "selected_singular_value_count": 9216,
  "export_complete": false,
  "selected_tensors": []
}
```

Allowed `status` values:

- `partial`
- `complete`
- `failed`

Each selected tensor entry:

```json
{
  "index": 1,
  "status": "complete",
  "path": "decoder.blocks.26.ffn.gate.kernel",
  "artifact_key": "decoder.blocks.26.ffn.gate.kernel",
  "shape": [1024, 3072],
  "type": "bf16",
  "backend_observed_during_export": "EXLA.Backend<cuda:0,...>",
  "singular_values": 1024,
  "offset_start": 0,
  "offset_end": 1024,
  "checkpoint_path": "checkpoints/0001_decoder.blocks.26.ffn.gate.kernel.safetensors",
  "checkpoint_sha256": "...",
  "decompose_elapsed_ms": 259925,
  "reconstruct_elapsed_ms": 1234,
  "u_backend": "EXLA.Backend<cuda:0,...>",
  "s_backend": "EXLA.Backend<cuda:0,...>",
  "v_backend": "EXLA.Backend<cuda:0,...>",
  "adapted_backend": "EXLA.Backend<cuda:0,...>",
  "error": null
}
```

Allowed tensor `status` values:

- `pending`
- `running`
- `complete`
- `failed`
- `skipped`

Manifest write rule:

- Write `manifest.json.tmp` first.
- Rename to `manifest.json` after successful JSON encoding and fsync-equivalent
  best effort where practical.
- Update manifest after every tensor checkpoint completes.

## Source Identity And Reproducibility

Record enough identity to know whether an artifact matches the current code and
inputs.

Required identity fields:

- base model repo: `Qwen/Qwen3-0.6B`
- Bumblebee module: `Bumblebee.Text.Qwen3`
- architecture: `:for_causal_language_modeling`
- source vector path
- source vector tensor name
- source vector SHA-256
- selected tensor paths
- selected tensor shapes
- selected singular-value count
- router head shape
- export backend
- `XLA_TARGET`

Optional identity fields if easy to obtain:

- Bumblebee git ref from `mix.lock`
- Nx version
- EXLA version
- CUDA target
- GPU name
- Qwen Hugging Face revision if pinned later

If any required identity changes, the exporter must refuse `--resume` unless
`--force` is passed.

## Exact Modules To Add

### `lib/mix/tasks/trinity.sakana.export_adapted.ex`

Public command:

```bash
XLA_TARGET=cuda12 mix trinity.sakana.export_adapted
```

Supported options:

```bash
--out priv/sakana_trinity/adapted_qwen3_0_6b_layer26
--source-vector priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors
--tensor-name trinity_router_es_vector
--resume
--force
--only-index 1
--skip-existing
```

Responsibilities:

- Parse CLI options.
- Validate output policy.
- Call exporter module.
- Print progress.
- Exit nonzero on invalid inputs, failed export, or incomplete canonical output.

### `lib/trinity_coordinator/sakana/exporter.ex`

Primary API:

```elixir
TrinityCoordinator.Sakana.Exporter.export_adapted(opts)
```

Expected options:

```elixir
[
  out_dir: "priv/sakana_trinity/adapted_qwen3_0_6b_layer26",
  source_vector_path: "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors",
  source_vector_tensor: "trinity_router_es_vector",
  resume: true,
  force: false,
  only_index: nil,
  skip_existing: true,
  progress: &IO.puts/1
]
```

Return shape:

```elixir
{:ok, manifest_map}
{:error, reason}
```

Responsibilities:

- Create output directories.
- Load source vector.
- Split source vector.
- Save router head artifact.
- Load Qwen profile.
- Select tensors.
- Validate selected count and singular count.
- Build or resume manifest.
- Export one tensor at a time.
- Save checkpoint after each tensor.
- Merge checkpoints into final `adapted_tensors.safetensors` after all complete.
- Mark manifest complete.

### `lib/trinity_coordinator/sakana/artifact.ex`

Primary APIs:

```elixir
TrinityCoordinator.Sakana.Artifact.load_manifest!(dir)
TrinityCoordinator.Sakana.Artifact.load_router_head!(dir)
TrinityCoordinator.Sakana.Artifact.load_adapted_tensors!(dir, opts \\ [])
TrinityCoordinator.Sakana.Artifact.patch_params!(params, manifest, tensors)
TrinityCoordinator.Sakana.Artifact.patch_model_info!(model_info, dir, opts \\ [])
```

Responsibilities:

- Read and validate manifest.
- Load final safetensors artifacts.
- Map artifact keys to params paths.
- Patch Qwen params with adapted tensors.
- Refuse missing paths.
- Refuse shape mismatch.
- Refuse incomplete manifest.

### Optional `lib/trinity_coordinator/sakana/manifest.ex`

Add if manifest logic gets large.

Responsibilities:

- Build manifest structs/maps.
- Validate identity.
- Validate completion.
- Atomic JSON writes.
- Checkpoint status updates.

## Exporter Resume Semantics

Resume mode must be deterministic and conservative.

Definitions:

- A tensor is complete only if manifest status is `complete`, checkpoint path
  exists, checkpoint loads successfully, shape matches, type matches, and SHA-256
  matches.
- If checkpoint exists but manifest is missing, treat it as untrusted unless
  `--recover-checkpoints` is implemented later.
- If manifest says complete but checkpoint is missing, fail and require manual
  cleanup or `--force`.
- If manifest identity does not match current source identity, fail unless
  `--force`.
- If `--only-index` is passed, export only that tensor and mark manifest partial.
- If `--force` is passed, delete or overwrite output safely.

Resume checklist:

- [x] Read existing manifest if present.
- [x] Validate source vector SHA-256.
- [x] Validate selected tensor paths/shapes against manifest.
- [x] Validate completed checkpoints before skipping.
- [x] Skip only verified complete tensors.
- [x] Recompute incomplete or failed tensors.
- [x] Never silently reuse mismatched checkpoint files.

## Failure Semantics

On tensor failure:

- Mark tensor status `failed`.
- Record error string and stage.
- Write manifest.
- Keep completed checkpoints.
- Exit nonzero.

On interruption:

- Completed checkpoint files remain.
- Manifest may show current tensor as `running` or `failed`.
- Resume should re-run `running` tensor unless its checkpoint verifies complete.

On final merge failure:

- Keep checkpoints.
- Mark manifest `partial` or `failed`.
- Do not create `export_complete: true`.

Atomic write rule:

- checkpoint: write `file.tmp`, then rename to final path.
- manifest: write `manifest.json.tmp`, then rename.
- final adapted artifact: write `adapted_tensors.safetensors.tmp`, then rename.

## Export Tensor Algorithm

For each selected tensor:

1. Compute `singular_values = min(shape)`.
2. Compute `offset_start` and `offset_end` from cumulative previous counts.
3. Slice scale offsets from the source vector.
4. Log tensor path, shape, type, backend, offset span.
5. Run `SVD.decompose_tensor/2`.
6. Assert `u`, `s`, and `v` backends include `EXLA.Backend<cuda:` for EXLA export.
7. Reconstruct adapted tensor with Sakana normalization.
8. Assert adapted tensor backend includes `EXLA.Backend<cuda:` for EXLA export.
9. Transfer adapted tensor to `Nx.BinaryBackend` for file writing.
10. Save one-tensor checkpoint with `Safetensors.write!/2`.
11. Hash checkpoint file.
12. Update manifest.
13. Drop references before continuing where practical.

## Router Head Export Algorithm

1. Load raw source vector.
2. Split with `SVD.split_router_vector(vector, 9216, 1024, 10)`.
3. Save `head_weights` to `router_head.safetensors`.
4. Hash the router head file.
5. Record shape and hash in manifest.

The router head artifact should use key:

```text
trinity_router_head
```

## Final Artifact Merge Algorithm

After all tensor checkpoints are complete:

1. Load checkpoint tensors lazily if possible.
2. Build map `%{artifact_key => tensor}`.
3. Write `adapted_tensors.safetensors.tmp` with `Safetensors.write!/2`.
4. Rename to `adapted_tensors.safetensors`.
5. Hash final file.
6. Mark manifest `status: complete` and `export_complete: true`.

If final merge is too memory-heavy, keep checkpoint-per-tensor loading as the
first runtime loader implementation and add merge later. If this fallback is
used, manifest must say `artifact_layout: checkpoint_directory`, not pretend a
single final file exists.

## Runtime Loader Design

Runtime must not run SVD.

Loader path:

```elixir
{:ok, {model_info, tokenizer}} = TrinityCoordinator.SLMProfile.load_profile(:qwen_coordinator)
model_info = TrinityCoordinator.Sakana.Artifact.patch_model_info!(model_info, artifact_dir)
head_weights = TrinityCoordinator.Sakana.Artifact.load_router_head!(artifact_dir)
```

Patch rules:

- Manifest must be complete.
- Every manifest tensor path must exist in Qwen params.
- Every loaded tensor shape must match the target param shape.
- Every loaded tensor type should match or be explicitly cast according to a
  documented rule.
- Patched tensors must be transferred to the same backend as the original target
  param when needed.

Runtime profile target:

```elixir
:qwen_sakana_adapted
```

Runtime acceptance:

- No call to `Nx.LinAlg.svd/2`.
- Qwen forward pass works after patching.
- Hidden vector shape is `{1, 1024}`.
- Router head logits shape is `{1, 10}`.
- Trace records artifact manifest hash or source vector hash.

## Handling Giant Embedding And LM Head Tensors

The two largest selected tensors are:

```text
embedder.token_embedding.kernel        {151936, 1024}
language_modeling_head.output.kernel  {151936, 1024}
```

Canonical export must include them unless Python reference inspection proves they
are not part of Sakana's selected tensor set.

Do not silently skip them.

If EXLA SVD is too slow or fails:

- Keep the manifest format.
- Add export backend option.
- Use Python/PyTorch/JAX/NumPy only for offline export.
- Save the same adapted tensor artifacts.
- Keep Elixir runtime unchanged.

Partial debug artifact rule:

- If the giant tensors are skipped for debugging, manifest status must not be
  `complete`.
- Mark `partial_debug_only: true`.
- Runtime loader must reject it unless an explicit debug override is passed.

## Python Reference And Parity Requirements

Before claiming parity with Sakana:

- Confirm Sakana Python selected tensor order.
- Confirm whether embeddings and LM head are included.
- Confirm offset spans match Python order.
- Compare at least one adapted tensor numerically against Python output.
- Compare router head reshape/order against Python.
- If possible, compare adapted Qwen hidden vector or route logits against Python.

Do not claim paper-score reproduction from artifact loading alone.

## Test Plan In Correct Order

### Unit tests first

- [x] Manifest JSON build/validate.
- [x] Atomic JSON write helper.
- [x] Checkpoint path/key generation.
- [x] Source vector checksum helper.
- [x] Tiny synthetic checkpoint write/read with `Safetensors.write!/2` and
      `Safetensors.read!/2`.
- [x] Params patching on a tiny nested container.
- [x] Shape mismatch refusal.
- [x] Incomplete manifest refusal.

### Exporter tests second

- [x] `--only-index` exports one tensor and writes partial manifest.
- [x] Resume skips a verified complete checkpoint.
- [x] Resume refuses checksum mismatch.
- [x] `--force` overwrites output directory.
- [x] Router head artifact is saved and reloads with shape `{10, 1024}`.

### Qwen tests third

- [x] Qwen profile loads on CUDA.
- [x] Selected tensor set matches manifest paths and shapes.
- [x] Artifact loader patches Qwen params without SVD.
- [x] Patched Qwen extracts `{1, 1024}` hidden vector.
- [x] Imported router head routes real Qwen vector.

### Expensive/manual tests last

- [ ] Full canonical export completes or fails with a useful manifest/log. (Use this
  manual GPU check after full-run stabilization.)
- [x] Giant tensor timings are recorded.
- [ ] If EXLA export is impractical, alternate export backend is implemented.

## Migration Of Existing `:expensive_qwen_svd` Test

Current issue:

- The test computes adapted tensors in memory.
- It discards the result.
- It is not the correct full-export path.

Required change:

- [x] Replace full in-memory expensive test with exporter-driven test.
- [x] The expensive test should call the exporter with `--only-index` for a
      bounded smoke, or with full export only when explicitly requested.
- [x] The full canonical export command should save artifacts and be resumable.
- [x] Tests should verify artifacts after export, not recompute full SVD.

## Implementation Checklist

### 1. Generated Artifact Git Policy

- [x] Ignore `/priv/sakana_trinity/adapted_qwen3_0_6b_layer26/` in `.gitignore`.
- [x] Document that the directory is generated and reproducible.
- [x] Ensure the exporter creates the ignored directory when needed.
- [x] Ensure no test requires generated artifacts unless explicitly tagged or
      guarded with a clear skip/error message.

### 2. Stop Treating Full SVD As A Throwaway Test

- [x] Replace or rename the existing full SVD test so it is exporter-driven.
- [x] Keep small SVD math tests as normal tests.
- [x] Keep representative Qwen tensor offset-span tests as normal `:qwen` tests.
- [x] Ensure `:expensive_qwen_svd` remains excluded from plain `mix test`.
- [x] Document that full model SVD work must produce artifacts or should not be
      run.

### 3. Add Manifest Module Or Helpers

- [x] Define manifest schema and validation.
- [x] Implement atomic manifest write.
- [x] Implement source vector SHA-256.
- [x] Implement checkpoint SHA-256.
- [x] Implement manifest identity validation for resume.

### 4. Add A Resumable Export Mix Task

- [x] Add `Mix.Tasks.Trinity.Sakana.ExportAdapted`.
- [x] Parse `--out`.
- [x] Parse `--source-vector`.
- [x] Parse `--tensor-name`.
- [x] Parse `--resume`.
- [x] Parse `--force`.
- [x] Parse `--only-index`.
- [x] Parse `--skip-existing`.
- [x] Refuse invalid option combinations.
- [x] Print command summary before doing expensive work.

### 5. Implement Exporter Core

- [x] Create output directory.
- [x] Create checkpoints directory.
- [x] Load raw source vector.
- [x] Split source vector.
- [x] Save router head artifact.
- [x] Load Qwen coordinator profile.
- [x] Select tensor entries.
- [x] Validate tensor count.
- [x] Validate singular-value count `9216`.
- [x] Build offset spans.
- [x] Resume or initialize manifest.
- [x] Export each tensor one at a time.
- [x] Save checkpoint immediately after each tensor.
- [x] Update manifest immediately after each tensor.
- [x] Merge checkpoints into final adapted artifact when complete.
- [x] Mark manifest complete.

### 6. Implement Checkpoint Write/Read

- [x] Transfer tensor to `Nx.BinaryBackend` before writing.
- [x] Write temp checkpoint with `Safetensors.write!/2`.
- [x] Rename temp checkpoint atomically.
- [x] Read checkpoint with `Safetensors.read!/2`.
- [x] Verify checkpoint key.
- [x] Verify shape.
- [x] Verify type.
- [x] Verify checksum.

### 7. Implement Artifact Loader

- [x] Add `TrinityCoordinator.Sakana.Artifact`.
- [x] Load manifest.
- [x] Validate manifest complete.
- [x] Load router head.
- [x] Load adapted tensors from final artifact or checkpoints.
- [x] Patch params using manifest paths.
- [x] Refuse missing path.
- [x] Refuse shape mismatch.
- [x] Refuse incomplete artifact by default.

### 8. Add Runtime Adapted Profile

- [x] Add profile or option `:qwen_sakana_adapted`.
- [x] Load base Qwen profile.
- [x] Apply adapted tensor artifact.
- [x] Load router head artifact.
- [x] Route real transcript.
- [x] Emit trace metadata with artifact identity.

### 9. Add Parity Checks

- [x] Inspect Sakana Python tensor order.
- [x] Confirm embeddings and LM head behavior.
- [x] Compare offset spans.
- [x] Compare one adapted tensor numerically.
- [x] Compare router head logits where practical.

### 10. Update Documentation

- [x] Update `README.md` to point to export command as canonical.
- [x] Update `docs/elixir_svd_decomposition.md` to separate math/export/runtime.
- [x] Update `docs/production_qwen_slm_profile.md` for adapted profile.
- [x] Document export run time after first successful canonical export.
- [x] Document resume/failure recovery.

## Main Roadmap To A Functional System

1. Build exporter that saves progress.
2. Generate or partially generate adapted artifacts with resume.
3. Add loader that patches Qwen params from saved artifacts.
4. Add adapted Qwen runtime profile.
5. Load router head from saved artifact.
6. Route real transcript through adapted Qwen plus imported head.
7. Trace profile/backend/artifact/logits/route metadata.
8. Compare against Python reference enough to validate tensor order and vector
   application.
9. Keep full training reproduction deferred until the artifact-based coordinator
   works.

## Correct End State

```text
one-time/export:
  Qwen base + Sakana vector -> adapted_qwen artifact + router_head artifact

runtime:
  Qwen base + adapted_qwen artifact + router_head artifact -> hidden vector -> route

tests:
  verify artifact loading and routing without recomputing full SVD
```

The full SVD operation is a build/import step. It is not request-time runtime
behavior, and it is not a throwaway test that discards reusable weights.
