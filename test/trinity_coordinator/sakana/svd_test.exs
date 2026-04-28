defmodule TrinityCoordinator.Sakana.SVDTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{CoordinationHead, Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, Exporter, ParityTrace, SVD}

  @router_vector_path "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"
  @python_reference_manifest_path "priv/sakana_trinity/reference/sakana_python_reference_manifest.json"
  @qwen_smoke_tensor_path "decoder.blocks.26.self_attention.query.kernel"

  test "decomposes and reconstructs a matrix with Sakana normalization" do
    matrix =
      Nx.tensor(
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 10.0]
        ],
        type: :f32
      )

    decomposition = SVD.decompose_tensor(matrix)
    zeros = Nx.broadcast(0.0, Nx.shape(decomposition.s))
    reconstructed = SVD.reconstruct(decomposition, zeros)

    max_error =
      reconstructed
      |> Nx.subtract(matrix)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert Nx.shape(decomposition.u) == {3, 3}
    assert Nx.shape(decomposition.s) == {3}
    assert Nx.shape(decomposition.v) == {3, 3}
    assert max_error < 1.0e-3
  end

  test "reconstruct supports Python torch.svd V layout" do
    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([2.0, 1.0], type: :f32)
    torch_v = Nx.tensor([[0.0, 1.0], [-1.0, 0.0]], type: :f32)
    zeros = Nx.tensor([0.0, 0.0], type: :f32)

    reconstructed = SVD.reconstruct(%{u: u, s: s, v: torch_v}, zeros, v_layout: :torch_v)
    expected = Nx.tensor([[0.0, -2.0], [1.0, 0.0]], type: :f32)

    assert Nx.to_number(Nx.all_close(reconstructed, expected, atol: 1.0e-6, rtol: 1.0e-6)) == 1
  end

  @tag :integration
  test "decomposition and reconstruction preserve CUDA backend" do
    Runtime.put_cuda_backend!()

    matrix = Nx.iota({4, 3}, type: :f32) |> Nx.backend_transfer({EXLA.Backend, client: :cuda})
    decomposition = SVD.decompose_tensor(matrix)
    zeros = Nx.broadcast(0.0, Nx.shape(decomposition.s))
    reconstructed = SVD.reconstruct(decomposition, zeros)

    assert Runtime.tensor_backend(reconstructed) =~ "EXLA.Backend<cuda:"
  end

  test "adapt_tensors keeps scale offsets at SVD precision before final source-type cast" do
    tensor = Nx.tensor([[1.0, 0.0], [0.0, 2.0]], type: :bf16)
    entry = %{path: "bf16.kernel", segments: [:kernel], tensor: tensor}
    # These values intentionally survive the final bf16 cast differently if
    # offsets are prematurely quantized to bf16.  A prior fixture used offsets
    # whose quantization error disappeared after the final cast, so it could not
    # catch the precision regression reliably.
    offsets = Nx.tensor([0.001, -0.67], type: :f32)

    adapted = SVD.adapt_tensors([entry], offsets, svd_compute_type: :f32)
    actual = adapted.tensors |> List.first() |> Map.fetch!(:tensor)

    decomposition = SVD.decompose_tensor(tensor, compute_type: :f32)

    expected =
      decomposition
      |> SVD.reconstruct(offsets)
      |> Nx.as_type(:bf16)

    quantized_offset_match? =
      Nx.all_close(
        actual,
        SVD.reconstruct(decomposition, Nx.as_type(offsets, :bf16)) |> Nx.as_type(:bf16),
        atol: 0.0,
        rtol: 0.0
      )
      |> Nx.to_number()

    exact_match? = Nx.all_close(actual, expected, atol: 0.0, rtol: 0.0) |> Nx.to_number()

    assert quantized_offset_match? == 0
    assert exact_match? == 1
  end

  test "selects only matrix-like tensors and flattens paths deterministically" do
    container = %{
      z: Nx.iota({2}, type: :f32),
      a: %{
        singleton: Nx.iota({2, 1}, type: :f32),
        matrix: Nx.iota({2, 3}, type: :f32)
      },
      b: [Nx.iota({3, 2}, type: :f32)]
    }

    flattened = SVD.flatten_tensors(container)
    selected = SVD.decomposable_tensors(container)

    assert Enum.map(flattened, &elem(&1, 0)) == ["a.matrix", "a.singleton", "b.0", "z"]
    assert Enum.map(selected, &elem(&1, 0)) == ["a.matrix", "b.0"]
    assert SVD.singular_value_count(selected) == 4

    entries = SVD.decomposable_tensor_entries(container)
    assert Enum.map(entries, & &1.segments) == [[:a, :matrix], [:b, 0]]
  end

  test "loads and splits the Sakana router vector safetensors artifact" do
    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    assert Nx.shape(vector) == {19_456}
    assert Nx.shape(split.scale_offsets) == {9216}
    assert Nx.shape(split.head_weights) == {10, 1024}
    assert split.scale_count == 9216
    assert split.head_count == 10_240
  end

  test "loads Sakana head weights into the linear Axon head" do
    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())
    updated = SVD.put_linear_head_weights(params, split.head_weights)

    assert Nx.shape(updated.data["routing_head"]["kernel"]) == {1024, 10}
    assert Nx.shape(updated.data["routing_head"]["bias"]) == {10}

    route = CoordinationHead.route(model, updated, Nx.broadcast(0.01, {1, 1024}), 7, 3)

    assert Nx.shape(route.logits) == {1, 10}
    assert Nx.shape(route.agent_logits) == {7}
    assert Nx.shape(route.role_logits) == {3}
  end

  test "applies scale offsets to selected decomposed tensors in deterministic order" do
    container = %{
      "layer.with.dots" => %{"kernel" => Nx.tensor([[1.0, 0.0], [0.0, 2.0]], type: :f32)},
      b: [Nx.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]], type: :f32)]
    }

    tensors = SVD.decomposable_tensor_entries(container)

    zero_offsets = Nx.broadcast(0.0, {4})
    zero_adapted = SVD.adapt_tensors(tensors, zero_offsets)

    assert zero_adapted.offset_count == 4
    assert Enum.map(zero_adapted.tensors, & &1.path) == ["b.0", "layer.with.dots.kernel"]

    first_error =
      zero_adapted.tensors
      |> hd()
      |> Map.fetch!(:tensor)
      |> Nx.subtract(hd(tensors).tensor)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert first_error < 1.0e-3

    nonzero_offsets = Nx.tensor([0.1, -0.1, 0.2, -0.2], type: :f32)
    nonzero_adapted = SVD.adapt_tensors(tensors, nonzero_offsets)

    assert Enum.map(nonzero_adapted.tensors, &Nx.shape(&1.tensor)) == [{3, 2}, {2, 2}]
    assert Enum.all?(nonzero_adapted.tensors, &(Nx.type(&1.tensor) == {:f, 32}))

    updated = SVD.put_tensor_entries(container, zero_adapted.tensors)
    assert Nx.shape(updated.b |> hd()) == {3, 2}
    assert Nx.shape(updated["layer.with.dots"]["kernel"]) == {2, 2}
  end

  @tag :qwen
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "selects Qwen SVF tensors for Sakana layer 26 on CUDA" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    selected =
      SVD.decomposable_tensors(model_info.params,
        path_filter: SVD.layer_index_filter([26])
      )

    assert SVD.singular_value_count(selected) == 9216

    assert Enum.any?(selected, fn {path, _tensor} ->
             String.contains?(path, "decoder.blocks.26.")
           end)

    assert Enum.any?(selected, fn {path, _tensor} ->
             not String.contains?(path, "decoder.blocks.")
           end)

    {_path, tensor} = hd(selected)
    assert Runtime.tensor_backend(tensor) =~ "EXLA.Backend<cuda:"

    manifest = SVD.tensor_manifest(selected)
    assert Enum.any?(manifest, &(&1.singular_values > 0))
    assert Enum.all?(manifest, &is_binary(&1.path))
  end

  @tag :qwen
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "maps a representative Qwen layer 26 tensor to its Sakana scale-offset span on CUDA" do
    Runtime.put_cuda_backend!()

    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    selected =
      SVD.decomposable_tensors(model_info.params,
        path_filter: SVD.layer_index_filter([26])
      )

    {offset_start, {_path, tensor}} =
      Enum.reduce_while(selected, 0, fn {path, tensor} = item, offset ->
        if String.contains?(path, "decoder.blocks.26.self_attention.query") do
          {:halt, {offset, item}}
        else
          count = tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min()
          {:cont, offset + count}
        end
      end)

    singular_values = tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min()
    offsets = Nx.slice(split.scale_offsets, [offset_start], [singular_values])

    assert offset_start >= 0
    assert offset_start + singular_values <= split.scale_count
    assert Nx.shape(offsets) == {singular_values}
    assert Runtime.tensor_backend(tensor) =~ "EXLA.Backend<cuda:"
  end

  @tag :expensive_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "exports one Sakana-selected Qwen tensor and validates partial artifacts" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_adapted_smoke_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: &expensive_svd_progress/1
             )

    assert manifest["status"] == "partial"
    assert manifest["export_complete"] == false
    assert manifest["selected_tensor_count"] == 9

    head = Artifact.load_router_head!(out_dir, manifest: manifest, allow_incomplete: true)
    assert Nx.shape(head) == {10, 1024}

    complete_entries =
      manifest["selected_tensors"]
      |> Enum.filter(fn entry -> entry["status"] == "complete" end)
      |> Enum.to_list()

    assert length(complete_entries) == 1

    completed = hd(complete_entries)
    checkpoint_path = Path.join(out_dir, completed["checkpoint_path"])

    assert File.exists?(checkpoint_path)

    tensors = Safetensors.read!(checkpoint_path)
    assert map_size(tensors) == 1
    assert Map.has_key?(tensors, completed["artifact_key"])

    events = load_export_log_events(out_dir)
    assert Enum.any?(events, &(&1["event"] == "export_started"))
    assert Enum.any?(events, &(&1["event"] == "tensor_export_started"))
    assert Enum.any?(events, &(&1["event"] == "tensor_export_finished"))
    assert Enum.any?(events, &(&1["event"] == "manifest_partial"))
  end

  @tag :expensive_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "resume refuses mismatched manifest identity" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_identity_mismatch_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: nil
             )

    mismatched =
      manifest
      |> Map.put("source_vector_path", "/this/path/is/not/the/real/source.safetensors")
      |> Map.put("source_vector_sha256", "000000")

    Artifact.write_manifest!(out_dir, mismatched)

    assert {:error, {:export_exception, message}} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               resume: true,
               skip_existing: true,
               progress: nil
             )

    assert message =~ "existing manifest identity mismatch"

    events = load_export_log_events(out_dir)
    assert Enum.any?(events, &(&1["event"] == "export_failed"))
  end

  @tag :expensive_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "resume reuses a verified one-tensor checkpoint in partial export" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_adapted_resume_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, first_manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: &expensive_svd_progress/1
             )

    first_entry =
      first_manifest["selected_tensors"]
      |> Enum.find(&(&1["index"] == smoke_tensor_index))

    assert first_entry["status"] == "complete"

    checkpoint_path = Path.join(out_dir, first_entry["checkpoint_path"])
    assert File.exists?(checkpoint_path)
    first_checksum = Artifact.file_sha256!(checkpoint_path)

    assert {:ok, second_manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               resume: true,
               skip_existing: true,
               progress: &expensive_svd_progress/1
             )

    second_entry =
      second_manifest["selected_tensors"]
      |> Enum.find(&(&1["index"] == smoke_tensor_index))

    assert second_entry["status"] == "complete"

    assert Artifact.file_sha256!(checkpoint_path) == first_checksum
  end

  @tag :expensive_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "resume refuses to reuse a checkpoint with a checksum mismatch" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_mismatch_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, first_manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: &expensive_svd_progress/1
             )

    first_entry =
      first_manifest["selected_tensors"]
      |> Enum.find(&(&1["index"] == smoke_tensor_index))

    assert first_entry["status"] == "complete"

    checkpoint_path = Path.join(out_dir, first_entry["checkpoint_path"])

    original_tensor =
      checkpoint_path
      |> Safetensors.read!()
      |> Map.fetch!(first_entry["artifact_key"])
      |> Nx.as_type(:f32)

    mismatched_tensor = Nx.negate(original_tensor)

    payload = %{
      first_entry["artifact_key"] => Nx.backend_transfer(mismatched_tensor, Nx.BinaryBackend)
    }

    Safetensors.write!(checkpoint_path, payload)
    mismatch_checksum = Artifact.file_sha256!(checkpoint_path)

    assert mismatch_checksum != first_entry["checkpoint_sha256"]

    assert {:ok, second_manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               resume: true,
               skip_existing: true,
               progress: &expensive_svd_progress/1
             )

    second_entry =
      second_manifest["selected_tensors"]
      |> Enum.find(&(&1["index"] == smoke_tensor_index))

    final_checksum = Artifact.file_sha256!(checkpoint_path)

    assert second_entry["status"] == "complete"
    assert second_entry["checkpoint_sha256"] == final_checksum
    assert final_checksum != mismatch_checksum
    assert second_entry["checkpoint_sha256"] == first_entry["checkpoint_sha256"]
  end

  @tag :expensive_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "force overwrites existing output directory for partial export" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_force_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(out_dir)
    stale_path = Path.join(out_dir, "stale.txt")
    File.write!(stale_path, "stale")

    assert {:ok, manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: &expensive_svd_progress/1
             )

    assert File.exists?(Path.join(out_dir, Artifact.manifest_file()))
    assert manifest["selected_tensor_count"] == 9
    assert manifest["export_complete"] == false
  end

  @tag :qwen
  @tag :expensive_qwen_svd
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "selected Qwen tensor manifest metadata matches live model selection" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    selected =
      SVD.decomposable_tensor_entries(
        model_info.params,
        path_filter: SVD.layer_index_filter([26])
      )

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_manifest_match_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: nil
             )

    expected_paths = Enum.map(selected, & &1.path) |> Enum.sort()

    manifest_paths =
      manifest["selected_tensors"]
      |> Enum.map(& &1["path"])
      |> Enum.sort()

    expected_shape_by_path =
      Map.new(selected, fn entry -> {entry.path, Nx.shape(entry.tensor)} end)

    assert manifest["selected_tensor_count"] == length(selected)
    assert manifest["selected_singular_value_count"] == SVD.singular_value_count(selected)
    assert manifest_paths == expected_paths

    Enum.each(manifest["selected_tensors"], fn entry ->
      manifest_shape = Nx.shape(Map.fetch!(expected_shape_by_path, entry["path"]))
      assert entry["shape"] == Tuple.to_list(manifest_shape)
    end)
  end

  @tag :qwen
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "selected Qwen tensor order and span match the Sakana Python reference" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    selected = qwen_selected_tensors(model_info)
    reference = python_reference_manifest!()

    reference_paths = Enum.map(reference["selected_tensors"], & &1["elixir_name"])
    actual_paths = Enum.map(selected, & &1.path)

    assert actual_paths == reference_paths
    assert Enum.at(reference_paths, 0) == "embedder.token_embedding.kernel"
    assert Enum.at(reference_paths, -1) == "language_modeling_head.output.kernel"
    assert "embedder.token_embedding.kernel" in actual_paths
    assert "language_modeling_head.output.kernel" in actual_paths

    Enum.each(reference["selected_tensors"], fn reference_entry ->
      path = reference_entry["elixir_name"]
      selected_entry = Enum.find(selected, &(&1.path == path))
      assert selected_entry
      selected_shape = Nx.shape(selected_entry.tensor) |> Tuple.to_list()
      reference_shape = reference_entry["shape"]

      assert selected_shape == reference_shape or selected_shape == Enum.reverse(reference_shape)

      assert Tuple.to_list(Nx.shape(selected_entry.tensor)) |> Enum.min() ==
               reference_entry["singular_values"]

      assert selected_tensor_offset_start(selected, path) == reference_entry["offset_start"]
      expected_end = reference_entry["offset_start"] + reference_entry["singular_values"]
      assert selected_tensor_offset_end(selected, path) == expected_end
    end)
  end

  @tag :qwen
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "reconstructs a Python reference tensor sample with expected hash" do
    # The Python reference hash is a PyTorch-SVD-component parity target, not a
    # native-SVD uniqueness target.  With non-zero singular-value offsets, two
    # mathematically valid SVD bases can reconstruct the source with zero offsets
    # and still produce different adapted tensors.  Emit full stage diagnostics
    # by setting TRINITY_SVD_PARITY_OUT.  Exact historical hash assertions are
    # opt-in because the stored reference hash requires the original Python SVD
    # component provenance, not a fresh SVD recomputation in a new runtime.
    components_dir = python_components_dir_from_env()

    report =
      ParityTrace.sample_report!(
        router_vector_path: @router_vector_path,
        reference_manifest_path: @python_reference_manifest_path,
        components_dir: components_dir,
        python_report_path: System.get_env("TRINITY_PYTHON_PARITY_REPORT")
      )

    case System.get_env("TRINITY_SVD_PARITY_OUT") do
      nil -> :ok
      "" -> :ok
      out -> ParityTrace.write_json!(out, report)
    end

    expected = get_in(report, ["reference", "expected_bf16_sha256"])
    assert expected == "600be6ab0f5a34325b9857182ccb5fce5971549a0ce8588cdacc992eda54014c"

    native_variants = Map.fetch!(report, "native_elixir_svd_variants")
    assert length(native_variants) >= 4
    assert Enum.all?(native_variants, &(byte_size(&1["observed_bf16_sha256"]) == 64))
    assert Enum.all?(native_variants, &(&1["zero_offset_max_abs_error_vs_source"] < 1.0))

    cond do
      is_binary(components_dir) ->
        semantic_variants = Map.fetch!(report, "semantic_python_component_variants")

        assert is_list(semantic_variants),
               "expected semantic Python component variants, got: #{inspect(semantic_variants)}"

        torch_v = Enum.find(semantic_variants, &(&1["v_layout"] == "torch_v"))
        assert torch_v, parity_failure_message(report)

        assert is_number(torch_v["zero_offset_max_abs_error_vs_source"]),
               parity_failure_message(report)

        assert torch_v["zero_offset_max_abs_error_vs_source"] < 1.0e-2,
               parity_failure_message(report)

        cond do
          strict_reference_hash?() ->
            assert Enum.any?(semantic_variants, & &1["matches_expected"]),
                   parity_failure_message(report)

          strict_python_current_hash?() ->
            assert Enum.any?(semantic_variants, & &1["matches_python_current"]),
                   parity_failure_message(report)

          true ->
            unless Enum.any?(
                     semantic_variants,
                     &(&1["matches_expected"] || &1["matches_python_current"])
                   ) do
              IO.puts(parity_failure_message(report))
            end
        end

      strict_native_svd_hash?() ->
        assert Enum.any?(native_variants, & &1["matches_expected"]),
               parity_failure_message(report)

      true ->
        unless Enum.any?(native_variants, & &1["matches_expected"]) do
          IO.puts(parity_failure_message(report))
        end
    end
  end

  @tag :qwen
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  test "router head logits match direct projection of split router head weights" do
    Runtime.put_cuda_backend!()

    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)
    hidden_vector = Nx.iota({1, 1024}, type: :f32)

    direct_logits = Nx.dot(hidden_vector, Nx.transpose(split.head_weights)) |> Nx.as_type(:f32)

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())
    params = SVD.put_linear_head_weights(params, split.head_weights)

    route = CoordinationHead.route(model, params, hidden_vector, 7, 3)

    assert Nx.shape(route.logits) == {1, 10}
    assert Nx.all_close(route.logits, direct_logits, atol: 1.0e-5, rtol: 1.0e-5)
  end

  @tag :qwen
  @tag :expensive_qwen_svd
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  @tag :qwen_sakana_adapted
  test "patches qwen model params from a one-tensor resumed artifact without recomputing SVD" do
    Runtime.put_cuda_backend!()

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_qwen_patch_#{System.unique_integer([:positive])}"
      )

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, full_manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: nil
             )

    index_entry =
      Enum.find(full_manifest["selected_tensors"], &(&1["index"] == smoke_tensor_index))

    assert index_entry["status"] == "complete"
    assert is_list(index_entry["segments"])

    manifest = Map.put(full_manifest, "selected_tensors", [index_entry])
    manifest = Map.put(manifest, "selected_tensor_count", 1)
    manifest = Map.put(manifest, "selected_singular_value_count", index_entry["singular_values"])

    original_path_tensor = fetch_tensor!(model_info.params, index_entry["segments"])

    patched_model_info =
      Artifact.patch_model_info!(model_info, out_dir,
        manifest: manifest,
        allow_incomplete: true,
        cast_head: false,
        patch_router_head: false
      )

    patched_path_tensor = fetch_tensor!(patched_model_info.params, index_entry["segments"])

    max_error =
      Nx.subtract(patched_path_tensor, original_path_tensor)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert max_error > 0.0
  end

  @tag :qwen
  @tag :expensive_qwen_svd
  @tag :slow_qwen_svd
  @tag timeout: 30 * 60 * 1000
  @tag :qwen_sakana_adapted
  test "routes a real Qwen hidden vector through the persisted Sakana router head on CUDA" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    smoke_tensor_index = qwen_smoke_tensor_index!(model_info)

    out_dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_sakana_head_route_#{System.unique_integer([:positive])}"
      )

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    assert {:ok, manifest} =
             Exporter.export_adapted(
               out_dir: out_dir,
               source_vector_path: @router_vector_path,
               source_vector_tensor: "trinity_router_es_vector",
               only_index: smoke_tensor_index,
               force: true,
               skip_existing: false,
               progress: nil
             )

    head_weights =
      Artifact.load_router_head!(out_dir, manifest: manifest, allow_incomplete: true)

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())
    params = SVD.put_linear_head_weights(params, head_weights)

    assert {:ok, metadata} =
             TrinityCoordinator.Extractor.extract_penultimate_hidden_state_with_metadata(
               model_info,
               tokenizer,
               [%{"role" => "user", "content" => "Route this request."}]
             )

    route = CoordinationHead.route(model, params, metadata.vector, 7, 3)

    assert metadata.vector_shape == {1, 1024}
    assert Runtime.tensor_backend(metadata.vector) =~ "EXLA.Backend<cuda:"
    assert Runtime.tensor_backend(route.logits) =~ "EXLA.Backend<cuda:"
    assert Nx.shape(route.logits) == {1, 10}
  end

  defp python_components_dir_from_env do
    path =
      System.get_env("TRINITY_PYTHON_COMPONENTS_DIR") ||
        System.get_env("TRINITY_PYTHON_SVD_COMPONENTS")

    cond do
      not is_binary(path) or path == "" -> nil
      Path.extname(path) == ".safetensors" -> Path.dirname(path)
      true -> path
    end
  end

  defp strict_native_svd_hash? do
    truthy_env?("TRINITY_STRICT_NATIVE_SVD_HASH")
  end

  defp strict_reference_hash? do
    truthy_env?("TRINITY_STRICT_REFERENCE_HASH")
  end

  defp strict_python_current_hash? do
    truthy_env?("TRINITY_STRICT_CURRENT_PYTHON_HASH")
  end

  defp truthy_env?(name) do
    System.get_env(name) in ["1", "true", "TRUE", "yes", "YES"]
  end

  defp parity_failure_message(report) do
    expected = get_in(report, ["reference", "expected_bf16_sha256"])
    python_current = get_in(report, ["python_current_baseline", "observed_bf16_sha256"])

    python_reproducible =
      get_in(report, ["python_current_baseline", "expected_hash_reproducible"])

    native =
      report
      |> Map.get("native_elixir_svd_variants", [])
      |> Enum.map_join("; ", fn variant ->
        "#{variant["label"]}=#{variant["observed_bf16_sha256"]} zero_error=#{variant["zero_offset_max_abs_error_vs_source"]}"
      end)

    semantic =
      case Map.get(report, "semantic_python_component_variants") do
        variants when is_list(variants) ->
          Enum.map_join(variants, "; ", fn variant ->
            "#{variant["label"]}=#{variant["observed_bf16_sha256"]} match_python=#{variant["matches_python_current"]} zero_error=#{variant["zero_offset_max_abs_error_vs_source"]}"
          end)

        other ->
          inspect(other)
      end

    "[qwen_svd_hash] stored_expected=#{expected}; current_python=#{python_current}; " <>
      "python_reproduces_stored=#{inspect(python_reproducible)}; " <>
      "native_variants=[#{native}]; semantic_variants=[#{semantic}]. " <>
      "Write TRINITY_SVD_PARITY_OUT=tmp/sakana_parity/elixir_sample_trace.json and compare with " <>
      "priv/sakana_trinity/scripts/debug_sakana_parity_sample.py output. " <>
      "Use TRINITY_STRICT_REFERENCE_HASH=1 only when the Python report itself reproduces the stored hash."
  end

  defp qwen_selected_tensors(model_info) do
    SVD.decomposable_tensor_entries(
      model_info.params,
      path_filter: SVD.layer_index_filter([26])
    )
  end

  defp qwen_smoke_tensor_index!(model_info) do
    qwen_selected_tensors(model_info)
    |> Enum.with_index(1)
    |> Enum.find_value(fn {entry, index} ->
      if selected_path(entry) == @qwen_smoke_tensor_path do
        index
      end
    end)
    |> case do
      nil -> raise ArgumentError, "expected #{@qwen_smoke_tensor_path} in selected tensors"
      index -> index
    end
  end

  defp python_reference_manifest! do
    @python_reference_manifest_path
    |> File.read!()
    |> Jason.decode!()
  end

  defp selected_tensor_offset_start(selected_tensors, target_path) do
    selected_tensors
    |> Enum.reduce_while({0, false}, fn entry, {offset, _found} ->
      if selected_path(entry) == target_path do
        {:halt, {offset, true}}
      else
        {:cont, {offset + tensor_singular_values(entry), false}}
      end
    end)
    |> case do
      {offset, true} -> offset
      {_, false} -> raise ArgumentError, "path #{inspect(target_path)} not selected"
    end
  end

  defp selected_tensor_offset_end(selected_tensors, target_path) do
    selected_tensor_offset_start(selected_tensors, target_path) +
      selected_tensor_singular_values!(selected_tensors, target_path)
  end

  defp selected_tensor_singular_values!(selected_tensors, target_path) do
    Enum.find_value(selected_tensors, fn entry ->
      if selected_path(entry) == target_path do
        tensor_singular_values(entry)
      end
    end) || raise(ArgumentError, "path #{inspect(target_path)} not selected")
  end

  defp tensor_singular_values(entry) do
    entry.tensor
    |> Nx.shape()
    |> Tuple.to_list()
    |> Enum.min()
  end

  defp selected_path(entry) do
    Map.get(entry, :path) || Map.get(entry, "path")
  end

  defp fetch_tensor!(%Axon.ModelState{} = container, segments) when is_list(segments) do
    fetch_tensor!(container.data, segments)
  end

  defp fetch_tensor!(container, [segment]) do
    fetch_child!(container, segment)
  end

  defp fetch_tensor!(container, [segment | rest]) do
    container
    |> fetch_child!(segment)
    |> fetch_tensor!(rest)
  end

  defp fetch_child!(container, segment) when is_list(container) and is_integer(segment) do
    Enum.fetch!(container, segment)
  end

  defp fetch_child!(container, segment) when is_tuple(container) and is_integer(segment) do
    elem(container, segment)
  end

  defp fetch_child!(container, segment) when is_map(container) do
    case resolve_test_map_key(container, segment) do
      nil -> raise ArgumentError, "missing map segment #{inspect(segment)}"
      key -> Map.fetch!(container, key)
    end
  end

  defp fetch_child!(container, segment) do
    raise ArgumentError,
          "cannot descend into #{inspect(container)} with segment #{inspect(segment)}"
  end

  defp resolve_test_map_key(container, segment) do
    cond do
      Map.has_key?(container, segment) -> segment
      is_binary(segment) -> existing_atom_key(container, segment)
      true -> nil
    end
  end

  defp existing_atom_key(container, segment) do
    atom = existing_atom(segment)
    if atom && Map.has_key?(container, atom), do: atom
  end

  defp existing_atom(key) when is_binary(key) do
    String.to_existing_atom(key)
  rescue
    ArgumentError -> nil
  end

  defp load_export_log_events(out_dir) do
    path = Artifact.export_log_path(out_dir)
    body = File.read!(path)

    body
    |> String.split("\n", trim: true)
    |> Enum.map(&Jason.decode!/1)
  end

  defp expensive_svd_progress(%{event: :tensor_export_started} = event) do
    expensive_log("tensor_export_started index=#{event.index} path=#{event.path}")
  end

  defp expensive_svd_progress(%{event: :tensor_export_progress} = event) do
    total = Map.get(event, :total)
    total_fragment = if total, do: "/#{total}", else: ""

    expensive_log(
      "tensor_export_progress #{event.index}#{total_fragment} path=#{event.path} decompose_ms=#{event.decompose_ms} reconstruct_ms=#{event.reconstruct_ms}"
    )
  end

  defp expensive_svd_progress(%{event: :tensor_export_finished} = event) do
    index = Map.get(event, :index, "?")
    total = Map.get(event, :total)
    total_fragment = if total, do: "/#{total}", else: ""

    expensive_log("tensor_export_finished #{index}#{total_fragment} path=#{event.path}")
  end

  defp expensive_svd_progress(%{event: :tensor_skipped} = event) do
    expensive_log("tensor_skipped path=#{event.path}")
  end

  defp expensive_svd_progress(%{event: :router_head_export_complete} = event) do
    expensive_log("router_head_export_complete path=#{event.path} sha256=#{event.sha256}")
  end

  defp expensive_svd_progress(%{event: :router_head_skipped} = event) do
    expensive_log("router_head_skipped path=#{event.path}")
  end

  defp expensive_svd_progress(%{event: :decompose_started} = event) do
    expensive_log(
      "decompose start #{event.index}/#{event.total} path=#{event.path} shape=#{inspect(event.shape)} type=#{inspect(event.type)} singular_values=#{event.singular_values}"
    )
  end

  defp expensive_svd_progress(%{event: :decompose_finished} = event) do
    expensive_log(
      "decompose done #{event.index}/#{event.total} path=#{event.path} u_backend=#{event.u_backend} s_backend=#{event.s_backend} v_backend=#{event.v_backend} elapsed_ms=#{event.elapsed_ms}"
    )
  end

  defp expensive_svd_progress(%{event: :reconstruct_started} = event) do
    expensive_log(
      "reconstruct start #{event.index}/#{event.total} path=#{event.path} offset_span=#{event.offset_start}..#{event.offset_end} singular_values=#{event.singular_values}"
    )
  end

  defp expensive_svd_progress(%{event: :reconstruct_finished} = event) do
    expensive_log(
      "reconstruct done #{event.index}/#{event.total} path=#{event.path} tensor_backend=#{event.tensor_backend} elapsed_ms=#{event.elapsed_ms}"
    )
  end

  defp expensive_svd_progress(event) do
    expensive_log("export progress: #{inspect(event)}")
  end

  defp expensive_log(message) do
    IO.puts("[expensive_qwen_svd] #{message}")
  end
end
