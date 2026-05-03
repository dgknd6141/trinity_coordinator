defmodule Mix.Tasks.Trinity.Sakana.ParitySampleTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Sakana.ParitySample
  alias TrinityCoordinator.Sakana.ParityTrace

  test "semantic-only disables native SVD diagnostics" do
    opts =
      ParitySample.parse_args!([
        "--semantic-only",
        "--components-dir",
        "tmp/sakana_parity/python_components",
        "--stage-dir",
        "tmp/sakana_parity/elixir_stages"
      ])

    assert opts[:components_dir] == "tmp/sakana_parity/python_components"
    assert opts[:stage_dir] == "tmp/sakana_parity/elixir_stages"
    refute opts[:native?]
  end

  test "fast semantic replay options are parsed explicitly" do
    opts =
      ParitySample.parse_args!([
        "--semantic-only",
        "--device-semantic-only",
        "--preferred-layout-only",
        "--source-from-python-stage"
      ])

    refute opts[:native?]
    assert opts[:device_semantic_only]
    assert opts[:preferred_layout_only]
    assert opts[:source_from_python_stage]
  end

  test "all-selected semantic replay option is parsed explicitly" do
    opts =
      ParitySample.parse_args!([
        "--semantic-only",
        "--all-selected-tensors",
        "--selected-source-filter",
        "model.layers.26.",
        "--source-from-python-stage"
      ])

    refute opts[:native?]
    assert opts[:all_selected_tensors]
    assert opts[:selected_source_filter] == "model.layers.26."
    assert opts[:source_from_python_stage]
  end

  test "native SVD diagnostics stay enabled by default" do
    opts = ParitySample.parse_args!([])

    assert opts[:native?]
  end

  @tag :tmp_dir
  test "all-selected semantic replay can filter selected tensors before namespaced stage checks",
       %{tmp_dir: tmp_dir} do
    components_dir = Path.join(tmp_dir, "components")
    stage_dir = Path.join(tmp_dir, "elixir_stages")
    File.mkdir_p!(components_dir)

    reference_path = Path.join(tmp_dir, "reference.json")
    router_vector_path = Path.join(tmp_dir, "router.safetensors")
    python_report_path = Path.join(tmp_dir, "python_report.json")
    sample_stage_path = Path.join(components_dir, "trinity_svf_stage_debug.safetensors")
    all_stage_path = Path.join(components_dir, "trinity_svf_all_selected_stage_debug.safetensors")

    entries = [
      %{
        "source_name" => "model.synthetic_a.weight",
        "elixir_name" => "synthetic.a.kernel",
        "shape" => [2, 2],
        "singular_values" => 2,
        "offset_start" => 0,
        "offset_end" => 2
      },
      %{
        "source_name" => "model.synthetic_b.weight",
        "elixir_name" => "synthetic.b.kernel",
        "shape" => [2, 2],
        "singular_values" => 2,
        "offset_start" => 2,
        "offset_end" => 4
      }
    ]

    sample =
      entries
      |> hd()
      |> Map.merge(%{
        "source_shape" => [2, 2],
        "sample_reconstructed_shape" => [2, 2],
        "sample_reconstructed_bf16_sha256" => String.duplicate("0", 64),
        "sample_reconstructed_bf16_min" => 0.0,
        "sample_reconstructed_bf16_max" => 2.0
      })

    File.write!(
      reference_path,
      Jason.encode!(%{"sample_adapted_tensor" => sample, "selected_tensors" => entries})
    )

    Safetensors.write!(router_vector_path, %{
      "trinity_router_es_vector" => Nx.broadcast(Nx.tensor(0.0, type: :f32), {19_456})
    })

    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)
    v = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    source = Nx.tensor([[1.0, 0.0], [0.0, 2.0]], type: :f32)
    generic_stages = synthetic_stage_tensors(source, offsets, s)

    Safetensors.write!(Path.join(components_dir, "trinity_svf_components.safetensors"), %{
      "svd.U.model.synthetic_a.weight" => u,
      "svd.S.model.synthetic_a.weight" => s,
      "svd.V.model.synthetic_a.weight" => v,
      "svd.U.model.synthetic_b.weight" => u,
      "svd.S.model.synthetic_b.weight" => s,
      "svd.V.model.synthetic_b.weight" => v
    })

    Safetensors.write!(Path.join(components_dir, "trinity_svf_scale_offsets.safetensors"), %{
      "svf.scale_offsets.model.synthetic_a.weight" => offsets,
      "svf.scale_offsets.model.synthetic_b.weight" => offsets
    })

    Safetensors.write!(sample_stage_path, generic_stages)

    all_stage_payload =
      entries
      |> Enum.flat_map(fn entry ->
        safe = entry["source_name"]

        Enum.map(generic_stages, fn {"stage." <> stage, tensor} ->
          {"tensor.#{safe}.#{stage}", tensor}
        end)
      end)
      |> Map.new()

    Safetensors.write!(all_stage_path, all_stage_payload)

    File.write!(
      python_report_path,
      Jason.encode!(%{
        "reference" => %{
          "current_python_baseline_label" => "synthetic",
          "current_python_baseline_bf16_sha256" => String.duplicate("f", 64),
          "expected_hash_reproducible" => false
        },
        "stage_debug" => %{
          "stage_tensor_file" => sample_stage_path,
          "all_selected_stage_tensor_file" => all_stage_path
        }
      })
    )

    report =
      ParityTrace.sample_report!(
        router_vector_path: router_vector_path,
        reference_manifest_path: reference_path,
        components_dir: components_dir,
        python_report_path: python_report_path,
        stage_dir: stage_dir,
        native?: false,
        semantic_host?: true,
        semantic_device?: false,
        semantic_layout_diagnostics?: false,
        source_from_python_stage?: true,
        all_selected_tensors?: true,
        selected_source_filter: "synthetic_b",
        require_cuda: false
      )

    variants = report["semantic_python_component_variants"]

    assert Enum.map(variants, & &1["source_name"]) == ["model.synthetic_b.weight"]

    assert Enum.all?(variants, &get_in(&1, ["stage_debug", "functional_parity_passed"]))
    assert Enum.all?(variants, &(length(get_in(&1, ["stage_debug", "checks"])) == 10))
  end

  @tag :tmp_dir
  test "parity JSON preserves booleans and nulls", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "report.json")

    ParityTrace.write_json!(path, %{
      "functional_passed" => true,
      "byte_match" => false,
      "stage_tensor_file" => nil,
      "atom_label" => :torch_v
    })

    assert %{
             "functional_passed" => true,
             "byte_match" => false,
             "stage_tensor_file" => nil,
             "atom_label" => "torch_v"
           } = path |> File.read!() |> Jason.decode!()
  end

  defp synthetic_stage_tensors(source, offsets, s) do
    normalization = Nx.tensor([1.0], type: :f32)

    %{
      "stage.source_f32" => source,
      "stage.offsets_f32" => offsets,
      "stage.scaled_s" => s,
      "stage.normalization" => normalization,
      "stage.u_scaled" => source,
      "stage.matmul_pre_norm" => source,
      "stage.zero_source_f32" => source,
      "stage.adapted_source_f32" => source,
      "stage.final_f32" => source,
      "stage.final_bf16" => Nx.as_type(source, :bf16)
    }
  end
end
