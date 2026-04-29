defmodule TrinityCoordinator.Sakana.LargeTensorChunksTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Sakana.LargeTensorChunks, as: LargeTensorChunksTask
  alias TrinityCoordinator.Sakana.LargeTensorChunks

  test "mix task parses explicit large-tensor chunk options" do
    opts =
      LargeTensorChunksTask.parse_args!([
        "--components-dir",
        "tmp/components",
        "--python-report",
        "tmp/python.json",
        "--chunk-rows",
        "2048",
        "--source",
        "model.embed_tokens.weight",
        "--no-cuda"
      ])

    assert opts[:components_dir] == "tmp/components"
    assert opts[:python_report] == "tmp/python.json"
    assert opts[:chunk_rows] == 2048
    assert opts[:source] == "model.embed_tokens.weight"
    assert opts[:no_cuda]
  end

  @tag :tmp_dir
  test "synthetic large tensor replay checks every row chunk", %{tmp_dir: tmp_dir} do
    components_dir = Path.join(tmp_dir, "components")
    File.mkdir_p!(components_dir)

    source_name = "model.embed_tokens.weight"
    safe_key = source_name
    u = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    s = Nx.tensor([1.0, 2.0, 3.0])
    v = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    offsets = Nx.tensor([0.0, 0.1, -0.2])
    stages = synthetic_stage_tensors(u, s, v, offsets)

    Safetensors.write!(Path.join(components_dir, "trinity_svf_components.safetensors"), %{
      "svd.U.#{safe_key}" => u,
      "svd.S.#{safe_key}" => s,
      "svd.V.#{safe_key}" => v
    })

    Safetensors.write!(Path.join(components_dir, "trinity_svf_scale_offsets.safetensors"), %{
      "svf.scale_offsets.#{safe_key}" => offsets
    })

    metadata_path = Path.join(components_dir, "trinity_svf_debug_manifest.json")

    File.write!(
      metadata_path,
      Jason.encode!(%{
        "selected_tensors" => [
          %{
            "source_name" => source_name,
            "elixir_name" => "embedder.token_embedding.kernel",
            "safe_key" => safe_key,
            "source_shape" => [4, 3],
            "component_tensors" => %{
              "u" => "svd.U.#{safe_key}",
              "s" => "svd.S.#{safe_key}",
              "v" => "svd.V.#{safe_key}"
            },
            "scale_tensor" => "svf.scale_offsets.#{safe_key}",
            "stage_tensors" =>
              Map.new(stages, fn {"stage." <> stage, _tensor} ->
                {stage, "tensor.#{safe_key}.#{stage}"}
              end)
          }
        ]
      })
    )

    stage_path = Path.join(components_dir, "trinity_svf_all_selected_stage_debug.safetensors")

    Safetensors.write!(
      stage_path,
      Map.new(stages, fn {"stage." <> stage, tensor} ->
        {"tensor.#{safe_key}.#{stage}", tensor}
      end)
    )

    python_report_path = Path.join(tmp_dir, "python_large_chunks.json")

    File.write!(
      python_report_path,
      Jason.encode!(%{
        "large_tensor_chunk_baselines" => [
          %{
            "source_name" => source_name,
            "elixir_name" => "embedder.token_embedding.kernel",
            "safe_key" => safe_key,
            "source_shape" => [4, 3],
            "component_tensors" => %{
              "u" => "svd.U.#{safe_key}",
              "s" => "svd.S.#{safe_key}",
              "v" => "svd.V.#{safe_key}"
            },
            "scale_tensor" => "svf.scale_offsets.#{safe_key}",
            "stage_tensor_file" => stage_path,
            "stage_tensors" =>
              Map.new(stages, fn {"stage." <> stage, _tensor} ->
                {stage, "tensor.#{safe_key}.#{stage}"}
              end),
            "chunks" => [
              %{"chunk_index" => 0, "row_start" => 0, "row_end" => 2, "row_count" => 2},
              %{"chunk_index" => 1, "row_start" => 2, "row_end" => 4, "row_count" => 2}
            ]
          }
        ]
      })
    )

    report =
      LargeTensorChunks.report!(
        components_dir: components_dir,
        python_report_path: python_report_path,
        chunk_rows: 2,
        require_cuda: false
      )

    assert get_in(report, ["summary", "chunk_count"]) == 2
    assert get_in(report, ["summary", "failed_required_count"]) == 0
    assert get_in(report, ["summary", "functional_parity_passed"])

    assert Enum.all?(report["large_tensor_chunk_checks"], fn chunk ->
             get_in(chunk, ["stage_debug", "functional_parity_passed"]) and
               length(chunk["checks"]) == 10
           end)
  end

  @tag :tmp_dir
  test "comparator strict stage gate accepts large tensor chunk checks", %{tmp_dir: tmp_dir} do
    python_path = Path.join(tmp_dir, "python.json")
    elixir_path = Path.join(tmp_dir, "elixir.json")

    File.write!(python_path, Jason.encode!(%{}))

    File.write!(
      elixir_path,
      Jason.encode!(%{
        "large_tensor_chunk_checks" => [
          %{
            "label" => "large_tensor_chunk_model.embed_tokens.weight_rows_0_2",
            "source_name" => "model.embed_tokens.weight",
            "chunk_index" => 0,
            "row_start" => 0,
            "row_end" => 2,
            "checks" => [
              %{
                "stage" => "stage.final_f32",
                "required_for_functional_parity" => true,
                "functional_passed" => true,
                "byte_match" => true,
                "shape_match" => true,
                "max_abs_error" => 0.0,
                "mean_abs_error" => 0.0,
                "mismatched_element_count" => 0,
                "tolerance" => %{"max_abs_error" => 0.001, "mean_abs_error" => 0.00001}
              }
            ]
          }
        ]
      })
    )

    {output, status} =
      System.cmd(
        "python3",
        [
          "priv/sakana_trinity/scripts/compare_sakana_parity_reports.py",
          python_path,
          elixir_path,
          "--strict-stage-tolerances"
        ],
        stderr_to_stdout: true
      )

    assert status == 0, output
    assert output =~ "Large tensor chunk checks against Python stage tensors"
  end

  @tag :tmp_dir
  test "comparator strict stage gate rejects failed large tensor chunk checks", %{
    tmp_dir: tmp_dir
  } do
    python_path = Path.join(tmp_dir, "python.json")
    elixir_path = Path.join(tmp_dir, "elixir.json")

    File.write!(python_path, Jason.encode!(%{}))

    File.write!(
      elixir_path,
      Jason.encode!(%{
        "large_tensor_chunk_checks" => [
          %{
            "label" => "large_tensor_chunk_model.embed_tokens.weight_rows_0_2",
            "source_name" => "model.embed_tokens.weight",
            "chunk_index" => 0,
            "row_start" => 0,
            "row_end" => 2,
            "checks" => [
              %{
                "stage" => "stage.final_f32",
                "required_for_functional_parity" => true,
                "functional_passed" => false,
                "byte_match" => false,
                "shape_match" => true,
                "max_abs_error" => 1.0,
                "mean_abs_error" => 0.5,
                "mismatched_element_count" => 1,
                "tolerance" => %{"max_abs_error" => 0.001, "mean_abs_error" => 0.00001}
              }
            ]
          }
        ]
      })
    )

    {output, status} =
      System.cmd(
        "python3",
        [
          "priv/sakana_trinity/scripts/compare_sakana_parity_reports.py",
          python_path,
          elixir_path,
          "--strict-stage-tolerances"
        ],
        stderr_to_stdout: true
      )

    assert status != 0
    assert output =~ "strict large-tensor chunk stage-tolerance comparison failed"
  end

  defp synthetic_stage_tensors(u, s, v, offsets) do
    scaled_s = Nx.multiply(s, Nx.add(offsets, 1.0))
    normalization = Nx.divide(Nx.sum(s), Nx.sum(scaled_s))
    u_scaled = Nx.multiply(u, Nx.reshape(scaled_s, {1, 3}))
    matmul_pre_norm = Nx.dot(u_scaled, Nx.transpose(v))
    adapted_source_f32 = Nx.multiply(matmul_pre_norm, normalization)
    zero_source_f32 = Nx.dot(Nx.multiply(u, Nx.reshape(s, {1, 3})), Nx.transpose(v))

    %{
      "stage.source_f32" => zero_source_f32,
      "stage.offsets_f32" => offsets,
      "stage.scaled_s" => scaled_s,
      "stage.normalization" => Nx.reshape(normalization, {1}),
      "stage.u_scaled" => u_scaled,
      "stage.matmul_pre_norm" => matmul_pre_norm,
      "stage.zero_source_f32" => zero_source_f32,
      "stage.adapted_source_f32" => adapted_source_f32,
      "stage.final_f32" => adapted_source_f32,
      "stage.final_bf16" => Nx.as_type(adapted_source_f32, :bf16)
    }
  end
end
