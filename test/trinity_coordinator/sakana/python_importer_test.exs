defmodule TrinityCoordinator.Sakana.PythonImporterTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.Sakana.{Artifact, ExportSpec, PythonImporter}

  test "imports a synthetic Python semantic bundle into canonical artifacts" do
    source_dir = unique_tmp_dir("python_source")
    out_dir = unique_tmp_dir("python_out")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    components_path = Path.join(source_dir, "trinity_svf_components.safetensors")
    scales_path = Path.join(source_dir, "trinity_svf_scale_offsets.safetensors")
    head_path = Path.join(source_dir, "trinity_router_head.safetensors")
    manifest_path = Path.join(source_dir, "trinity_sakana_export_manifest.json")

    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)
    v = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    head = Nx.iota({4, 2}, type: :f32)

    Safetensors.write!(components_path, %{
      "svd.U.model.embed_tokens.weight" => u,
      "svd.S.model.embed_tokens.weight" => s,
      "svd.V.model.embed_tokens.weight" => v
    })

    Safetensors.write!(scales_path, %{"svf.scale_offsets.model.embed_tokens.weight" => offsets})
    Safetensors.write!(head_path, %{"trinity.router_head.linear.weight" => head})

    python_manifest = %{
      "format" => "trinity_sakana_safetensors_export",
      "components_path" => Path.basename(components_path),
      "scale_offsets_path" => Path.basename(scales_path),
      "router_head_path" => Path.basename(head_path),
      "source_vector_sha256" => "synthetic",
      "selected_tensors" => [
        %{
          "source_name" => "model.embed_tokens.weight",
          "elixir_name" => "embedder.token_embedding.kernel",
          "shape" => [2, 2],
          "singular_values" => 2,
          "offset_start" => 0,
          "offset_end" => 2
        }
      ]
    }

    File.write!(manifest_path, Jason.encode!(python_manifest))

    spec = synthetic_spec(out_dir, 2)

    assert {:ok, manifest} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               manifest: Path.basename(manifest_path),
               out_dir: out_dir,
               force: true,
               load_qwen: false,
               spec: spec
             )

    assert manifest["status"] == "complete"
    assert manifest["export_complete"] == true
    assert manifest["selected_tensor_count"] == 1
    assert manifest["router_head_shape"] == [4, 2]
    assert [entry] = manifest["selected_tensors"]
    refute entry["target_verified"]

    assert {:ok, loaded_manifest} = Artifact.load_manifest(out_dir)
    assert loaded_manifest["artifact_layout"] == Artifact.artifact_layout_checkpoint_directory()

    head_tensor = Artifact.load_router_head!(out_dir, manifest: loaded_manifest)
    assert Nx.shape(head_tensor) == {4, 2}

    tensors = Artifact.load_adapted_tensors!(out_dir, manifest: loaded_manifest)
    adapted = Map.fetch!(tensors, "embedder.token_embedding.kernel")

    assert Nx.shape(adapted) == {2, 2}
    assert_all_close(adapted, Nx.tensor([[1.0, 0.0], [0.0, 2.0]], type: :f32), atol: 1.0e-6)
  end

  test "imports the full Python semantic exporter manifest schema" do
    source_dir = unique_tmp_dir("python_source_export_schema")
    out_dir = unique_tmp_dir("python_out_export_schema")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)
    v = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    head = Nx.iota({4, 2}, type: :f32)

    Safetensors.write!(Path.join(source_dir, "trinity_svf_components.safetensors"), %{
      "svd.U.model.layers.26.mlp.gate_proj.weight" => u,
      "svd.S.model.layers.26.mlp.gate_proj.weight" => s,
      "svd.V.model.layers.26.mlp.gate_proj.weight" => v
    })

    Safetensors.write!(
      Path.join(source_dir, "trinity_svf_scale_offsets.safetensors"),
      %{"svf.scale_offsets.model.layers.26.mlp.gate_proj.weight" => offsets}
    )

    Safetensors.write!(Path.join(source_dir, "trinity_router_head.safetensors"), %{
      "trinity.router_head.linear.weight" => head
    })

    manifest_path = Path.join(source_dir, "trinity_sakana_export_manifest.json")

    File.write!(
      manifest_path,
      Jason.encode!(%{
        "format" => "trinity_sakana_safetensors_export",
        "version" => 1,
        "source" => %{
          "router_vector" => "priv/sakana_trinity/artifacts/sakana_model_iter_60.npy",
          "router_vector_sha256" => String.duplicate("a", 64)
        },
        "outputs" => %{
          "components" => "trinity_svf_components.safetensors",
          "scale_offsets" => "trinity_svf_scale_offsets.safetensors",
          "head" => "trinity_router_head.safetensors"
        },
        "routing" => %{
          "head_tensor" => "trinity.router_head.linear.weight",
          "head_shape" => [4, 2]
        },
        "svf" => %{
          "entries" => [
            %{
              "source_parameter" => "model.layers.26.mlp.gate_proj.weight",
              "safe_parameter" => "model.layers.26.mlp.gate_proj.weight",
              "scale_tensor" => "svf.scale_offsets.model.layers.26.mlp.gate_proj.weight",
              "component_tensors" => %{
                "u" => "svd.U.model.layers.26.mlp.gate_proj.weight",
                "s" => "svd.S.model.layers.26.mlp.gate_proj.weight",
                "v" => "svd.V.model.layers.26.mlp.gate_proj.weight"
              },
              "offset_start" => 0,
              "offset_end" => 2,
              "num_singular_values" => 2,
              "shape" => [2]
            }
          ]
        }
      })
    )

    spec = synthetic_spec(out_dir, 2)

    assert {:ok, manifest} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               manifest: Path.basename(manifest_path),
               out_dir: out_dir,
               force: true,
               load_qwen: false,
               spec: spec
             )

    assert manifest["status"] == "complete"
    assert manifest["source_vector_sha256"] == String.duplicate("a", 64)
    assert manifest["selected_tensor_count"] == 1
    assert manifest["selected_singular_value_count"] == 2
    assert manifest["router_head_shape"] == [4, 2]

    [entry] = manifest["selected_tensors"]
    assert entry["path"] == "decoder.blocks.26.ffn.gate.kernel"
    assert entry["shape"] == [2, 2]
    assert entry["singular_values"] == 2
    assert entry["target_verified"] == false

    tensors = Artifact.load_adapted_tensors!(out_dir, manifest: manifest)
    assert Map.has_key?(tensors, "decoder.blocks.26.ffn.gate.kernel")

    head_tensor =
      Artifact.load_router_head!(out_dir, manifest: manifest, expected_shape: [4, 2])

    assert Nx.shape(head_tensor) == {4, 2}
  end

  test "default loader rejects wrong router head shape" do
    source_dir = unique_tmp_dir("python_source_bad_head")
    out_dir = unique_tmp_dir("python_out_bad_head")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    write_minimal_export_bundle!(source_dir,
      head: Nx.iota({9, 1024}, type: :f32),
      component_tensors: %{
        "u" => "svd.U.model.layers.26.mlp.gate_proj.weight",
        "s" => "svd.S.model.layers.26.mlp.gate_proj.weight",
        "v" => "svd.V.model.layers.26.mlp.gate_proj.weight"
      }
    )

    assert {:error, {:router_head_shape_mismatch, {10, 1024}, {9, 1024}}} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               out_dir: out_dir,
               force: true,
               load_qwen: false
             )
  end

  test "component key mismatch reports the source tensor name" do
    source_dir = unique_tmp_dir("python_source_bad_component")
    out_dir = unique_tmp_dir("python_out_bad_component")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    write_minimal_export_bundle!(source_dir,
      head: Nx.iota({4, 2}, type: :f32),
      component_tensors: %{
        "u" => "svd.U.missing",
        "s" => "svd.S.model.layers.26.mlp.gate_proj.weight",
        "v" => "svd.V.model.layers.26.mlp.gate_proj.weight"
      }
    )

    assert {:error, {:reconstruct_error, message}} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               out_dir: out_dir,
               force: true,
               load_qwen: false,
               spec: synthetic_spec(out_dir, 2)
             )

    assert message =~ "model.layers.26.mlp.gate_proj.weight"
    assert message =~ "svd.U.missing"
  end

  test "defaults Python semantic V tensors to torch.svd V layout without live Qwen target" do
    source_dir = unique_tmp_dir("python_source_torch_v")
    out_dir = unique_tmp_dir("python_out_torch_v")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    components_path = Path.join(source_dir, "trinity_svf_components.safetensors")
    scales_path = Path.join(source_dir, "trinity_svf_scale_offsets.safetensors")
    head_path = Path.join(source_dir, "trinity_router_head.safetensors")
    manifest_path = Path.join(source_dir, "trinity_sakana_export_manifest.json")

    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)

    # This is a non-symmetric orthogonal V in legacy torch.svd layout.  A wrong
    # Vh/Nx interpretation flips both off-diagonal signs.
    v = Nx.tensor([[0.0, -1.0], [1.0, 0.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    head = Nx.iota({4, 2}, type: :f32)

    Safetensors.write!(components_path, %{
      "svd.U.model.synthetic.weight" => u,
      "svd.S.model.synthetic.weight" => s,
      "svd.V.model.synthetic.weight" => v
    })

    Safetensors.write!(scales_path, %{"svf.scale_offsets.model.synthetic.weight" => offsets})
    Safetensors.write!(head_path, %{"trinity.router_head.linear.weight" => head})

    File.write!(
      manifest_path,
      Jason.encode!(%{
        "format" => "trinity_sakana_safetensors_export",
        "components_path" => Path.basename(components_path),
        "scale_offsets_path" => Path.basename(scales_path),
        "router_head_path" => Path.basename(head_path),
        "source_vector_sha256" => "synthetic",
        "selected_tensors" => [
          %{
            "source_name" => "model.synthetic.weight",
            "elixir_name" => "synthetic.kernel",
            "shape" => [2, 2],
            "singular_values" => 2,
            "offset_start" => 0,
            "offset_end" => 2
          }
        ]
      })
    )

    spec = synthetic_spec(out_dir, 2)

    assert {:ok, manifest} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               manifest: Path.basename(manifest_path),
               out_dir: out_dir,
               force: true,
               load_qwen: false,
               spec: spec
             )

    [entry] = manifest["selected_tensors"]
    assert entry["python_v_layout"] == "torch_v"

    tensors = Artifact.load_adapted_tensors!(out_dir, manifest: manifest)
    adapted = Map.fetch!(tensors, "synthetic.kernel")

    expected = Nx.tensor([[0.0, 1.0], [-2.0, 0.0]], type: :f32)
    wrong_vh = Nx.tensor([[0.0, -1.0], [2.0, 0.0]], type: :f32)

    assert_all_close(adapted, expected, atol: 1.0e-6)
    refute_all_close(adapted, wrong_vh, atol: 1.0e-6)
  end

  test "transposes square Qwen layer kernels by semantic path, not only by shape" do
    source_dir = unique_tmp_dir("python_source_square_orientation")
    out_dir = unique_tmp_dir("python_out_square_orientation")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    components_path = Path.join(source_dir, "trinity_svf_components.safetensors")
    scales_path = Path.join(source_dir, "trinity_svf_scale_offsets.safetensors")
    head_path = Path.join(source_dir, "trinity_router_head.safetensors")
    manifest_path = Path.join(source_dir, "trinity_sakana_export_manifest.json")

    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)
    v = Nx.tensor([[0.0, -1.0], [1.0, 0.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    head = Nx.iota({4, 2}, type: :f32)

    Safetensors.write!(components_path, %{
      "svd.U.model.layers.26.self_attn.k_proj.weight" => u,
      "svd.S.model.layers.26.self_attn.k_proj.weight" => s,
      "svd.V.model.layers.26.self_attn.k_proj.weight" => v
    })

    Safetensors.write!(scales_path, %{
      "svf.scale_offsets.model.layers.26.self_attn.k_proj.weight" => offsets
    })

    Safetensors.write!(head_path, %{"trinity.router_head.linear.weight" => head})

    File.write!(
      manifest_path,
      Jason.encode!(%{
        "format" => "trinity_sakana_safetensors_export",
        "components_path" => Path.basename(components_path),
        "scale_offsets_path" => Path.basename(scales_path),
        "router_head_path" => Path.basename(head_path),
        "source_vector_sha256" => "synthetic",
        "selected_tensors" => [
          %{
            "source_name" => "model.layers.26.self_attn.k_proj.weight",
            "elixir_name" => "decoder.blocks.26.self_attention.key.kernel",
            "shape" => [2, 2],
            "singular_values" => 2,
            "offset_start" => 0,
            "offset_end" => 2
          }
        ]
      })
    )

    spec = synthetic_spec(out_dir, 2)

    assert {:ok, manifest} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               manifest: Path.basename(manifest_path),
               out_dir: out_dir,
               force: true,
               load_qwen: false,
               spec: spec
             )

    tensors = Artifact.load_adapted_tensors!(out_dir, manifest: manifest)
    adapted = Map.fetch!(tensors, "decoder.blocks.26.self_attention.key.kernel")

    source_oriented = Nx.tensor([[0.0, 1.0], [-2.0, 0.0]], type: :f32)
    target_oriented = Nx.transpose(source_oriented)

    assert_all_close(adapted, target_oriented, atol: 1.0e-6)
    refute_all_close(adapted, source_oriented, atol: 1.0e-6)
  end

  defp synthetic_spec(out_dir, scale_count) do
    %ExportSpec{
      name: :synthetic_python_import,
      base_model_repo: "synthetic",
      bumblebee_module: Bumblebee.Text.Gpt2,
      architecture: :base,
      hidden_size: 2,
      num_agents: 1,
      num_roles: 3,
      selected_layer_indices: [],
      scale_offset_count: scale_count,
      source_vector_tensor: "synthetic",
      router_head_tensor_key: Artifact.router_head_tensor_key(),
      source_vector_path: "synthetic",
      out_dir: out_dir,
      xla_target: "host",
      export_backend: "test"
    }
  end

  defp write_minimal_export_bundle!(source_dir, opts) do
    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)
    v = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    head = Keyword.fetch!(opts, :head)
    component_tensors = Keyword.fetch!(opts, :component_tensors)

    Safetensors.write!(Path.join(source_dir, "trinity_svf_components.safetensors"), %{
      "svd.U.model.layers.26.mlp.gate_proj.weight" => u,
      "svd.S.model.layers.26.mlp.gate_proj.weight" => s,
      "svd.V.model.layers.26.mlp.gate_proj.weight" => v
    })

    Safetensors.write!(
      Path.join(source_dir, "trinity_svf_scale_offsets.safetensors"),
      %{"svf.scale_offsets.model.layers.26.mlp.gate_proj.weight" => offsets}
    )

    Safetensors.write!(Path.join(source_dir, "trinity_router_head.safetensors"), %{
      "trinity.router_head.linear.weight" => head
    })

    File.write!(
      Path.join(source_dir, "trinity_sakana_export_manifest.json"),
      Jason.encode!(%{
        "format" => "trinity_sakana_safetensors_export",
        "outputs" => %{
          "components" => "trinity_svf_components.safetensors",
          "scale_offsets" => "trinity_svf_scale_offsets.safetensors",
          "head" => "trinity_router_head.safetensors"
        },
        "routing" => %{"head_tensor" => "trinity.router_head.linear.weight"},
        "svf" => %{
          "entries" => [
            %{
              "source_parameter" => "model.layers.26.mlp.gate_proj.weight",
              "safe_parameter" => "model.layers.26.mlp.gate_proj.weight",
              "scale_tensor" => "svf.scale_offsets.model.layers.26.mlp.gate_proj.weight",
              "component_tensors" => component_tensors,
              "offset_start" => 0,
              "offset_end" => 2,
              "num_singular_values" => 2,
              "shape" => [2]
            }
          ]
        }
      })
    )
  end

  defp assert_all_close(left, right, opts) do
    assert Nx.to_number(Nx.all_close(left, right, opts)) == 1
  end

  defp refute_all_close(left, right, opts) do
    assert Nx.to_number(Nx.all_close(left, right, opts)) == 0
  end

  defp unique_tmp_dir(prefix) do
    Path.join(System.tmp_dir!(), "#{prefix}_#{System.unique_integer([:positive])}")
    |> tap(&File.mkdir_p!/1)
  end
end
