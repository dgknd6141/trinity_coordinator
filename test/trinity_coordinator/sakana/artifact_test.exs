defmodule TrinityCoordinator.Sakana.ArtifactTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.Sakana.Artifact

  @tmp_prefix "trinity_artifact_unit_test"

  test "writes and reads synthetic checkpoint artifacts" do
    out_dir = unique_tmp_dir()

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    path = Path.join(out_dir, "checkpoint.safetensors")
    original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)

    Safetensors.write!(path, %{"synthetic" => Nx.backend_transfer(original, Nx.BinaryBackend)})

    tensors = Safetensors.read!(path)
    restored = tensors["synthetic"]

    assert Nx.shape(restored) == Nx.shape(original)
    assert Nx.to_flat_list(restored) == Nx.to_flat_list(original)
  end

  test "patches tensors inside a tiny nested params container" do
    params = %{
      "layers" => [
        %{"kernel" => Nx.iota({2, 2}, type: :f32)},
        %{"kernel" => Nx.iota({2, 2}, type: :f32) |> Nx.multiply(10.0)}
      ],
      "attention" => %{"output" => Nx.broadcast(0.0, {2, 2})}
    }

    manifest = %{
      "artifact_version" => Artifact.manifest_version(),
      "status" => "complete",
      "export_complete" => true,
      "selected_tensors" => [
        %{
          "path" => "layers.0.kernel",
          "artifact_key" => "layers.0.kernel",
          "segments" => ["layers", 0, "kernel"]
        },
        %{
          "path" => "attention.output",
          "artifact_key" => "attention.output",
          "segments" => ["attention", "output"]
        }
      ]
    }

    tensors = %{
      "layers.0.kernel" => Nx.add(Nx.iota({2, 2}, type: :f32), 100),
      "attention.output" => Nx.broadcast(Nx.tensor(7.0, type: :f32), {2, 2})
    }

    patched = Artifact.patch_params!(params, manifest, tensors)

    assert Nx.to_flat_list(Enum.at(patched["layers"], 0)["kernel"]) == [
             100.0,
             101.0,
             102.0,
             103.0
           ]

    assert Nx.to_flat_list(patched["attention"]["output"]) == [7.0, 7.0, 7.0, 7.0]
  end

  test "raises when a patched tensor shape does not match target" do
    params = %{"head" => %{"kernel" => Nx.broadcast(0.0, {2, 2})}}

    manifest = %{
      "artifact_version" => Artifact.manifest_version(),
      "status" => "complete",
      "export_complete" => true,
      "selected_tensors" => [
        %{
          "path" => "head.kernel",
          "artifact_key" => "head.kernel",
          "segments" => ["head", "kernel"]
        }
      ]
    }

    tensors = %{
      "head.kernel" => Nx.broadcast(0.0, {3, 3})
    }

    assert_raise ArgumentError, ~r/shape mismatch/, fn ->
      Artifact.patch_params!(params, manifest, tensors)
    end
  end

  test "rejects incomplete manifest loads by default" do
    out_dir = unique_tmp_dir()

    on_exit(fn ->
      File.rm_rf(out_dir)
    end)

    manifest = %{
      "artifact_version" => Artifact.manifest_version(),
      "status" => "partial",
      "export_complete" => false,
      "selected_tensors" => []
    }

    Artifact.write_manifest!(out_dir, manifest)

    assert_raise ArgumentError, fn ->
      Artifact.load_adapted_tensors!(out_dir)
    end

    assert Artifact.trace_metadata(%{}) == %{}
  end

  test "identity comparison ignores ephemeral tensor statuses" do
    expected = %{
      "artifact_version" => 1,
      "base_model_repo" => "Qwen/Qwen3-0.6B",
      "bumblebee_module" => "Bumblebee.Text.Qwen3",
      "architecture" => "for_causal_language_modeling",
      "xla_target" => "cuda12",
      "export_backend" => "elixir_nx_exla_cuda",
      "source_vector_path" =>
        "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors",
      "source_vector_tensor" => "trinity_router_es_vector",
      "source_vector_sha256" => "abc123",
      "selected_tensors" => [
        %{
          "path" => "decoder.blocks.0.test.weight",
          "artifact_key" => "decoder.blocks.0.test.weight",
          "shape" => [4, 4],
          "singular_values" => 4,
          "type" => "{:f, 32}",
          "segments" => ["decoder", "blocks", 0, "test", "weight"],
          "status" => "pending",
          "checkpoint_path" => "checkpoints/0001_decoder.blocks.0.test.weight.safetensors",
          "checkpoint_sha256" => nil,
          "decompose_elapsed_ms" => nil,
          "reconstruct_elapsed_ms" => nil,
          "u_backend" => nil,
          "s_backend" => nil,
          "v_backend" => nil,
          "adapted_backend" => nil,
          "error" => nil
        }
      ],
      "selected_singular_value_count" => 4,
      "scale_offset_count" => 9_216,
      "router_head_shape" => [10, 1_024],
      "selected_tensor_count" => 1,
      "source_vector_shape" => [19_456]
    }

    observed =
      Map.put(expected, "selected_tensors", [
        expected["selected_tensors"]
        |> List.first()
        |> Map.put("status", "running")
        |> Map.put("error", "interrupted")
        |> Map.put("checkpoint_sha256", "different")
      ])

    assert Artifact.identity_matches?(expected, observed)

    observed_mismatch =
      Map.put(observed, "selected_tensors", [
        observed["selected_tensors"]
        |> List.first()
        |> Map.put("singular_values", 3)
      ])

    refute Artifact.identity_matches?(expected, observed_mismatch)
  end

  defp unique_tmp_dir do
    Path.join(System.tmp_dir!(), "#{@tmp_prefix}_#{System.unique_integer([:positive])}")
    |> then(fn path ->
      File.mkdir_p!(path)
      path
    end)
  end
end
