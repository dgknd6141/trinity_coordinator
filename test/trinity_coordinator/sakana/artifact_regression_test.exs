defmodule TrinityCoordinator.Sakana.ArtifactRegressionTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Sakana.Artifact

  test "patches nested tuple segments and treats manifest type as optional" do
    original = Nx.broadcast(0.0, {2, 2})
    replacement = Nx.broadcast(1.0, {2, 2})

    params = %{
      "tuple_container" => {Nx.broadcast(-1.0, {2, 2}), original}
    }

    manifest = %{
      "artifact_version" => Artifact.manifest_version(),
      "status" => "complete",
      "export_complete" => true,
      "selected_tensors" => [
        %{
          "path" => "tuple_container.1",
          "artifact_key" => "tuple_container.1",
          "segments" => ["tuple_container", 1],
          "shape" => [2, 2]
        }
      ]
    }

    patched = Artifact.patch_params!(params, manifest, %{"tuple_container.1" => replacement})

    assert Nx.to_flat_list(elem(patched["tuple_container"], 1)) == [1.0, 1.0, 1.0, 1.0]
  end
end
