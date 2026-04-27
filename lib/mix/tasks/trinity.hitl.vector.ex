defmodule Mix.Tasks.Trinity.Hitl.Vector do
  @moduledoc """
  HITL gate: prove the Sakana router vector can be loaded and split.

      XLA_TARGET=cuda12 mix trinity.hitl.vector
  """

  use Mix.Task

  alias TrinityCoordinator.HITL
  alias TrinityCoordinator.Sakana.{Artifact, SVD}

  @shortdoc "HITL Sakana router-vector split check"
  @default_path "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")
    path = List.first(args) || @default_path

    HITL.banner("TRINITY HITL ROUTER VECTOR CHECK")
    HITL.kv("Source vector path", path)
    HITL.kv("Source vector sha256", Artifact.file_sha256!(path))

    vector = SVD.load_router_vector!(path)
    split = SVD.split_router_vector(vector, 9_216, 1_024, 10)

    HITL.ensure_shape!(vector, {19_456}, "router vector")
    HITL.ensure_shape!(split.scale_offsets, {9_216}, "scale offsets")
    HITL.ensure_shape!(split.head_weights, {10, 1_024}, "router head weights")
    HITL.kv("scale_count", split.scale_count)
    HITL.kv("head_count", split.head_count)

    HITL.pass("TRINITY HITL ROUTER VECTOR CHECK")
  end
end
