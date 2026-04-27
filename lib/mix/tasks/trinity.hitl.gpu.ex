defmodule Mix.Tasks.Trinity.Hitl.Gpu do
  @moduledoc """
  HITL gate: prove EXLA can see CUDA and allocate a CUDA tensor.

      XLA_TARGET=cuda12 mix trinity.hitl.gpu
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Runtime}

  @shortdoc "HITL GPU/EXLA CUDA visibility check"

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    HITL.banner("TRINITY HITL GPU CHECK")
    HITL.kv("XLA_TARGET", System.get_env("XLA_TARGET", "(unset)"))

    platforms = HITL.require_cuda!()
    HITL.kv("CUDA platform", Map.get(platforms, :cuda))

    Runtime.put_cuda_backend!()

    tensor =
      Nx.iota({8, 8}, type: :f32)
      |> Nx.dot(Nx.iota({8, 8}, type: :f32))

    HITL.ensure_shape!(tensor, {8, 8}, "CUDA smoke tensor")
    HITL.ensure_cuda_tensor!(tensor, "CUDA smoke tensor")
    HITL.pass("TRINITY HITL GPU CHECK")
  end
end
