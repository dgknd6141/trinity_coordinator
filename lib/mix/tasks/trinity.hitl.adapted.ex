defmodule Mix.Tasks.Trinity.Hitl.Adapted do
  @moduledoc """
  HITL gate: load the adapted Qwen coordinator and route a live hidden vector.

      XLA_TARGET=cuda12 mix trinity.hitl.adapted

  Requires a complete canonical artifact directory:
  `priv/sakana_trinity/adapted_qwen3_0_6b_layer26`.
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Runtime}
  alias TrinityCoordinator.Sakana.Coordinator

  @shortdoc "HITL adapted-Qwen coordinator route check"

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    HITL.banner("TRINITY HITL ADAPTED COORDINATOR CHECK")
    Runtime.put_cuda_backend!()

    {:ok, coordinator} = Coordinator.load()

    HITL.kv("Artifact dir", coordinator.artifact_dir)
    HITL.kv("Artifact status", coordinator.manifest["status"])
    HITL.kv("Selected tensor count", coordinator.manifest["selected_tensor_count"])
    HITL.kv("Hidden size", coordinator.hidden_size)
    HITL.kv("Num agents", coordinator.num_agents)
    HITL.kv("Num roles", coordinator.num_roles)

    {:ok, routed} =
      Coordinator.route_messages(coordinator, [
        %{"role" => "user", "content" => "Select a TRINITY role for this reasoning task."}
      ])

    HITL.ensure_shape!(routed.extraction.vector_shape, {1, 1_024}, "adapted Qwen vector")
    HITL.ensure_cuda_tensor!(routed.extraction.vector, "adapted Qwen vector")
    HITL.ensure_shape!(routed.route.logits, {1, 10}, "adapted route logits")
    HITL.ensure_cuda_tensor!(routed.route.logits, "adapted route logits")
    HITL.kv("Agent id", routed.route.agent_id)
    HITL.kv("Role id", routed.route.role_id)
    HITL.kv("Role name", HITL.role_name(routed.route.role_id))

    HITL.pass("TRINITY HITL ADAPTED COORDINATOR CHECK")
  end
end
