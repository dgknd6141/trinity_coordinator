defmodule Mix.Tasks.Trinity.Hitl.HeadRoute do
  @moduledoc """
  HITL gate: prove the Sakana router head routes a live base-Qwen hidden vector.

      XLA_TARGET=cuda12 mix trinity.hitl.head_route
  """

  use Mix.Task

  alias TrinityCoordinator.{CoordinationHead, Extractor, HITL, Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Head, SVD}

  @shortdoc "HITL live hidden-state to Sakana-head routing check"
  @vector_path "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    HITL.banner("TRINITY HITL HEAD ROUTE CHECK")
    Runtime.put_cuda_backend!()

    vector = SVD.load_router_vector!(@vector_path)
    split = SVD.split_router_vector(vector, 9_216, 1_024, 10)

    {:ok, head_state} =
      Head.build_routing_state(split.head_weights, backend: {EXLA.Backend, client: :cuda})

    {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    {:ok, meta} =
      Extractor.extract_penultimate_hidden_state_with_metadata(
        model_info,
        tokenizer,
        [%{"role" => "user", "content" => "Route this request through the Sakana head."}]
      )

    route =
      CoordinationHead.route(
        head_state.model,
        head_state.params,
        meta.vector,
        head_state.num_agents,
        head_state.num_roles
      )

    HITL.ensure_shape!(meta.vector_shape, {1, 1_024}, "Qwen penultimate vector")
    HITL.ensure_cuda_tensor!(meta.vector, "Qwen penultimate vector")
    HITL.ensure_shape!(route.logits, {1, 10}, "routing logits")
    HITL.ensure_cuda_tensor!(route.logits, "routing logits")
    HITL.kv("Agent id", route.agent_id)
    HITL.kv("Role id", route.role_id)
    HITL.kv("Role name", HITL.role_name(route.role_id))
    HITL.kv("Agent logits", HITL.short_logits(route.agent_logits))
    HITL.kv("Role logits", HITL.short_logits(route.role_logits))

    HITL.pass("TRINITY HITL HEAD ROUTE CHECK")
  end
end
