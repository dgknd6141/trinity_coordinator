defmodule Mix.Tasks.Trinity.Hitl.BaseQwen do
  @moduledoc """
  HITL gate: prove Qwen3-0.6B is loaded and served from Elixir on CUDA.

      XLA_TARGET=cuda12 mix trinity.hitl.base_qwen
  """

  use Mix.Task

  alias TrinityCoordinator.{Extractor, HITL, Runtime, SLMProfile}

  @shortdoc "HITL base Qwen CUDA hidden-state check"

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    HITL.banner("TRINITY HITL BASE QWEN CHECK")
    Runtime.put_cuda_backend!()

    {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    HITL.kv("Profile", :qwen_coordinator)
    HITL.kv("Model hidden size", model_info.spec.hidden_size)

    {:ok, meta} =
      Extractor.extract_penultimate_hidden_state_with_metadata(
        model_info,
        tokenizer,
        [%{"role" => "user", "content" => "Hello TRINITY. Prove the router can see me."}]
      )

    HITL.kv("Tokenizer input shapes", meta.input_shapes)
    HITL.kv("Hidden state shape", meta.hidden_state_shape)
    HITL.kv("Vector shape", meta.vector_shape)

    HITL.ensure_shape!(meta.vector_shape, {1, 1_024}, "Qwen penultimate vector")
    HITL.ensure_cuda_tensor!(meta.vector, "Qwen penultimate vector")
    HITL.pass("TRINITY HITL BASE QWEN CHECK")
  end
end
