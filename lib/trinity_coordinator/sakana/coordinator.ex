defmodule TrinityCoordinator.Sakana.Coordinator do
  @moduledoc """
  High-level loader for the artifact-driven TRINITY coordinator.

  It returns a single struct-like map containing:

    * the Qwen model_info/tokenizer used for hidden-state extraction,
    * the standalone Axon routing-head model and params,
    * artifact manifest metadata,
    * inferred `num_agents`, `num_roles`, and hidden size.

  Provider LLM calls are not performed here.
  """

  alias TrinityCoordinator.{Extractor, Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, Head}

  @type t :: %{
          required(:model_info) => map(),
          required(:tokenizer) => map(),
          required(:routing_model) => Axon.t(),
          required(:routing_params) => struct(),
          required(:manifest) => map(),
          required(:artifact_dir) => String.t(),
          required(:num_agents) => pos_integer(),
          required(:num_roles) => pos_integer(),
          required(:hidden_size) => pos_integer()
        }

  @doc """
  Loads the Sakana-adapted Qwen backbone and routing head.

  Options:

    * `:artifact_dir` - defaults to `Artifact.default_output_dir/0`.
    * `:num_roles` - defaults to `3`.
    * `:backend` - defaults to CUDA.
    * `:require_cuda` - default `true`.
  """
  @spec load(keyword()) :: {:ok, t()} | {:error, term()}
  def load(opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        artifact_dir: Artifact.default_output_dir(),
        num_roles: 3,
        backend: {EXLA.Backend, client: :cuda},
        require_cuda: true
      )

    if opts[:require_cuda] do
      Runtime.put_cuda_backend!()
    end

    profile =
      SLMProfile.qwen_coordinator()
      |> Map.put(:adapted_artifact_dir, opts[:artifact_dir])
      |> Map.put(:artifact_patch_options,
        patch_router_head: false,
        allow_incomplete: false,
        cast_tensors: true
      )

    with {:ok, {model_info, tokenizer}} <- SLMProfile.load_profile(profile),
         {:ok, manifest} <- Artifact.load_manifest(opts[:artifact_dir]),
         head_weights <- Artifact.load_router_head!(opts[:artifact_dir], manifest: manifest),
         {:ok, head_state} <-
           Head.build_routing_state(head_weights,
             num_roles: opts[:num_roles],
             backend: opts[:backend]
           ) do
      {:ok,
       %{
         model_info: model_info,
         tokenizer: tokenizer,
         routing_model: head_state.model,
         routing_params: head_state.params,
         manifest: manifest,
         artifact_dir: opts[:artifact_dir],
         num_agents: head_state.num_agents,
         num_roles: head_state.num_roles,
         hidden_size: head_state.hidden_size
       }}
    else
      {:error, reason} -> {:error, reason}
    end
  rescue
    e ->
      {:error, {:coordinator_load_error, Exception.message(e)}}
  end

  @doc """
  Extracts a vector with the adapted model and routes it through the artifact head.
  """
  def route_messages(%{} = coordinator, messages) when is_list(messages) do
    with {:ok, extraction} <-
           Extractor.extract_penultimate_hidden_state_with_metadata(
             coordinator.model_info,
             coordinator.tokenizer,
             messages
           ) do
      route =
        TrinityCoordinator.CoordinationHead.route(
          coordinator.routing_model,
          coordinator.routing_params,
          extraction.vector,
          coordinator.num_agents,
          coordinator.num_roles
        )

      {:ok, %{extraction: extraction, route: route}}
    end
  end
end
