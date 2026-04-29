defmodule TrinityCoordinator.AgentPool do
  @moduledoc """
  Provider dispatch for selected coordinator agents.

  Agents are selected via a configurable pool of provider specs. Existing tests and
  callers that still pass `:agents` continue to work as a compatibility path.
  """

  @legacy_agents %{
    0 => %{provider: :openai, model: "gpt-4o-mini"},
    1 => %{provider: :openai, model: "gpt-4o-mini"},
    2 => %{provider: :openai, model: "gpt-4o-mini"},
    3 => %{provider: :openai, model: "gpt-4o-mini"},
    4 => %{provider: :openai, model: "gpt-4o-mini"},
    5 => %{provider: :openai, model: "gpt-4o-mini"},
    6 => %{provider: :openai, model: "gpt-4o-mini"}
  }

  alias TrinityCoordinator.AgentPool.{Inference, Mock}
  alias TrinityCoordinator.ProviderPool

  @doc """
  Routes the message list to the mapped provider for the selected agent.
  """
  def call_agent(agent_id, messages, opts \\ []) do
    with {:ok, messages} <- normalize_messages(messages),
         {:ok, spec} <- fetch_agent_spec(agent_id, opts),
         {:ok, adapter} <- adapter_for(spec, opts),
         {:ok, response} <- call_adapter(adapter, spec, messages, opts) do
      {:ok, response}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  def call_agent_with_spec(spec, messages, opts \\ [])

  def call_agent_with_spec(%TrinityCoordinator.ProviderPool.Spec{} = spec, messages, opts)
      when is_list(opts) do
    call_agent_with_spec(Map.from_struct(spec), messages, opts)
  end

  def call_agent_with_spec(spec, messages, opts) when is_map(spec) and is_list(opts) do
    with {:ok, messages} <- normalize_messages(messages),
         {:ok, adapter} <- adapter_for(spec, opts),
         {:ok, response} <- call_adapter(adapter, spec, messages, opts) do
      {:ok, response}
    else
      {:error, reason} -> {:error, reason}
      _ -> {:error, :provider_dispatch_failed}
    end
  end

  @doc "Returns the default provider-pool specification as a map keyed by id."
  def agent_specs, do: @legacy_agents

  @doc "Returns the number of providers in the configured default, explicit list, or named pool."
  def agent_count, do: map_size(@legacy_agents)

  def agent_count(pool) when is_list(pool), do: ProviderPool.size(pool)

  def agent_count(pool_name) when is_atom(pool_name) or is_binary(pool_name) do
    TrinityCoordinator.ProviderPool.size(pool_name)
  end

  @doc "Returns normalized provider specs for an explicit pool."
  def fetch_pool(opts \\ []) do
    resolve_pool(opts)
  end

  @doc "Returns a provider spec for an agent in the selected pool."
  def fetch_agent_spec(agent_id, opts) when is_integer(agent_id) do
    with {:ok, pool} <- resolve_pool(opts, :maybe),
         {:ok, spec} <- fetch_from_pool(agent_id, pool) do
      {:ok, normalize_spec_fields(spec)}
    else
      {:error, :fallback_legacy} ->
        case @legacy_agents[agent_id] do
          nil -> {:error, {:unknown_agent, agent_id}}
          spec -> {:ok, Map.put(spec, :id, agent_id)}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp resolve_pool(opts) do
    resolve_pool(opts, :strict)
  end

  defp resolve_pool(opts, mode) do
    with {:ok, pool} <- find_explicit_pool(opts) do
      resolve_explicit_pool(pool, mode)
    end
  end

  defp resolve_explicit_pool(:default, mode), do: resolve_default_pool(mode)

  defp resolve_explicit_pool(:fallback_legacy, :maybe), do: {:error, :fallback_legacy}

  defp resolve_explicit_pool(:fallback_legacy, _mode), do: {:ok, :default}

  defp resolve_explicit_pool(pool, _mode) when is_list(pool), do: {:ok, pool}

  defp resolve_default_pool(:maybe), do: {:error, :fallback_legacy}

  defp resolve_default_pool(_mode), do: {:ok, ProviderPool.fetch!(:default)}

  defp find_explicit_pool(opts) do
    cond do
      Keyword.has_key?(opts, :agents) ->
        {:ok, :fallback_legacy}

      Keyword.has_key?(opts, :provider_pool) ->
        parse_provider_pool(Keyword.get(opts, :provider_pool))

      Keyword.has_key?(opts, :provider_pool_name) ->
        {:ok, ProviderPool.fetch!(Keyword.get(opts, :provider_pool_name))}

      true ->
        {:ok, :default}
    end
  end

  defp parse_provider_pool(pool) when is_list(pool), do: {:ok, pool}

  defp parse_provider_pool(pool) when is_atom(pool) or is_binary(pool),
    do: {:ok, ProviderPool.fetch!(pool)}

  defp parse_provider_pool(pool), do: {:error, {:invalid_provider_pool, pool}}

  defp fetch_from_pool(agent_id, pool) when is_list(pool) do
    spec = ProviderPool.spec_for_agent(pool, agent_id)

    if is_nil(spec) do
      {:error, {:unknown_agent, agent_id}}
    else
      {:ok, spec}
    end
  end

  defp fetch_from_pool(_agent_id, _), do: {:error, :invalid_provider_pool}

  defp adapter_for(spec, opts) when is_map(spec) do
    adapter_for(spec[:provider] || spec["provider"], opts)
  end

  defp adapter_for(provider, opts) do
    case Keyword.get(opts, :adapter) do
      nil -> adapter_from_provider(provider)
      adapter -> {:ok, adapter}
    end
  end

  defp adapter_from_provider(:openai), do: {:ok, Inference}
  defp adapter_from_provider("openai"), do: {:ok, Inference}

  defp adapter_from_provider(:openai_compatible), do: {:ok, Inference}
  defp adapter_from_provider("openai_compatible"), do: {:ok, Inference}

  defp adapter_from_provider(:gemini), do: {:ok, Inference}
  defp adapter_from_provider("gemini"), do: {:ok, Inference}

  defp adapter_from_provider(:gemini_ex), do: {:ok, Inference}
  defp adapter_from_provider("gemini_ex"), do: {:ok, Inference}

  defp adapter_from_provider(:anthropic), do: {:ok, Inference}
  defp adapter_from_provider("anthropic"), do: {:ok, Inference}

  defp adapter_from_provider(:asm), do: {:ok, Inference}
  defp adapter_from_provider("asm"), do: {:ok, Inference}

  defp adapter_from_provider(:agent_session_manager), do: {:ok, Inference}
  defp adapter_from_provider("agent_session_manager"), do: {:ok, Inference}

  defp adapter_from_provider(:mock), do: {:ok, Mock}
  defp adapter_from_provider("mock"), do: {:ok, Mock}

  defp adapter_from_provider(_), do: {:error, :unsupported_provider}

  defp call_adapter(adapter, spec, messages, opts) do
    adapter_opts =
      [
        openai_api_key: Keyword.get(opts, :openai_api_key),
        openai_base_url: spec[:base_url] || Keyword.get(opts, :openai_base_url),
        openai_timeout_ms: spec[:timeout_ms] || Keyword.get(opts, :openai_timeout_ms),
        openai_max_tokens: spec[:max_tokens] || Keyword.get(opts, :openai_max_tokens),
        openai_temperature: spec[:temperature] || Keyword.get(opts, :openai_temperature),
        mock_agent_fn: Keyword.get(opts, :mock_agent_fn),
        mock_response: Keyword.get(opts, :mock_response),
        mock_responses: Keyword.get(opts, :mock_responses)
      ]
      |> Keyword.merge(opts)
      |> Keyword.reject(fn {_k, v} -> is_nil(v) end)

    adapter.call(spec, messages, adapter_opts)
  end

  defp normalize_spec_fields(%TrinityCoordinator.ProviderPool.Spec{} = spec),
    do: Map.from_struct(spec)

  defp normalize_spec_fields(spec) when is_map(spec), do: spec

  defp normalize_messages(messages) when is_list(messages) do
    normalized =
      Enum.map(messages, fn message ->
        role = Map.get(message, :role, Map.get(message, "role"))
        content = Map.get(message, :content, Map.get(message, "content"))

        if is_binary(role) and is_binary(content) do
          %{role: role, content: content}
        else
          {:error, {:invalid_message, message}}
        end
      end)

    case Enum.find(normalized, &match?({:error, _}, &1)) do
      nil -> {:ok, normalized}
      {:error, reason} -> {:error, reason}
    end
  end

  defp normalize_messages(_), do: {:error, :invalid_messages}
end
