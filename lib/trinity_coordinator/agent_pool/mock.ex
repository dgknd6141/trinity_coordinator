defmodule TrinityCoordinator.AgentPool.Mock do
  @moduledoc """
  Deterministic in-process provider adapter.

  This adapter exists for implementation gates and tests where the coordinator
  must exercise routing, role injection, state update, and trace persistence
  without making external LLM calls.

  Supported options:

    * `:mock_response` - fixed response string.
    * `:mock_responses` - map keyed by agent id, model, provider, or `:default`.
    * `:mock_agent_fn` - function called as `(agent_spec, messages, opts)`.
  """

  @behaviour TrinityCoordinator.AgentPool.Adapter

  @impl true
  def call(agent_spec, messages, opts) when is_map(agent_spec) and is_list(messages) do
    cond do
      is_function(opts[:mock_agent_fn], 3) ->
        normalize_response(opts[:mock_agent_fn].(agent_spec, messages, opts))

      is_binary(opts[:mock_response]) ->
        {:ok, opts[:mock_response]}

      is_map(opts[:mock_responses]) ->
        response_from_map(agent_spec, opts[:mock_responses])

      true ->
        {:ok, default_response(agent_spec, messages)}
    end
  end

  def call(_agent_spec, _messages, _opts), do: {:error, :invalid_mock_provider_call}

  defp response_from_map(agent_spec, responses) do
    keys = [
      Map.get(agent_spec, :id),
      Map.get(agent_spec, "id"),
      Map.get(agent_spec, :name),
      Map.get(agent_spec, "name"),
      Map.get(agent_spec, :model),
      Map.get(agent_spec, "model"),
      :default,
      "default"
    ]

    case Enum.find_value(keys, &Map.get(responses, &1)) do
      response when is_binary(response) -> {:ok, response}
      nil -> {:ok, default_response(agent_spec, [])}
      other -> normalize_response(other)
    end
  end

  defp normalize_response({:ok, response}) when is_binary(response), do: {:ok, response}
  defp normalize_response({:error, reason}), do: {:error, reason}
  defp normalize_response(response) when is_binary(response), do: {:ok, response}
  defp normalize_response(other), do: {:error, {:invalid_mock_response, other}}

  defp default_response(agent_spec, messages) do
    model = Map.get(agent_spec, :model, Map.get(agent_spec, "model", "mock"))
    id = Map.get(agent_spec, :id, Map.get(agent_spec, "id", "?"))

    last_user =
      messages
      |> Enum.reverse()
      |> Enum.find_value(fn message ->
        role = Map.get(message, :role, Map.get(message, "role"))
        content = Map.get(message, :content, Map.get(message, "content"))

        if role == "user", do: content, else: nil
      end)

    prompt =
      case last_user do
        nil -> "no user message"
        text -> text
      end

    "MOCK agent=#{id} model=#{model}: #{prompt}"
  end
end
