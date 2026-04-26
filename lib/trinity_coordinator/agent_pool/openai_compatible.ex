defmodule TrinityCoordinator.AgentPool.OpenAICompatible do
  @moduledoc """
  Adapter for OpenAI-compatible local endpoints (Ollama, vLLM, LM Studio, ...).
  """

  @behaviour TrinityCoordinator.AgentPool.Adapter
  alias TrinityCoordinator.AgentPool.OpenAI

  @impl true
  def call(agent_spec, messages, opts) do
    api_key = Keyword.get(opts, :openai_api_key)

    with :ok <- validate_api_key(api_key),
         :ok <- validate_base_url(agent_spec[:base_url]),
         {:ok, payload} <- build_payload(agent_spec, messages),
         {:ok, response} <-
           request(agent_spec[:base_url], payload, api_key, agent_spec[:timeout_ms]) do
      OpenAI.parse_response(response)
    end
  end

  defp validate_api_key(api_key) when is_binary(api_key) and byte_size(api_key) > 0, do: :ok
  defp validate_api_key(_), do: {:error, :missing_openai_api_key}

  defp build_payload(agent_spec, messages) do
    model = to_string(agent_spec[:model])
    max_tokens = agent_spec[:max_tokens] || 200
    temperature = agent_spec[:temperature] || 0.2
    {:ok, %{model: model, messages: messages, max_tokens: max_tokens, temperature: temperature}}
  end

  defp validate_base_url(base_url) when is_binary(base_url) and byte_size(base_url) > 0, do: :ok
  defp validate_base_url(_), do: {:error, :missing_provider_base_url}

  defp request(url, payload, api_key, timeout) do
    endpoint = Path.join(url, "chat/completions")

    case Req.post(endpoint,
           json: payload,
           headers: [
             {"authorization", "Bearer #{api_key}"},
             {"content-type", "application/json"}
           ],
           receive_timeout: timeout,
           connect_options: [timeout: timeout]
         ) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
