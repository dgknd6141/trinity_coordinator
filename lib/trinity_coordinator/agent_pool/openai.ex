defmodule TrinityCoordinator.AgentPool.OpenAI do
  @moduledoc """
  OpenAI-compatible provider adapter used by the agent pool.
  """

  @behaviour TrinityCoordinator.AgentPool.Adapter

  @default_base_url "https://api.openai.com/v1"

  @impl true
  def call(agent_spec, messages, opts) do
    api_key = Keyword.get(opts, :openai_api_key)
    base_url = Keyword.get(opts, :openai_base_url, @default_base_url)

    timeout = Keyword.get(opts, :openai_timeout_ms, 30_000)

    max_tokens = Keyword.get(opts, :openai_max_tokens, agent_spec[:max_tokens] || 200)
    temperature = Keyword.get(opts, :openai_temperature, agent_spec[:temperature] || 0.2)

    with :ok <- validate_api_key(api_key),
         {:ok, payload} <- build_payload(agent_spec[:model], messages, max_tokens, temperature),
         {:ok, response} <- request(base_url, payload, api_key, timeout) do
      parse_response(response)
    end
  end

  defp validate_api_key(api_key) when is_binary(api_key) and byte_size(api_key) > 0, do: :ok
  defp validate_api_key(_), do: {:error, :missing_openai_api_key}

  defp build_payload(model, messages, max_tokens, temperature) when is_binary(model) do
    with :ok <- validate_positive_integer(max_tokens),
         :ok <- validate_positive_number(temperature) do
      {:ok, %{model: model, messages: messages, max_tokens: max_tokens, temperature: temperature}}
    end
  end

  defp build_payload(_, _, _, _), do: {:error, :invalid_model}

  defp request(base_url, payload, api_key, timeout) do
    url = Path.join(base_url, "chat/completions")

    case Req.post(url,
           json: payload,
           headers: [{"authorization", "Bearer #{api_key}"}, {"content-type", "application/json"}],
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

  def parse_response(%{"choices" => [%{"message" => %{"content" => response}} | _]})
      when is_binary(response) do
    {:ok, response}
  end

  def parse_response(%{"choices" => [%{"text" => response} | _]}) when is_binary(response) do
    {:ok, response}
  end

  def parse_response(_), do: {:error, :invalid_provider_response}

  defp validate_positive_integer(value) when is_integer(value) and value > 0, do: :ok
  defp validate_positive_integer(_), do: {:error, :invalid_payload}

  defp validate_positive_number(value) when is_number(value) and value >= 0, do: :ok
  defp validate_positive_number(_), do: {:error, :invalid_payload}
end
