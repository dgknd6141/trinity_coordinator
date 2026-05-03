defmodule TrinityCoordinator.AgentPool.Inference do
  @moduledoc """
  Generic `:inference` consumer adapter for `TrinityCoordinator.AgentPool`.

  Provider-specific transport, SDK, and CLI-agent behavior lives in the shared
  `:inference` package. Trinity only maps selected agent specs into
  `Inference.Client` and `Inference.Request` data, then returns the response
  text expected by the existing orchestrator.
  """

  @behaviour TrinityCoordinator.AgentPool.Adapter

  alias Inference.{Client, Error, Response}

  @hosted_providers [:openai, :gemini, :anthropic, :openai_compatible]
  @asm_providers [:asm, :agent_session_manager]

  @impl true
  def call(agent_spec, messages, opts) when is_map(agent_spec) and is_list(messages) do
    with :ok <- validate_credentials(agent_spec, opts),
         {:ok, client} <- build_client(agent_spec, opts),
         {:ok, request_opts} <- build_request_opts(agent_spec, opts),
         {:ok, %Response{} = response} <- Inference.complete(client, messages, request_opts) do
      {:ok, Response.text(response)}
    else
      {:error, reason} -> {:error, normalize_error(reason)}
    end
  end

  def call(_agent_spec, _messages, _opts), do: {:error, :invalid_inference_provider_call}

  defp build_client(agent_spec, opts) do
    provider = provider(agent_spec)
    adapter = inference_adapter(provider, agent_spec, opts)
    inference_provider = inference_provider(provider, agent_spec, opts)

    Client.new(
      adapter: adapter,
      provider: inference_provider,
      model: model(agent_spec),
      backend: backend(provider, adapter),
      defaults: client_defaults(agent_spec, opts),
      metadata: client_metadata(agent_spec, opts),
      adapter_opts: adapter_opts(provider, agent_spec, opts)
    )
  end

  defp build_request_opts(agent_spec, opts) do
    {:ok,
     [
       model: model(agent_spec),
       temperature:
         number_field(agent_spec, :temperature, Keyword.get(opts, :inference_temperature)),
       max_tokens:
         integer_field(agent_spec, :max_tokens, Keyword.get(opts, :inference_max_tokens)),
       metadata: request_metadata(agent_spec, opts),
       session: Keyword.get(opts, :inference_session),
       options: request_options(agent_spec, opts)
     ]
     |> Enum.reject(fn {_key, value} -> is_nil(value) end)}
  end

  defp inference_adapter(provider, agent_spec, opts) do
    Keyword.get(opts, :inference_adapter) ||
      metadata(agent_spec)[:inference_adapter] ||
      default_adapter(provider)
  end

  defp default_adapter(:gemini_ex), do: Inference.Adapters.GeminiEx
  defp default_adapter(provider) when provider in @asm_providers, do: Inference.Adapters.ASM
  defp default_adapter(:mock), do: Inference.Adapters.Mock
  defp default_adapter(_provider), do: Inference.Adapters.ReqLLM

  defp inference_provider(provider, agent_spec, opts) do
    Keyword.get(opts, :inference_provider) ||
      metadata(agent_spec)[:inference_provider] ||
      default_inference_provider(provider)
  end

  defp default_inference_provider(:openai_compatible), do: :openai
  defp default_inference_provider(:gemini_ex), do: :gemini
  defp default_inference_provider(provider) when provider in @asm_providers, do: provider
  defp default_inference_provider(provider), do: provider

  defp client_defaults(agent_spec, opts) do
    [
      max_tokens:
        integer_field(agent_spec, :max_tokens, Keyword.get(opts, :inference_max_tokens)),
      temperature:
        number_field(agent_spec, :temperature, Keyword.get(opts, :inference_temperature)),
      timeout: integer_field(agent_spec, :timeout_ms, Keyword.get(opts, :inference_timeout_ms))
    ]
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp adapter_opts(provider, agent_spec, opts) do
    opts
    |> Keyword.get(:inference_adapter_opts, [])
    |> Keyword.merge(metadata(agent_spec)[:inference_adapter_opts] || [])
    |> maybe_put(:api_key, api_key(provider, opts))
    |> maybe_put(:env, Keyword.get(opts, :inference_env))
    |> maybe_put(:session, Keyword.get(opts, :inference_session))
    |> maybe_put(:model_spec, model_spec(provider, agent_spec, opts))
  end

  defp request_options(agent_spec, opts) do
    [
      api_key: api_key(provider(agent_spec), opts),
      timeout: integer_field(agent_spec, :timeout_ms, Keyword.get(opts, :inference_timeout_ms)),
      base_url: string_field(agent_spec, :base_url, Keyword.get(opts, :inference_base_url))
    ]
    |> Keyword.merge(Keyword.get(opts, :inference_options, []))
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp model_spec(provider, agent_spec, opts) do
    Keyword.get(opts, :inference_model_spec) ||
      metadata(agent_spec)[:inference_model_spec] ||
      default_model_spec(provider, model(agent_spec))
  end

  defp default_model_spec(provider, model) when is_binary(model) do
    case provider do
      :gemini -> "google:" <> model
      :gemini_ex -> nil
      provider when provider in @asm_providers -> nil
      _provider -> nil
    end
  end

  defp default_model_spec(_provider, _model), do: nil

  defp validate_credentials(agent_spec, opts) do
    provider = provider(agent_spec)

    cond do
      provider in @hosted_providers and is_nil(api_key(provider, opts)) ->
        {:error, missing_key_error(provider)}

      provider == :gemini_ex and is_nil(api_key(:gemini, opts)) ->
        {:error, :missing_gemini_api_key}

      true ->
        :ok
    end
  end

  defp api_key(provider, opts) do
    case provider do
      :openai -> first_present(opts, [:api_key, :openai_api_key])
      :openai_compatible -> first_present(opts, [:api_key, :openai_api_key])
      :gemini -> first_present(opts, [:api_key, :gemini_api_key, :google_api_key])
      :gemini_ex -> first_present(opts, [:api_key, :gemini_api_key, :google_api_key])
      :anthropic -> first_present(opts, [:api_key, :anthropic_api_key])
      _other -> Keyword.get(opts, :api_key)
    end
  end

  defp first_present(opts, keys) do
    Enum.find_value(keys, fn key ->
      case Keyword.get(opts, key) do
        value when is_binary(value) and value != "" -> value
        _other -> nil
      end
    end)
  end

  defp missing_key_error(:openai), do: :missing_openai_api_key
  defp missing_key_error(:openai_compatible), do: :missing_openai_api_key
  defp missing_key_error(:gemini), do: :missing_gemini_api_key
  defp missing_key_error(:anthropic), do: :missing_anthropic_api_key
  defp missing_key_error(provider), do: {:missing_provider_api_key, provider}

  defp normalize_error(%Error{} = error),
    do: {:inference_error, error.category, error.reason, error.message}

  defp normalize_error(reason), do: reason

  defp client_metadata(agent_spec, opts) do
    agent_spec
    |> metadata()
    |> Map.merge(%{
      agent_id: field(agent_spec, :id),
      agent_name: field(agent_spec, :name),
      provider_adapter: __MODULE__,
      provider: provider(agent_spec)
    })
    |> Map.merge(Keyword.get(opts, :inference_metadata, %{}))
  end

  defp request_metadata(agent_spec, opts) do
    %{
      agent_id: field(agent_spec, :id),
      agent_name: field(agent_spec, :name),
      provider: provider(agent_spec)
    }
    |> Map.merge(Keyword.get(opts, :inference_request_metadata, %{}))
  end

  defp backend(provider, adapter) do
    cond do
      adapter == Inference.Adapters.Mock -> :mock
      provider in @asm_providers -> :agent_session_manager
      true -> provider
    end
  end

  defp provider(agent_spec), do: field(agent_spec, :provider)
  defp model(agent_spec), do: field(agent_spec, :model)

  defp metadata(agent_spec) do
    case field(agent_spec, :metadata) do
      metadata when is_map(metadata) -> metadata
      _other -> %{}
    end
  end

  defp field(map, key), do: Map.get(map, key, Map.get(map, to_string(key)))

  defp string_field(map, key, fallback) do
    case field(map, key) || fallback do
      value when is_binary(value) and value != "" -> value
      _other -> nil
    end
  end

  defp integer_field(map, key, fallback) do
    case field(map, key) || fallback do
      value when is_integer(value) and value > 0 -> value
      _other -> nil
    end
  end

  defp number_field(map, key, fallback) do
    case field(map, key) || fallback do
      value when is_number(value) and value >= 0 -> value
      _other -> nil
    end
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
