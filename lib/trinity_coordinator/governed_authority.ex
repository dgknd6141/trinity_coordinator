defmodule TrinityCoordinator.GovernedAuthority do
  @moduledoc """
  Explicit authority materialization for governed coordinator runs.

  Standalone callers can keep using configured provider pools and local
  provider defaults. Governed callers must provide this packet and cannot pair
  it with direct provider-pool or credential options.
  """

  alias TrinityCoordinator.ProviderPool.Spec

  @enforce_keys [
    :authority_ref,
    :workflow_ref,
    :runtime_ref,
    :provider_pool_ref,
    :credential_ref,
    :provider_pool,
    :agent_pool_opts,
    :redaction_values
  ]
  defstruct [
    :authority_ref,
    :workflow_ref,
    :runtime_ref,
    :provider_pool_ref,
    :credential_ref,
    :provider_pool,
    :agent_pool_opts,
    :redaction_values
  ]

  @direct_top_fields [
    :agents,
    :mock_agent_fn,
    :provider_pool,
    :provider_pool_name
  ]
  @required_refs [
    :authority_ref,
    :workflow_ref,
    :runtime_ref,
    :provider_pool_ref,
    :credential_ref
  ]

  @type t :: %__MODULE__{
          authority_ref: String.t(),
          workflow_ref: String.t(),
          runtime_ref: String.t(),
          provider_pool_ref: String.t(),
          credential_ref: String.t(),
          provider_pool: [Spec.t()],
          agent_pool_opts: keyword(),
          redaction_values: [String.t()]
        }

  @doc "Materializes orchestrator options when `:governed_authority` is present."
  def materialize_orchestrator_opts(opts) when is_list(opts) do
    case Keyword.fetch(opts, :governed_authority) do
      :error ->
        {:ok, opts}

      {:ok, authority_input} ->
        with :ok <- reject_direct_fields(opts),
             {:ok, authority} <- new(authority_input) do
          {:ok, merge_authority_opts(opts, authority)}
        end
    end
  end

  def materialize_orchestrator_opts(_opts), do: {:error, :invalid_orchestrator_opts}

  @doc "Builds a governed authority from a keyword list or map."
  def new(input) when is_list(input) or is_map(input) do
    with {:ok, refs} <- refs(input),
         {:ok, provider_pool} <- provider_pool(input),
         {:ok, agent_pool_opts} <- agent_pool_opts(input, refs),
         redaction_values <- redaction_values(input, agent_pool_opts) do
      {:ok,
       %__MODULE__{
         authority_ref: refs.authority_ref,
         workflow_ref: refs.workflow_ref,
         runtime_ref: refs.runtime_ref,
         provider_pool_ref: refs.provider_pool_ref,
         credential_ref: refs.credential_ref,
         provider_pool: provider_pool,
         agent_pool_opts: agent_pool_opts,
         redaction_values: redaction_values
       }}
    end
  end

  def new(_input), do: {:error, :invalid_governed_authority}

  defp reject_direct_fields(opts) do
    case direct_top_fields(opts) ++ direct_agent_fields(opts) do
      [] -> :ok
      fields -> {:error, {:governed_direct_fields_rejected, fields}}
    end
  end

  defp direct_top_fields(opts) do
    Enum.filter(@direct_top_fields, &direct_top_field?(opts, &1))
  end

  defp direct_top_field?(opts, field) do
    case Keyword.fetch(opts, field) do
      {:ok, nil} -> false
      {:ok, []} -> false
      {:ok, _value} -> true
      :error -> false
    end
  end

  defp direct_agent_fields(opts) do
    case Keyword.get(opts, :agent_pool_opts, []) do
      value when value in [nil, []] -> []
      _value -> [:agent_pool_opts]
    end
  end

  defp merge_authority_opts(opts, authority) do
    trace_opts =
      opts
      |> Keyword.get(:trace, [])
      |> merge_trace_redaction(authority.redaction_values)

    opts
    |> Keyword.drop([:governed_authority, :provider_pool, :provider_pool_name, :agent_pool_opts])
    |> Keyword.put(:provider_pool, authority.provider_pool)
    |> Keyword.put(:agent_pool_opts, authority.agent_pool_opts)
    |> Keyword.put(:trace, trace_opts)
    |> Keyword.put(:governed_authority_ref, authority.authority_ref)
    |> Keyword.put(:governed_workflow_ref, authority.workflow_ref)
    |> Keyword.put(:governed_runtime_ref, authority.runtime_ref)
    |> Keyword.put(:governed_provider_pool_ref, authority.provider_pool_ref)
  end

  defp merge_trace_redaction(trace_opts, values) when is_list(trace_opts) do
    existing = Keyword.get(trace_opts, :redaction_values, [])

    trace_opts
    |> Keyword.put(:redaction_values, Enum.uniq(normalize_values(existing) ++ values))
  end

  defp merge_trace_redaction(_trace_opts, values), do: [redaction_values: values]

  defp refs(input) do
    Enum.reduce_while(@required_refs, {:ok, %{}}, fn field, {:ok, acc} ->
      case string_field(input, field) do
        {:ok, value} -> {:cont, {:ok, Map.put(acc, field, value)}}
        {:error, _reason} -> {:halt, {:error, {:missing_governed_ref, field}}}
      end
    end)
  end

  defp provider_pool(input) do
    case field(input, :provider_pool) do
      nil ->
        {:error, :governed_provider_pool_required}

      raw_pool ->
        with {:ok, pool} <- Spec.normalize(raw_pool),
             :ok <- Spec.validate(pool) do
          {:ok, pool}
        end
    end
  end

  defp agent_pool_opts(input, refs) do
    opts =
      []
      |> maybe_put(:api_key, string_value(input, :api_key))
      |> maybe_put(:inference_base_url, string_value(input, :base_url))
      |> Keyword.put(:credential_ref, refs.credential_ref)
      |> Keyword.put(:governed_authority_ref, refs.authority_ref)
      |> Keyword.put(:inference_metadata, %{
        authority_ref: refs.authority_ref,
        credential_ref: refs.credential_ref,
        provider_pool_ref: refs.provider_pool_ref,
        runtime_ref: refs.runtime_ref,
        workflow_ref: refs.workflow_ref
      })
      |> Keyword.put(:inference_request_metadata, %{
        authority_ref: refs.authority_ref,
        credential_ref: refs.credential_ref,
        provider_pool_ref: refs.provider_pool_ref
      })

    {:ok, opts}
  end

  defp redaction_values(input, agent_pool_opts) do
    input
    |> field(:redaction_values)
    |> normalize_values()
    |> Kernel.++(normalize_values([Keyword.get(agent_pool_opts, :api_key)]))
    |> Kernel.++(normalize_values([Keyword.get(agent_pool_opts, :inference_base_url)]))
    |> Enum.uniq()
  end

  defp string_field(input, field_name) do
    case string_value(input, field_name) do
      value when is_binary(value) -> {:ok, value}
      nil -> {:error, field_name}
    end
  end

  defp string_value(input, field_name) do
    case field(input, field_name) do
      value when is_binary(value) ->
        value = String.trim(value)
        if value == "", do: nil, else: value

      _other ->
        nil
    end
  end

  defp field(input, field_name) when is_map(input) do
    Map.get(input, field_name, Map.get(input, Atom.to_string(field_name)))
  end

  defp field(input, field_name) when is_list(input) do
    case Keyword.fetch(input, field_name) do
      {:ok, value} -> value
      :error -> list_value(input, Atom.to_string(field_name))
    end
  end

  defp list_value(input, field_name) do
    Enum.find_value(input, fn
      {^field_name, value} -> value
      _other -> nil
    end)
  end

  defp normalize_values(values) when is_list(values) do
    values
    |> Enum.filter(&(is_binary(&1) and &1 != ""))
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end

  defp normalize_values(value) when is_binary(value), do: normalize_values([value])
  defp normalize_values(_), do: []

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
