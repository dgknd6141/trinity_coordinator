defmodule TrinityCoordinator.ProviderPool do
  @moduledoc """
  Provider pool configuration and agent-spec resolution.

  Pools are configured via application env:

      config :trinity_coordinator,
        provider_pools: [
          default: [
            [id: 0, name: :fast_openai, provider: :openai, model: "gpt-4o-mini"],
            ...
          ],
          mock: [
            [id: 0, name: :mock_0, provider: :mock, model: "mock-agent-0"],
            ...
          ]
        ]
  """

  alias TrinityCoordinator.ProviderPool.Spec

  @type pool_name :: atom() | String.t()
  @type normalized_pool :: [Spec.t()]

  @default_pools %{
    default: [
      [id: 0, name: :fast_openai, provider: :openai, model: "gpt-4o-mini"],
      [id: 1, name: :default_reasoning, provider: :openai, model: "gpt-4o-mini"],
      [id: 2, name: :compact_reasoning, provider: :openai, model: "gpt-4o-mini"],
      [id: 3, name: :backup_openai, provider: :openai, model: "gpt-4o-mini"],
      [id: 4, name: :fast_openai_2, provider: :openai, model: "gpt-4o-mini"],
      [id: 5, name: :reasoner_2, provider: :openai, model: "gpt-4o-mini"],
      [id: 6, name: :fallback_openai, provider: :openai, model: "gpt-4o-mini"]
    ],
    mock: [
      [id: 0, name: :mock_0, provider: :mock, model: "mock-agent-0"],
      [id: 1, name: :mock_1, provider: :mock, model: "mock-agent-1"],
      [id: 2, name: :mock_2, provider: :mock, model: "mock-agent-2"],
      [id: 3, name: :mock_3, provider: :mock, model: "mock-agent-3"],
      [id: 4, name: :mock_4, provider: :mock, model: "mock-agent-4"],
      [id: 5, name: :mock_5, provider: :mock, model: "mock-agent-5"],
      [id: 6, name: :mock_6, provider: :mock, model: "mock-agent-6"]
    ]
  }

  @doc "Returns the normalized provider pool for the given name."
  def fetch!(pool_name \\ :default) do
    with {:ok, raw} <- lookup_pool(pool_name),
         {:ok, pool} <- Spec.normalize(raw),
         :ok <- Spec.validate(pool) do
      pool
    else
      {:error, reason} ->
        raise ArgumentError,
          message: "invalid provider pool #{inspect(pool_name)}: #{inspect(reason)}"
    end
  end

  @doc "Returns normalized pools for all configured names."
  def all_pools do
    config()
    |> Enum.map(fn {name, raw_pool} ->
      {name,
       Spec.normalize(raw_pool)
       |> case do
         {:ok, pool} -> pool
         {:error, _} -> nil
       end}
    end)
    |> Enum.reject(fn {_name, pool} -> is_nil(pool) end)
    |> Enum.into(%{})
  end

  @doc "Returns the pool size for the given name or list spec."
  def size(pool_or_name \\ :default)

  def size(name) when is_atom(name) or is_binary(name) do
    fetch!(name) |> Enum.count()
  end

  def size(pool) when is_list(pool) do
    Spec.validate!(pool)
    Enum.count(pool)
  end

  @doc "Returns the normalized spec for a specific agent id."
  def spec_for_agent(pool_or_name, id) when is_integer(id) do
    pool = normalize_pool(pool_or_name)
    Enum.find(pool, fn spec -> spec.id == id end)
  end

  defp normalize_pool(pool) when is_atom(pool) or is_binary(pool), do: fetch!(pool)

  defp normalize_pool(pool) when is_list(pool) do
    if Enum.all?(pool, &is_struct(&1, Spec)) do
      pool
    else
      Spec.normalize!(pool)
    end
  end

  defp normalize_pool(pool),
    do: raise(ArgumentError, "invalid provider pool input: #{inspect(pool)}")

  defp lookup_pool(pool_name) when is_atom(pool_name) or is_binary(pool_name) do
    pools = config()

    normalized_name = normalize_pool_name(pool_name)

    case pools[normalized_name] do
      nil -> {:error, {:unknown_pool, pool_name}}
      raw_pool -> {:ok, raw_pool}
    end
  end

  defp config do
    Application.get_env(:trinity_coordinator, :provider_pools, @default_pools)
    |> normalize_config_keys()
  end

  defp normalize_config_keys(pools) do
    pools
    |> Enum.reduce(%{}, fn
      {name, spec}, acc when is_atom(name) or is_binary(name) ->
        Map.put(acc, normalize_pool_name(name), spec)

      _, acc ->
        acc
    end)
  end

  defp normalize_pool_name(name) when is_binary(name), do: String.to_atom(name)
  defp normalize_pool_name(name), do: name
end

defmodule TrinityCoordinator.ProviderPool.Spec do
  @moduledoc "Typed provider spec normalization and validation."

  @enforce_keys [:id, :provider, :model]
  defstruct [
    :id,
    :name,
    :provider,
    :model,
    :base_url,
    :timeout_ms,
    :max_tokens,
    :temperature,
    :metadata,
    :enabled
  ]

  @type t :: %__MODULE__{
          id: non_neg_integer(),
          name: atom(),
          provider: atom(),
          model: String.t(),
          base_url: String.t() | nil,
          timeout_ms: pos_integer(),
          max_tokens: pos_integer(),
          temperature: float(),
          metadata: map(),
          enabled: boolean()
        }

  @known_providers [:openai, :openai_compatible, :mock]

  def normalize(raw) when is_list(raw) do
    with {:ok, specs} <- normalize_list(raw, []) do
      {:ok, Enum.sort_by(specs, & &1.id)}
    end
  end

  def normalize(_), do: {:error, :invalid_pool}

  def normalize!(raw) when is_list(raw) do
    case normalize(raw) do
      {:ok, pool} ->
        pool

      {:error, reason} ->
        raise ArgumentError, message: "invalid provider pool spec: #{inspect(reason)}"
    end
  end

  def validate(pool) when is_list(pool) do
    with :ok <- validate_duplicate_ids(pool),
         :ok <- validate_duplicates(pool),
         :ok <- validate_contiguous_ids(pool),
         :ok <- validate_specs(pool) do
      :ok
    else
      {:error, reason} -> {:error, reason}
    end
  end

  def validate!(pool) when is_list(pool) do
    case validate(pool) do
      :ok ->
        true

      {:error, reason} ->
        raise ArgumentError, message: "invalid provider pool: #{inspect(reason)}"
    end
  end

  def validate!(pool),
    do: raise(ArgumentError, message: "invalid provider pool: #{inspect(pool)}")

  defp validate_duplicate_ids(pool) do
    ids = Enum.map(pool, & &1.id)
    unique = MapSet.new(ids)

    if MapSet.size(unique) == length(ids) do
      :ok
    else
      {:error, :duplicate_provider_ids}
    end
  end

  defp validate_duplicates(pool) do
    names = pool |> Enum.map(& &1.name) |> Enum.reject(&is_nil/1)
    unique = MapSet.new(names)

    if MapSet.size(unique) == length(names) do
      :ok
    else
      {:error, :duplicate_provider_names}
    end
  end

  defp validate_contiguous_ids(pool) do
    ids = pool |> Enum.map(& &1.id) |> Enum.sort()
    expected = Enum.to_list(0..(length(ids) - 1))

    if ids == expected do
      :ok
    else
      {:error, :non_contiguous_ids}
    end
  end

  defp validate_specs(pool) do
    invalid =
      Enum.find(pool, fn spec ->
        spec.provider == :openai_compatible and
          (is_nil(spec.base_url) or not is_binary(spec.base_url))
      end)

    if invalid do
      {:error, {:invalid_openai_compatible_spec, invalid}}
    else
      :ok
    end
  end

  defp normalize_list([], acc), do: {:ok, Enum.reverse(acc)}

  defp normalize_list([entry | rest], acc) do
    case normalize_entry(entry) do
      {:ok, spec} ->
        normalize_list(rest, [spec | acc])

      {:error, _reason} = error ->
        error
    end
  end

  defp normalize_entry(%__MODULE__{} = spec) do
    normalize_struct(spec)
  end

  defp normalize_entry(entry) when is_list(entry) or is_map(entry) do
    normalized = Map.new(entry)

    with {:ok, id} <- coerce_id(normalized[:id]),
         {:ok, provider} <- coerce_atom(normalized[:provider]),
         {:ok, model} <- coerce_non_empty_binary(normalized[:model], :model),
         {:ok, timeout_ms} <- coalesce_positive_integer(normalized[:timeout_ms], 30_000),
         {:ok, max_tokens} <- coalesce_positive_integer(normalized[:max_tokens], 200),
         {:ok, temperature} <- coalesce_non_negative_number(normalized[:temperature], 0.2) do
      with {:ok, name} <- normalize_name(normalized[:name], id) do
        spec = %__MODULE__{
          id: id,
          name: name,
          provider: provider,
          model: model,
          base_url: coalesce_binary(normalized[:base_url]),
          timeout_ms: timeout_ms,
          max_tokens: max_tokens,
          temperature: temperature,
          metadata: Map.get(normalized, :metadata, %{}),
          enabled: Map.get(normalized, :enabled, true)
        }

        normalize_struct(spec)
      end
    end
  end

  defp normalize_entry(other), do: {:error, "invalid provider entry #{inspect(other)}"}

  defp normalize_struct(spec) do
    with {:ok, name} <- normalize_name(spec.name, spec.id),
         {:ok, provider} <- ensure_known_provider(spec.provider),
         {:ok, model} <- coerce_non_empty_binary(spec.model, :model),
         {:ok, _id} <- coerce_id(spec.id),
         {:ok, timeout_ms} <- coalesce_positive_integer(spec.timeout_ms, 30_000),
         {:ok, max_tokens} <- coalesce_positive_integer(spec.max_tokens, 200),
         {:ok, temperature} <- coalesce_non_negative_number(spec.temperature, 0.2) do
      {:ok,
       %__MODULE__{
         id: spec.id,
         name: name,
         provider: provider,
         model: model,
         base_url: coalesce_binary(spec.base_url),
         timeout_ms: timeout_ms,
         max_tokens: max_tokens,
         temperature: temperature,
         metadata: Map.get(spec, :metadata, %{}),
         enabled: spec.enabled == true
       }}
    end
  end

  defp normalize_name(nil, id), do: {:ok, String.to_atom("agent_#{id}")}
  defp normalize_name(name, _id) when is_atom(name), do: {:ok, name}
  defp normalize_name(name, _id) when is_binary(name), do: {:ok, String.to_atom(name)}
  defp normalize_name(name, _id), do: {:error, "invalid provider name #{inspect(name)}"}

  defp ensure_known_provider(provider) when provider in @known_providers, do: {:ok, provider}
  defp ensure_known_provider(provider), do: {:error, "unknown provider #{inspect(provider)}"}

  defp coerce_id(value) when is_integer(value) and value >= 0, do: {:ok, value}

  defp coerce_id(value) when is_binary(value) do
    case Integer.parse(value) do
      {id, ""} when id >= 0 -> {:ok, id}
      _ -> {:error, "invalid id #{inspect(value)}"}
    end
  end

  defp coerce_id(other), do: {:error, "invalid id #{inspect(other)}"}

  defp coerce_atom(value) when is_atom(value), do: {:ok, value}
  defp coerce_atom(value) when is_binary(value), do: {:ok, String.to_atom(value)}
  defp coerce_atom(other), do: {:error, "invalid atom value #{inspect(other)}"}

  defp coerce_non_empty_binary(value, field) when is_binary(value) do
    if String.trim(value) == "" do
      {:error, "#{field} cannot be empty"}
    else
      {:ok, value}
    end
  end

  defp coerce_non_empty_binary(other, field),
    do: {:error, "#{field} invalid: #{inspect(other)}"}

  defp coalesce_binary(nil), do: nil
  defp coalesce_binary(value) when is_binary(value) and value != "", do: value
  defp coalesce_binary(_), do: nil

  defp coalesce_positive_integer(nil, default), do: {:ok, default}

  defp coalesce_positive_integer(value, _default) when is_integer(value) and value > 0,
    do: {:ok, value}

  defp coalesce_positive_integer(_, _), do: {:error, "invalid integer value"}

  defp coalesce_non_negative_number(nil, default), do: {:ok, default}

  defp coalesce_non_negative_number(value, _default) when is_number(value) and value >= 0,
    do: {:ok, value / 1}

  defp coalesce_non_negative_number(_, _), do: {:error, "invalid number value"}
end
