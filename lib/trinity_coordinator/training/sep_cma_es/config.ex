defmodule TrinityCoordinator.Training.SepCMAES.Config do
  @moduledoc "Configuration and validation for sep-CMA-ES optimization runs."

  @type t :: %__MODULE__{
          population_size: pos_integer(),
          sigma: float(),
          generations: non_neg_integer(),
          replications: pos_integer(),
          top_candidates: pos_integer(),
          seed: {integer(), integer(), integer()},
          stop_threshold: float(),
          evaluation_budget: pos_integer() | nil,
          provider_budget_usd: float() | nil,
          cancellation_fn: (-> boolean) | nil
        }

  defstruct population_size: 32,
            sigma: 0.05,
            generations: 10,
            replications: 4,
            top_candidates: 8,
            seed: {42, 11, 7},
            stop_threshold: 1.0,
            evaluation_budget: nil,
            provider_budget_usd: nil,
            cancellation_fn: nil

  @doc "Builds and validates a config struct."
  @spec new(keyword() | map() | t()) :: {:ok, t()} | {:error, term()}
  def new(opts) when is_list(opts), do: opts |> normalize() |> validate()

  def new(%__MODULE__{} = config), do: validate(%{config | top_candidates: config.top_candidates})

  def new(map) when is_map(map) do
    map
    |> struct(__MODULE__)
    |> validate()
  end

  defp normalize(opts) when is_list(opts), do: struct(__MODULE__, opts)

  defp validate(%__MODULE__{} = config) do
    with :ok <- validate_integer(config.population_size, :population_size),
         :ok <- validate_integer(config.generations, :generations),
         :ok <- validate_float(config.sigma, :sigma),
         :ok <- validate_integer(config.replications, :replications),
         :ok <- validate_integer(config.top_candidates, :top_candidates),
         :ok <- validate_seed(config.seed),
         :ok <- validate_stop_threshold(config.stop_threshold),
         :ok <- validate_top_candidates_bounds(config.population_size, config.top_candidates),
         :ok <- validate_optional_budget(config.evaluation_budget),
         :ok <- validate_optional_budget(config.provider_budget_usd),
         :ok <- validate_cancellation_fn(config.cancellation_fn) do
      {:ok, config}
    end
  end

  defp validate_integer(value, _name) when is_integer(value) and value > 0 do
    :ok
  end

  defp validate_integer(_value, _name), do: {:error, :invalid_int}

  defp validate_float(value, _name) when is_number(value) and value > 0 do
    :ok
  end

  defp validate_float(_value, _name), do: {:error, :invalid_float}

  defp validate_seed({a, b, c}) when is_integer(a) and is_integer(b) and is_integer(c), do: :ok
  defp validate_seed(_), do: {:error, :invalid_seed}

  defp validate_stop_threshold(value) when is_number(value) and value >= -1.0 and value <= 1.0,
    do: :ok

  defp validate_stop_threshold(_), do: {:error, :invalid_stop_threshold}

  defp validate_top_candidates_bounds(population, top) when top <= population and top > 0, do: :ok
  defp validate_top_candidates_bounds(_, _), do: {:error, :invalid_top_candidates}

  defp validate_optional_budget(nil), do: :ok
  defp validate_optional_budget(value) when is_number(value) and value > 0, do: :ok
  defp validate_optional_budget(_), do: {:error, :invalid_budget}

  defp validate_cancellation_fn(nil), do: :ok
  defp validate_cancellation_fn(fnc) when is_function(fnc, 0), do: :ok
  defp validate_cancellation_fn(_), do: {:error, :invalid_cancellation_fn}
end
