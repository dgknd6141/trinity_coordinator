defmodule TrinityCoordinator.Training.SepCMAES.State do
  @moduledoc """
  Internal optimizer state passed between generations.
  """

  @type trace_entry :: map()

  @enforce_keys [:generation, :mean_vector, :sigma, :best_reward, :seed]
  defstruct [
    :generation,
    :mean_vector,
    :sigma,
    :seed,
    :template_state,
    :model_metadata,
    :best_reward,
    :best_vector,
    :evaluations,
    :provider_cost_usd,
    :trace,
    :history
  ]

  @type t :: %__MODULE__{
          generation: non_neg_integer(),
          mean_vector: Nx.Tensor.t(),
          sigma: float(),
          seed: {integer(), integer(), integer()},
          template_state: map(),
          model_metadata: list(),
          best_reward: float(),
          best_vector: Nx.Tensor.t(),
          evaluations: non_neg_integer(),
          provider_cost_usd: float(),
          trace: [trace_entry()],
          history: [map()]
        }

  @doc "Builds the mutable optimization state from an initial model."
  @spec new(
          generation: non_neg_integer(),
          mean_vector: Nx.Tensor.t(),
          sigma: float(),
          seed: {integer(), integer(), integer()},
          template_state: map(),
          model_metadata: list()
        ) :: t()
  def new(opts) do
    opts_map = opts |> Enum.into(%{})

    %__MODULE__{
      generation: Map.get(opts_map, :generation, 0),
      mean_vector: Map.get(opts_map, :mean_vector),
      sigma: Map.get(opts_map, :sigma),
      seed: Map.get(opts_map, :seed),
      template_state: Map.get(opts_map, :template_state),
      model_metadata: Map.get(opts_map, :model_metadata),
      best_reward: -1.0,
      best_vector: Map.get(opts_map, :mean_vector),
      evaluations: 0,
      provider_cost_usd: 0.0,
      trace: [],
      history: []
    }
  end

  @doc "Builds a normalized state map returned to callers after training."
  @spec result(t(), map(), map()) :: map()
  def result(state, trained_state, metrics) do
    %{
      model_state: trained_state,
      metrics: metrics,
      trace: Enum.reverse(state.trace)
    }
  end
end
