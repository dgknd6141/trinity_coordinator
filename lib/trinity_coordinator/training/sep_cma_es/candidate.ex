defmodule TrinityCoordinator.Training.SepCMAES.Candidate do
  @moduledoc """
  Candidate record for a single sampled router parameter set.
  """

  alias TrinityCoordinator.Training.SepCMAES.Reward

  @type t :: %__MODULE__{
          id: non_neg_integer(),
          generation: non_neg_integer(),
          vector: Nx.Tensor.t(),
          rewards: [float()],
          mean_reward: float(),
          metadata: map()
        }

  @enforce_keys [:id, :generation, :vector, :rewards, :metadata]
  defstruct [:id, :generation, :vector, :rewards, :metadata, mean_reward: 0.0]

  @doc "Builds a candidate from a raw sampled vector and a list of reward values."
  @spec new(non_neg_integer(), non_neg_integer(), Nx.Tensor.t(), [number()], map()) :: t()
  def new(id, generation, vector, rewards, metadata \\ %{}) do
    mean_reward = Reward.mean(rewards)

    %__MODULE__{
      id: id,
      generation: generation,
      vector: vector,
      rewards: Enum.map(rewards, &Float.round(&1, 6)),
      metadata: metadata,
      mean_reward: mean_reward
    }
  end
end
