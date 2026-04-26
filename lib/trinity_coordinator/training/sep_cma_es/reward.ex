defmodule TrinityCoordinator.Training.SepCMAES.Reward do
  @moduledoc """
  Utilities for normalizing and validating rewards used by sep-CMA-ES.
  """

  @doc """
  Aggregates per-replication reward samples into a mean scalar.
  """
  @spec aggregate([number()]) :: float()
  def aggregate([_ | _] = samples) do
    {sum, count} =
      samples
      |> Enum.reduce({0.0, 0}, fn sample, {sum, count} ->
        {sum + normalize(sample), count + 1}
      end)

    sum / count
  end

  @doc """
  Keeps reward values in `[0.0, 1.0]` for binary label-free objectives.
  """
  @spec normalize(number()) :: float()
  def normalize(value) when is_number(value) and value <= 1 and value >= 0, do: value / 1

  def normalize(value) when is_number(value),
    do: raise(ArgumentError, "reward must be in [0, 1], got #{value}")

  def normalize(_), do: raise(ArgumentError, "invalid reward value")

  @doc false
  def mean(samples), do: aggregate(samples)
end
