defmodule TrinityCoordinator.Training.SepCMAES.Sampling do
  @moduledoc """
  Sampling helpers for sep-CMA-ES candidates.

  The module intentionally keeps sampling deterministic and reproducible for test
  and checkpointing workflows. It uses seeded pseudorandom generation and maps
  a 1-D mean vector into `population_size` candidates.
  """

  alias Nx.Tensor

  @type seed :: {integer(), integer(), integer()}
  @type sampling_spec :: %{
          required(:population_size) => pos_integer(),
          required(:sigma) => float(),
          required(:seed) => seed(),
          required(:candidate_shape) => tuple()
        }

  @doc """
  Draws a matrix of shape `{population_size, dim}` with values
  `mean + sigma * randn`.
  """
  @spec sample(Nx.Tensor.t(), float(), pos_integer(), seed()) :: {Nx.Tensor.t(), sampling_spec()}
  def sample(mean, sigma, population_size, seed \\ {42, 11, 7}) do
    validate_mean_tensor(mean)
    validate_positive(population_size, :population_size)
    validate_positive_number(sigma, :sigma)

    mean_flat = Nx.to_flat_list(Nx.reshape(mean, {:auto}))
    dim = length(mean_flat)
    candidate_shape = {population_size, dim}

    :rand.seed(:exsss, seed)

    candidates =
      Enum.map(1..population_size, fn _ ->
        Enum.map(mean_flat, fn value ->
          value + sigma * :rand.normal()
        end)
      end)
      |> Nx.tensor(type: Nx.type(mean))

    spec = %{
      population_size: population_size,
      sigma: sigma,
      seed: seed,
      candidate_shape: candidate_shape
    }

    {candidates, spec}
  end

  defp validate_mean_tensor(%Tensor{} = tensor) do
    shape = Nx.shape(tensor)

    case shape do
      {0, _} -> raise ArgumentError, "mean tensor must have at least one element"
      {_, _} when tuple_size(shape) > 2 -> raise ArgumentError, "mean tensor should be 1-D"
      _ -> :ok
    end
  end

  defp validate_mean_tensor(_), do: raise(ArgumentError, "mean must be an Nx tensor")

  defp validate_positive(value, _name) when is_integer(value) and value > 0, do: :ok

  defp validate_positive(value, name),
    do: raise(ArgumentError, "#{name} must be a positive integer, got #{inspect(value)}")

  defp validate_positive_number(value, _name) when is_number(value) and value > 0, do: :ok

  defp validate_positive_number(value, name),
    do: raise(ArgumentError, "#{name} must be a positive number, got #{inspect(value)}")
end
