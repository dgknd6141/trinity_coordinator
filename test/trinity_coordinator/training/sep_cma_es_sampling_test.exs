defmodule TrinityCoordinator.Training.SepCMAESSamplingTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.Training.SepCMAES.Sampling

  test "sampling is deterministic for the same seed" do
    mean = Nx.tensor([0.0, 1.0, -1.0], type: :f32)

    {first, first_spec} = Sampling.sample(mean, 0.25, 4, {99, 88, 77})
    {second, second_spec} = Sampling.sample(mean, 0.25, 4, {99, 88, 77})

    assert Sampling.sample(mean, 0.25, 4, {99, 88, 77}) == {first, first_spec}
    assert Nx.to_flat_list(first) == Nx.to_flat_list(second)
    assert first_spec == second_spec
  end

  test "different seeds usually change samples" do
    mean = Nx.tensor([1.0, 2.0, 3.0], type: :f32)

    {first, _} = Sampling.sample(mean, 0.25, 4, {1, 2, 3})
    {second, _} = Sampling.sample(mean, 0.25, 4, {3, 2, 1})

    assert Nx.to_flat_list(first) != Nx.to_flat_list(second)
  end

  test "sample metadata captures generation shape" do
    mean = Nx.tensor([1.0, 2.0, 3.0], type: :f32)

    {_samples, spec} = Sampling.sample(mean, 0.1, 3, {11, 22, 33})

    assert spec.population_size == 3
    assert spec.sigma == 0.1
    assert spec.seed == {11, 22, 33}
    assert spec.candidate_shape == {3, 3}
  end
end
