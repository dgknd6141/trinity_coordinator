defmodule TrinityCoordinator.Training.SepCMAESRecombinationTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Training.SepCMAES

  test "rank-weighted recombination weights are normalized and descending" do
    weights = SepCMAES.recombination_weights(4, :rank_weighted)
    values = Nx.to_flat_list(weights)

    assert length(values) == 4
    assert_in_delta Enum.sum(values), 1.0, 1.0e-6
    assert Enum.at(values, 0) > Enum.at(values, 1)
    assert Enum.at(values, 1) > Enum.at(values, 2)
    assert Enum.at(values, 2) > Enum.at(values, 3)
  end

  test "uniform recombination preserves arithmetic mean" do
    top_vectors = Nx.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]], type: :f32)
    weights = SepCMAES.recombination_weights(3, :uniform)
    mean = SepCMAES.weighted_mean(top_vectors, weights)

    assert_all_close(mean, Nx.tensor([10.0 / 3.0, 10.0 / 3.0]), atol: 1.0e-6)
  end

  test "rank-weighted mean favors higher ranked vectors" do
    top_vectors = Nx.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]], type: :f32)
    uniform = SepCMAES.weighted_mean(top_vectors, SepCMAES.recombination_weights(3, :uniform))

    weighted =
      SepCMAES.weighted_mean(top_vectors, SepCMAES.recombination_weights(3, :rank_weighted))

    refute_all_close(uniform, weighted, atol: 1.0e-6)
    [weighted_x, weighted_y] = Nx.to_flat_list(weighted)
    [uniform_x, uniform_y] = Nx.to_flat_list(uniform)

    assert weighted_x > uniform_x
    assert weighted_y < uniform_y
  end

  defp assert_all_close(left, right, opts) do
    assert Nx.to_number(Nx.all_close(left, right, opts)) == 1
  end

  defp refute_all_close(left, right, opts) do
    assert Nx.to_number(Nx.all_close(left, right, opts)) == 0
  end
end
