defmodule TrinityCoordinator.BenchmarkSeparabilityTest do
  use ExUnit.Case

  alias TrinityCoordinator.Benchmark.{Dataset, Separability}

  test "computes separability metrics from synthetic vectors" do
    cases = [
      %Dataset{id: "a1", family: "math", messages: [], expected_agent: nil, expected_role: nil},
      %Dataset{id: "a2", family: "math", messages: [], expected_agent: nil, expected_role: nil},
      %Dataset{id: "b1", family: "code", messages: [], expected_agent: nil, expected_role: nil}
    ]

    features = Nx.tensor([[1.0, 1.0], [0.9, 1.1], [2.0, -1.0]])
    assert {:ok, metrics} = Separability.run(cases, features)

    assert metrics.dataset_size == 3
    assert metrics.family_count == 2
    assert metrics.within_distance >= 0.0
    assert metrics.between_distance >= 0.0
    assert metrics.nearest_centroid_accuracy >= 0.0
  end

  test "rejects feature mismatches" do
    cases = [
      %Dataset{id: "x", family: "math", messages: [], expected_agent: nil, expected_role: nil}
    ]

    features = Nx.tensor([[1.0, 1.0], [2.0, 2.0]])
    assert {:error, :feature_count_mismatch} = Separability.run(cases, features)
  end
end
