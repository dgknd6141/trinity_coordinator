defmodule TrinityCoordinator.BenchmarkDatasetTest do
  use ExUnit.Case

  alias TrinityCoordinator.Benchmark.Dataset

  test "loads and validates benchmark fixtures" do
    path = Path.join(__DIR__, "../../fixtures/benchmark_cases.jsonl")
    path = Path.expand(path)

    assert {:ok, cases} = Dataset.load!(path)
    assert length(cases) >= 5
    assert Enum.all?(cases, &is_struct(&1, Dataset))

    first = hd(cases)
    assert first.id != nil
    assert first.family in ["math", "coding", "proof"]
    assert is_list(first.messages)
    assert is_integer(first.expected_agent)
    assert is_integer(first.expected_role)
  end

  test "rejects malformed jsonl lines" do
    data = """
    {"id":"good","family":"math","messages":[{"role":"user","content":"Hi"}],"expected_agent":0}
    {this is bad}
    """

    assert {:error, :invalid_jsonl} = Dataset.parse_dataset(data)
  end

  test "rejects malformed records" do
    data = """
    {"id":"","family":"math","messages":[]}
    {"id":"x","family":"","messages":[{"role":"user","content":"x"}]}
    """

    assert {:error, :invalid_record_schema} = Dataset.parse_dataset(data)
  end
end
