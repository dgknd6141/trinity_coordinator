defmodule TrinityCoordinator.BenchmarkFeatureExtractorTest do
  use ExUnit.Case

  alias TrinityCoordinator.{Benchmark.Dataset, Benchmark.FeatureExtractor, Extractor, Runtime}

  test "extractor path validates cases list shape" do
    assert {:error, :invalid_inputs} = FeatureExtractor.run({%{}, %{}}, [])
  end

  @tag :integration
  test "extracts real vectors from a tiny model with one vector per case" do
    Runtime.put_cuda_backend!()

    path = Path.expand("../../fixtures/benchmark_cases.jsonl", __DIR__)
    assert {:ok, cases} = Dataset.load!(path)
    cases = Enum.take(cases, 2)

    {:ok, {model_info, tokenizer}} =
      Extractor.load_slm_model(
        {:hf, "hf-internal-testing/tiny-random-gpt2"},
        Bumblebee.Text.Gpt2,
        :base
      )

    assert {:ok, features} = FeatureExtractor.run(model_info, tokenizer, cases)
    assert Nx.shape(features) == {2, 32}
  end
end
