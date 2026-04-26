defmodule TrinityCoordinator.Benchmark.Separability do
  @moduledoc """
  Computes task-family separability metrics from extracted benchmark features.

  The default metrics are intentionally lightweight and deterministic:

  * within-family cosine-distance average
  * between-family cosine-distance average
  * nearest-centroid family-assignment accuracy
  """

  alias TrinityCoordinator.Benchmark.Dataset

  @type metrics :: %{
          dataset_size: non_neg_integer(),
          family_count: non_neg_integer(),
          within_distance: float(),
          between_distance: float(),
          nearest_centroid_accuracy: float(),
          centroid_count: non_neg_integer(),
          family_sizes: %{String.t() => non_neg_integer()}
        }

  @doc """
  Computes separability metrics for one dataset.
  """
  @spec run([Dataset.t()], Nx.Tensor.t()) :: {:ok, metrics()} | {:error, term()}
  def run(cases, feature_tensor) do
    feature_rows = feature_rows(feature_tensor)
    families = Enum.map(cases, & &1.family)

    with :ok <- validate_inputs(cases, feature_rows) do
      by_family = group_by_family(cases, feature_rows)
      centroid_info = centroids(by_family)
      within = within_family_distance(by_family)
      between = between_family_distance(centroid_info)
      nearest = nearest_centroid_accuracy(by_family, families, feature_rows, centroid_info)

      {:ok,
       %{
         dataset_size: length(cases),
         family_count: map_size(by_family),
         centroid_count: map_size(by_family),
         family_sizes: map_sizes(by_family),
         within_distance: Float.round(within, 6),
         between_distance: Float.round(between, 6),
         nearest_centroid_accuracy: Float.round(nearest, 6)
       }}
    end
  end

  defp validate_inputs(cases, features) when is_list(features) do
    cond do
      length(cases) != length(features) ->
        {:error, :feature_count_mismatch}

      Enum.empty?(cases) ->
        {:error, :empty_cases}

      true ->
        :ok
    end
  end

  defp validate_inputs(_, _), do: {:error, :invalid_inputs}

  defp feature_rows(%Nx.Tensor{} = features) do
    Nx.to_list(features)
  end

  defp group_by_family(cases, feature_rows) do
    cases
    |> Enum.zip(feature_rows)
    |> Enum.group_by(fn {case, _vector} -> case.family end, fn {_case, vector} -> vector end)
  end

  defp centroids(by_family) do
    by_family
    |> Enum.map(fn {family, vectors} ->
      {family, centroid(vectors)}
    end)
    |> Map.new()
  end

  defp centroid([]), do: []

  defp centroid(vectors) when is_list(vectors) do
    dim = vectors |> hd() |> length()

    vectors
    |> Enum.reduce(List.duplicate(0.0, dim), fn vector, acc ->
      vector
      |> Enum.zip(acc)
      |> Enum.map(fn {value, sum} -> value + sum end)
    end)
    |> Enum.map(&(&1 / length(vectors)))
  end

  defp centroid(_), do: []

  defp within_family_distance(by_family) do
    scores =
      by_family
      |> Enum.flat_map(fn {_family, vectors} ->
        vectors
        |> pair_indices()
        |> Enum.map(fn {left, right} ->
          1.0 - cosine_similarity(Enum.at(vectors, left), Enum.at(vectors, right))
        end)
      end)

    if Enum.empty?(scores), do: 0.0, else: Enum.sum(scores) / length(scores)
  end

  defp between_family_distance(centroids) do
    centroid_list = Map.values(centroids)

    scores =
      centroid_list
      |> pair_indices()
      |> Enum.map(fn {left, right} ->
        1.0 - cosine_similarity(Enum.at(centroid_list, left), Enum.at(centroid_list, right))
      end)

    if Enum.empty?(scores), do: 0.0, else: Enum.sum(scores) / length(scores)
  end

  defp nearest_centroid_accuracy(_by_family, families, feature_rows, centroids) do
    case map_size(centroids) do
      0 ->
        0.0

      _ ->
        cases =
          families
          |> Enum.with_index()
          |> Enum.zip(feature_rows)
          |> Enum.map(fn {{family, _index}, vector} ->
            pred = nearest_family(family, vector, centroids)
            {family, pred}
          end)

        correct =
          cases
          |> Enum.count(fn {expected, predicted} -> expected == predicted end)

        correct / length(cases)
    end
  end

  defp nearest_family(_target_family, vector, centroids) do
    centroids
    |> Enum.map(fn {family, centroid} ->
      {family, cosine_similarity(vector, centroid)}
    end)
    |> Enum.max_by(fn {_family, score} -> score end, fn -> nil end)
    |> case do
      {family, _score} -> family
      _ -> ""
    end
  end

  defp pair_indices(list) when is_list(list), do: pair_indices(length(list), [])

  defp pair_indices(count, _)
       when count < 2 do
    []
  end

  defp pair_indices(count, _acc) do
    for i <- 0..(count - 2), j <- (i + 1)..(count - 1), do: {i, j}
  end

  defp map_sizes(by_family),
    do: Map.new(by_family, fn {family, vectors} -> {family, length(vectors)} end)

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) and length(a) == length(b) do
    {dot, norm_a, norm_b} =
      Enum.reduce(Enum.zip(a, b), {0.0, 0.0, 0.0}, fn {left, right},
                                                      {dot_acc, norm_a_acc, norm_b_acc} ->
        left_f = normalize_number(left)
        right_f = normalize_number(right)

        {
          dot_acc + left_f * right_f,
          norm_a_acc + left_f * left_f,
          norm_b_acc + right_f * right_f
        }
      end)

    denom = :math.sqrt(norm_a) * :math.sqrt(norm_b)

    if denom == 0.0 do
      0.0
    else
      dot / denom
    end
  end

  defp cosine_similarity(_, _), do: 0.0

  defp normalize_number(value) when is_float(value), do: value
  defp normalize_number(value) when is_integer(value), do: value / 1.0
  defp normalize_number(_), do: 0.0
end
