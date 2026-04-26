defmodule TrinityCoordinator.Benchmark.Routing do
  @moduledoc """
  Routing accuracy metrics and confusion matrices for benchmark cases.

  Labels are expected to come from fixture fields:
  `expected_agent` and `expected_role`.
  """

  alias TrinityCoordinator.{Benchmark.Dataset, CoordinationHead}

  @type confusion_matrix :: %{optional(integer()) => %{optional(integer()) => non_neg_integer()}}

  @type metrics :: %{
          considered: non_neg_integer(),
          skipped: non_neg_integer(),
          agent_accuracy: float(),
          role_accuracy: float(),
          joint_accuracy: float(),
          logits_mean: float(),
          agent_confusion: %{integer() => %{integer() => non_neg_integer()}},
          role_confusion: %{integer() => %{integer() => non_neg_integer()}},
          margin_mean: float(),
          predictions: [map()]
        }

  @doc """
  Computes routing metrics from real model routes.
  """
  @spec run([Dataset.t()], Nx.Tensor.t(), Axon.t(), map(), keyword()) ::
          {:ok, metrics()} | {:error, term()}
  def run(cases, feature_tensor, model, params, opts \\ []) do
    num_agents = Keyword.get(opts, :num_agents, 7)
    num_roles = Keyword.get(opts, :num_roles, 3)
    feature_rows = Nx.to_list(feature_tensor)

    with :ok <- validate_sizes(cases, feature_rows) do
      feature_rows
      |> Enum.zip(cases)
      |> Enum.reduce(
        init_metrics(),
        &accumulate_case(&1, &2, model, params, num_agents, num_roles)
      )
      |> metrics_from_totals()
    end
  end

  defp init_metrics,
    do: {[], 0, 0, 0, 0, 0, %{}, %{}, 0, 0.0}

  defp accumulate_case(
         {_vector, %Dataset{expected_agent: nil, expected_role: nil}},
         {results, considered, skipped, a_ok, r_ok, j_ok, a_conf, r_conf, logits_count,
          margin_sum},
         _model,
         _params,
         _num_agents,
         _num_roles
       ) do
    {results, considered, skipped + 1, a_ok, r_ok, j_ok, a_conf, r_conf, logits_count, margin_sum}
  end

  defp accumulate_case(
         {vector,
          %Dataset{expected_agent: expected_agent, expected_role: expected_role, id: case_id}},
         {results, considered, skipped, a_ok, r_ok, j_ok, a_conf, r_conf, logits_count,
          margin_sum},
         model,
         params,
         num_agents,
         num_roles
       ) do
    route = route_with_shape(model, params, vector, num_agents, num_roles)
    selected_agent = route.agent_id
    selected_role = route.role_id
    logits = Nx.to_flat_list(Nx.squeeze(route.logits, axes: [0]))
    route_margin = route_margin(route.agent_logits, route.role_logits)
    a_correct = selected_agent == expected_agent
    r_correct = selected_role == expected_role
    j_correct = a_correct and r_correct

    {
      [
        %{
          id: case_id,
          expected_agent: expected_agent,
          expected_role: expected_role,
          agent_id: selected_agent,
          role_id: selected_role,
          agent_correct: a_correct,
          role_correct: r_correct,
          joint_correct: j_correct,
          logits: logits,
          route_margin: route_margin
        }
        | results
      ],
      considered + 1,
      skipped,
      a_ok + bool_to_int(a_correct),
      r_ok + bool_to_int(r_correct),
      j_ok + bool_to_int(j_correct),
      bump_confusion(a_conf, expected_agent, selected_agent),
      bump_confusion(r_conf, expected_role, selected_role),
      logits_count + 1,
      margin_sum + route_margin
    }
  end

  defp accumulate_case({_vector, _case}, metrics, _model, _params, _num_agents, _num_roles),
    do: metrics

  defp metrics_from_totals(
         {predictions, considered, skipped, a_ok, r_ok, j_ok, a_conf, r_conf, logits_count,
          margin_sum}
       ) do
    if considered == 0 do
      {:error, :no_labeled_cases}
    else
      {:ok,
       %{
         considered: considered,
         skipped: skipped,
         agent_accuracy: Float.round(a_ok / considered, 6),
         role_accuracy: Float.round(r_ok / considered, 6),
         joint_accuracy: Float.round(j_ok / considered, 6),
         margin_mean: Float.round(margin_sum / max(logits_count, 1), 6),
         logits_mean: Float.round(avg_logit_abs(predictions), 6),
         agent_confusion: a_conf,
         role_confusion: r_conf,
         predictions: Enum.reverse(predictions)
       }}
    end
  end

  defp validate_sizes(cases, feature_rows) when is_list(cases) and is_list(feature_rows) do
    if length(cases) == length(feature_rows) do
      :ok
    else
      {:error, :feature_count_mismatch}
    end
  end

  defp validate_sizes(_, _), do: {:error, :invalid_inputs}

  defp route_with_shape(model, params, vector, num_agents, num_roles) do
    vector_tensor =
      vector
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({1, length(vector)})

    route = CoordinationHead.route(model, params, vector_tensor, num_agents, num_roles)

    if route.agent_id in 0..(num_agents - 1) and route.role_id in 0..(num_roles - 1) do
      route
    else
      raise ArgumentError, "route ids out of range for configured head"
    end
  end

  defp route_margin(agent_logits, role_logits) do
    top_two = top_two(to_list_floats(agent_logits))
    role_top_two = top_two(to_list_floats(role_logits))
    (agent_margin(top_two) + role_margin(role_top_two)) / 2.0
  end

  defp top_two(values) do
    values
    |> Enum.sort(:desc)
    |> Enum.take(2)
    |> pad_top_two()
  end

  defp pad_top_two([first, second]), do: [first, second]
  defp pad_top_two([first]), do: [first, 0.0]
  defp pad_top_two([]), do: [0.0, 0.0]

  defp agent_margin([first, second]), do: first - second
  defp agent_margin([first]), do: first
  defp role_margin([first, second]), do: first - second
  defp role_margin([first]), do: first

  defp bump_confusion(confusion, expected, predicted) do
    row = Map.get(confusion, expected, %{})
    Map.put(confusion, expected, Map.update(row, predicted, 1, &(&1 + 1)))
  end

  defp bool_to_int(true), do: 1
  defp bool_to_int(false), do: 0

  defp to_list_floats(tensor), do: tensor |> Nx.to_flat_list() |> Enum.map(&to_float/1)

  defp to_float(value) when is_float(value), do: value
  defp to_float(value) when is_integer(value), do: value / 1
  defp to_float(_), do: 0.0

  defp avg_logit_abs(predictions) do
    flat = Enum.flat_map(predictions, & &1[:logits])

    if Enum.empty?(flat) do
      0.0
    else
      Enum.sum(Enum.map(flat, &abs/1)) / length(flat)
    end
  end
end
