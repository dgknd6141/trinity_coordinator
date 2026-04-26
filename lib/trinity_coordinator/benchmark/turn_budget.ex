defmodule TrinityCoordinator.Benchmark.TurnBudget do
  @moduledoc """
  Turn-budget benchmark suite.

  This suite reuses the real feature extractor and real routing policy, then
  simulates verifier outcomes from selected roles and labeled targets to keep the
  benchmark fully deterministic without provider spend.
  """

  alias TrinityCoordinator.{Benchmark.Dataset, CoordinationHead, Runtime}
  alias TrinityCoordinator.Benchmark.Report

  @doc """
  Evaluates routing turn behavior using route decisions and a local safety policy.
  """
  @spec run(
          [Dataset.t()],
          {any(), any()} | %{model_info: any(), tokenizer: any()},
          Nx.Tensor.t(),
          Axon.t(),
          map(),
          keyword()
        ) ::
          {:ok, map()} | {:error, term()}
  def run(cases, slm_context, features, model, params, opts \\ []) do
    max_turns = Keyword.get(opts, :max_turns, 5)
    num_agents = Keyword.get(opts, :num_agents, 7)
    num_roles = Keyword.get(opts, :num_roles, 3)
    include_trace = Keyword.get(opts, :include_trace, false)

    features_list = Nx.to_list(features)

    with :ok <- validate_run_inputs(cases, features_list, max_turns) do
      outcomes =
        cases
        |> Enum.with_index()
        |> Enum.map(fn {case, index} ->
          run_case(case, index, features_list, model, params, max_turns, num_agents, num_roles)
        end)

      summary = summarize_outcomes(outcomes)
      report = maybe_write_trace(outcomes, include_trace, Keyword.get(opts, :trace_path))

      runtime = %{
        xla_platforms: Runtime.supported_platforms(),
        trace_events: report,
        vector_backend: Runtime.tensor_backend(features)
      }

      {:ok,
       %{
         summary: summary,
         runtime: runtime,
         outcomes: outcomes,
         run_options: %{
           max_turns: max_turns,
           num_agents: num_agents,
           num_roles: num_roles
         },
         slm_context: slm_context
       }}
    end
  end

  defp validate_run_inputs(cases, features, max_turns)
       when is_list(cases) and is_list(features) and is_integer(max_turns) and max_turns > 0 do
    if length(cases) == length(features) do
      :ok
    else
      {:error, :feature_count_mismatch}
    end
  end

  defp validate_run_inputs(_, _, _), do: {:error, :invalid_inputs}

  defp run_case(case, index, features, model, params, max_turns, num_agents, num_roles) do
    feature = Enum.at(features, index)
    route = route_for_vector(model, params, feature, num_agents, num_roles)

    result =
      simulate_turns(
        case,
        route,
        max_turns
      )

    Map.merge(result, %{
      id: case.id,
      expected_role: case.expected_role,
      family: case.family,
      selected_role: route.role_id,
      selected_agent: route.agent_id,
      selected_role_name: route.role_name,
      vector_shape: Tuple.to_list(Nx.shape(feature_as_tensor(feature))),
      run_backend: "EXLA.Backend"
    })
  end

  defp route_for_vector(model, params, feature, num_agents, num_roles) do
    route =
      feature
      |> feature_as_tensor()
      |> then(fn tensor ->
        CoordinationHead.route(model, params, tensor, num_agents, num_roles)
      end)
      |> then(fn route ->
        role_name =
          case route.role_id do
            0 -> "Thinker"
            1 -> "Worker"
            2 -> "Verifier"
            _ -> "Unknown"
          end

        Map.put(route, :role_name, role_name)
      end)

    route
  end

  defp simulate_turns(case, route, max_turns) do
    simulate_turns(route, case.expected_role, max_turns, 1, [])
  end

  defp simulate_turns(route, expected_role, max_turns, turn, acc) when turn <= max_turns do
    status =
      cond do
        route.role_id == 2 and (expected_role == nil or expected_role == 2) ->
          :accepted

        route.role_id == 2 ->
          :revised

        true ->
          :revised
      end

    event = %{
      turn: turn - 1,
      role_id: route.role_id,
      role_name: route.role_name,
      status: status
    }

    acc = [event | acc]

    if status == :accepted do
      final_result(acc)
    else
      simulate_turns(route, expected_role, max_turns, turn + 1, acc)
    end
  end

  defp simulate_turns(_route, _expected_role, _max_turns, _turn, acc) do
    final_result(acc)
  end

  defp final_result(events) do
    final_status =
      events
      |> Enum.any?(&(&1.status == :accepted))
      |> if(do: :accepted, else: :max_turns_reached)

    turns_taken =
      case final_status do
        :accepted ->
          events |> Enum.find_index(&(&1.status == :accepted)) |> Kernel.+(1)

        :max_turns_reached ->
          length(events)
      end

    %{
      turns: turns_taken,
      final_status: final_status,
      provider_calls: turns_taken,
      events: Enum.reverse(events)
    }
  end

  defp summarize_outcomes(outcomes) do
    totals = Enum.count(outcomes)
    accepted = Enum.count(outcomes, &(&1.final_status == :accepted))
    revised = Enum.count(outcomes, &(&1.final_status == :revised))
    max_turn = Enum.count(outcomes, &(&1.final_status == :max_turns_reached))
    total_turns = outcomes |> Enum.map(& &1.turns) |> Enum.sum()

    avg_turns = if totals > 0, do: total_turns / totals, else: 0.0

    provider_calls = outcomes |> Enum.map(& &1.provider_calls) |> Enum.sum()

    %{
      total_cases: totals,
      accepted: accepted,
      revised: revised,
      max_turn_reached: max_turn,
      accept_rate: if(totals > 0, do: accepted / totals, else: 0.0),
      revise_rate: if(totals > 0, do: revised / totals, else: 0.0),
      max_turn_hit_rate: if(totals > 0, do: max_turn / totals, else: 0.0),
      average_turns: Float.round(avg_turns, 4),
      provider_calls: provider_calls,
      avg_cost_estimate_usd: 0.0
    }
  end

  defp maybe_write_trace(outcomes, true, path) when is_binary(path) do
    schema = %{version: 1, suite: :turn_budget, generated_at: DateTime.utc_now()}
    Report.write_json(path, %{schema: schema, outcomes: outcomes})
  end

  defp maybe_write_trace(_, _, _), do: nil

  defp feature_as_tensor(vector) when is_list(vector), do: Nx.tensor([vector], type: :f32)
  defp feature_as_tensor(%Nx.Tensor{} = tensor), do: tensor
end
