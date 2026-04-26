defmodule TrinityCoordinator.Training.Evaluator do
  @moduledoc "
  Evaluation boundary for complete routing trajectories used by sep-CMA-ES.

  The evaluator owns task execution and reward shaping while delegating
  trajectory execution to the orchestrator.
  "

  alias TrinityCoordinator.{Orchestrator, StateManager}
  alias TrinityCoordinator.Training.SepCMAES.Reward

  @type task_spec :: %{
          required(:id) => String.t() | atom() | integer(),
          required(:messages) => [map()]
        }

  @doc """
  Evaluates a candidate model state against a task batch.

  Returns either `{:ok, reward}` or `{:ok, [reward]}` on success.
  """
  def evaluate_candidate(candidate_model_state, candidate_metadata, opts) when is_list(opts) do
    tasks = Keyword.get(opts, :tasks, [])
    model = Keyword.fetch!(opts, :model)
    reward_fn = Keyword.fetch!(opts, :reward_fn)
    slm_context = Keyword.fetch!(opts, :slm_context)
    run_orchestrate = Keyword.get(opts, :run_candidate, &run_candidate/5)
    run_opts = Keyword.get(opts, :run_opts, [])

    context = %{
      model: model,
      candidate_model_state: candidate_model_state,
      candidate_metadata: candidate_metadata,
      reward_fn: reward_fn,
      run_orchestrate: run_orchestrate,
      run_opts: run_opts,
      slm_context: slm_context
    }

    tasks
    |> Enum.reduce_while({:ok, [], []}, &evaluate_single_candidate_task(&1, &2, context))
    |> finalize_evaluation()
  end

  def evaluate_candidate(_model_state, _metadata, _opts) do
    {:error, :invalid_options}
  end

  defp finalize_evaluation({:ok, rewards, _outcomes}), do: {:ok, Enum.reverse(rewards)}
  defp finalize_evaluation(other), do: other

  defp normalize_task(task) when is_map(task) do
    id = Map.get(task, :id, Map.get(task, "id", :unknown))
    messages = Map.get(task, :messages, Map.get(task, "messages", []))
    %{id: id, messages: messages}
  end

  defp normalize_task(_), do: %{id: :invalid, messages: []}

  defp reward_from_outcome(reward_fn, task, trajectory, _candidate_metadata, _slm_context)
       when is_function(reward_fn, 2) do
    reward_fn.(task, trajectory)
    |> normalize_reward_result()
  end

  defp reward_from_outcome(reward_fn, task, trajectory, candidate_metadata, _slm_context)
       when is_function(reward_fn, 3) do
    reward_fn.(task, trajectory, candidate_metadata)
    |> normalize_reward_result()
  end

  defp reward_from_outcome(reward_fn, task, trajectory, candidate_metadata, _slm_context) do
    reward_fn.(task, trajectory, candidate_metadata, :ignored)
    |> normalize_reward_result()
  end

  defp evaluate_single_candidate_task(
         task,
         {:ok, rewards, outcomes},
         context
       ) do
    model = context.model
    candidate_model_state = context.candidate_model_state
    candidate_metadata = context.candidate_metadata
    run_orchestrate = context.run_orchestrate
    run_opts = context.run_opts
    reward_fn = context.reward_fn
    slm_context = context.slm_context

    task_spec = normalize_task(task)

    case run_orchestrate.(model, candidate_model_state, task_spec, run_opts, slm_context) do
      {:ok, trajectory} ->
        reward_from_outcome(
          reward_fn,
          task_spec,
          trajectory,
          candidate_metadata,
          slm_context
        )
        |> append_task_result(task_spec, rewards, outcomes, trajectory)

      {:error, reason} ->
        {:halt, {:error, {:trajectory_error, reason}}}
    end
  end

  defp evaluate_single_candidate_task(_, error, _context) do
    error
  end

  defp append_task_result(
         {:ok, reward_value},
         task_spec,
         rewards,
         outcomes,
         trajectory
       ) do
    normalized = Reward.normalize(reward_value)

    {:cont,
     {:ok, [normalized | rewards],
      [%{task: task_spec.id, reward: normalized, trajectory: trajectory} | outcomes]}}
  end

  defp append_task_result({:error, reason}, _task_spec, _rewards, _outcomes, _trajectory) do
    {:halt, {:error, {:reward_error, reason}}}
  end

  defp normalize_reward_result(value) when is_number(value), do: {:ok, value}

  defp normalize_reward_result({:ok, value}) when is_number(value), do: {:ok, value}

  defp normalize_reward_result({:ok, %{reward: value}}) when is_number(value) do
    {:ok, value}
  end

  defp normalize_reward_result(_), do: {:error, :invalid_reward}

  defp run_candidate(model, candidate_model_state, task, run_opts, slm_context) do
    case Keyword.get(run_opts, :run_candidate, nil) do
      nil ->
        run_full_trajectory(model, candidate_model_state, task, run_opts, slm_context)

      provided_fn when is_function(provided_fn, 5) ->
        provided_fn.(model, candidate_model_state, task, run_opts, slm_context)

      _ ->
        {:error, :invalid_run_candidate_hook}
    end
  end

  defp run_full_trajectory(model, candidate_model_state, task, run_opts, slm_context) do
    with :ok <- validate_provider_backed_mode(run_opts) do
      max_turns = Keyword.get(run_opts, :max_turns, 3)
      orchestrator_opts = Keyword.get(run_opts, :orchestrator_opts, [])

      with {:ok, pid} <- StateManager.start_link(task.messages),
           {:ok, response} <-
             Orchestrator.run_loop(
               pid,
               model,
               candidate_model_state,
               Keyword.put(orchestrator_opts, :slm_context, slm_context)
               |> Keyword.put(:max_turns, max_turns)
             ) do
        {:ok, %{status: :ok, task_id: task.id, response: response}}
      else
        {:error, reason} ->
          {:ok, %{status: :error, task_id: task.id, reason: reason}}
      end
    end
  end

  defp validate_provider_backed_mode(run_opts) do
    if Keyword.get(run_opts, :provider_enabled, false) do
      case validate_provider_budget(Keyword.get(run_opts, :provider_budget_usd)) do
        :ok ->
          validate_provider_credentials(Keyword.get(run_opts, :provider_credentials, nil))

        error ->
          error
      end
    else
      {:error, :provider_backed_training_disabled}
    end
  end

  defp validate_provider_budget(nil), do: {:error, :provider_budget_missing}

  defp validate_provider_budget(value) when is_number(value) and value > 0.0, do: :ok

  defp validate_provider_budget(_), do: {:error, :provider_budget_missing}

  defp validate_provider_credentials(creds) when is_map(creds) do
    if map_size(creds) > 0 do
      :ok
    else
      {:error, :provider_credentials_missing}
    end
  end

  defp validate_provider_credentials(creds) when is_list(creds) do
    if creds == [] do
      {:error, :provider_credentials_missing}
    else
      :ok
    end
  end

  defp validate_provider_credentials(_), do: {:error, :provider_credentials_missing}
end
