defmodule TrinityCoordinator.Training.EvaluatorTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.Training.Evaluator

  @provider_env_key "TRINITY_ENABLE_PROVIDER_TESTS"
  @provider_budget_env_key "TRINITY_PROVIDER_BUDGET_USD"
  @provider_api_env_keys ["OPENAI_API_KEY", "OPENAI_API_KEY_ENV", "TRINITY_OPENAI_API_KEY"]

  test "evaluates tasks with complete-trajectory hooks" do
    model = :dummy_model
    candidate_model_state = :candidate_state
    metadata = %{candidate_id: 1}

    tasks = [
      %{id: :task_alpha, messages: [%{"role" => "user", "content" => "route"}]},
      %{id: :task_beta, messages: [%{"role" => "user", "content" => "verify"}]}
    ]

    run_candidate = fn _model, _candidate_model_state, task, _opts, _slm_context ->
      {:ok, %{status: :ok, task_id: task.id, response: "done"}}
    end

    reward_fn = fn _task, %{status: :ok, task_id: task_id} ->
      reward = if task_id == :task_alpha, do: 1.0, else: 0.0
      {:ok, reward}
    end

    assert {:ok, [reward_alpha, reward_beta]} =
             Evaluator.evaluate_candidate(
               candidate_model_state,
               metadata,
               tasks: tasks,
               model: model,
               reward_fn: reward_fn,
               slm_context: %{},
               run_candidate: run_candidate
             )

    assert reward_alpha == 1.0
    assert reward_beta == 0.0
  end

  test "returns trajectory errors from evaluator boundary" do
    model = :dummy_model
    candidate_model_state = :candidate_state

    tasks = [%{id: :task_alpha, messages: [%{"role" => "user", "content" => "route"}]}]

    run_candidate = fn _model, _candidate_model_state, _task, _opts, _slm_context ->
      {:error, :route_failed}
    end

    reward_fn = fn _task, _trajectory ->
      {:ok, 1.0}
    end

    assert {:error, {:trajectory_error, :route_failed}} =
             Evaluator.evaluate_candidate(
               candidate_model_state,
               %{},
               tasks: tasks,
               model: model,
               reward_fn: reward_fn,
               slm_context: %{},
               run_candidate: run_candidate
             )
  end

  test "requires explicit provider-gated mode for default trajectory evaluation" do
    model = :dummy_model
    candidate_model_state = :candidate_state

    tasks = [%{id: :task_alpha, messages: [%{"role" => "user", "content" => "route"}]}]

    reward_fn = fn _task, _trajectory ->
      {:ok, +0.0}
    end

    assert {:error, {:trajectory_error, :provider_backed_training_disabled}} =
             Evaluator.evaluate_candidate(
               candidate_model_state,
               %{},
               tasks: tasks,
               model: model,
               reward_fn: reward_fn,
               slm_context: %{},
               run_opts: []
             )
  end

  @tag :provider_training
  test "provider-backed evaluation requires explicit, budgeted credentials when enabled" do
    case provider_training_config() do
      {:ok, conf} ->
        model = :dummy_model
        candidate_model_state = :candidate_state

        tasks = [%{id: :task_alpha, messages: [%{"role" => "user", "content" => "route"}]}]

        reward_fn = fn
          _task, %{status: :ok} ->
            {:ok, 1.0}

          _task, _trajectory ->
            {:ok, +0.0}
        end

        assert {:ok, [reward_value]} =
                 Evaluator.evaluate_candidate(
                   candidate_model_state,
                   %{},
                   tasks: tasks,
                   model: model,
                   reward_fn: reward_fn,
                   slm_context: %{},
                   run_opts: [
                     provider_enabled: true,
                     provider_budget_usd: conf.budget_usd,
                     provider_credentials: %{openai_api_key: conf.api_key}
                   ]
                 )

        assert reward_value == 1.0

      {:skip, _reason} ->
        assert true
    end
  end

  test "requires explicit provider budget and credentials before trajectory execution" do
    model = :dummy_model
    candidate_model_state = :candidate_state

    tasks = [%{id: :task_alpha, messages: [%{"role" => "user", "content" => "route"}]}]

    reward_fn = fn _task, _trajectory ->
      {:ok, +0.0}
    end

    assert {:error, {:trajectory_error, :provider_budget_missing}} =
             Evaluator.evaluate_candidate(
               candidate_model_state,
               %{},
               tasks: tasks,
               model: model,
               reward_fn: reward_fn,
               slm_context: %{},
               run_opts: [provider_enabled: true, provider_credentials: %{openai_api_key: "x"}]
             )

    assert {:error, {:trajectory_error, :provider_credentials_missing}} =
             Evaluator.evaluate_candidate(
               candidate_model_state,
               %{},
               tasks: tasks,
               model: model,
               reward_fn: reward_fn,
               slm_context: %{},
               run_opts: [
                 provider_enabled: true,
                 provider_budget_usd: 1.0
               ]
             )
  end

  defp provider_training_config do
    with "1" <- System.get_env(@provider_env_key),
         {:ok, budget_usd} <- parse_provider_budget(),
         {:ok, api_key} <- parse_provider_api_key() do
      {:ok, %{budget_usd: budget_usd, api_key: api_key}}
    else
      nil -> {:skip, :disabled}
      "0" -> {:skip, :disabled}
      {:error, reason} -> {:skip, reason}
      _ -> {:skip, :disabled}
    end
  end

  defp parse_provider_budget do
    case System.get_env(@provider_budget_env_key) do
      nil ->
        {:error, :budget_missing}

      value ->
        case Float.parse(String.trim(value)) do
          {budget, ""} when budget > 0.0 ->
            {:ok, budget}

          _ ->
            {:error, :invalid_budget}
        end
    end
  end

  defp parse_provider_api_key do
    case Enum.find_value(@provider_api_env_keys, &System.get_env/1) do
      nil -> {:error, :api_key_missing}
      key when is_binary(key) and byte_size(key) > 0 -> {:ok, key}
      _ -> {:error, :api_key_invalid}
    end
  end
end
