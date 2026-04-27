defmodule TrinityCoordinator.Training.EvaluatorMockTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.{CoordinationHead, Training.Evaluator}

  test "default trajectory evaluation can run with mocked provider and extractor hooks" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 0.0, 10.0])

    tasks = [
      %{id: :mock_task, messages: [%{"role" => "user", "content" => "verify"}]}
    ]

    extractor_fn = fn _messages, _slm_context ->
      {:ok,
       %{
         vector: Nx.tensor([[0.0, 0.0, 0.0, 0.0]], type: :f32),
         vector_shape: {1, input_dim},
         hidden_state_shape: {1, 2, input_dim},
         input_shapes: %{}
       }}
    end

    mock_agent_fn = fn :verifier, _messages, _metadata ->
      {:ok, "ACCEPT: mocked evaluator trajectory accepted."}
    end

    reward_fn = fn _task, %{status: :ok, response: response} ->
      if String.starts_with?(response, "ACCEPT"), do: {:ok, 1.0}, else: {:ok, 0.0}
    end

    assert {:ok, [1.0]} =
             Evaluator.evaluate_candidate(
               params,
               %{candidate_id: 0},
               tasks: tasks,
               model: model,
               reward_fn: reward_fn,
               slm_context: :mock_context,
               run_opts: [
                 max_turns: 5,
                 orchestrator_opts: [
                   num_agents: num_agents,
                   num_roles: num_roles,
                   extractor_fn: extractor_fn,
                   mock_agent_fn: mock_agent_fn,
                   provider_pool: :mock
                 ]
               ]
             )
  end

  defp force_role_bias(%Axon.ModelState{} = params, values) do
    bias = Nx.tensor(values, type: :f32)
    kernel = Nx.broadcast(0.0, {4, length(values)})

    put_in(params.data["routing_head"], %{
      "kernel" => kernel,
      "bias" => bias
    })
  end
end
