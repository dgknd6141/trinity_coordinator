defmodule TrinityCoordinator.OrchestratorMockTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.{CoordinationHead, Orchestrator, StateManager}

  test "runs verifier ACCEPT path through mock provider without live credentials" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 0.0, 10.0])

    extractor_fn = fn _messages, _slm_context ->
      {:ok,
       %{
         vector: Nx.tensor([[0.0, 0.0, 0.0, 0.0]], type: :f32),
         vector_shape: {1, input_dim},
         hidden_state_shape: {1, 2, input_dim},
         input_shapes: %{"input_ids" => {1, 2}}
       }}
    end

    mock_agent_fn = fn :verifier, messages, metadata ->
      assert metadata.agent_id == 0
      assert hd(messages).role == "system"
      {:ok, "ACCEPT: smoke-test verifier accepted."}
    end

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "check this"}])

    assert {:ok, "ACCEPT: smoke-test verifier accepted."} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 3,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock
             )

    messages = StateManager.get_messages(pid)
    assert List.last(messages).content =~ "ACCEPT"
  end

  test "mock worker path executes provider turn and then reaches max turns" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 10.0, 0.0])

    extractor_fn = fn _messages ->
      %{
        vector: Nx.tensor([[0.0, 0.0, 0.0, 0.0]], type: :f32),
        vector_shape: {1, input_dim},
        hidden_state_shape: {1, 2, input_dim},
        input_shapes: %{}
      }
    end

    counter = :counters.new(1, [])

    mock_agent_fn = fn :worker, _messages ->
      :counters.add(counter, 1, 1)
      {:ok, "Result: one worker turn."}
    end

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "do work"}])

    assert {:error, :max_turns_reached} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 1,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock
             )

    assert :counters.get(counter, 1) == 1
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
