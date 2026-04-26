defmodule TrinityCoordinator.OrchestratorTest do
  use ExUnit.Case
  alias TrinityCoordinator.{CoordinationHead, Extractor, Orchestrator, Runtime, StateManager}

  test "requires an SLM context before routing" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())

    assert Orchestrator.run_loop(pid, model, params, max_turns: 3) ==
             {:error, :missing_slm_context}
  end

  @tag :integration
  test "runs real SLM extraction and router forward pass before provider dispatch" do
    Runtime.put_cuda_backend!()

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(32, 5, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())

    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(
               {:hf, "hf-internal-testing/tiny-random-gpt2"},
               Bumblebee.Text.Gpt2,
               :base
             )

    result =
      Orchestrator.run_loop(
        pid,
        model,
        params,
        max_turns: 1,
        num_agents: 5,
        num_roles: 3,
        slm_context: {model_info, tokenizer},
        agent_pool_opts: [openai_api_key: ""]
      )

    assert result == {:error, :missing_openai_api_key}
  end

  @tag :integration
  test "emits trace events for real extraction and provider selection errors" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(32, 5, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())

    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(
               {:hf, "hf-internal-testing/tiny-random-gpt2"},
               Bumblebee.Text.Gpt2,
               :base
             )

    trace_path =
      Path.join(
        System.tmp_dir!(),
        "trinity_trace_run_#{System.unique_integer([:positive])}.jsonl"
      )

    File.rm(trace_path)

    assert {:error, :missing_openai_api_key} =
             Orchestrator.run_loop(
               pid,
               model,
               params,
               max_turns: 1,
               num_agents: 5,
               num_roles: 3,
               slm_context: {model_info, tokenizer},
               agent_pool_opts: [openai_api_key: ""],
               trace: [enabled: true, sink: {:jsonl, trace_path}]
             )

    lines = File.read!(trace_path) |> String.split("\n", trim: true)
    assert length(lines) >= 5

    events =
      lines
      |> Enum.map(&Jason.decode!/1)
      |> Enum.map(& &1["event"])

    assert events == [
             "run_started",
             "turn_started",
             "slm_extracted",
             "route_selected",
             "provider_called",
             "provider_called",
             "run_failed"
           ]

    # Ensure backend metadata and redaction defaults are present.
    parsed = lines |> Enum.map(&Jason.decode!/1)

    assert Enum.any?(parsed, fn event ->
             event["event"] == "slm_extracted" and is_binary(event["vector_backend"])
           end)

    assert Enum.any?(parsed, fn event ->
             event["event"] == "provider_called" and event["status"] == "error"
           end)
  end
end
