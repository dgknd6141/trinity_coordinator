defmodule Mix.Tasks.Trinity.Hitl.MockLoop do
  @moduledoc """
  HITL gate: run the adapted coordinator through the orchestrator with mocked LLM calls.

      XLA_TARGET=cuda12 mix trinity.hitl.mock_loop --trace-out tmp/trinity_mock_trace.jsonl

  This intentionally performs no live provider calls. By default it writes a
  hash-redacted JSONL trace and validates that the trace file exists.
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Orchestrator, StateManager}
  alias TrinityCoordinator.Sakana.Coordinator

  @shortdoc "HITL adapted coordinator mock-orchestrator check"
  @default_trace_path "tmp/trinity_mock_trace.jsonl"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    args
    |> parse_args!()
    |> prepare_trace!()
    |> run_mock_orchestrator!()
    |> report_result!()

    HITL.pass("TRINITY HITL MOCK ORCHESTRATOR LOOP")
  end

  defp parse_args!(args) do
    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [trace_out: :string, trace_content: :string, run_id: :string]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    %{
      trace_path: Keyword.get(opts, :trace_out, @default_trace_path),
      trace_content: parse_trace_content(Keyword.get(opts, :trace_content, "hash")),
      run_id: Keyword.get(opts, :run_id, "hitl_mock")
    }
  end

  defp prepare_trace!(context) do
    File.mkdir_p!(Path.dirname(context.trace_path))
    File.rm(context.trace_path)

    HITL.banner("TRINITY HITL MOCK ORCHESTRATOR LOOP")
    HITL.kv("Trace path", context.trace_path)
    HITL.kv("Trace content", context.trace_content)

    context
  end

  defp run_mock_orchestrator!(context) do
    {:ok, coordinator} = Coordinator.load()
    {:ok, pid} = start_state_manager!()
    turn_counter = :counters.new(1, [])

    result =
      Orchestrator.run_loop(
        pid,
        coordinator.routing_model,
        coordinator.routing_params,
        orchestrator_opts(coordinator, context, turn_counter)
      )

    Map.merge(context, %{result: result, turns: :counters.get(turn_counter, 1)})
  end

  defp start_state_manager! do
    StateManager.start_link([
      %{role: "user", content: "Solve a tiny arithmetic task: compute 6 * 7."}
    ])
  end

  defp orchestrator_opts(coordinator, context, turn_counter) do
    [
      max_turns: 5,
      num_agents: coordinator.num_agents,
      num_roles: coordinator.num_roles,
      slm_context: {coordinator.model_info, coordinator.tokenizer},
      mock_agent_fn: mock_agent_fn(turn_counter),
      provider_pool: :mock,
      trace: [
        enabled: true,
        sink: {:jsonl, context.trace_path},
        run_id: context.run_id,
        content: context.trace_content
      ]
    ]
  end

  defp mock_agent_fn(turn_counter) do
    fn role, messages, metadata ->
      record_mock_turn!(turn_counter, role, messages, metadata)
      {:ok, mock_response(role)}
    end
  end

  defp record_mock_turn!(turn_counter, role, messages, metadata) do
    :counters.add(turn_counter, 1, 1)
    turn = :counters.get(turn_counter, 1)

    HITL.kv("Mock turn #{turn}", %{
      role: role,
      agent_id: metadata.agent_id,
      messages: length(messages)
    })
  end

  defp mock_response(:verifier),
    do: "ACCEPT: The current answer is complete enough for the smoke test."

  defp mock_response(:thinker), do: "Plan: multiply 6 by 7 and ask a verifier to check it."
  defp mock_response(:worker), do: "Result: 6 * 7 = 42."
  defp mock_response(_role), do: "Proceed."

  defp report_result!(context) do
    HITL.kv("Mock turns executed", context.turns)
    HITL.kv("Loop result", context.result)
    HITL.kv("Trace path", context.trace_path)

    validate_turns!(context.turns)
    validate_trace_file!(context.trace_path)
    validate_trace_events!(context.trace_path)
    report_termination!(context.result)
  end

  defp validate_turns!(turns) when turns > 0, do: :ok
  defp validate_turns!(_turns), do: raise("mock loop did not execute any provider turn")

  defp validate_trace_file!(trace_path) do
    unless File.exists?(trace_path) do
      raise "trace file was not written: #{trace_path}"
    end
  end

  defp validate_trace_events!(trace_path) do
    trace_events = trace_event_names(trace_path)
    HITL.kv("Trace events", trace_events)

    unless Enum.member?(trace_events, "slm_extracted") and
             Enum.member?(trace_events, "route_selected") do
      raise "trace file did not include extraction and route events"
    end
  end

  defp report_termination!({:ok, _response}), do: HITL.kv("Termination", "Verifier ACCEPT")

  defp report_termination!({:error, :max_turns_reached}) do
    HITL.kv("Termination", "max_turns_reached after successful mock dispatch")
  end

  defp report_termination!({:error, reason}) do
    raise "mock loop failed: #{inspect(reason)}"
  end

  defp parse_trace_content("full"), do: :full
  defp parse_trace_content(:full), do: :full
  defp parse_trace_content(_), do: :hash

  defp trace_event_names(path) do
    path
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&trace_event_name/1)
    |> Enum.reject(&is_nil/1)
  end

  defp trace_event_name(line) do
    case Jason.decode(line) do
      {:ok, %{"event" => event}} -> event
      _ -> nil
    end
  end
end
