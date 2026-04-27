defmodule Mix.Tasks.Trinity.Hitl.MockLoop do
  @moduledoc """
  HITL gate: run the adapted coordinator through the orchestrator with mocked LLM calls.

      XLA_TARGET=cuda12 mix trinity.hitl.mock_loop

  This intentionally performs no live provider calls.
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Orchestrator, StateManager}
  alias TrinityCoordinator.Sakana.Coordinator

  @shortdoc "HITL adapted coordinator mock-orchestrator check"

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    HITL.banner("TRINITY HITL MOCK ORCHESTRATOR LOOP")
    {:ok, coordinator} = Coordinator.load()

    {:ok, pid} =
      StateManager.start_link([
        %{role: "user", content: "Solve a tiny arithmetic task: compute 6 * 7."}
      ])

    turn_counter = :counters.new(1, [])

    mock_agent_fn = fn role, messages, metadata ->
      :counters.add(turn_counter, 1, 1)
      turn = :counters.get(turn_counter, 1)

      HITL.kv("Mock turn #{turn}", %{
        role: role,
        agent_id: metadata.agent_id,
        messages: length(messages)
      })

      case role do
        :verifier -> {:ok, "ACCEPT: The current answer is complete enough for the smoke test."}
        :thinker -> {:ok, "Plan: multiply 6 by 7 and ask a verifier to check it."}
        :worker -> {:ok, "Result: 6 * 7 = 42."}
        _ -> {:ok, "Proceed."}
      end
    end

    result =
      Orchestrator.run_loop(
        pid,
        coordinator.routing_model,
        coordinator.routing_params,
        max_turns: 5,
        num_agents: coordinator.num_agents,
        num_roles: coordinator.num_roles,
        slm_context: {coordinator.model_info, coordinator.tokenizer},
        mock_agent_fn: mock_agent_fn,
        provider_pool: :mock
      )

    turns = :counters.get(turn_counter, 1)
    HITL.kv("Mock turns executed", turns)
    HITL.kv("Loop result", result)

    unless turns > 0 do
      raise "mock loop did not execute any provider turn"
    end

    case result do
      {:ok, _response} ->
        HITL.kv("Termination", "Verifier ACCEPT")

      {:error, :max_turns_reached} ->
        HITL.kv("Termination", "max_turns_reached after successful mock dispatch")

      {:error, reason} ->
        raise "mock loop failed: #{inspect(reason)}"
    end

    HITL.pass("TRINITY HITL MOCK ORCHESTRATOR LOOP")
  end
end
