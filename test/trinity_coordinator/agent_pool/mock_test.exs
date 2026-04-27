defmodule TrinityCoordinator.AgentPoolMockTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.AgentPool

  test "mock provider returns fixed response without credentials" do
    spec = %{id: 0, provider: :mock, model: "mock-agent"}

    assert {:ok, "fixed"} =
             AgentPool.call_agent_with_spec(spec, [%{role: "user", content: "hello"}],
               mock_response: "fixed"
             )
  end

  test "mock provider can dispatch from response map" do
    spec = %{id: 2, provider: :mock, model: "mock-agent-2"}

    assert {:ok, "agent two"} =
             AgentPool.call_agent_with_spec(spec, [%{role: "user", content: "hello"}],
               mock_responses: %{2 => "agent two"}
             )
  end

  test "mock provider has deterministic fallback text" do
    spec = %{id: 1, provider: :mock, model: "mock-agent-1"}

    assert {:ok, response} =
             AgentPool.call_agent_with_spec(spec, [%{role: "user", content: "hello"}])

    assert response =~ "MOCK agent=1"
    assert response =~ "hello"
  end
end
