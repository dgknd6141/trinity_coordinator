defmodule TrinityCoordinator.ProviderPoolMockTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.{AgentPool, ProviderPool}
  alias TrinityCoordinator.ProviderPool.Spec

  test "default mock pool contains seven mock agents" do
    pool = ProviderPool.fetch!(:mock)

    assert length(pool) == 7
    assert Enum.all?(pool, &(&1.provider == :mock))
    assert ProviderPool.size(:mock) == 7
  end

  test "normalizes explicit mock specs" do
    specs =
      Spec.normalize!([
        [id: 0, name: :mock_zero, provider: :mock, model: "mock-zero"],
        [id: 1, name: :mock_one, provider: "mock", model: "mock-one"]
      ])

    assert Enum.map(specs, & &1.provider) == [:mock, :mock]
    assert :ok == Spec.validate(specs)
  end

  test "AgentPool fetches and calls mock provider by pool name" do
    assert {:ok, response} =
             AgentPool.call_agent(0, [%{role: "user", content: "route this"}],
               provider_pool: :mock,
               mock_response: "mocked"
             )

    assert response == "mocked"
  end
end
