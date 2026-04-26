defmodule TrinityCoordinator.BenchmarkTurnBudgetTest do
  use ExUnit.Case

  alias TrinityCoordinator.{
    Benchmark.Dataset,
    Benchmark.TurnBudget,
    CoordinationHead
  }

  test "simulates turn-budget outcomes from routing decisions" do
    num_agents = 3
    num_roles = 3
    model = CoordinationHead.build_model(4, num_agents, num_roles)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.broadcast(0.1, {1, 4}), Axon.ModelState.empty())

    cases = [
      %Dataset{
        id: "case-accepted",
        family: "math",
        messages: [%{role: "user", content: "x"}],
        expected_agent: 0,
        expected_role: 2
      },
      %Dataset{
        id: "case-revised",
        family: "code",
        messages: [%{role: "user", content: "y"}],
        expected_agent: 0,
        expected_role: 1
      }
    ]

    features =
      Nx.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
      ])

    assert {:ok, result} =
             TurnBudget.run(
               cases,
               %{model_info: nil, tokenizer: nil},
               features,
               model,
               params,
               max_turns: 2,
               include_trace: false,
               num_agents: num_agents,
               num_roles: num_roles
             )

    assert result.summary.total_cases == 2
    assert result.summary.max_turn_hit_rate >= 0.0
  end
end
