defmodule TrinityCoordinator.BenchmarkRoutingTest do
  use ExUnit.Case

  alias TrinityCoordinator.{
    Benchmark.Dataset,
    Benchmark.Routing,
    CoordinationHead
  }

  test "computes route accuracy and confusion on real forward passes" do
    num_agents = 3
    num_roles = 3
    model = CoordinationHead.build_model(4, num_agents, num_roles)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.broadcast(0.1, {1, 4}), Axon.ModelState.empty())

    features =
      Nx.stack([
        Nx.iota({1, 4}, type: :f32),
        Nx.add(Nx.iota({1, 4}, type: :f32), 1.0),
        Nx.add(Nx.iota({1, 4}, type: :f32), 2.0)
      ])
      |> Nx.reshape({3, 4})

    feature_rows = Nx.to_list(features)

    cases =
      feature_rows
      |> Enum.with_index()
      |> Enum.map(fn {vector, index} ->
        route =
          CoordinationHead.route(
            model,
            params,
            Nx.tensor([vector], type: :f32),
            num_agents,
            num_roles
          )

        %Dataset{
          id: "case-#{index}",
          family: "synthetic",
          messages: [%{role: "user", content: "x"}],
          expected_agent: route.agent_id,
          expected_role: route.role_id
        }
      end)

    assert {:ok, metrics} =
             Routing.run(cases, features, model, params,
               num_agents: num_agents,
               num_roles: num_roles
             )

    assert metrics.considered == 3
    assert metrics.skipped == 0
    assert metrics.agent_accuracy == 1.0
    assert metrics.role_accuracy == 1.0
    assert metrics.joint_accuracy == 1.0
  end

  test "rejects feature count mismatch" do
    cases = [
      %Dataset{id: "x", family: "a", messages: [], expected_agent: 1, expected_role: 2}
    ]

    features = Nx.broadcast(0.0, {2, 4})

    assert {:error, :feature_count_mismatch} =
             Routing.run(
               cases,
               features,
               CoordinationHead.build_model(4),
               Nx.template({1, 4}, :f32)
             )
  end
end
