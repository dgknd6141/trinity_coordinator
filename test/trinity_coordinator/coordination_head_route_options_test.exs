defmodule TrinityCoordinator.CoordinationHeadRouteOptionsTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.CoordinationHead

  test "route/6 default options match route/5" do
    model = CoordinationHead.build_model(4, 2, 3)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
    input = Nx.iota({1, 4}, type: :f32)

    a = CoordinationHead.route(model, params, input, 2, 3)
    b = CoordinationHead.route(model, params, input, 2, 3, [])

    assert a.agent_id == b.agent_id
    assert a.role_id == b.role_id
    assert Nx.all_close(a.logits, b.logits)
    assert b.agent_selection_mode == :argmax
    assert b.role_selection_mode == :argmax
  end

  test "softmax returns probability tensors" do
    model = CoordinationHead.build_model(4, 2, 3)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
    input = Nx.tensor([[0.1, 0.2, 0.3, 0.4]], type: :f32)

    route =
      CoordinationHead.route(model, params, input, 2, 3,
        agent_selection: :softmax,
        role_selection: :softmax,
        temperature: 1.0
      )

    assert route.agent_selection_mode == :softmax
    assert route.role_selection_mode == :softmax
    assert Nx.shape(route.agent_probabilities) == {2}
    assert Nx.shape(route.role_probabilities) == {3}
    assert Nx.to_number(Nx.sum(route.agent_probabilities)) > 0.999
    assert Nx.to_number(Nx.sum(route.role_probabilities)) > 0.999
  end

  test "string selection modes are bounded to known route modes" do
    model = CoordinationHead.build_model(4, 2, 3)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
    input = Nx.tensor([[0.1, 0.2, 0.3, 0.4]], type: :f32)

    route =
      CoordinationHead.route(model, params, input, 2, 3,
        agent_selection: "softmax-argmax",
        role_selection: "softmax",
        temperature: 1.0
      )

    assert route.agent_selection_mode == :softmax
    assert route.role_selection_mode == :softmax

    error =
      assert_raise ArgumentError, fn ->
        CoordinationHead.route(model, params, input, 2, 3,
          agent_selection: "external-runtime-mode"
        )
      end

    assert String.contains?(Exception.message(error), "agent_selection must be")
    assert String.contains?(Exception.message(error), "external-runtime-mode")
  end

  test "seeded sampling is deterministic for the same logits" do
    model = CoordinationHead.build_model(4, 2, 3)
    {init_fn, _} = Axon.build(model)
    params = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
    input = Nx.iota({1, 4}, type: :f32)

    a =
      CoordinationHead.route(model, params, input, 2, 3,
        agent_selection: :sample,
        role_selection: :sample,
        seed: {1, 2, 3}
      )

    b =
      CoordinationHead.route(model, params, input, 2, 3,
        agent_selection: :sample,
        role_selection: :sample,
        seed: {1, 2, 3}
      )

    assert a.agent_id == b.agent_id
    assert a.role_id == b.role_id
    assert Nx.shape(a.agent_probabilities) == {2}
    assert Nx.shape(a.role_probabilities) == {3}
  end
end
