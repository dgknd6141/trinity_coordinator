defmodule TrinityCoordinator.CoordinationHeadTest do
  use ExUnit.Case
  alias TrinityCoordinator.CoordinationHead

  test "builds model and returns bounded route details from a real Axon forward pass" do
    input_dim = 10
    num_agents = 3
    num_roles = 2

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    tensor = Nx.broadcast(0.5, {1, input_dim})

    route = CoordinationHead.route(model, params, tensor, num_agents, num_roles)

    assert Nx.shape(route.logits) == {1, num_agents + num_roles}
    assert Nx.shape(route.agent_logits) == {num_agents}
    assert Nx.shape(route.role_logits) == {num_roles}

    assert is_integer(route.agent_id)
    assert route.agent_id >= 0 and route.agent_id < num_agents

    assert is_integer(route.role_id)
    assert route.role_id >= 0 and route.role_id < num_roles
  end

  test "builds combined one-hot labels for agent and role supervision" do
    labels = CoordinationHead.build_labels([0, 2], [1, 0], 3, 2)

    assert Nx.shape(labels) == {2, 5}
    assert Nx.to_flat_list(labels) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
  end

  @tag :integration
  test "trains the real Axon coordination head on tensors and routes with trained parameters" do
    input_dim = 4
    num_agents = 2
    num_roles = 2

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)

    features =
      Nx.tensor(
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ],
        type: :f32
      )

    labels = CoordinationHead.build_labels([0, 1, 0, 1], [0, 0, 1, 1], num_agents, num_roles)

    trained_state =
      CoordinationHead.train_supervised(model, features, labels,
        num_agents: num_agents,
        num_roles: num_roles,
        epochs: 40,
        learning_rate: 0.1,
        compiler: EXLA
      )

    route =
      CoordinationHead.route(
        model,
        trained_state,
        Nx.slice(features, [0, 0], [1, input_dim]),
        num_agents,
        num_roles
      )

    assert route.agent_id == 0
    assert route.role_id == 0
    assert inspect(route.logits) =~ "EXLA.Backend<"
  end

  test "supports linear variant as the default with unchanged semantics" do
    model = CoordinationHead.build_model(10, 4, 3)
    metadata = CoordinationHead.variant_metadata(10, 4, 3)

    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())

    route = CoordinationHead.route(model, params, Nx.iota({1, 10}, type: :f32), 4, 3)

    assert is_integer(route.agent_id)
    assert metadata.head == :linear
    assert metadata.parameter_count == 10 * (4 + 3) + (4 + 3)
    assert metadata.input_partitions == [{0, 10}]
    assert metadata.output_partitions == [{0, 7}]
  end

  test "supports unknown head variants with a clear error" do
    assert_raise ArgumentError, fn ->
      CoordinationHead.build_model(10, 4, 3, head: :unknown)
    end
  end

  test "builds block-diagonal models with metadata and route bounds" do
    input_dim = 7
    num_agents = 3
    num_roles = 2
    blocks = 3

    model =
      CoordinationHead.build_model(input_dim, num_agents, num_roles,
        head: :block_diagonal,
        blocks: blocks
      )

    metadata =
      CoordinationHead.variant_metadata(input_dim, num_agents, num_roles,
        head: :block_diagonal,
        blocks: blocks
      )

    assert metadata.head == :block_diagonal
    assert metadata.blocks == blocks
    assert metadata.input_partitions == [{0, 3}, {3, 2}, {5, 2}]
    assert metadata.output_partitions == [{0, 2}, {2, 2}, {4, 1}]

    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    route =
      CoordinationHead.route(
        model,
        params,
        Nx.iota({1, input_dim}, type: :f32),
        num_agents,
        num_roles
      )

    assert route.agent_id in 0..(num_agents - 1)
    assert route.role_id in 0..(num_roles - 1)
    expected_params = 17
    assert metadata.parameter_count == expected_params
  end

  test "supports sparse models with fixed projection width and metadata" do
    input_dim = 10
    num_agents = 2
    num_roles = 3

    model =
      CoordinationHead.build_model(input_dim, num_agents, num_roles,
        head: :sparse,
        sparse_k: 4
      )

    metadata =
      CoordinationHead.variant_metadata(input_dim, num_agents, num_roles,
        head: :sparse,
        sparse_k: 4
      )

    assert metadata.head == :sparse
    assert metadata.effective_sparse_k == 4

    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    route =
      CoordinationHead.route(
        model,
        params,
        Nx.iota({1, input_dim}, type: :f32),
        num_agents,
        num_roles
      )

    assert route.agent_id in 0..(num_agents - 1)
    assert route.role_id in 0..(num_roles - 1)
    assert metadata.parameter_count == 4 * (num_agents + num_roles) + (num_agents + num_roles)
  end

  test "validates sparse_k to fit input width" do
    assert_raise ArgumentError, fn ->
      CoordinationHead.build_model(6, 2, 3, head: :sparse, sparse_k: 8)
    end
  end
end
