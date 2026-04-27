defmodule TrinityCoordinator.SakanaHeadTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.{CoordinationHead, Runtime}
  alias TrinityCoordinator.Sakana.Head

  test "builds standalone routing head params from Sakana layout weights" do
    head_weights = Nx.iota({4, 8}, type: :f32)

    assert {:ok, head_state} = Head.build_routing_state(head_weights, num_roles: 3)

    assert head_state.hidden_size == 8
    assert head_state.output_count == 4
    assert head_state.num_agents == 1
    assert head_state.num_roles == 3
    assert Nx.shape(head_state.params.data["routing_head"]["kernel"]) == {8, 4}
    assert Nx.shape(head_state.params.data["routing_head"]["bias"]) == {4}

    route =
      CoordinationHead.route(
        head_state.model,
        head_state.params,
        Nx.broadcast(0.1, {1, 8}),
        1,
        3
      )

    assert Nx.shape(route.logits) == {1, 4}
  end

  @tag :integration
  test "can build standalone routing head on CUDA" do
    Runtime.put_cuda_backend!()

    head_weights =
      Nx.iota({4, 8}, type: :f32) |> Nx.backend_transfer({EXLA.Backend, client: :cuda})

    assert {:ok, head_state} =
             Head.build_routing_state(head_weights,
               num_roles: 3,
               backend: {EXLA.Backend, client: :cuda}
             )

    assert Runtime.tensor_backend(head_state.params.data["routing_head"]["kernel"]) =~
             "EXLA.Backend<cuda:"
  end
end
