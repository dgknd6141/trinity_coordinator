defmodule TrinityCoordinator.Training.SepCMAESCodecTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{
    CoordinationHead,
    Training.SepCMAES.Codec
  }

  test "flattens and restores the same model state deterministically" do
    model = CoordinationHead.build_model(8, 3, 2)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 8}, :f32), Axon.ModelState.empty())

    {flat_one, metadata_one} = Codec.flatten_model_state(params)
    {flat_two, metadata_two} = Codec.flatten_model_state(params)

    assert Nx.shape(flat_one) == Nx.shape(flat_two)
    assert metadata_one == metadata_two

    restored = Codec.unflatten_model_state(flat_one, metadata_one, params)

    assert Nx.to_flat_list(params.data["routing_head"]["bias"]) ==
             Nx.to_flat_list(restored.data["routing_head"]["bias"])

    assert Nx.to_flat_list(params.data["routing_head"]["kernel"]) ==
             Nx.to_flat_list(restored.data["routing_head"]["kernel"])

    input = Nx.broadcast(0.25, {1, 8})
    expected = CoordinationHead.route(model, params, input, 3, 2)
    observed = CoordinationHead.route(model, restored, input, 3, 2)

    assert expected == observed
  end

  test "unflatten validates size against flattened vector" do
    model = CoordinationHead.build_model(8, 3, 2)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 8}, :f32), Axon.ModelState.empty())

    {_flat, metadata} = Codec.flatten_model_state(params)
    truncated = Nx.iota({5}, type: :f32)

    assert_raise ArgumentError,
                 fn ->
                   Codec.unflatten_model_state(truncated, metadata, params)
                 end
  end
end
