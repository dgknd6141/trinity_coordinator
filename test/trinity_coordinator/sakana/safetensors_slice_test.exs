defmodule TrinityCoordinator.Sakana.SafetensorsSliceTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Sakana.SafetensorsSlice

  @tag :tmp_dir
  test "reads bounded row slices from a lazy rank-2 safetensors tensor", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "matrix.safetensors")
    matrix = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], type: :f32)
    Safetensors.write!(path, %{"matrix" => matrix})

    lazy = path |> Safetensors.read!(lazy: true) |> Map.fetch!("matrix")

    slice = SafetensorsSlice.row_slice!(lazy, 1, 2)

    assert Nx.shape(slice) == {2, 3}
    assert Nx.to_flat_list(slice) == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  end

  @tag :tmp_dir
  test "preserves low-precision element layout when slicing", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "bf16_matrix.safetensors")
    matrix = Nx.tensor([[1.25, 2.5], [3.75, 4.5], [5.25, 6.5]], type: :bf16)
    Safetensors.write!(path, %{"matrix" => matrix})

    lazy = path |> Safetensors.read!(lazy: true) |> Map.fetch!("matrix")

    slice = SafetensorsSlice.row_slice!(lazy, 2, 1)

    assert Nx.type(slice) == {:bf, 16}
    assert slice |> Nx.as_type(:f32) |> Nx.to_flat_list() == [5.25, 6.5]
  end

  @tag :tmp_dir
  test "rejects out-of-bounds or non-rank-2 slices", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "vectors.safetensors")

    Safetensors.write!(path, %{
      "matrix" => Nx.broadcast(0.0, {2, 2}),
      "vector" => Nx.broadcast(0.0, {2})
    })

    tensors = Safetensors.read!(path, lazy: true)

    assert_raise ArgumentError, fn ->
      SafetensorsSlice.row_slice!(Map.fetch!(tensors, "matrix"), 1, 2)
    end

    assert_raise ArgumentError, fn ->
      SafetensorsSlice.row_slice!(Map.fetch!(tensors, "vector"), 0, 1)
    end
  end
end
