defmodule TrinityCoordinator.Training.SepCMAES.Codec do
  @moduledoc """
  Parameter codec helpers for sep-CMA-ES.

  The codec flattens an `Axon.ModelState` into a single vector and restores it
  deterministically from a metadata description of tensor paths and shapes.
  """

  @type path :: [binary() | atom() | integer()]
  @type metadata_entry :: %{
          required(:path) => path(),
          required(:shape) => tuple(),
          required(:size) => non_neg_integer(),
          required(:type) => atom(),
          required(:offset) => non_neg_integer()
        }
  @type codec_metadata :: [metadata_entry()]

  @doc """
  Flattens all tensors in an `Axon.ModelState` into a single 1-D tensor.

  Returns `{flat_vector, metadata}` where metadata preserves deterministic tensor
  path and shape information needed for round-tripping.
  """
  @spec flatten_model_state(model_state :: struct()) :: {Nx.Tensor.t(), codec_metadata()}
  def flatten_model_state(%Axon.ModelState{} = model_state) do
    entries = collect_tensors(model_state.data, [])

    {slices, metadata, _} =
      Enum.reduce(entries, {[], [], 0}, fn {path, tensor}, {slices, metadata, offset} ->
        shape = Nx.shape(tensor)
        size = Nx.size(tensor)
        flat = Nx.reshape(tensor, {size})

        entry = %{
          path: path,
          shape: shape,
          size: size,
          type: Nx.type(tensor),
          offset: offset
        }

        {[flat | slices], [entry | metadata], offset + size}
      end)

    flat = Nx.concatenate(Enum.reverse(slices), axis: 0)

    {flat, Enum.reverse(metadata)}
  end

  @doc """
  Rebuilds an `Axon.ModelState` from a flattened vector and metadata.
  """
  @spec unflatten_model_state(Nx.Tensor.t(), codec_metadata(), struct()) :: struct()
  def unflatten_model_state(flat_vector, metadata, %Axon.ModelState{} = template_state) do
    flat_size = Nx.size(flat_vector)
    expected = Enum.reduce(metadata, 0, &(&1.size + &2))

    if flat_size != expected do
      raise ArgumentError,
            "flattened parameter size mismatch: got #{flat_size}, expected #{expected}"
    end

    data =
      Enum.reduce(metadata, template_state.data, fn %{
                                                      path: path,
                                                      shape: shape,
                                                      size: size,
                                                      offset: offset
                                                    },
                                                    acc ->
        slice = Nx.slice(flat_vector, [offset], [size])
        value = Nx.reshape(slice, shape)

        put_nested(acc, path, value)
      end)

    %{template_state | data: data}
  end

  defp collect_tensors(%Nx.Tensor{} = tensor, path), do: [{path, tensor}]

  defp collect_tensors(map, path) when is_map(map) do
    map
    |> Enum.to_list()
    |> Enum.sort_by(fn {key, _} -> normalize_key(key) end)
    |> Enum.flat_map(fn {key, value} ->
      collect_tensors(value, path ++ [key])
    end)
  end

  defp collect_tensors(_other, _path), do: []

  defp normalize_key(key) when is_atom(key), do: to_string(key)
  defp normalize_key(key), do: key

  defp put_nested(map, [key], value), do: Map.put(map, key, value)

  defp put_nested(map, [key | rest], value) when is_map(map) do
    Map.put(map, key, put_nested(Map.get(map, key, %{}), rest, value))
  end
end
