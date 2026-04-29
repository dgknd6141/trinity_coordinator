defmodule TrinityCoordinator.Sakana.SafetensorsSlice do
  @moduledoc """
  Bounded safetensors reads for large rank-2 tensors.

  The upstream safetensors lazy container loads an entire tensor when converted
  with `Nx.to_tensor/1`. This module reads contiguous row ranges directly from a
  `%Safetensors.FileTensor{}` so large embedding/LM-head matrices can be replayed
  in chunks.
  """

  alias Safetensors.Shared

  @doc "Reads a contiguous row slice from a rank-2 lazy safetensors tensor."
  def row_slice!(
        %Safetensors.FileTensor{
          shape: shape,
          type: type,
          path: path,
          byte_offset: byte_offset,
          byte_size: byte_size
        },
        row_start,
        row_count
      )
      when is_integer(row_start) and row_start >= 0 and is_integer(row_count) and
             row_count > 0 do
    {rows, cols} = rank2_shape!(shape)

    if row_start + row_count > rows do
      raise ArgumentError,
            "row slice #{inspect({row_start, row_count})} exceeds tensor rows #{rows}"
    end

    elem_bytes = element_bytes!(type)
    row_bytes = cols * elem_bytes
    read_offset = byte_offset + row_start * row_bytes
    read_size = row_count * row_bytes
    max_read_size = byte_size - row_start * row_bytes

    if read_size > max_read_size do
      raise ArgumentError,
            "row slice byte range exceeds safetensors payload for #{inspect(path)}"
    end

    File.open!(path, [:read, :raw], fn file ->
      file
      |> read_slice_binary!(read_offset, read_size, path)
      |> build_tensor!({row_count, cols}, type)
    end)
  end

  def row_slice!(%Safetensors.FileTensor{} = lazy_tensor, row_start, row_count) do
    raise ArgumentError,
          "invalid row slice #{inspect({row_start, row_count})} for #{inspect(lazy_tensor.shape)}"
  end

  @doc "Materializes a tensor or lazy safetensors value onto the BinaryBackend."
  def materialize!(%Nx.Tensor{} = tensor), do: host_snapshot(tensor)

  def materialize!(%Safetensors.FileTensor{} = lazy_tensor) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      lazy_tensor
      |> Nx.to_tensor()
      |> host_snapshot()
    end)
  end

  defp rank2_shape!(shape) when is_tuple(shape) and tuple_size(shape) == 2 do
    {elem(shape, 0), elem(shape, 1)}
  end

  defp rank2_shape!(shape) do
    raise ArgumentError, "expected rank-2 safetensors tensor, got shape #{inspect(shape)}"
  end

  defp element_bytes!({_kind, bits}) when is_integer(bits) and rem(bits, 8) == 0,
    do: div(bits, 8)

  defp element_bytes!(type),
    do: raise(ArgumentError, "unsupported safetensors type #{inspect(type)}")

  defp read_slice_binary!(file, read_offset, read_size, path) do
    case :file.pread(file, read_offset, read_size) do
      {:ok, binary} when byte_size(binary) == read_size ->
        binary

      {:ok, binary} ->
        raise "short safetensors read from #{inspect(path)}: expected #{read_size} bytes, got #{byte_size(binary)}"

      {:error, reason} ->
        raise File.Error, reason: reason, action: "read", path: path
    end
  end

  defp build_tensor!(binary, shape, type) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      Shared.build_tensor(binary, shape, type)
    end)
  end

  defp host_snapshot(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)
end
