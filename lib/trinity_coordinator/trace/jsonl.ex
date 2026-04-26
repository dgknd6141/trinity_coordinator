defmodule TrinityCoordinator.Trace.JSONL do
  @moduledoc """
  Append-only JSONL sink for trace events.
  """

  @behaviour TrinityCoordinator.Trace.Sink

  alias TrinityCoordinator.Trace.Hash

  @enforce_keys [:path]
  defstruct path: nil

  @type t :: %__MODULE__{path: String.t()}

  @impl true
  def write_event(%__MODULE__{path: path}, event) do
    :ok = ensure_directory(path)

    {:ok, line} = encode_event(event)
    File.write(path, line, [:append])
  end

  defp ensure_directory(path) do
    path
    |> Path.dirname()
    |> File.mkdir_p()
  end

  defp encode_event(event) do
    event
    |> normalize_for_json()
    |> Jason.encode()
    |> case do
      {:ok, encoded} -> {:ok, encoded <> "\n"}
      {:error, _} = error -> error
    end
  end

  defp normalize_for_json(event) when is_map(event) do
    event
    |> Map.new(fn {k, v} -> {normalize_key(k), normalize_value(v)} end)
  end

  defp normalize_for_json(event), do: event

  defp normalize_key(key) when is_atom(key), do: Atom.to_string(key)
  defp normalize_key(key), do: to_string(key)

  defp normalize_value(value) when is_binary(value), do: value
  defp normalize_value(value) when is_number(value), do: value
  defp normalize_value(true), do: true
  defp normalize_value(false), do: false
  defp normalize_value(nil), do: nil

  defp normalize_value(%Nx.Tensor{} = tensor) do
    %{
      tensor_shape: Nx.shape(tensor),
      tensor_backend: Nx.backend_transfer(tensor) |> inspect(),
      hash: Hash.tensor(tensor)
    }
    |> Map.new(fn {key, value} -> {normalize_key(key), normalize_value(value)} end)
  end

  defp normalize_value(value) when is_atom(value), do: Atom.to_string(value)

  defp normalize_value(value) when is_list(value) do
    Enum.map(value, &normalize_value/1)
  end

  defp normalize_value(value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.map(&normalize_value/1)
  end

  defp normalize_value(value) when is_map(value) do
    Map.new(value, fn {k, v} -> {normalize_key(k), normalize_value(v)} end)
  end

  defp normalize_value(value), do: inspect(value)
end
