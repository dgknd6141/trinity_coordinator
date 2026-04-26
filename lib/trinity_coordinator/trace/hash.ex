defmodule TrinityCoordinator.Trace.Hash do
  @moduledoc """
  Stable hashing helpers for traces.
  """

  @type hash :: String.t()

  @doc "Hashes any term with deterministic serialization."
  @spec term(term()) :: hash()
  def term(value) do
    value
    |> normalize_for_hash()
    |> :erlang.term_to_binary([:compressed])
    |> sha256_hex()
  end

  @doc "Hashes a transcript represented as role/content maps."
  @spec messages([map()]) :: hash()
  def messages(messages) when is_list(messages) do
    normalized =
      Enum.map(messages, fn message ->
        role = Map.get(message, :role, Map.get(message, "role"))
        content = Map.get(message, :content, Map.get(message, "content"))

        %{role: role, content: content}
      end)

    term(normalized)
  end

  @doc "Hashes text or response payloads."
  @spec text(String.t()) :: hash()
  def text(value) when is_binary(value), do: term(%{text: value})

  @doc "Hashes an Nx tensor deterministically by value and shape."
  @spec tensor(Nx.Tensor.t()) :: hash()
  def tensor(%Nx.Tensor{} = tensor) do
    data =
      tensor
      |> Nx.to_flat_list()
      |> Enum.map(&Float.round(&1, 10))

    term(%{shape: Nx.shape(tensor), data: data})
  end

  @doc "Hashes any module-friendly event shape and runtime metadata map."
  @spec metadata(map()) :: hash()
  def metadata(map) when is_map(map), do: term(map)

  defp sha256_hex(binary) when is_binary(binary) do
    :crypto.hash(:sha256, binary)
    |> Base.encode16(case: :lower)
  end

  defp normalize_for_hash(%Nx.Tensor{} = tensor), do: tensor_digest(tensor)

  defp normalize_for_hash(value) when is_map(value) do
    value
    |> Enum.to_list()
    |> Enum.sort_by(fn {k, _} -> normalize_key(k) end)
    |> Enum.map(fn {k, v} -> {normalize_key(k), normalize_for_hash(v)} end)
    |> Map.new()
  end

  defp normalize_for_hash(value) when is_list(value), do: Enum.map(value, &normalize_for_hash/1)

  defp normalize_for_hash(value) when is_tuple(value),
    do: Tuple.to_list(value) |> Enum.map(&normalize_for_hash/1)

  defp normalize_for_hash(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_for_hash(value) when is_number(value), do: value
  defp normalize_for_hash(value) when is_binary(value), do: value
  defp normalize_for_hash(value), do: inspect(value)

  defp normalize_key(nil), do: ""

  defp normalize_key(key) when is_atom(key), do: Atom.to_string(key)
  defp normalize_key(key) when is_binary(key), do: key
  defp normalize_key(key) when is_integer(key), do: Integer.to_string(key)
  defp normalize_key(key), do: inspect(key)

  defp tensor_digest(tensor) do
    {
      Nx.shape(tensor),
      Nx.type(tensor),
      Enum.map(Nx.to_flat_list(tensor), &Float.round(&1, 10))
    }
  end
end
