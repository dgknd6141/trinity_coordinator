defmodule TrinityCoordinator.Benchmark.Dataset do
  @moduledoc """
  Utilities for loading TRINITY benchmark fixtures.

  Fixtures are JSONL lines with at minimum:

    - `id`
    - `family`
    - `messages`

  Optional fields:

    - `expected_agent`
    - `expected_role`
    - `difficulty`
    - `source`
    - `metadata`
  """

  alias __MODULE__

  defstruct [
    :id,
    :family,
    :messages,
    :expected_agent,
    :expected_role,
    :difficulty,
    :source,
    :metadata
  ]

  @type t :: %Dataset{
          id: String.t(),
          family: String.t(),
          messages: [map()],
          expected_agent: non_neg_integer() | nil,
          expected_role: non_neg_integer() | nil,
          difficulty: String.t() | nil,
          source: String.t() | nil,
          metadata: map()
        }

  @type load_error ::
          :file_not_found | :invalid_jsonl | :invalid_record | :invalid_record_schema

  @doc """
  Loads and validates a benchmark dataset.
  """
  @spec load!(String.t()) :: {:ok, [t()]} | {:error, load_error()}
  def load!(path) when is_binary(path) do
    with {:ok, raw} <- File.read(path) do
      parse_dataset(raw)
    end
  end

  @doc """
  Loads and validates a benchmark dataset, returning structured records.
  """
  @spec parse_dataset(String.t()) :: {:ok, [t()]} | {:error, load_error()}
  def parse_dataset(contents) when is_binary(contents) do
    lines =
      contents
      |> String.split("\n", trim: true)
      |> Enum.reject(&(&1 == ""))

    parsed =
      Enum.reduce_while(lines, [], fn line, acc ->
        case decode_case(line) do
          {:ok, case_map} -> {:cont, [case_map | acc]}
          {:error, reason} -> {:halt, {:error, reason}}
        end
      end)

    case parsed do
      {:error, reason} -> {:error, reason}
      cases -> {:ok, Enum.reverse(cases)}
    end
  end

  defp decode_case(line) do
    with {:ok, raw} <- Jason.decode(line),
         {:ok, case_map} <- validate_case(raw) do
      {:ok, case_map}
    else
      {:error, :invalid_record_schema} = reason ->
        reason

      {:error, _} ->
        {:error, :invalid_jsonl}
    end
  end

  defp validate_case(raw) when is_map(raw) do
    with {:ok, id} <- string_field(raw, "id"),
         {:ok, family} <- string_field(raw, "family"),
         {:ok, messages} <- messages_field(raw) do
      expected_agent = integer_or_nil(raw["expected_agent"] || raw[:expected_agent])
      expected_role = integer_or_nil(raw["expected_role"] || raw[:expected_role])

      {:ok,
       %Dataset{
         id: id,
         family: family,
         messages: messages,
         expected_agent: expected_agent,
         expected_role: expected_role,
         difficulty: optional_string(raw["difficulty"] || raw[:difficulty]),
         source: optional_string(raw["source"] || raw[:source]),
         metadata: normalize_metadata(raw["metadata"] || raw[:metadata])
       }}
    else
      {:error, _} = error -> error
    end
  end

  defp validate_case(_), do: {:error, :invalid_record}

  defp string_field(raw, key) do
    case raw[key] || raw[String.to_atom(key)] do
      value when is_binary(value) and byte_size(value) > 0 ->
        {:ok, value}

      _ ->
        {:error, :invalid_record_schema}
    end
  end

  defp messages_field(raw) do
    messages = raw["messages"] || raw[:messages] || []

    if is_list(messages) and messages != [] and Enum.all?(messages, &message_valid?/1) do
      {:ok, messages}
    else
      {:error, :invalid_record_schema}
    end
  end

  defp message_valid?(message) when is_map(message) do
    role = message[:role] || message["role"]
    content = message[:content] || message["content"]
    is_binary(role) and is_binary(content)
  end

  defp message_valid?(_), do: false

  defp integer_or_nil(nil), do: nil

  defp integer_or_nil(value) when is_integer(value) and value >= 0, do: value
  defp integer_or_nil(_), do: nil

  defp optional_string(nil), do: nil
  defp optional_string(value) when is_binary(value), do: value
  defp optional_string(_), do: nil

  defp normalize_metadata(meta) when is_map(meta), do: meta
  defp normalize_metadata(_), do: %{}
end
