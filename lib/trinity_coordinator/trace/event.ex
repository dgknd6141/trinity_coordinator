defmodule TrinityCoordinator.Trace.Event do
  @moduledoc """
  Structured trace event payloads with lightweight schema validation.
  """

  @events [
    :run_started,
    :turn_started,
    :slm_extracted,
    :route_selected,
    :provider_called,
    :turn_completed,
    :run_completed,
    :run_failed
  ]

  @base_required [:schema_version, :event, :run_id, :timestamp_ms]

  @doc "Schema version used by all serialized events."
  def schema_version, do: 1

  @doc "Builds an event map with required metadata."
  def new(event, run_id, fields \\ %{})
      when is_atom(event) and is_binary(run_id) and is_map(fields) do
    fields_map =
      fields
      |> Map.put(:event, event)
      |> Map.put(:run_id, run_id)
      |> Map.put(:schema_version, schema_version())
      |> Map.put(:timestamp_ms, System.system_time(:millisecond))

    validate!(fields_map)
    fields_map
  end

  @doc "Validates required core fields and returns validated event."
  def validate(%{} = event) do
    :ok = validate_base(event)
    validate_event(event)
  end

  @doc false
  def validate!(event) do
    case validate(event) do
      :ok -> :ok
      {:error, reason} -> raise(ArgumentError, "invalid trace event: #{inspect(reason)}")
    end
  end

  defp validate_base(event) when is_map(event) do
    Enum.reduce_while(@base_required, :ok, fn field, :ok ->
      if Map.has_key?(event, field) do
        {:cont, :ok}
      else
        {:halt, {:error, {:missing_required_field, field}}}
      end
    end)
  end

  defp validate_event(%{schema_version: 1} = event) do
    with :ok <- validate_event_name(event.event),
         :ok <- validate_turn(event) do
      :ok
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_event(%{schema_version: _version}) do
    {:error, :unsupported_schema_version}
  end

  defp validate_event_name(event) when event in @events, do: :ok
  defp validate_event_name(_), do: {:error, :invalid_event_type}

  defp validate_turn(%{event: :run_started, turn: nil}), do: :ok
  defp validate_turn(%{turn: turn}) when is_integer(turn) and turn >= 0, do: :ok
  defp validate_turn(%{}), do: :ok
end
