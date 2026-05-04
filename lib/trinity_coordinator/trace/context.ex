defmodule TrinityCoordinator.Trace.Context do
  @moduledoc "Trace context and option parsing for orchestrator runs."

  alias TrinityCoordinator.Trace.Event
  alias TrinityCoordinator.Trace.JSONL
  alias TrinityCoordinator.Trace.Redactor

  defstruct [:run_id, :content, :sink, :enabled, :schema_version, redaction_values: []]

  @default_run_id "run_" <> Integer.to_string(System.unique_integer([:positive]))

  @doc """
  Builds a trace context from orchestrator options.
  """
  def new(opts) when is_list(opts) do
    enabled = Keyword.get(opts, :enabled, false)
    run_id = Keyword.get(opts, :run_id, @default_run_id)
    content = Keyword.get(opts, :content, :hash)
    redaction_values = normalize_redaction_values(Keyword.get(opts, :redaction_values, []))

    context =
      %__MODULE__{
        run_id: run_id,
        content: content,
        enabled: enabled,
        sink: nil,
        schema_version: 1,
        redaction_values: redaction_values
      }

    case build_sink(enabled, opts) do
      {:ok, sink} -> %{context | sink: sink}
      :disabled -> context
      {:error, reason} -> raise(ArgumentError, "invalid trace sink: #{inspect(reason)}")
    end
  end

  def write(%__MODULE__{enabled: false}, _event), do: :ok

  def write(%__MODULE__{enabled: true, sink: nil}, _event), do: {:error, :trace_sink_missing}

  def write(%__MODULE__{enabled: true} = context, event) when is_map(event) do
    enriched =
      event
      |> Map.put_new(:run_id, context.run_id)
      |> Map.put_new(:schema_version, Event.schema_version())

    with :ok <- Event.validate(enriched) do
      context.sink.__struct__.write_event(
        context.sink,
        redact_event(context.content, context.redaction_values, enriched)
      )
    end
  end

  def write(%__MODULE__{}, _event), do: {:error, :invalid_trace_event}

  defp build_sink(false, _opts), do: :disabled

  defp build_sink(true, opts) do
    case Keyword.get(opts, :sink) do
      nil ->
        {:error, :sink_required}

      {:jsonl, path} when is_binary(path) ->
        {:ok, %JSONL{path: path}}

      other ->
        {:error, {:unsupported_sink, other}}
    end
  end

  defp redact_event(:full, values, event), do: Redactor.redact_values(event, values)

  defp redact_event(:hash, values, event) do
    event
    |> Redactor.redact(:redacted)
    |> Redactor.redact_values(values)
  end

  defp redact_event(_unknown, values, event) do
    event
    |> Redactor.redact(:redacted)
    |> Redactor.redact_values(values)
  end

  defp normalize_redaction_values(values) when is_list(values) do
    values
    |> Enum.filter(&(is_binary(&1) and &1 != ""))
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.uniq()
  end

  defp normalize_redaction_values(value) when is_binary(value),
    do: normalize_redaction_values([value])

  defp normalize_redaction_values(_), do: []
end
