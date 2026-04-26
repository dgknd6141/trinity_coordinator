defmodule TrinityCoordinator.Trace.Sink do
  @moduledoc """
  Behaviour for pluggable trace sinks.
  """

  @type sink :: term()

  @callback write_event(sink, map()) :: :ok | {:error, term()}
end
