defmodule Mix.Tasks.Trinity.Demo do
  @moduledoc """
  Compatibility wrapper for the active adapted-coordinator route demo.

      XLA_TARGET=cuda12 mix trinity.demo --mock

  The old `trinity.demo` task trained a supervised routing head inline. That
  experiment-reproduction path is no longer part of the active service lane, so
  this task now delegates to `mix trinity.route.demo`.
  """

  use Mix.Task

  alias Mix.Tasks.Trinity.Route.Demo, as: RouteDemo

  @shortdoc "Runs the active adapted-coordinator route demo"

  @impl Mix.Task
  def run(args) do
    RouteDemo.run(args)
  end
end
