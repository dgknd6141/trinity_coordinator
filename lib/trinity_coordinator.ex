defmodule TrinityCoordinator do
  @moduledoc """
  Public entry points and metadata for the TRINITY coordinator.

  The core implementation lives in the focused modules:

  - `TrinityCoordinator.Extractor` for real SLM hidden-state extraction.
  - `TrinityCoordinator.CoordinationHead` for real Axon routing.
  - `TrinityCoordinator.Orchestrator` for multi-turn routing.
  - `TrinityCoordinator.Runtime` for EXLA/CUDA checks.
  """

  @roles %{0 => "Worker", 1 => "Thinker", 2 => "Verifier"}
  @gpu_demo_command "XLA_TARGET=cuda12 mix trinity.route.demo --mock"

  @doc """
  Returns the canonical TRINITY role map.

  ## Examples

      iex> TrinityCoordinator.roles()
      %{0 => "Worker", 1 => "Thinker", 2 => "Verifier"}

  """
  def roles, do: @roles

  @doc """
  Returns the command used by this repository's real GPU demo.
  """
  def gpu_demo_command, do: @gpu_demo_command
end
