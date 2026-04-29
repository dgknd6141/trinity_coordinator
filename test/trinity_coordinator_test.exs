defmodule TrinityCoordinatorTest do
  use ExUnit.Case
  doctest TrinityCoordinator

  test "exposes canonical role metadata" do
    assert TrinityCoordinator.roles() == %{0 => "Worker", 1 => "Thinker", 2 => "Verifier"}
  end

  test "exposes the real GPU demo command" do
    assert TrinityCoordinator.gpu_demo_command() ==
             "XLA_TARGET=cuda12 mix trinity.route.demo --mock"
  end
end
