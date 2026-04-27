defmodule TrinityCoordinator.RoleInjectorTest do
  use ExUnit.Case
  alias TrinityCoordinator.RoleInjector

  test "injects Thinker role correctly" do
    messages = [%{role: "user", content: "Hello"}]
    injected = RoleInjector.inject_role(messages, "Thinker")

    assert length(injected) == 2
    assert Enum.at(injected, 0).role == "system"
    assert Enum.at(injected, 0).content =~ "Analyze the current state"
    assert Enum.at(injected, 1).content == "Hello"
  end

  test "injects Worker role correctly" do
    messages = []
    injected = RoleInjector.inject_role(messages, "Worker")

    assert length(injected) == 1
    assert Enum.at(injected, 0).content =~ "Execute the next"
  end

  test "injects Verifier role correctly" do
    messages = []
    injected = RoleInjector.inject_role(messages, "Verifier")

    assert length(injected) == 1
    assert Enum.at(injected, 0).content =~ "Start your response with exactly ACCEPT or REVISE"
  end

  test "accepts role ids and atoms" do
    assert RoleInjector.role_name(0) == "Thinker"
    assert RoleInjector.role_name(:worker) == "Worker"
    assert RoleInjector.role_name("v") == "Verifier"
    assert RoleInjector.role_atom("Verifier") == :verifier
    assert RoleInjector.role_id(:thinker) == 0

    assert RoleInjector.inject_role([], :verifier)
           |> hd()
           |> Map.fetch!(:content)
           |> String.contains?("ACCEPT")
  end

  test "defaults to a helpful assistant if role is unknown" do
    messages = []
    injected = RoleInjector.inject_role(messages, "UnknownRole")

    assert length(injected) == 1
    assert Enum.at(injected, 0).content == "You are a helpful assistant."
  end
end
