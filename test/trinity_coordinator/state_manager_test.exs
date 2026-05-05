defmodule TrinityCoordinator.StateManagerTest do
  use ExUnit.Case
  alias TrinityCoordinator.StateManager

  test "starts with an empty transcript or initial messages" do
    {:ok, pid} = StateManager.start_link([])
    assert StateManager.get_messages(pid) == []
  end

  test "appends user and assistant messages" do
    {:ok, pid} = StateManager.start_link([])
    StateManager.append_user(pid, "Hello")
    StateManager.append_assistant(pid, "Hi there")

    messages = StateManager.get_messages(pid)
    assert length(messages) == 2
    assert Enum.at(messages, 0) == %{role: "user", content: "Hello"}
    assert Enum.at(messages, 1) == %{role: "assistant", content: "Hi there"}
  end

  test "rejects unknown coordinator message roles" do
    trap_exit = Process.flag(:trap_exit, true)

    try do
      assert {:error, {%ArgumentError{} = error, _stack}} =
               StateManager.start_link([%{role: "external-runtime-role", content: "Hello"}])

      assert String.contains?(Exception.message(error), "unsupported message role")
      assert String.contains?(Exception.message(error), "external-runtime-role")
    after
      Process.flag(:trap_exit, trap_exit)
    end
  end
end
