defmodule TrinityCoordinator.TraceTest do
  use ExUnit.Case
  alias TrinityCoordinator.Trace.{Context, Event, Hash, JSONL, Redactor}

  test "Trace.Event validates required core fields" do
    event =
      Event.new(:run_started, "run_1", %{
        turn: nil
      })

    assert event.event == :run_started
    assert event.schema_version == 1
    assert event.run_id == "run_1"
    assert is_integer(event.timestamp_ms)

    assert :ok =
             Event.validate(%{
               schema_version: 1,
               event: :turn_started,
               run_id: "run_2",
               timestamp_ms: System.system_time(:millisecond),
               turn: 1
             })
  end

  test "Trace.Event rejects invalid event type" do
    assert_raise ArgumentError, fn ->
      Event.new(:unsupported_event, "run_1", %{})
    end
  end

  test "trace hash is deterministic for equivalent messages" do
    message_list = [
      %{role: "user", content: "Hello"},
      %{"role" => "assistant", "content" => "Bye"}
    ]

    canonicalized_messages = [
      %{role: "user", content: "Hello"},
      %{"role" => "assistant", "content" => "Bye"}
    ]

    assert Hash.messages(message_list) == Hash.messages(canonicalized_messages)
  end

  test "redactor masks secrets recursively" do
    payload = %{
      role: "user",
      api_key: "sk-very-secret",
      nested: %{Authorization: "Bearer deadbeef", token: "abc"},
      headers: ["Bearer secret", "safe"]
    }

    redacted = Redactor.redact(payload, :redacted)
    assert redacted[:api_key] == "<redacted>"
    assert redacted[:nested][:Authorization] == "<redacted>"
    assert redacted[:nested][:token] == "<redacted>"
    assert Enum.any?(redacted[:headers], &(&1 == "<redacted>"))
  end

  test "JSONL sink appends one normalized line per event" do
    output =
      Path.join(
        System.tmp_dir!(),
        "trinity_trace_jsonl_#{System.unique_integer([:positive])}.jsonl"
      )

    File.rm(output)

    sink = %JSONL{path: output}

    :ok =
      JSONL.write_event(sink, %{event: :run_started, value: "ok", tensor: Nx.tensor([1.0, 2.0])})

    :ok = JSONL.write_event(sink, %{event: :turn_completed, nested: %{items: [1, 2, 3]}})

    lines = output |> File.read!() |> String.split("\n", trim: true)
    assert length(lines) == 2

    decoded =
      lines
      |> Enum.map(&Jason.decode!/1)
      |> Enum.map(&Map.get(&1, "nested"))

    assert decoded == [nil, %{"items" => [1, 2, 3]}]
  end

  test "Trace.Context applies redaction on write by default" do
    output =
      Path.join(
        System.tmp_dir!(),
        "trinity_trace_context_#{System.unique_integer([:positive])}.jsonl"
      )

    File.rm(output)

    context = Context.new(enabled: true, run_id: "run_redact", sink: {:jsonl, output})

    event = %{
      event: :run_failed,
      turn: 1,
      timestamp_ms: 1,
      run_id: "run_redact",
      schema_version: 1,
      reason: :failure,
      api_key: "se-cre",
      authorization: "Bearer token"
    }

    assert :ok = Context.write(context, event)

    [logged] =
      output |> File.read!() |> String.split("\n", trim: true) |> Enum.map(&Jason.decode!/1)

    assert Map.get(logged, "api_key") == "<redacted>"
    assert Map.get(logged, "authorization") == "<redacted>"
  end
end
