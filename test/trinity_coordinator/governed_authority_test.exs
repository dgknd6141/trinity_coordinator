defmodule TrinityCoordinator.GovernedAuthorityTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{
    CoordinationHead,
    GovernedAuthority,
    Orchestrator,
    StateManager,
    Trace
  }

  alias Mix.Tasks.Trinity.Route.Demo

  test "materializes provider pool and credentials only from authority input" do
    run_with_env([{"OPENAI_API_KEY", "ambient-env-token"}], fn ->
      assert {:ok, opts} =
               GovernedAuthority.materialize_orchestrator_opts(
                 governed_authority: authority_packet()
               )

      assert Keyword.get(opts[:agent_pool_opts], :api_key) == "authority-token"
      assert Keyword.get(opts[:agent_pool_opts], :credential_ref) == "cred-trinity-1"
      assert opts[:governed_authority_ref] == "auth-trinity-1"

      [spec] = opts[:provider_pool]
      assert spec.provider == :openai
      assert spec.model == "gpt-4o-mini"

      refute inspect(opts) =~ "ambient-env-token"
    end)
  end

  test "orchestrator rejects direct provider authority beside governed authority" do
    {model, params} = routing_model()
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "route"}])

    assert {:error, {:governed_direct_fields_rejected, fields}} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 1,
               slm_context: :mock_context,
               extractor_fn: extractor(),
               governed_authority: authority_packet(),
               provider_pool: :default,
               agent_pool_opts: [openai_api_key: "direct-token"]
             )

    assert :provider_pool in fields
    assert :agent_pool_opts in fields
  end

  test "trace context redacts authority materialized values from JSONL output" do
    output =
      Path.join(
        System.tmp_dir!(),
        "trinity_governed_redaction_#{System.unique_integer([:positive])}.jsonl"
      )

    File.rm(output)

    context =
      Trace.Context.new(
        enabled: true,
        run_id: "governed-redaction",
        sink: {:jsonl, output},
        redaction_values: ["authority-token", "env-derived-value"]
      )

    event =
      Trace.Event.new(:provider_called, "governed-redaction", %{
        turn: 0,
        status: :error,
        error: "provider rejected authority-token and env-derived-value"
      })

    assert :ok = Trace.Context.write(context, event)

    [line] = File.read!(output) |> String.split("\n", trim: true)
    refute line =~ "authority-token"
    refute line =~ "env-derived-value"
    assert line =~ "<redacted>"
  end

  test "route demo governed arguments do not import ambient provider env" do
    run_with_env([{"OPENAI_API_KEY", "ambient-env-token"}], fn ->
      opts =
        Demo.parse_args!([
          "--governed-authority-ref",
          "auth-trinity-1",
          "--governed-workflow-ref",
          "workflow-trinity-1",
          "--governed-runtime-ref",
          "runtime-trinity-1",
          "--governed-provider-pool-ref",
          "pool-trinity-1",
          "--governed-credential-ref",
          "cred-trinity-1",
          "--governed-api-key",
          "authority-token",
          "--governed-provider",
          "openai",
          "--governed-model",
          "gpt-4o-mini",
          "--profile",
          "qwen_sakana_adapted"
        ])

      assert opts.provider_pool == nil
      assert opts.governed_authority[:api_key] == "authority-token"
      assert opts.governed_authority[:authority_ref] == "auth-trinity-1"
      refute inspect(opts) =~ "ambient-env-token"
    end)
  end

  defp authority_packet do
    [
      authority_ref: "auth-trinity-1",
      workflow_ref: "workflow-trinity-1",
      runtime_ref: "runtime-trinity-1",
      provider_pool_ref: "pool-trinity-1",
      credential_ref: "cred-trinity-1",
      api_key: "authority-token",
      redaction_values: ["authority-token"],
      provider_pool: [
        [
          id: 0,
          provider: :openai,
          model: "gpt-4o-mini",
          max_tokens: 8,
          temperature: 0.0
        ]
      ]
    ]
  end

  defp routing_model do
    model = CoordinationHead.build_model(4, 1, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
    {model, params}
  end

  defp extractor do
    fn _messages ->
      %{
        vector: Nx.broadcast(0.0, {1, 4}),
        vector_shape: {1, 4},
        hidden_state_shape: {1, 2, 4},
        input_shapes: %{}
      }
    end
  end

  defp run_with_env(updates, fun) do
    original = Enum.map(updates, fn {key, _value} -> {key, System.get_env(key)} end)

    Enum.each(updates, fn
      {key, nil} -> System.delete_env(key)
      {key, value} -> System.put_env(key, value)
    end)

    try do
      fun.()
    after
      Enum.each(original, fn
        {key, nil} -> System.delete_env(key)
        {key, value} -> System.put_env(key, value)
      end)
    end
  end
end
