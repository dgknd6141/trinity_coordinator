defmodule Mix.Tasks.Trinity.Route.Demo do
  @moduledoc """
  Runs the adapted coordinator through the runtime provider boundary.

  Mock mode is the default safe smoke:

      XLA_TARGET=cuda12 mix trinity.route.demo \
        --mock \
        --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
        --trace-out tmp/trinity_route_demo.jsonl

  Live provider mode must be explicit:

      TRINITY_ENABLE_PROVIDER_DEMO=1 XLA_TARGET=cuda12 mix trinity.route.demo \
        --profile qwen_sakana_adapted \
        --provider-pool gemini_cli_asm \
        --trace-out tmp/trinity_route_demo.jsonl

  This task never bypasses provider adapters. Missing credentials or provider
  errors fail the task instead of being reported as a pass.
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Orchestrator, StateManager}
  alias TrinityCoordinator.Sakana.{Artifact, Coordinator}

  @shortdoc "Runs a gated adapted-coordinator route demo"
  @default_message "Select a TRINITY role for this reasoning task."
  @default_trace_path "tmp/trinity_route_demo.jsonl"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    args
    |> parse_args!()
    |> validate_live_gate!()
    |> prepare_trace!()
    |> run_route_demo!()
    |> report_result!()

    HITL.pass("TRINITY ROUTE DEMO")
  end

  @doc false
  def parse_args!(args) do
    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [
          allow_live: :boolean,
          artifact_dir: :string,
          max_turns: :integer,
          message: :string,
          mock: :boolean,
          profile: :string,
          provider_pool: :string,
          run_id: :string,
          trace_content: :string,
          trace_out: :string
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    mock? = Keyword.get(opts, :mock, false)

    %{
      allow_live?: Keyword.get(opts, :allow_live, false),
      artifact_dir: Keyword.get(opts, :artifact_dir, Artifact.default_output_dir()),
      max_turns: Keyword.get(opts, :max_turns, 5),
      message: Keyword.get(opts, :message, @default_message),
      mock?: mock?,
      profile: Keyword.get(opts, :profile, "qwen_sakana_adapted"),
      provider_pool: Keyword.get(opts, :provider_pool, if(mock?, do: "mock", else: "default")),
      run_id: Keyword.get(opts, :run_id, "route_demo"),
      trace_content: parse_trace_content(Keyword.get(opts, :trace_content, "hash")),
      trace_path: Keyword.get(opts, :trace_out, @default_trace_path)
    }
  end

  defp validate_live_gate!(%{profile: "qwen_sakana_adapted", mock?: true} = opts), do: opts

  defp validate_live_gate!(%{profile: "qwen_sakana_adapted"} = opts) do
    enabled? =
      opts.allow_live? or
        System.get_env("TRINITY_ENABLE_PROVIDER_DEMO") in ["1", "true", "TRUE", "yes", "YES"]

    unless enabled? do
      Mix.raise(
        "live provider demo is gated; pass --mock for local smoke or set TRINITY_ENABLE_PROVIDER_DEMO=1"
      )
    end

    opts
  end

  defp validate_live_gate!(%{profile: profile}) do
    Mix.raise("unsupported route demo profile: #{inspect(profile)}")
  end

  defp prepare_trace!(opts) do
    File.mkdir_p!(Path.dirname(opts.trace_path))
    File.rm(opts.trace_path)

    HITL.banner("TRINITY ROUTE DEMO")
    HITL.kv("Profile", opts.profile)
    HITL.kv("Artifact dir", opts.artifact_dir)
    HITL.kv("Provider pool", opts.provider_pool)
    HITL.kv("Provider mode", if(opts.mock?, do: :mock, else: :live))
    HITL.kv("Trace path", opts.trace_path)

    opts
  end

  defp run_route_demo!(opts) do
    {:ok, coordinator} = Coordinator.load(artifact_dir: opts.artifact_dir)
    {:ok, pid} = StateManager.start_link([%{role: "user", content: opts.message}])

    result =
      Orchestrator.run_loop(
        pid,
        coordinator.routing_model,
        coordinator.routing_params,
        orchestrator_opts(coordinator, opts)
      )

    Map.put(opts, :result, result)
  end

  defp orchestrator_opts(coordinator, opts) do
    [
      max_turns: opts.max_turns,
      num_agents: coordinator.num_agents,
      num_roles: coordinator.num_roles,
      slm_context: {coordinator.model_info, coordinator.tokenizer},
      provider_pool: opts.provider_pool,
      agent_pool_opts: agent_pool_opts(),
      trace: [
        enabled: true,
        sink: {:jsonl, opts.trace_path},
        run_id: opts.run_id,
        content: opts.trace_content
      ]
    ]
    |> maybe_put_mock_agent(opts)
  end

  defp maybe_put_mock_agent(opts, %{mock?: true}),
    do: Keyword.put(opts, :mock_agent_fn, &mock_agent/3)

  defp maybe_put_mock_agent(opts, _), do: opts

  defp agent_pool_opts do
    [
      openai_api_key:
        first_env(["OPENAI_API_KEY", "TRINITY_OPENAI_API_KEY", "OPENAI_API_KEY_ENV"]),
      openai_max_tokens: 128,
      openai_timeout_ms: 30_000
    ]
    |> Keyword.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp mock_agent(:verifier, _messages, _metadata),
    do: {:ok, "ACCEPT: mock route demo verified the worker response."}

  defp mock_agent(:thinker, _messages, _metadata),
    do:
      {:ok,
       "<suggestion>Ask the solver for the concrete answer.</suggestion><suggested_role>solver</suggested_role>"}

  defp mock_agent(:worker, _messages, _metadata), do: {:ok, "Result: 6 * 7 = 42."}
  defp mock_agent(_role, _messages, _metadata), do: {:ok, "Proceed."}

  defp report_result!(%{result: {:ok, response}} = opts) do
    validate_trace!(opts.trace_path)
    HITL.kv("Result", response)
    HITL.kv("Termination", trace_final_status(opts.trace_path) || "ok")
  end

  defp report_result!(%{result: {:error, reason}} = opts) do
    validate_trace!(opts.trace_path)
    Mix.raise("route demo failed: #{inspect(reason)}")
  end

  defp validate_trace!(trace_path) do
    unless File.exists?(trace_path), do: raise("trace file was not written: #{trace_path}")

    events =
      trace_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)

    unless Enum.any?(events, &(&1["event"] == "provider_called")) do
      raise "trace file did not include provider dispatch events"
    end
  end

  defp trace_final_status(path) do
    path
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.reverse()
    |> Enum.find_value(fn line ->
      case Jason.decode(line) do
        {:ok, %{"event" => "run_completed", "final_status" => status}} -> status
        _ -> nil
      end
    end)
  end

  defp first_env(keys) do
    Enum.find_value(keys, fn key ->
      case System.get_env(key) do
        value when is_binary(value) and value != "" -> value
        _ -> nil
      end
    end)
  end

  defp parse_trace_content("full"), do: :full
  defp parse_trace_content(:full), do: :full
  defp parse_trace_content(_), do: :hash
end
