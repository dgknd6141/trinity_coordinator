defmodule TrinityCoordinator.ProviderSmokeTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{CoordinationHead, Extractor, Orchestrator, Runtime, StateManager}

  @smoke_env_key "TRINITY_ENABLE_PROVIDER_TESTS"
  @budget_env_key "TRINITY_PROVIDER_BUDGET_USD"
  @api_env_keys ["OPENAI_API_KEY", "OPENAI_API_KEY_ENV", "TRINITY_OPENAI_API_KEY"]

  @tag :provider_smoke
  test "executes real provider calls under explicit budget and credentials guard" do
    case smoke_test_config() do
      {:skip, _reason} ->
        assert true

      {:ok, smoke_conf} ->
        Runtime.put_cuda_backend!()

        {:ok, pid} = StateManager.start_link([%{role: "user", content: "Answer briefly: 1+1."}])

        model = CoordinationHead.build_model(32, 5, 3)
        {init_fn, _predict_fn} = Axon.build(model)
        params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())

        assert {:ok, {model_info, tokenizer}} =
                 Extractor.load_slm_model(
                   {:hf, "hf-internal-testing/tiny-random-gpt2"},
                   Bumblebee.Text.Gpt2,
                   :base
                 )

        trace_path =
          Path.join(
            System.tmp_dir!(),
            "trinity_smoke_run_#{System.unique_integer([:positive])}.jsonl"
          )

        File.rm(trace_path)

        run_opts =
          [
            max_turns: 2,
            num_agents: 5,
            num_roles: 3,
            slm_context: {model_info, tokenizer},
            trace: [enabled: true, sink: {:jsonl, trace_path}],
            agent_pool_opts: [
              openai_api_key: smoke_conf.api_key,
              openai_max_tokens: smoke_conf.max_tokens,
              openai_timeout_ms: smoke_conf.timeout_ms
            ]
          ]

        result = Orchestrator.run_loop(pid, model, params, run_opts)
        assert match?({:ok, _}, result) or result == {:error, :max_turns_reached}

        lines =
          trace_path
          |> File.read!()
          |> String.split("\n", trim: true)
          |> Enum.map(&Jason.decode!/1)

        assert Enum.any?(lines, &(&1["event"] == "provider_called" and is_binary(&1["status"])))

        assert Enum.any?(
                 lines,
                 &(&1["event"] == "provider_called" and &1["status"] in ["ok", "error"])
               )

        assert Enum.all?(lines, &(&1["authorization"] != smoke_conf.api_key))

        provider_events = Enum.filter(lines, &(&1["event"] == "provider_called"))

        refute Enum.empty?(provider_events)
        assert total_estimated_cost(lines) <= smoke_conf.budget_usd
    end
  end

  test "smoke configuration is skipped when env var is disabled" do
    run_with_isolated_env(
      [
        {@smoke_env_key, nil},
        {@budget_env_key, nil},
        {"OPENAI_API_KEY", nil}
      ],
      fn ->
        assert {:skip, :disabled} = smoke_test_config()
      end
    )
  end

  test "smoke configuration rejects missing or invalid budget" do
    run_with_isolated_env(
      [
        {@smoke_env_key, "1"},
        {@budget_env_key, nil},
        {"OPENAI_API_KEY", "test-key"}
      ],
      fn ->
        assert {:skip, :budget_missing} = smoke_test_config()
      end
    )

    run_with_isolated_env(
      [
        {@smoke_env_key, "1"},
        {@budget_env_key, "0"},
        {"OPENAI_API_KEY", "test-key"}
      ],
      fn ->
        assert {:skip, :invalid_budget} = smoke_test_config()
      end
    )
  end

  test "smoke configuration rejects missing credentials" do
    run_with_isolated_env(
      [
        {@smoke_env_key, "1"},
        {@budget_env_key, "1"},
        {"OPENAI_API_KEY", nil},
        {"OPENAI_API_KEY_ENV", nil},
        {"TRINITY_OPENAI_API_KEY", nil}
      ],
      fn ->
        assert {:skip, :api_key_missing} = smoke_test_config()
      end
    )
  end

  test "smoke config parses valid settings" do
    run_with_isolated_env(
      [
        {@smoke_env_key, "1"},
        {@budget_env_key, "2.5"},
        {"OPENAI_API_KEY", "test-key"},
        {"OPENAI_API_KEY_ENV", nil},
        {"TRINITY_OPENAI_API_KEY", nil}
      ],
      fn ->
        assert {:ok, conf} = smoke_test_config()
        assert conf.budget_usd == 2.5
        assert conf.api_key == "test-key"
        assert conf.max_tokens == 16
        assert conf.timeout_ms == 30_000
      end
    )
  end

  test "smoke run enforces low call limits" do
    run_with_isolated_env(
      [
        {@smoke_env_key, "1"},
        {@budget_env_key, "1"},
        {"OPENAI_API_KEY", "test-key"},
        {"OPENAI_API_KEY_ENV", nil},
        {"TRINITY_OPENAI_API_KEY", nil}
      ],
      fn ->
        {:ok, conf} = smoke_test_config()
        assert conf.max_tokens <= 16
        assert conf.timeout_ms == 30_000
      end
    )
  end

  defp run_with_isolated_env(updates, fun) do
    original =
      updates
      |> Enum.map(fn {key, _value} ->
        {key, System.get_env(key)}
      end)

    clear_or_set = fn key, value ->
      if is_nil(value), do: System.delete_env(key), else: System.put_env(key, value)
    end

    Enum.each(updates, fn {key, value} -> clear_or_set.(key, value) end)

    try do
      fun.()
    after
      Enum.each(original, fn {key, value} ->
        clear_or_set.(key, value)
      end)
    end
  end

  defp smoke_test_config do
    enabled = System.get_env(@smoke_env_key)

    with "1" <- enabled,
         {:ok, budget_usd} <- parse_budget(),
         {:ok, api_key} <- parse_api_key(),
         {:ok, max_tokens} <- parse_max_tokens() do
      {:ok,
       %{
         budget_usd: budget_usd,
         api_key: api_key,
         max_tokens: max_tokens,
         timeout_ms: 30_000
       }}
    else
      "0" -> {:skip, :disabled}
      nil -> {:skip, :disabled}
      {:error, reason} -> {:skip, reason}
      _ -> {:skip, :disabled}
    end
  end

  defp parse_budget do
    case System.get_env(@budget_env_key) do
      nil ->
        {:error, :budget_missing}

      value ->
        case Float.parse(String.trim(value)) do
          {budget, ""} when budget > 0.0 ->
            {:ok, budget}

          {_, ""} ->
            {:error, :invalid_budget}

          _ ->
            {:error, :invalid_budget}
        end
    end
  end

  defp parse_api_key do
    case Enum.find_value(@api_env_keys, fn key -> System.get_env(key) end) do
      nil -> {:error, :api_key_missing}
      key when is_binary(key) and byte_size(key) > 0 -> {:ok, key}
      _ -> {:error, :api_key_invalid}
    end
  end

  defp parse_max_tokens do
    {:ok, 16}
  end

  defp total_estimated_cost(lines) do
    model_rate_usd_per_1k_tokens = 0.000_03

    Enum.reduce(lines, 0.0, fn
      %{"event" => "provider_called", "status" => "ok", "provider_max_tokens" => nil}, acc ->
        acc

      %{"event" => "provider_called", "status" => "ok", "provider_max_tokens" => max_tokens},
      acc ->
        acc + max_tokens / 1000.0 * model_rate_usd_per_1k_tokens

      _other, acc ->
        acc
    end)
  end
end
