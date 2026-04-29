defmodule TrinityCoordinator.ProviderPoolTest do
  use ExUnit.Case

  alias TrinityCoordinator.AgentPool
  alias TrinityCoordinator.ProviderPool
  alias TrinityCoordinator.ProviderPool.Spec

  test "normalizes valid specs and applies defaults" do
    normalized =
      Spec.normalize!([
        [id: "0", provider: "openai", model: "gpt-4o-mini", temperature: 0.2],
        [
          id: 1,
          provider: :openai_compatible,
          model: "llama",
          base_url: "http://127.0.0.1:11434/v1"
        ]
      ])

    assert length(normalized) == 2
    assert Enum.at(normalized, 0).id == 0
    assert Enum.at(normalized, 0).name == :agent_0
    assert Enum.at(normalized, 1).provider == :openai_compatible
    assert is_integer(Enum.at(normalized, 1).timeout_ms)
    assert Enum.at(normalized, 1).enabled == true
  end

  test "rejects duplicate ids" do
    assert {:error, :duplicate_provider_ids} ==
             Spec.validate([
               %Spec{
                 id: 0,
                 provider: :openai,
                 model: "a",
                 timeout_ms: 1,
                 max_tokens: 1,
                 temperature: 0.2,
                 name: :a,
                 metadata: %{},
                 enabled: true
               },
               %Spec{
                 id: 0,
                 provider: :openai,
                 model: "b",
                 timeout_ms: 1,
                 max_tokens: 1,
                 temperature: 0.2,
                 name: :b,
                 metadata: %{},
                 enabled: true
               }
             ])
  end

  test "rejects openai-compatible specs missing base URL" do
    assert {:error, {:invalid_openai_compatible_spec, spec}} =
             Spec.normalize([
               [id: 0, provider: :openai_compatible, model: "x"]
             ])
             |> then(fn
               {:ok, specs} -> Spec.validate(specs)
               err -> err
             end)

    assert spec.base_url == nil
  end

  test "fetches configured pool by name and computes size" do
    assert is_integer(ProviderPool.size(:default))
    assert ProviderPool.size(:default) > 0
  end

  test "ships a Gemini CLI ASM pool for live route smokes" do
    pool = ProviderPool.fetch!(:gemini_cli_asm)

    assert length(pool) == 7
    assert Enum.map(pool, & &1.id) == Enum.to_list(0..6)
    assert Enum.all?(pool, &(&1.provider == :asm))
    assert Enum.all?(pool, &(&1.model == "gemini-3.1-flash-lite-preview"))

    spec = hd(pool)
    query_opts = spec.metadata.inference_adapter_opts[:query_opts]
    payload = query_opts[:model_payload]

    assert spec.metadata.inference_provider == :gemini
    assert query_opts[:lane] == :sdk
    assert query_opts[:stream_timeout_ms] == 180_000
    assert payload.requested_model == "gemini-3.1-flash-lite-latest"
    assert payload.resolved_model == "gemini-3.1-flash-lite-preview"
    assert payload.provider == :gemini
  end

  test "supports explicit list pools" do
    explicit_pool = [
      [id: 0, provider: :openai, model: "gpt-4o-mini", base_url: nil],
      [id: 1, provider: :openai, model: "gpt-4o-mini", base_url: nil]
    ]

    assert {:ok, normalized} = Spec.normalize(explicit_pool)
    assert is_list(normalized)
    assert length(normalized) == 2
    assert ProviderPool.size(normalized) == 2
    assert Enum.at(normalized, 0).id == 0
    assert Enum.at(normalized, 1).id == 1
  end

  test "normalizes shared inference provider specs" do
    specs =
      Spec.normalize!([
        [id: 0, provider: :gemini, model: "gemini-3.1-flash-lite-preview"],
        [id: 1, provider: :gemini_ex, model: "gemini-3.1-flash-lite-preview"],
        [id: 2, provider: :asm, model: "codex-local"],
        [id: 3, provider: :agent_session_manager, model: "gemini-cli"],
        [id: 4, provider: :anthropic, model: "claude-haiku-4-5"]
      ])

    assert Enum.map(specs, & &1.provider) == [
             :gemini,
             :gemini_ex,
             :asm,
             :agent_session_manager,
             :anthropic
           ]

    assert :ok == Spec.validate(specs)
  end

  test "openai-compatible base URL is carried through spec normalization" do
    normalized =
      Spec.normalize!([
        [id: 0, provider: :openai_compatible, model: "llama", base_url: "http://127.0.0.1:9999"]
      ])

    assert spec = Enum.at(normalized, 0)
    assert spec.provider == :openai_compatible
    assert spec.base_url == "http://127.0.0.1:9999"
  end

  test "adapter selection honors openai-compatible provider mapping" do
    pool = [
      [
        id: 0,
        provider: :openai_compatible,
        model: "llama",
        base_url: "http://127.0.0.1:9999",
        timeout_ms: 10
      ]
    ]

    messages = [%{role: "user", content: "Hi"}]

    assert {:error, _reason} =
             AgentPool.call_agent(0, messages, provider_pool: pool, openai_api_key: "test-key")
  end

  test "resolves a specific agent spec from a list pool" do
    explicit_pool = [
      [id: 0, provider: :openai, model: "gpt-4o-mini", base_url: nil],
      [id: 1, provider: :openai, model: "gpt-4o-mini", base_url: nil]
    ]

    assert spec = ProviderPool.spec_for_agent(explicit_pool, 1)
    assert spec.id == 1
    assert spec.provider == :openai
  end
end
