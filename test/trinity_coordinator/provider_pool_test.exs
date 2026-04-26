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
