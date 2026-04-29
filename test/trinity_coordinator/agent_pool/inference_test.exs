defmodule TrinityCoordinator.AgentPoolInferenceTest do
  use ExUnit.Case, async: true

  alias Inference.{Client, Request, Response}
  alias TrinityCoordinator.AgentPool
  alias TrinityCoordinator.ProviderPool

  defmodule FakeInferenceAdapter do
    @behaviour Inference.Adapter

    @impl true
    def complete(%Client{} = client, %Request{} = request) do
      if pid = Keyword.get(client.adapter_opts, :test_pid) do
        send(pid, {:inference_call, client, request})
      end

      {:ok,
       Response.new(
         provider: client.provider,
         model: request.model || client.model,
         text: Keyword.get(client.adapter_opts, :response_text, "fake inference response"),
         metadata: request.metadata
       )}
    end

    @impl true
    def stream(_client, _request), do: {:error, :not_used}
  end

  test "routes hosted provider specs through the shared inference adapter" do
    spec = %{
      id: 1,
      name: :reviewer,
      provider: :gemini,
      model: "gemini-test",
      max_tokens: 64,
      temperature: 0.3,
      metadata: %{lane: :review}
    }

    messages = [
      %{role: "system", content: "You are concise."},
      %{role: "user", content: "Review this route."}
    ]

    assert {:ok, "routed by inference"} =
             AgentPool.call_agent_with_spec(spec, messages,
               adapter: TrinityCoordinator.AgentPool.Inference,
               inference_adapter: FakeInferenceAdapter,
               inference_adapter_opts: [test_pid: self(), response_text: "routed by inference"],
               gemini_api_key: "test-key"
             )

    assert_receive {:inference_call, %Client{} = client, %Request{} = request}

    assert client.adapter == FakeInferenceAdapter
    assert client.provider == :gemini
    assert client.model == "gemini-test"
    assert client.metadata.agent_id == 1
    assert client.metadata.agent_name == :reviewer
    assert client.metadata.provider_adapter == TrinityCoordinator.AgentPool.Inference
    assert request.max_tokens == 64
    assert request.temperature == 0.3
    assert Enum.map(request.messages, & &1.role) == [:system, :user]
    assert Inference.Request.user_prompt(request) == "Review this route."
  end

  test "preserves explicit inference adapter options for ASM provider specs" do
    spec = %{
      id: 2,
      provider: :asm,
      model: "codex-local",
      metadata: %{inference_provider: :codex}
    }

    assert {:ok, "asm response"} =
             AgentPool.call_agent_with_spec(spec, [%{role: "user", content: "solve"}],
               adapter: TrinityCoordinator.AgentPool.Inference,
               inference_adapter: FakeInferenceAdapter,
               inference_adapter_opts: [test_pid: self(), response_text: "asm response"],
               inference_session: "session-1"
             )

    assert_receive {:inference_call, %Client{} = client, %Request{} = request}

    assert client.provider == :codex
    assert request.session == "session-1"
  end

  test "passes Gemini CLI ASM pool metadata through the inference boundary" do
    spec = ProviderPool.spec_for_agent(:gemini_cli_asm, 0)

    assert {:ok, "gemini asm response"} =
             AgentPool.call_agent_with_spec(spec, [%{role: "user", content: "ping"}],
               adapter: TrinityCoordinator.AgentPool.Inference,
               inference_adapter: FakeInferenceAdapter,
               inference_adapter_opts: [
                 test_pid: self(),
                 response_text: "gemini asm response"
               ]
             )

    assert_receive {:inference_call, %Client{} = client, %Request{} = request}

    query_opts = client.adapter_opts[:query_opts]
    payload = query_opts[:model_payload]

    assert client.provider == :gemini
    assert client.backend == :agent_session_manager
    assert request.model == "gemini-3.1-flash-lite-preview"
    assert query_opts[:lane] == :sdk
    assert payload.provider == :gemini
    assert payload.requested_model == "gemini-3.1-flash-lite-latest"
    assert payload.resolved_model == "gemini-3.1-flash-lite-preview"
  end

  test "hosted provider specs still fail before dispatch without credentials" do
    spec = %{id: 0, provider: :openai, model: "gpt-test"}

    assert {:error, :missing_openai_api_key} =
             AgentPool.call_agent_with_spec(spec, [%{role: "user", content: "hello"}],
               adapter: TrinityCoordinator.AgentPool.Inference,
               inference_adapter: FakeInferenceAdapter,
               inference_adapter_opts: [test_pid: self()]
             )

    refute_received {:inference_call, _client, _request}
  end
end
