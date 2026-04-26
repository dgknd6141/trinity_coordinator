defmodule TrinityCoordinator.Orchestrator do
  @moduledoc """
  Orchestrates a real TRINITY multi-turn routing loop.
  """
  alias TrinityCoordinator.{
    AgentPool,
    CoordinationHead,
    Extractor,
    RoleInjector,
    Runtime,
    StateManager,
    Trace
  }

  @roles %{0 => "Thinker", 1 => "Worker", 2 => "Verifier"}
  @default_max_turns 5

  @doc """
  Run loop with keyword options:

  - `:max_turns` – stop after this many turns if no termination.
  - `:slm_context` – `%{model_info: ...}` or `{model_info, tokenizer}` for real extraction.
  - `:stop_token` – verifier termination token (default `"ACCEPT"`).
  - `:agent_pool_opts` – custom options passed through to `AgentPool`.
  - `:provider_pool` – pool name or explicit pool spec list.
  - `:roles` – optional role-map for index->name decoding.
  - `:num_agents` – number of agent logits in the coordination head.
  - `:num_roles` – number of role logits in the coordination head.
  - `:trace` – trace options (enabled, sink, run_id, content).
  """
  def run_loop(pid, model, params, opts \\ [])

  def run_loop(pid, model, params, opts) when is_list(opts) do
    max_turns = Keyword.get(opts, :max_turns, @default_max_turns)
    slm_context = Keyword.get(opts, :slm_context)
    stop_token = Keyword.get(opts, :stop_token, "ACCEPT")
    roles = Keyword.get(opts, :roles, @roles)
    agent_pool_opts = Keyword.get(opts, :agent_pool_opts, [])

    num_agents =
      Keyword.get(opts, :num_agents) ||
        AgentPool.agent_count(Keyword.get(opts, :provider_pool, :default))

    num_roles = Keyword.get(opts, :num_roles, 3)
    trace = Trace.Context.new(Keyword.get(opts, :trace, []))

    run_ctx = %{
      roles: roles,
      stop_token: stop_token,
      agent_pool_opts: agent_pool_opts,
      provider_pool: Keyword.get(opts, :provider_pool),
      num_agents: num_agents,
      num_roles: num_roles
    }

    case validate_loop_input(pid, model, params) do
      {:ok, _} ->
        emit_trace(
          trace,
          :run_started,
          %{
            max_turns: max_turns,
            num_agents: num_agents,
            num_roles: num_roles
          }
        )

        do_run_loop(
          pid,
          model,
          params,
          0,
          max_turns,
          slm_context,
          run_ctx,
          trace
        )

      error ->
        error
    end
  end

  def run_loop(pid, model, params, max_turns) when is_integer(max_turns),
    do: run_loop(pid, model, params, max_turns: max_turns)

  def run_loop(pid, model, params, max_turns, slm_context) when is_integer(max_turns),
    do:
      run_loop(pid, model, params,
        max_turns: max_turns,
        slm_context: slm_context
      )

  defp validate_loop_input(pid, model, params) do
    cond do
      not is_pid(pid) -> {:error, :invalid_state_pid}
      model == nil -> {:error, :invalid_model}
      params == nil -> {:error, :invalid_params}
      true -> {:ok, :ok}
    end
  end

  defp do_run_loop(_pid, _model, _params, turn, max_turns, _slm_context, _run_ctx, trace)
       when turn >= max_turns do
    emit_trace(trace, :run_failed, %{reason: :max_turns_reached})
    {:error, :max_turns_reached}
  end

  defp do_run_loop(pid, model, params, turn, max_turns, slm_context, run_ctx, trace) do
    messages = StateManager.get_messages(pid)
    transcript_hash = Trace.Hash.messages(messages)

    with :ok <-
           emit_trace(trace, :turn_started, %{
             turn: turn,
             max_turns: max_turns,
             transcript_hash: transcript_hash,
             message_count: length(messages)
           }),
         {:ok, extraction} <- extract_router_tensor(messages, slm_context),
         :ok <- emit_extraction_trace(trace, extraction, messages, turn),
         route <-
           CoordinationHead.route(
             model,
             params,
             extraction.vector,
             run_ctx.num_agents,
             run_ctx.num_roles
           ),
         :ok <- emit_route_trace(trace, route, run_ctx.roles, turn),
         role_name = Map.get(run_ctx.roles, route.role_id, "Worker"),
         injected_messages <- RoleInjector.inject_role(messages, role_name),
         {:ok, spec} <-
           AgentPool.fetch_agent_spec(
             route.agent_id,
             put_provider_pool(run_ctx.agent_pool_opts, run_ctx.provider_pool)
           ),
         :ok <-
           emit_trace(trace, :provider_called, %{
             turn: turn,
             provider: spec.provider,
             provider_model: spec.model,
             provider_base_url: Map.get(spec, :base_url),
             provider_timeout_ms: Map.get(spec, :timeout_ms),
             provider_max_tokens: Map.get(spec, :max_tokens),
             provider_temperature: Map.get(spec, :temperature),
             selected_agent: route.agent_id,
             selected_role: route.role_id
           }),
         {:ok, response_text} <-
           AgentPool.call_agent_with_spec(
             spec,
             injected_messages,
             put_provider_pool(run_ctx.agent_pool_opts, run_ctx.provider_pool)
           ) do
      StateManager.append_assistant(pid, response_text)

      verifier_status = verifier_accept?(role_name, response_text, run_ctx.stop_token)

      emit_trace(
        trace,
        :provider_called,
        %{
          turn: turn,
          provider: spec.provider,
          provider_model: spec.model,
          selected_agent: route.agent_id,
          selected_role: route.role_id,
          response_hash: Trace.Hash.text(response_text),
          status: :ok
        }
      )

      emit_turn_completed(trace, %{
        turn: turn,
        transcript_hash: transcript_hash,
        selected_agent: route.agent_id,
        selected_role: role_name,
        provider: spec.provider,
        provider_model: spec.model,
        response_hash: Trace.Hash.text(response_text),
        selected_agent_logits: Nx.to_flat_list(route.agent_logits),
        selected_role_logits: Nx.to_flat_list(route.role_logits),
        logits: Nx.to_flat_list(Nx.squeeze(route.logits, axes: [0])),
        vector_shape: extraction.vector_shape,
        hidden_state_shape: extraction.hidden_state_shape,
        vector_backend: Runtime.tensor_backend(extraction.vector),
        verifier_status: if(verifier_status, do: :accepted, else: :revised)
      })

      if verifier_status do
        emit_trace(trace, :run_completed, %{
          turn: turn,
          final_status: :accepted,
          response_hash: Trace.Hash.text(response_text)
        })

        {:ok, response_text}
      else
        do_run_loop(pid, model, params, turn + 1, max_turns, slm_context, run_ctx, trace)
      end
    else
      {:error, reason} ->
        emit_trace(
          trace,
          :provider_called,
          %{
            turn: turn,
            status: :error,
            error: inspect(reason)
          }
        )

        emit_trace(trace, :run_failed, %{turn: turn, reason: reason})
        {:error, reason}

      _ ->
        emit_trace(trace, :run_failed, %{turn: turn, reason: :unexpected_orchestrator_state})
        {:error, :unexpected_orchestrator_state}
    end
  end

  defp emit_extraction_trace(trace, extraction, messages, turn) do
    emit_trace(trace, :slm_extracted, %{
      turn: turn,
      input_shapes: extraction.input_shapes,
      hidden_state_shape: extraction.hidden_state_shape,
      vector_shape: extraction.vector_shape,
      vector_backend: Runtime.tensor_backend(extraction.vector),
      transcript_hash: Trace.Hash.messages(messages)
    })
  end

  defp emit_route_trace(trace, route, roles, turn) do
    emit_trace(trace, :route_selected, %{
      turn: turn,
      logits: Nx.to_flat_list(Nx.squeeze(route.logits, axes: [0])),
      agent_logits: Nx.to_flat_list(route.agent_logits),
      role_logits: Nx.to_flat_list(route.role_logits),
      selected_agent: route.agent_id,
      selected_role: Map.get(roles, route.role_id, "Worker"),
      selected_role_id: route.role_id
    })
  end

  defp emit_turn_completed(trace, fields), do: emit_trace(trace, :turn_completed, fields)

  defp emit_trace(%Trace.Context{} = trace, event, fields) do
    Trace.Context.write(trace, Trace.Event.new(event, trace.run_id, fields))
  end

  defp put_provider_pool(opts, nil), do: opts

  defp put_provider_pool(opts, provider_pool),
    do: Keyword.put(opts, :provider_pool, provider_pool)

  defp verifier_accept?(role_name, response_text, stop_token) when role_name == "Verifier" do
    response_text
    |> String.trim()
    |> String.upcase()
    |> String.starts_with?(String.upcase(stop_token))
  end

  defp verifier_accept?(_role_name, _response_text, _stop_token), do: false

  defp extract_router_tensor(_messages, nil), do: {:error, :missing_slm_context}

  defp extract_router_tensor(messages, {model_info, tokenizer}) do
    Extractor.extract_penultimate_hidden_state_with_metadata(model_info, tokenizer, messages)
  end

  defp extract_router_tensor(messages, %{model_info: model_info, tokenizer: tokenizer}) do
    extract_router_tensor(messages, {model_info, tokenizer})
  end

  defp extract_router_tensor(_messages, _context), do: {:error, :invalid_slm_context}
end
