defmodule TrinityCoordinator.Training.SepCMAES do
  @moduledoc """
  sep-CMA-ES optimization for the coordination-head router using terminal rewards.

  The trainer is deterministic: model states are flattened, sampled candidates are
  evaluated through the evaluator boundary, and the parent mean updates from top
  candidates.
  """

  alias TrinityCoordinator.Trace.Hash
  alias TrinityCoordinator.Training.SepCMAES.{Candidate, Codec, Config, Reward, Sampling, State}

  @type evaluator_fun ::
          (map(), map() ->
             {:ok, number()}
             | {:ok, [number()]}
             | {:ok, %{reward: number() | [number()], provider_cost_usd: number() | nil}}
             | {:error, term()})

  @type trained_result :: %{
          model_state: map(),
          metrics: map(),
          trace: [map()]
        }

  @doc """
  Runs a seeded sep-CMA-ES loop over model parameters.
  """
  @spec train(
          keyword() | map() | {Axon.t(), map()},
          keyword() | Config.t() | map(),
          evaluator_fun()
        ) :: {:ok, trained_result()} | {:error, term()}
  def train(initial_state, config_opts, evaluator) when is_function(evaluator, 2) do
    with {:ok, config} <- Config.new(config_opts),
         {:ok, _model, initial_params} <- extract_model_and_params(initial_state) do
      init_state = init_training_state(initial_params, config)
      train_generations(init_state, config, evaluator)
    end
  end

  def train(_initial_state, _config_opts, _evaluator) do
    {:error, :invalid_arguments}
  end

  defp extract_model_and_params({model, %Axon.ModelState{} = params}) when not is_nil(model) do
    {:ok, model, params}
  end

  defp extract_model_and_params(%{model: model, model_state: %Axon.ModelState{} = params})
       when not is_nil(model) do
    {:ok, model, params}
  end

  defp extract_model_and_params(%{model: model, params: %Axon.ModelState{} = params})
       when not is_nil(model) do
    {:ok, model, params}
  end

  defp extract_model_and_params(_), do: {:error, :invalid_model_state}

  defp init_training_state(%Axon.ModelState{} = params, config) do
    {mean_vector, metadata} = Codec.flatten_model_state(params)

    State.new(
      generation: 0,
      mean_vector: Nx.as_type(mean_vector, :f32),
      sigma: config.sigma,
      seed: config.seed,
      template_state: params,
      model_metadata: metadata
    )
  end

  defp train_generations(state, config, evaluator) do
    case stop_reason(state, config) do
      :continue ->
        with {:ok, candidates, evaluations_delta, provider_cost_delta} <-
               evaluate_generation(state, config, evaluator),
             {:ok, next_state} <-
               recombine_generation(
                 state,
                 candidates,
                 config,
                 evaluations_delta,
                 provider_cost_delta
               ) do
          train_generations(next_state, config, evaluator)
        end

      {:stop, reason} ->
        {:ok, finish(state, config, reason)}
    end
  end

  defp stop_reason(state, config) do
    cond do
      state.best_reward >= config.stop_threshold ->
        {:stop, :threshold}

      cancellation_requested?(config.cancellation_fn) ->
        {:stop, :cancelled}

      state.generation >= config.generations ->
        {:stop, :generation_budget}

      budget_hit?(state.evaluations, config.evaluation_budget) ->
        {:stop, :evaluation_budget}

      budget_hit?(state.provider_cost_usd, config.provider_budget_usd) ->
        {:stop, :provider_budget}

      true ->
        :continue
    end
  end

  defp budget_hit?(_, nil), do: false

  defp budget_hit?(value, budget) when is_number(value) and is_number(budget) do
    value >= budget
  end

  defp budget_hit?(_, _), do: false

  defp cancellation_requested?(nil), do: false

  defp cancellation_requested?(cancellation_fn) when is_function(cancellation_fn, 0) do
    cancellation_fn.() == true
  end

  defp evaluate_generation(state, config, evaluator) do
    sample_seed = evolve_seed(state.seed, state.generation)
    dim = Nx.size(state.mean_vector)

    {samples, sample_meta} =
      Sampling.sample(state.mean_vector, state.sigma, config.population_size, sample_seed)

    context = %{
      state: state,
      samples: samples,
      dim: dim,
      sample_seed: sample_seed,
      sample_meta: sample_meta,
      evaluator: evaluator,
      config: config
    }

    candidates_result =
      0..(config.population_size - 1)
      |> Enum.reduce_while({[], 0.0, 0.0}, &evaluate_single_candidate(&1, &2, context))

    finalize_generation_candidates(candidates_result)
  end

  defp finalize_generation_candidates({candidates, eval_delta, cost_delta, reason}) do
    finalize_candidate_window({candidates, eval_delta, cost_delta, reason})
  end

  defp finalize_generation_candidates({:ok, candidates, eval_delta, cost_delta, reason}) do
    finalize_candidate_window({candidates, eval_delta, cost_delta, reason})
  end

  defp finalize_generation_candidates({candidates, eval_delta, cost_delta}) do
    finalize_candidate_window({candidates, eval_delta, cost_delta, :none})
  end

  defp finalize_generation_candidates({:error, reason}), do: {:error, reason}

  defp evaluate_single_candidate(
         index,
         {acc, evals, provider_cost},
         context
       ) do
    state = context.state

    if generation_budget_exhausted?(state, evals, context.config) do
      {:halt, {:ok, Enum.reverse(acc), evals, provider_cost, :evaluation_budget}}
    else
      vector =
        Nx.slice(context.samples, [index, 0], [1, context.dim]) |> Nx.squeeze(axes: [0])

      candidate_metadata =
        candidate_metadata(state.generation, index, context.sample_seed, context.sample_meta)

      evaluate_decoded_candidate(context, vector, candidate_metadata, acc, evals, provider_cost)
    end
  end

  defp generation_budget_exhausted?(state, evals, config) do
    budget_hit?(state.evaluations + evals, config.evaluation_budget) or
      not budget_allows?(config.replications, state.evaluations + evals, config.evaluation_budget)
  end

  defp candidate_metadata(generation, candidate_id, sample_seed, sample_meta) do
    %{
      generation: generation,
      candidate_id: candidate_id,
      sample_seed: sample_seed,
      sample_meta: sample_meta
    }
  end

  defp evaluate_decoded_candidate(
         context,
         vector,
         candidate_metadata,
         acc,
         evals,
         provider_cost
       ) do
    case decode_and_evaluate(
           context.state,
           vector,
           candidate_metadata,
           context.evaluator,
           context.config
         ) do
      {:ok, candidate, candidate_evals, candidate_cost} ->
        {:cont, {[candidate | acc], evals + candidate_evals, provider_cost + candidate_cost}}

      {:error, _} = error ->
        {:halt, error}
    end
  end

  defp finalize_candidate_window({candidates, eval_delta, cost_delta, reason}) do
    candidates = Enum.reverse(candidates)

    case candidates do
      [] ->
        if reason == :none, do: {:error, :no_valid_candidates}, else: {:error, reason}

      _ ->
        {:ok, candidates, eval_delta, cost_delta}
    end
  end

  defp decode_and_evaluate(state, vector, metadata, evaluator, config) do
    model_state = restore_candidate_model(state, vector)

    with {:ok, rewards, provider_cost, used_evals} <-
           evaluate_replicates(model_state, metadata, evaluator, config.replications) do
      candidate_metadata =
        metadata
        |> Map.put(:vector_hash, Hash.tensor(vector))
        |> Map.put(:model_state_hash, Hash.metadata(model_state.data))

      {:ok,
       Candidate.new(
         metadata.candidate_id,
         metadata.generation,
         vector,
         rewards,
         candidate_metadata
       ), used_evals, provider_cost}
    end
  end

  defp evaluate_replicates(model_state, metadata, evaluator, replications)
       when is_integer(replications) and replications > 0 do
    1..replications
    |> Enum.reduce_while({[], 0.0}, fn replication, {acc_rewards, acc_cost} ->
      case parse_evaluator_result(
             evaluator.(model_state, Map.put(metadata, :replication, replication))
           ) do
        {:ok, rewards, provider_cost} ->
          {:cont, {reward_concat(rewards, acc_rewards), acc_cost + provider_cost}}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:error, {:invalid_evaluator_result, _}} = error ->
        error

      {:error, _} = error ->
        error

      {rewards, provider_cost} ->
        {:ok, rewards, provider_cost, replications}
    end
  end

  defp budget_allows?(_required, _consumed, nil), do: true

  defp budget_allows?(required, consumed, budget)
       when is_number(required) and is_number(consumed) and is_number(budget),
       do: consumed + required <= budget

  defp budget_allows?(_, _, _), do: false

  defp parse_evaluator_result({:ok, result}) when is_list(result) do
    parse_reward_values(result, 0.0)
  end

  defp parse_evaluator_result({:ok, result}) when is_number(result) do
    parse_reward_values(result, 0.0)
  end

  defp parse_evaluator_result({:ok, %{reward: reward} = payload})
       when is_number(reward) or is_list(reward) do
    parse_reward_values(reward, Map.get(payload, :provider_cost_usd, 0.0))
  end

  defp parse_evaluator_result({:error, reason}) do
    {:error, {:invalid_evaluator_result, reason}}
  end

  defp parse_evaluator_result(other) do
    {:error, {:invalid_evaluator_result, other}}
  end

  defp parse_reward_values(reward, provider_cost) when is_number(reward) do
    if reward >= 0 and reward <= 1 do
      {:ok, [Reward.normalize(reward)], normalize_provider_cost(provider_cost)}
    else
      {:error, {:invalid_evaluator_result, reward}}
    end
  end

  defp parse_reward_values(rewards, provider_cost) when is_list(rewards) and rewards != [] do
    if Enum.all?(rewards, &valid_reward_value?/1) do
      {:ok, Enum.map(rewards, &Reward.normalize/1), normalize_provider_cost(provider_cost)}
    else
      {:error, {:invalid_evaluator_result, rewards}}
    end
  end

  defp parse_reward_values([], _), do: {:error, :invalid_evaluator_result}

  defp valid_reward_value?(value) when is_number(value), do: value >= 0 and value <= 1
  defp valid_reward_value?(_), do: false

  defp normalize_provider_cost(nil), do: 0.0
  defp normalize_provider_cost(cost) when is_number(cost), do: max(cost, 0.0)
  defp normalize_provider_cost(_), do: 0.0

  defp reward_concat(rewards, acc) do
    acc ++ rewards
  end

  defp recombine_generation(state, candidates, config, eval_delta, provider_delta) do
    ranked = Enum.sort_by(candidates, & &1.mean_reward, :desc)
    top = Enum.take(ranked, config.top_candidates)
    best_candidate = hd(top)

    top_vectors =
      top
      |> Enum.map(&Nx.reshape(&1.vector, {1, Nx.size(&1.vector)}))
      |> Nx.concatenate(axis: 0)

    mean_vector = Nx.mean(top_vectors, axes: [0])
    best_reward = max(best_candidate.mean_reward, state.best_reward)

    best_vector =
      if best_candidate.mean_reward > state.best_reward,
        do: best_candidate.vector,
        else: state.best_vector

    entry =
      Map.merge(
        %{
          generation: state.generation,
          best_candidate_id: best_candidate.id,
          best_candidate_mean_reward: best_candidate.mean_reward,
          top_candidate_ids: Enum.map(top, & &1.id),
          sigma: state.sigma,
          evaluations_delta: eval_delta,
          provider_cost_delta_usd: provider_delta
        },
        best_candidate_metadata_entry(best_candidate)
      )

    {:ok,
     %State{
       state
       | mean_vector: mean_vector,
         sigma: next_sigma(state.sigma, config.population_size),
         best_reward: best_reward,
         best_vector: best_vector,
         seed: evolve_seed(state.seed, state.generation),
         evaluations: state.evaluations + eval_delta,
         provider_cost_usd: state.provider_cost_usd + provider_delta,
         trace: [entry | state.trace],
         generation: state.generation + 1
     }}
  end

  defp best_candidate_metadata_entry(candidate) do
    metadata = candidate.metadata || %{}

    %{
      best_candidate_vector_hash: Map.get(metadata, :vector_hash),
      best_candidate_model_state_hash: Map.get(metadata, :model_state_hash)
    }
  end

  defp next_sigma(sigma, population_size),
    do: Float.round(sigma * :math.pow(0.999, 1 / population_size), 12)

  defp finish(state, config, reason) do
    trained_state = restore_candidate_model(state, state.best_vector)

    metrics =
      %{best_reward: state.best_reward, generations: state.generation, stop_reason: reason}
      |> Map.merge(%{
        populations_evaluated: state.evaluations,
        provider_cost_usd: state.provider_cost_usd,
        evaluation_budget: config.evaluation_budget,
        provider_budget_usd: config.provider_budget_usd
      })

    %{
      model_state: trained_state,
      metrics: metrics,
      trace: [
        %{event: :training_stopped, stop_reason: reason, generation: state.generation}
        | Enum.reverse(state.trace)
      ]
    }
  end

  defp restore_candidate_model(
         %State{template_state: template_state, model_metadata: metadata},
         vector
       ) do
    Codec.unflatten_model_state(vector, metadata, template_state)
  end

  defp evolve_seed({a, b, c}, generation), do: {a + generation, b + generation, c + generation}
end
