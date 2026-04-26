defmodule TrinityCoordinator.Training.SepCMAESTrainingTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{
    CoordinationHead,
    Extractor,
    Runtime,
    Training.SepCMAES
  }

  @tiny_model {:hf, "hf-internal-testing/tiny-random-gpt2"}

  @tag :integration
  test "runs one generation with real extractor vectors and real route scoring" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(@tiny_model, Bumblebee.Text.Gpt2, :base)

    assert {:ok, vectors} =
             Extractor.extract_batch_penultimate_hidden_states(
               model_info,
               tokenizer,
               [
                 [%{"role" => "user", "content" => "Plan a minimal test response."}],
                 [%{"role" => "user", "content" => "Summarize a short theorem."}]
               ]
             )

    input_dim = Nx.axis_size(vectors, 1)
    num_agents = 3
    num_roles = 3
    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    initial_state = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    target_vector = Nx.slice(vectors, [0, 0], [1, input_dim])

    target_route =
      CoordinationHead.route(model, initial_state, target_vector, num_agents, num_roles)

    evaluator = fn candidate_model_state, _metadata ->
      route =
        CoordinationHead.route(model, candidate_model_state, target_vector, num_agents, num_roles)

      reward =
        if route.agent_id == target_route.agent_id and route.role_id == target_route.role_id do
          1.0
        else
          0.0
        end

      {:ok, reward}
    end

    assert {:ok, trained} =
             SepCMAES.train(
               {model, initial_state},
               [
                 population_size: 6,
                 generations: 1,
                 replications: 1,
                 top_candidates: 3,
                 sigma: 0.08,
                 seed: {7, 8, 9},
                 stop_threshold: 0.95
               ],
               evaluator
             )

    assert is_struct(trained.model_state, Axon.ModelState)
    assert trained.metrics.generations in 0..1
    assert trained.metrics.stop_reason in [:generation_budget, :threshold]
    assert Enum.any?(trained.trace, &(&1[:event] == :training_stopped))
  end

  test "trains from a real Axon model state using real router forward passes" do
    input_dim = 6
    num_agents = 3
    num_roles = 3
    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    initial_state = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    eval_vector = Nx.iota({1, input_dim}, type: :f32)

    target_route =
      CoordinationHead.route(model, initial_state, eval_vector, num_agents, num_roles)

    evaluator = fn candidate_model_state, _metadata ->
      route =
        CoordinationHead.route(model, candidate_model_state, eval_vector, num_agents, num_roles)

      reward =
        if route.agent_id == target_route.agent_id and route.role_id == target_route.role_id do
          1.0
        else
          0.0
        end

      {:ok, reward}
    end

    assert {:ok, trained} =
             SepCMAES.train(
               {model, initial_state},
               [
                 population_size: 6,
                 generations: 2,
                 replications: 1,
                 top_candidates: 2,
                 sigma: 0.08,
                 seed: {9, 9, 9},
                 stop_threshold: 0.95
               ],
               evaluator
             )

    assert is_number(trained.metrics.best_reward)
    assert trained.metrics.generations in 0..2
    assert length(trained.trace) <= 3
    assert is_struct(trained.model_state, Axon.ModelState)
  end

  test "returns an error for invalid evaluator outputs" do
    model = CoordinationHead.build_model(4, 2, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    initial_state = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())

    bad_evaluator = fn _candidate, _metadata -> :invalid end

    assert {:error, {:invalid_evaluator_result, :invalid}} =
             SepCMAES.train(
               {model, initial_state},
               [
                 population_size: 4,
                 generations: 1,
                 replications: 1,
                 top_candidates: 2,
                 sigma: 0.1
               ],
               bad_evaluator
             )
  end

  test "stops on evaluation and provider budgets with trace evidence" do
    input_dim = 4
    model = CoordinationHead.build_model(input_dim, 2, 2)
    {init_fn, _predict_fn} = Axon.build(model)
    initial_state = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    evaluator = fn _candidate_model_state, metadata ->
      case metadata.replication do
        1 -> {:ok, %{reward: 1.0, provider_cost_usd: 0.06}}
        2 -> {:ok, %{reward: 0.0, provider_cost_usd: 0.06}}
      end
    end

    assert {:ok, trained} =
             SepCMAES.train(
               {model, initial_state},
               [
                 population_size: 2,
                 generations: 10,
                 replications: 2,
                 top_candidates: 1,
                 sigma: 0.1,
                 stop_threshold: 0.99,
                 evaluation_budget: 2,
                 provider_budget_usd: 0.05
               ],
               evaluator
             )

    assert trained.metrics.stop_reason in [:evaluation_budget, :provider_budget]
    assert trained.metrics.provider_cost_usd > 0.0
    assert trained.metrics.evaluation_budget == 2
    assert trained.metrics.provider_budget_usd == 0.05
    assert Enum.any?(trained.trace, &Map.has_key?(&1, :best_candidate_vector_hash))
    assert Enum.any?(trained.trace, &Map.has_key?(&1, :best_candidate_model_state_hash))
    assert Enum.any?(trained.trace, &(&1[:event] == :training_stopped))
  end

  test "stops on cancellation callback before next generation" do
    input_dim = 4
    model = CoordinationHead.build_model(input_dim, 2, 2)
    {init_fn, _predict_fn} = Axon.build(model)
    initial_state = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    evaluator = fn _candidate_model_state, _metadata ->
      {:ok, 0.0}
    end

    on_exit(fn -> :erlang.erase(:sep_cancel_count) end)

    cancel_fn = fn ->
      count = :erlang.get(:sep_cancel_count)
      count = if count == :undefined, do: 0, else: count
      :erlang.put(:sep_cancel_count, count + 1)
      count >= 1
    end

    assert {:ok, trained} =
             SepCMAES.train(
               {model, initial_state},
               [
                 population_size: 2,
                 generations: 5,
                 replications: 1,
                 top_candidates: 1,
                 sigma: 0.1,
                 stop_threshold: 0.99,
                 cancellation_fn: cancel_fn
               ],
               evaluator
             )

    assert trained.metrics.stop_reason == :cancelled
    assert trained.metrics.generations == 1
    assert is_list(trained.trace)
  end

  test "aggregates evaluator rewards across replications" do
    input_dim = 6
    num_agents = 3
    num_roles = 3
    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    initial_state = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    pid = self()

    evaluator = fn _candidate_model_state, metadata ->
      send(pid, {:replication, metadata.replication})
      reward = if rem(metadata.replication, 2) == 0, do: 1.0, else: 0.0
      {:ok, reward}
    end

    assert {:ok, _trained} =
             SepCMAES.train(
               {model, initial_state},
               [
                 population_size: 4,
                 generations: 1,
                 replications: 3,
                 top_candidates: 2,
                 sigma: 0.08,
                 seed: {9, 9, 9},
                 stop_threshold: 0.0
               ],
               evaluator
             )

    assert Enum.count(receiveable_messages()) == 12
  end

  defp receiveable_messages do
    Stream.repeatedly(fn ->
      receive do
        {:replication, _replication} -> {:replication, :observed}
      after
        0 -> :done
      end
    end)
    |> Stream.take_while(&(&1 != :done))
    |> Enum.to_list()
  end
end
