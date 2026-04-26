defmodule Mix.Tasks.Trinity.Benchmark do
  @moduledoc """
  Run TRINITY benchmark suites.

  Example:

      XLA_TARGET=cuda12 mix trinity.benchmark --suite separability --dataset test/fixtures/benchmark_cases.jsonl --out tmp/separability.json
  """

  use Mix.Task

  alias TrinityCoordinator.{
    Benchmark.Dataset,
    Benchmark.FeatureExtractor,
    Benchmark.Report,
    Benchmark.Routing,
    Benchmark.Separability,
    Benchmark.TurnBudget,
    CoordinationHead,
    Runtime,
    SLMProfile
  }

  @shortdoc "Run TRINITY benchmark harnesses"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _} = parse_args(args)

    suite = normalize_suite(opts[:suite] || "all")
    dataset_path = Keyword.get(opts, :dataset, default_dataset_path())
    out_path = Keyword.get(opts, :out)
    profile_name = Keyword.get(opts, :profile, "tiny_gpt2")
    _provider_pool = opts[:provider_pool] || "default"
    num_agents = Keyword.get(opts, :num_agents, 7)
    num_roles = Keyword.get(opts, :num_roles, 3)
    max_turns = Keyword.get(opts, :max_turns, 5)
    blocks = Keyword.get(opts, :blocks)
    sparse_k = Keyword.get(opts, :sparse_k)
    head = parse_head(opts[:head] || "linear")

    head_opts =
      []
      |> maybe_put(:blocks, blocks)
      |> maybe_put(:sparse_k, sparse_k)
      |> Keyword.put(:head, head)

    IO.puts("TRINITY Benchmark Task")
    IO.puts("  Suite: #{suite}")
    IO.puts("  Profile: #{profile_name}")
    IO.puts("  Dataset: #{dataset_path}")

    profile = resolve_profile(profile_name)
    {model_info, tokenizer} = load_profile_or_raise(profile)

    cases = load_cases_or_raise(dataset_path)
    dataset_id = Path.basename(dataset_path)
    feature_limit = Keyword.get(opts, :limit)
    cases = if feature_limit, do: Enum.take(cases, feature_limit), else: cases

    {:ok, features} =
      run_feature_extractor_or_raise(
        %{model_info: model_info, tokenizer: tokenizer},
        cases
      )

    input_dim = Nx.axis_size(features, 1)
    model = CoordinationHead.build_model(input_dim, num_agents, num_roles, head_opts)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, Nx.type(features)), Axon.ModelState.empty())

    payload_ctx = %{
      cases: cases,
      features: features,
      model_info: model_info,
      tokenizer: tokenizer,
      max_turns: max_turns,
      model: model,
      params: params,
      num_agents: num_agents,
      num_roles: num_roles
    }

    payload = build_payload(suite, payload_ctx)

    report =
      Report.envelope(suite, payload,
        profile: profile.name,
        head: head,
        head_opts: head_opts,
        dataset_id: dataset_id,
        dataset_hash: dataset_hash(cases),
        platform: Runtime.supported_platforms(),
        xla_target: System.get_env("XLA_TARGET", "")
      )

    if out_path do
      :ok = Report.write(out_path, report)
      IO.puts("Wrote report to #{out_path}")
    else
      Mix.shell().info(Jason.encode!(payload, pretty: true))
    end

    :ok
  end

  defp default_dataset_path do
    Path.expand("test/fixtures/benchmark_cases.jsonl", File.cwd!())
  end

  defp normalize_suite("separability"), do: :separability
  defp normalize_suite("routing"), do: :routing
  defp normalize_suite("turn-budget"), do: :turn_budget
  defp normalize_suite("turn_budget"), do: :turn_budget
  defp normalize_suite("all"), do: :all
  defp normalize_suite(_), do: :all

  defp parse_args(args) do
    case OptionParser.parse(args,
           strict: [
             suite: :string,
             dataset: :string,
             profile: :string,
             provider_pool: :string,
             head: :string,
             blocks: :integer,
             sparse_k: :integer,
             limit: :integer,
             num_agents: :integer,
             num_roles: :integer,
             max_turns: :integer,
             out: :string
           ]
         ) do
      {opts, rest, []} when is_list(rest) ->
        {opts, rest}

      {_opts, _rest, errors} ->
        raise ArgumentError, "Invalid benchmark options: #{inspect(errors)}"
    end
  end

  defp parse_head(nil), do: :linear

  defp parse_head(name) when is_binary(name) do
    name
    |> String.trim()
    |> String.downcase()
    |> String.replace("-", "_")
    |> String.to_atom()
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

  defp resolve_profile(value) when is_binary(value) do
    value
    |> String.to_atom()
    |> resolve_profile()
  end

  defp resolve_profile(other) do
    case SLMProfile.profile(other) do
      {:ok, profile} ->
        profile

      {:error, reason} ->
        raise ArgumentError, "unsupported benchmark profile #{inspect(other)}: #{inspect(reason)}"
    end
  end

  defp build_payload(:separability, %{cases: cases, features: features}) do
    {:ok, metrics} = Separability.run(cases, features)
    %{separability: metrics}
  end

  defp build_payload(
         :routing,
         %{
           cases: cases,
           features: features,
           model: model,
           params: params,
           num_agents: num_agents,
           num_roles: num_roles
         }
       ) do
    {:ok, metrics} =
      Routing.run(cases, features, model, params,
        num_agents: num_agents,
        num_roles: num_roles
      )

    %{routing: metrics}
  end

  defp build_payload(
         :turn_budget,
         %{
           cases: cases,
           features: features,
           model: model,
           params: params,
           model_info: model_info,
           tokenizer: tokenizer,
           max_turns: max_turns,
           num_agents: num_agents,
           num_roles: num_roles
         }
       ) do
    {:ok, result} =
      TurnBudget.run(
        cases,
        %{model_info: model_info, tokenizer: tokenizer},
        features,
        model,
        params,
        max_turns: max_turns,
        num_agents: num_agents,
        num_roles: num_roles
      )

    %{turn_budget: result.summary}
  end

  defp build_payload(
         :all,
         %{
           cases: cases,
           features: features,
           model: model,
           params: params,
           model_info: model_info,
           tokenizer: tokenizer,
           max_turns: max_turns,
           num_agents: num_agents,
           num_roles: num_roles
         }
       ) do
    {:ok, separability} = Separability.run(cases, features)

    {:ok, routing} =
      Routing.run(cases, features, model, params,
        num_agents: num_agents,
        num_roles: num_roles
      )

    {:ok, turn_budget} =
      TurnBudget.run(
        cases,
        %{model_info: model_info, tokenizer: tokenizer},
        features,
        model,
        params,
        max_turns: max_turns,
        num_agents: num_agents,
        num_roles: num_roles
      )

    %{separability: separability, routing: routing, turn_budget: turn_budget.summary}
  end

  defp dataset_hash(cases) do
    :crypto.hash(:sha256, :erlang.term_to_binary(%{count: length(cases)}))
    |> Base.encode16(case: :lower)
  end

  defp load_profile_or_raise(profile) do
    case SLMProfile.load_profile(profile) do
      {:ok, payload} -> payload
      {:error, reason} -> raise ArgumentError, "Unable to load profile: #{inspect(reason)}"
    end
  end

  defp load_cases_or_raise(path) do
    case Dataset.load!(path) do
      {:ok, cases} -> cases
      {:error, reason} -> raise ArgumentError, "Unable to load dataset: #{inspect(reason)}"
    end
  end

  defp run_feature_extractor_or_raise(slm_context, cases) do
    case FeatureExtractor.run(slm_context, cases) do
      {:ok, features} -> {:ok, features}
      {:error, reason} -> raise ArgumentError, "Unable to extract features: #{inspect(reason)}"
    end
  end
end
