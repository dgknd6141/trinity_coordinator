defmodule Mix.Tasks.Trinity.Demo do
  @moduledoc """
  Demonstrates the real GPU-backed TRINITY router path.

      XLA_TARGET=cuda12 mix trinity.demo
  """

  use Mix.Task

  alias TrinityCoordinator.{CoordinationHead, Extractor, ProviderPool, Runtime, SLMProfile}

  @shortdoc "Runs a real GPU-backed TRINITY router demonstration"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")
    Logger.configure(level: :error)
    :logger.set_primary_config(:level, :error)

    {opts, []} =
      OptionParser.parse!(args,
        strict: [
          pool: :string,
          provider_pool: :string,
          profile: :string,
          head: :string,
          blocks: :integer,
          sparse_k: :integer
        ]
      )

    pool_name =
      opts
      |> Keyword.get(:provider_pool, Keyword.get(opts, :pool, "default"))
      |> to_atom_name()

    head = opts |> Keyword.get(:head, "linear") |> parse_head_name()

    head_opts =
      []
      |> maybe_put_head_option(:blocks, Keyword.get(opts, :blocks))
      |> maybe_put_head_option(:sparse_k, Keyword.get(opts, :sparse_k))

    selected_head =
      Keyword.put(head_opts, :head, head)

    pool_size = ProviderPool.size(pool_name)

    profile =
      opts
      |> Keyword.get(:profile, "tiny_gpt2")
      |> resolve_profile()

    info("TRINITY Coordinator GPU demo")
    info("==============================")
    info("")

    Runtime.put_cuda_backend!()

    platforms = Runtime.supported_platforms()
    info("1. EXLA runtime")
    info("   XLA_TARGET: #{System.get_env("XLA_TARGET", "(unset)")}")
    info("   Supported platforms: #{inspect(platforms)}")
    info("")

    info("2. Provider pool")
    info("   name: #{inspect(pool_name)}")
    info("   size: #{pool_size}")
    info("")

    info("3. Loading real SLM and tokenizer")

    {:ok, model_info, tokenizer} =
      case SLMProfile.load_profile(profile) do
        {:ok, {loaded_model_info, loaded_tokenizer}} ->
          {:ok, loaded_model_info, loaded_tokenizer}

        {:error, {:unsupported_profile, profile_name, reason}} ->
          Mix.raise(
            "Profile #{inspect(profile_name)} is currently unsupported: #{inspect(reason)}. " <>
              "Set up a supported router profile or wait for dependency compatibility."
          )

        {:error, reason} ->
          Mix.raise("Failed to load profile #{inspect(profile.name)}: #{inspect(reason)}")
      end

    info("   Profile: #{profile.name}")
    info("   Repository: #{inspect(profile.repo)}")
    info("   Model module: #{inspect(profile.module)}")
    info("   Expected hidden size: #{profile.expected_hidden_size}")
    info("")

    messages = [%{"role" => "user", "content" => "Find a strategy for a short algebra proof."}]

    info("4. Formatting transcript and running real SLM forward pass")

    {:ok, metadata} =
      Extractor.extract_penultimate_hidden_state_with_metadata(model_info, tokenizer, messages)

    info("   Transcript:")
    info(indent(metadata.transcript, 6))
    info("   Tokenizer input shapes: #{inspect(metadata.input_shapes)}")
    info("   Final hidden-state tensor shape: #{inspect(metadata.hidden_state_shape)}")
    info("   Second-to-last token vector shape: #{inspect(metadata.vector_shape)}")
    info("   Vector backend: #{Runtime.tensor_backend(metadata.vector)}")
    info("")

    training_batches = [
      [%{"role" => "user", "content" => "Solve a symbolic algebra problem."}],
      [%{"role" => "user", "content" => "Write and debug a small function."}],
      [%{"role" => "user", "content" => "Check whether this proof is complete."}],
      [%{"role" => "user", "content" => "Plan a multi-step reasoning solution."}]
    ]

    info("5. Extracting real SLM vectors for supervised head training")

    {:ok, features} =
      Extractor.extract_batch_penultimate_hidden_states(model_info, tokenizer, training_batches)

    info("   Training examples: #{length(training_batches)}")
    info("   Feature tensor shape: #{inspect(Nx.shape(features))}")
    info("   Feature backend: #{Runtime.tensor_backend(features)}")
    info("")

    num_agents = 3
    num_roles = 3

    model =
      CoordinationHead.build_model(
        Nx.axis_size(features, 1),
        num_agents,
        num_roles,
        selected_head
      )

    model_metadata =
      CoordinationHead.variant_metadata(
        Nx.axis_size(features, 1),
        num_agents,
        num_roles,
        selected_head
      )

    feature_params =
      CoordinationHead.parameter_count(
        Nx.axis_size(features, 1),
        num_agents,
        num_roles,
        selected_head
      )

    labels = CoordinationHead.build_labels([0, 1, 2, 0], [1, 1, 2, 0], num_agents, num_roles)

    info("6. Training real Axon coordination head")

    info("   Head variant: #{selected_head[:head]}")
    info("   Head blocks: #{inspect(model_metadata[:blocks])}")
    info("   Head effective sparse_k: #{inspect(model_metadata[:effective_sparse_k])}")
    info("   Head parameter count: #{feature_params}")

    trained_state =
      CoordinationHead.train_supervised(model, features, labels,
        num_agents: num_agents,
        num_roles: num_roles,
        epochs: 30,
        learning_rate: 0.05,
        compiler: EXLA
      )

    info("   Head input dimension: #{Nx.axis_size(features, 1)}")

    info(
      "   Output logits: #{num_agents + num_roles} (#{num_agents} agents + #{num_roles} roles)"
    )

    info("   Trained state: #{inspect(trained_state)}")
    info("")

    info("7. Routing the original transcript")
    route = CoordinationHead.route(model, trained_state, metadata.vector, num_agents, num_roles)
    info("   Logits backend: #{Runtime.tensor_backend(route.logits)}")
    info("   Agent logits: #{inspect_rounded(route.agent_logits)}")
    info("   Role logits: #{inspect_rounded(route.role_logits)}")
    info("   Selected agent id: #{route.agent_id}")
    info("   Selected role id: #{route.role_id} (#{role_name(route.role_id)})")
    info("")

    info(
      "Demo complete: real Bumblebee SLM forward pass -> second-to-last hidden-state vector ->"
    )

    info("real Axon training -> real Axon routing forward pass, all on EXLA CUDA.")
  end

  defp inspect_rounded(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&Float.round(&1, 4))
    |> inspect()
  end

  defp role_name(0), do: "Thinker"
  defp role_name(1), do: "Worker"
  defp role_name(2), do: "Verifier"
  defp role_name(_), do: "Unknown"

  defp to_atom_name(nil), do: :default
  defp to_atom_name(value) when is_atom(value), do: value
  defp to_atom_name(value) when is_binary(value), do: String.to_atom(value)

  defp maybe_put_head_option(options, _key, nil), do: options

  defp maybe_put_head_option(options, key, value) when is_atom(key) do
    Keyword.put(options, key, value)
  end

  defp parse_head_name(name) when is_binary(name) do
    name
    |> String.trim()
    |> String.downcase()
    |> String.replace("-", "_")
    |> String.to_atom()
  end

  defp resolve_profile(name) when is_binary(name) do
    name
    |> String.to_atom()
    |> resolve_profile()
  end

  defp resolve_profile(name) when is_atom(name) do
    case SLMProfile.profile(name) do
      {:error, reason} ->
        raise ArgumentError, "unsupported profile #{inspect(name)}: #{inspect(reason)}"

      {:ok, profile} ->
        profile
    end
  end

  defp resolve_profile(_), do: raise(ArgumentError, "invalid profile")

  defp indent(text, spaces) do
    prefix = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map_join("\n", &(prefix <> &1))
  end

  defp info(message), do: Mix.shell().info(message)
end
