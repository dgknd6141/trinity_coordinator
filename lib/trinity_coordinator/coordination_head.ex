defmodule TrinityCoordinator.CoordinationHead do
  @moduledoc """
  A routing head that maps SLM hidden states to agent/role logits.

  The default is a single dense projection (`:linear`). Optional variants include
  `:block_diagonal` and `:sparse` for ablation work.
  """

  @known_head_variants [:linear, :block_diagonal, :sparse]

  @doc """
  Builds the Axon model structure.

  Supported options:

    * `:head` - one of `:linear`, `:block_diagonal`, or `:sparse` (default `:linear`).
    * `:blocks` - number of blocks for `:block_diagonal` (default `1`).
    * `:sparse_k` - fixed feature width for `:sparse`.
  """
  def build_model(input_dim \\ 1024, num_agents \\ 7, num_roles \\ 3, opts \\ [])
      when is_integer(input_dim) and input_dim > 0 and
             is_integer(num_agents) and num_agents > 0 and
             is_integer(num_roles) and num_roles > 0 and
             is_list(opts) do
    head_opts = parse_head_options!(opts)
    total_outputs = num_agents + num_roles

    validate_head_dimensions!(input_dim, total_outputs, head_opts)

    build_model_for_head_variant(input_dim, total_outputs, head_opts)
  end

  defp build_model_for_head_variant(input_dim, total_outputs, head_opts) do
    case head_opts[:head] do
      :linear ->
        Axon.input("hidden_state", shape: {nil, input_dim})
        |> Axon.dense(total_outputs, name: "routing_head")

      :block_diagonal ->
        build_block_diagonal_model(input_dim, total_outputs, head_opts[:blocks])

      :sparse ->
        build_sparse_model(input_dim, total_outputs, head_opts[:sparse_k])
    end
  end

  @doc """
  Returns variant metadata and partition layout used for diagnostics and tests.
  """
  def variant_metadata(input_dim, num_agents, num_roles, opts \\ [])
      when is_integer(input_dim) and is_integer(num_agents) and is_integer(num_roles) and
             is_list(opts) do
    head_opts = parse_head_options!(opts)
    total_outputs = num_agents + num_roles

    validate_head_dimensions!(input_dim, total_outputs, head_opts)

    base = %{
      input_dim: input_dim,
      num_agents: num_agents,
      num_roles: num_roles,
      output_dim: total_outputs,
      head: head_opts[:head]
    }

    case head_opts[:head] do
      :linear ->
        base
        |> Map.put(:blocks, 1)
        |> Map.put(:effective_sparse_k, input_dim)
        |> Map.put(:input_partitions, [{0, input_dim}])
        |> Map.put(:output_partitions, [{0, total_outputs}])
        |> Map.put(:parameter_count, dense_param_count(input_dim, total_outputs))

      :block_diagonal ->
        in_counts = partition_counts(input_dim, head_opts[:blocks])
        out_counts = partition_counts(total_outputs, head_opts[:blocks])

        base
        |> Map.put(:blocks, head_opts[:blocks])
        |> Map.put(:input_partitions, partitions_with_offsets(in_counts))
        |> Map.put(:output_partitions, partitions_with_offsets(out_counts))
        |> Map.put(:parameter_count, block_diagonal_param_count(in_counts, out_counts))
        |> Map.put(:effective_sparse_k, nil)

      :sparse ->
        sparse_k = effective_sparse_k(head_opts[:sparse_k], input_dim)

        base
        |> Map.put(:blocks, 1)
        |> Map.put(:effective_sparse_k, sparse_k)
        |> Map.put(:input_partitions, [{0, sparse_k}])
        |> Map.put(:output_partitions, [{0, total_outputs}])
        |> Map.put(:parameter_count, dense_param_count(sparse_k, total_outputs))
    end
  end

  @doc """
  Returns trainable parameter count for the selected variant.
  """
  def parameter_count(input_dim, num_agents, num_roles, opts \\ []) when is_list(opts) do
    variant_metadata(input_dim, num_agents, num_roles, opts).parameter_count
  end

  @doc "Returns raw logits as a rank-2 tensor with shape {batch, num_agents+num_roles}."
  def output_logits(model, params, penultimate_tensor) do
    Axon.predict(model, params, %{"hidden_state" => penultimate_tensor})
  end

  @doc """
  Runs the real Axon forward pass and returns route details.
  """
  def route(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    logits = output_logits(model, params, penultimate_tensor)
    validate_logits!(logits, num_agents, num_roles)

    logits_1d = Nx.squeeze(logits, axes: [0])

    agent_logits = Nx.slice(logits_1d, [0], [num_agents])
    role_logits = Nx.slice(logits_1d, [num_agents], [num_roles])

    agent_id = Nx.to_number(Nx.argmax(agent_logits))
    role_id = Nx.to_number(Nx.argmax(role_logits))

    if not is_integer(agent_id) or not is_integer(role_id) do
      raise ArgumentError, "invalid argmax output from coordination head"
    end

    %{
      agent_id: agent_id,
      role_id: role_id,
      logits: logits,
      agent_logits: agent_logits,
      role_logits: role_logits
    }
  end

  @doc "Runs the forward pass and returns `{agent_id, role_id}`."
  def forward(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    route = route(model, params, penultimate_tensor, num_agents, num_roles)
    {route.agent_id, route.role_id}
  end

  @doc """
  Builds a combined one-hot label tensor for supervised head training.

  Each label row is `[agent_one_hot, role_one_hot]`.
  """
  def build_labels(agent_ids, role_ids, num_agents \\ 7, num_roles \\ 3)
      when is_list(agent_ids) and is_list(role_ids) do
    if length(agent_ids) != length(role_ids) do
      raise ArgumentError, "agent_ids and role_ids must have the same length"
    end

    agent_ids
    |> Enum.zip(role_ids)
    |> Enum.map(fn {agent_id, role_id} ->
      validate_label_id!(agent_id, num_agents, :agent_id)
      validate_label_id!(role_id, num_roles, :role_id)

      one_hot(agent_id, num_agents) ++ one_hot(role_id, num_roles)
    end)
    |> Nx.tensor(type: :f32)
  end

  @doc """
  Trains the coordination head with real Axon/Polaris supervised optimization.

  This is the direct supervised path described in the paper appendix: the SLM is
  frozen, extracted hidden-state vectors are provided as `features`, and only the
  lightweight routing head is trained.
  """
  def train_supervised(model, features, labels, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        num_agents: 7,
        num_roles: 3,
        epochs: 20,
        learning_rate: 0.01,
        compiler: EXLA,
        log: 0,
        initial_model_state: Axon.ModelState.empty()
      )

    validate_training_tensors!(features, labels, opts[:num_agents], opts[:num_roles])

    data = [{%{"hidden_state" => features}, labels}]
    optimizer = Polaris.Optimizers.adam(learning_rate: opts[:learning_rate])
    loss = supervised_loss(opts[:num_agents], opts[:num_roles])

    run_opts = [epochs: opts[:epochs]]

    run_opts =
      if opts[:compiler], do: Keyword.put(run_opts, :compiler, opts[:compiler]), else: run_opts

    model
    |> Axon.Loop.trainer(loss, optimizer, log: opts[:log])
    |> Axon.Loop.run(data, opts[:initial_model_state], run_opts)
  end

  defp supervised_loss(num_agents, num_roles) do
    fn y_true, y_pred ->
      batch_size = Nx.axis_size(y_true, 0)

      agent_true = Nx.slice(y_true, [0, 0], [batch_size, num_agents])
      role_true = Nx.slice(y_true, [0, num_agents], [batch_size, num_roles])

      agent_pred = Nx.slice(y_pred, [0, 0], [batch_size, num_agents])
      role_pred = Nx.slice(y_pred, [0, num_agents], [batch_size, num_roles])

      agent_loss =
        Axon.Losses.categorical_cross_entropy(agent_true, agent_pred,
          from_logits: true,
          reduction: :mean
        )

      role_loss =
        Axon.Losses.categorical_cross_entropy(role_true, role_pred,
          from_logits: true,
          reduction: :mean
        )

      Nx.add(agent_loss, role_loss)
    end
  end

  defp validate_logits!(logits, num_agents, num_roles) do
    output_dim = num_agents + num_roles
    logits_shape = Nx.shape(logits)

    case logits_shape do
      {1, ^output_dim} ->
        :ok

      {_batch, ^output_dim} ->
        raise ArgumentError,
              "coordination routing expects a single example, got shape #{inspect(logits_shape)}"

      {_batch, _dim} ->
        raise ArgumentError,
              "coordination head must output #{output_dim} logits, got shape #{inspect(logits_shape)}"

      _ ->
        raise ArgumentError, "invalid coordination head output shape #{inspect(logits_shape)}"
    end
  end

  defp validate_training_tensors!(features, labels, num_agents, num_roles) do
    output_dim = num_agents + num_roles

    case {Nx.shape(features), Nx.shape(labels)} do
      {{batch_size, _input_dim}, {batch_size, ^output_dim}} when batch_size > 0 ->
        :ok

      {feature_shape, label_shape} ->
        raise ArgumentError,
              "invalid training tensor shapes, got features #{inspect(feature_shape)} and labels #{inspect(label_shape)}"
    end
  end

  defp validate_label_id!(id, limit, _name) when is_integer(id) and id >= 0 and id < limit,
    do: :ok

  defp validate_label_id!(id, limit, name) do
    raise ArgumentError, "#{name} must be an integer in 0..#{limit - 1}, got #{inspect(id)}"
  end

  defp parse_head_options!(opts) do
    head = Keyword.get(opts, :head, :linear)
    blocks = Keyword.get(opts, :blocks, 1)
    sparse_k = Keyword.get(opts, :sparse_k, nil)

    unless head in @known_head_variants do
      raise ArgumentError, "invalid head variant #{inspect(head)}"
    end

    blocks = normalize_blocks(head, blocks)
    sparse_k = normalize_sparse_k(sparse_k)

    %{head: head, blocks: blocks, sparse_k: sparse_k}
  end

  defp normalize_blocks(:block_diagonal, value) do
    if is_integer(value) and value > 0 do
      value
    else
      raise ArgumentError, "blocks must be a positive integer"
    end
  end

  defp normalize_blocks(_head, _value), do: 1

  defp normalize_sparse_k(value) when is_integer(value) and value > 0, do: value
  defp normalize_sparse_k(nil), do: nil

  defp normalize_sparse_k(_value),
    do: raise(ArgumentError, "sparse_k must be nil or positive integer")

  defp validate_head_dimensions!(input_dim, output_dim, head_opts) do
    if head_opts[:head] == :block_diagonal &&
         (head_opts[:blocks] > input_dim || head_opts[:blocks] > output_dim) do
      raise ArgumentError,
            "block_diagonal requires blocks <= input_dim and <= num_agents+num_roles, got blocks=#{head_opts[:blocks]}, input_dim=#{input_dim}, output_dim=#{output_dim}"
    end

    if head_opts[:head] == :sparse do
      sparse_k = head_opts[:sparse_k] || input_dim

      unless is_integer(sparse_k) and sparse_k >= 1 and sparse_k <= input_dim do
        raise ArgumentError,
              "sparse_k must be between 1 and input_dim (#{input_dim}), got #{sparse_k}"
      end
    end

    :ok
  end

  defp build_block_diagonal_model(input_dim, output_dim, blocks) do
    input_node = Axon.input("hidden_state", shape: {nil, input_dim})
    in_counts = partition_counts(input_dim, blocks)
    out_counts = partition_counts(output_dim, blocks)

    in_partitions = partitions_with_offsets(in_counts)
    out_partitions = partitions_with_offsets(out_counts)

    block_layers =
      Enum.zip(in_partitions, out_partitions)
      |> Enum.with_index()
      |> Enum.map(fn {{{in_start, in_count}, {_, out_count}}, idx} ->
        input_slice =
          Axon.nx(input_node, fn x ->
            Nx.slice(x, [0, in_start], [Nx.axis_size(x, 0), in_count])
          end)

        Axon.dense(input_slice, out_count, name: "routing_head_block_#{idx}")
      end)

    Axon.concatenate(block_layers, axis: 1, name: "routing_head")
  end

  defp build_sparse_model(input_dim, output_dim, sparse_k) do
    input_node = Axon.input("hidden_state", shape: {nil, input_dim})
    keep = effective_sparse_k(sparse_k, input_dim)

    sliced_input =
      Axon.nx(input_node, fn x ->
        Nx.slice(x, [0, 0], [Nx.axis_size(x, 0), keep])
      end)

    Axon.dense(sliced_input, output_dim, name: "routing_head")
  end

  defp effective_sparse_k(nil, input_dim), do: input_dim
  defp effective_sparse_k(sparse_k, _input_dim), do: sparse_k

  defp partition_counts(total, parts) do
    base = div(total, parts)
    remainder = rem(total, parts)

    0..(parts - 1)
    |> Enum.map(fn index ->
      if index < remainder do
        base + 1
      else
        base
      end
    end)
  end

  defp partitions_with_offsets(counts) do
    {_size, partitions} =
      Enum.reduce(counts, {0, []}, fn count, {offset, acc} ->
        {offset + count, [{offset, count} | acc]}
      end)

    Enum.reverse(partitions)
  end

  defp block_diagonal_param_count(in_counts, out_counts) do
    Enum.zip(in_counts, out_counts)
    |> Enum.reduce(0, fn {in_count, out_count}, acc ->
      acc + in_count * out_count + out_count
    end)
  end

  defp dense_param_count(input_dim, output_dim), do: input_dim * output_dim + output_dim

  defp one_hot(index, size) do
    Enum.map(0..(size - 1), fn
      ^index -> 1.0
      _ -> 0.0
    end)
  end
end
