defmodule TrinityCoordinator.Sakana.Head do
  @moduledoc """
  Utilities for building a standalone Axon routing head from Sakana head weights.

  The Qwen/Bumblebee causal-LM params do not contain a `routing_head`; the routing
  head is a separate Axon model.  This module makes that separation explicit.
  """

  alias TrinityCoordinator.{CoordinationHead, Runtime}

  @routing_head_layer "routing_head"

  @type build_result :: %{
          required(:model) => Axon.t(),
          required(:params) => struct(),
          required(:hidden_size) => pos_integer(),
          required(:output_count) => pos_integer(),
          required(:num_agents) => pos_integer(),
          required(:num_roles) => pos_integer()
        }

  @doc """
  Builds a routing head model and initialized params from `{output_count, hidden_size}` weights.

  Options:

    * `:num_roles` - default `3`.
    * `:backend` - optional backend for initialized params, e.g. `{EXLA.Backend, client: :cuda}`.
    * `:model_opts` - forwarded to `CoordinationHead.build_model/4`.
  """
  @spec build_routing_state(term(), keyword()) :: {:ok, build_result()} | {:error, term()}
  def build_routing_state(head_weights, opts \\ [])

  def build_routing_state(%Nx.Tensor{} = head_weights, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, num_roles: 3, backend: nil, model_opts: [])

    with {:ok, output_count, hidden_size, num_agents, num_roles} <-
           infer_head_dimensions(head_weights, opts[:num_roles]) do
      model = CoordinationHead.build_model(hidden_size, num_agents, num_roles, opts[:model_opts])
      {init_fn, _predict_fn} = Axon.build(model)

      input =
        Nx.broadcast(0.0, {1, hidden_size})
        |> Nx.as_type(:f32)
        |> maybe_transfer(opts[:backend])

      params = init_fn.(input, Axon.ModelState.empty())

      {:ok,
       %{
         model: model,
         params: put_head_weights!(params, head_weights, backend: opts[:backend]),
         hidden_size: hidden_size,
         output_count: output_count,
         num_agents: num_agents,
         num_roles: num_roles
       }}
    end
  end

  def build_routing_state(_head_weights, _opts), do: {:error, :invalid_head_weights}

  @doc """
  Replaces `routing_head.kernel` and `routing_head.bias` in an Axon state.

  `head_weights` must be shaped `{output_count, hidden_size}` and is transposed
  to Axon's dense-kernel layout `{hidden_size, output_count}`.
  """
  def put_head_weights!(%Axon.ModelState{} = params, %Nx.Tensor{} = head_weights, opts \\ []) do
    opts = Keyword.validate!(opts, backend: nil, cast: true)

    {output_count, hidden_size} = Nx.shape(head_weights)
    data = params.data

    layer_key = resolve_map_key!(data, @routing_head_layer)
    layer = Map.fetch!(data, layer_key)

    kernel_key = resolve_map_key!(layer, "kernel")
    bias_key = resolve_map_key!(layer, "bias")

    existing_kernel = Map.fetch!(layer, kernel_key)
    existing_bias = Map.fetch!(layer, bias_key)

    expected_kernel_shape = {hidden_size, output_count}

    unless Nx.shape(existing_kernel) == expected_kernel_shape do
      raise ArgumentError,
            "routing head kernel shape mismatch: expected #{inspect(expected_kernel_shape)}, got #{inspect(Nx.shape(existing_kernel))}"
    end

    backend = opts[:backend] || backend_from_tensor(existing_kernel)
    target_type = Nx.type(existing_kernel)

    kernel =
      head_weights
      |> Nx.transpose()
      |> maybe_cast(target_type, opts[:cast])
      |> maybe_transfer(backend)

    bias =
      Nx.broadcast(0.0, Nx.shape(existing_bias))
      |> maybe_cast(Nx.type(existing_bias), true)
      |> maybe_transfer(backend)

    patched_layer =
      layer
      |> Map.put(kernel_key, kernel)
      |> Map.put(bias_key, bias)

    %{params | data: Map.put(data, layer_key, patched_layer)}
  end

  defp infer_head_dimensions(head_weights, num_roles)
       when is_integer(num_roles) and num_roles > 0 do
    case Nx.shape(head_weights) do
      {output_count, hidden_size}
      when output_count > num_roles and hidden_size > 0 ->
        {:ok, output_count, hidden_size, output_count - num_roles, num_roles}

      shape ->
        {:error, {:invalid_head_shape, shape}}
    end
  end

  defp infer_head_dimensions(_head_weights, num_roles),
    do: {:error, {:invalid_num_roles, num_roles}}

  defp resolve_map_key!(container, key) when is_map(container) and is_binary(key) do
    if Map.has_key?(container, key) do
      key
    else
      existing_atom_key(container, key) || raise_missing_map_key!(key)
    end
  end

  defp resolve_map_key!(container, key) when is_map(container) do
    if Map.has_key?(container, key) do
      key
    else
      raise_missing_map_key!(key)
    end
  end

  defp raise_missing_map_key!(key), do: raise(ArgumentError, "missing map key #{inspect(key)}")

  defp existing_atom_key(container, key) do
    atom = String.to_existing_atom(key)
    if Map.has_key?(container, atom), do: atom
  rescue
    ArgumentError -> nil
  end

  defp maybe_cast(tensor, target_type, true), do: Nx.as_type(tensor, target_type)

  defp maybe_cast(tensor, target_type, false) do
    if Nx.type(tensor) == target_type do
      tensor
    else
      raise ArgumentError,
            "routing head type mismatch: expected #{inspect(target_type)}, got #{inspect(Nx.type(tensor))}"
    end
  end

  defp maybe_transfer(tensor, nil), do: tensor
  defp maybe_transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp backend_from_tensor(tensor) do
    case Runtime.tensor_backend(tensor) do
      "EXLA.Backend<cuda" <> _ -> {EXLA.Backend, client: :cuda}
      "EXLA.Backend<host" <> _ -> {EXLA.Backend, client: :host}
      _ -> nil
    end
  end
end
