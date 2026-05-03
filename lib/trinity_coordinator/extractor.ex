defmodule TrinityCoordinator.Extractor do
  @moduledoc """
  Extracts routing-relevant hidden-state tensors from a coordinator model.
  """

  @default_slm_repo {:hf, "hf-internal-testing/tiny-random-gpt2"}
  @default_slm_architecture :base
  @default_slm_module Bumblebee.Text.Gpt2

  @doc """
  Extracts the hidden state vector corresponding to the second-to-last token.

  If the sequence length is 1, the final token is used as a safe fallback.
  """
  def extract_penultimate_hidden_state(hidden_states) when is_struct(hidden_states, Nx.Tensor) do
    shape = Nx.shape(hidden_states)

    case shape do
      {batch, seq_len, hidden_dim} ->
        index = if seq_len <= 1, do: 0, else: seq_len - 2
        sliced = Nx.slice(hidden_states, [0, index, 0], [batch, 1, hidden_dim])
        Nx.squeeze(sliced, axes: [1])

      _ ->
        {:error, :invalid_hidden_shape}
    end
  end

  @doc """
  Loads an SLM and tokenizer.
  """
  def load_slm_model(
        slm_repo \\ @default_slm_repo,
        slm_module \\ @default_slm_module,
        architecture \\ @default_slm_architecture,
        opts \\ []
      ) do
    with {:ok, model_info} <-
           Bumblebee.load_model(
             slm_repo,
             Keyword.merge([module: slm_module, architecture: architecture], opts)
           ),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(slm_repo) do
      {:ok, {model_info, tokenizer}}
    end
  end

  @doc """
  Extracts the penultimate hidden-state vector from structured message transcripts.
  """
  def extract_penultimate_hidden_state_from_messages(
        messages,
        slm_repo \\ @default_slm_repo,
        slm_module \\ @default_slm_module,
        architecture \\ @default_slm_architecture,
        opts \\ []
      ) do
    with {:ok, {model_info, tokenizer}} <-
           load_slm_model(slm_repo, slm_module, architecture, opts) do
      extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages)
    end
  end

  @doc """
  Extracts the penultimate hidden-state vector from preloaded model/tokenizer objects.
  """
  def extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages) do
    with {:ok, result} <-
           extract_penultimate_hidden_state_with_metadata(model_info, tokenizer, messages) do
      {:ok, result.vector}
    end
  end

  @doc """
  Extracts a batch of penultimate hidden-state vectors.
  """
  def extract_batch_penultimate_hidden_states(model_info, tokenizer, message_batches)
      when is_list(message_batches) do
    message_batches
    |> Enum.reduce_while([], fn messages, acc ->
      case extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages) do
        {:ok, vector} -> {:cont, [vector | acc]}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:error, reason} ->
        {:error, reason}

      vectors ->
        {:ok, vectors |> Enum.reverse() |> Nx.concatenate(axis: 0)}
    end
  end

  def extract_batch_penultimate_hidden_states(_model_info, _tokenizer, _message_batches) do
    {:error, :invalid_message_batches}
  end

  @doc """
  Extracts the penultimate hidden-state vector and metadata useful for demos.
  """
  def extract_penultimate_hidden_state_with_metadata(model_info, tokenizer, messages) do
    with {:ok, transcript} <- format_messages(messages),
         inputs <- Bumblebee.apply_tokenizer(tokenizer, transcript),
         outputs <-
           Axon.predict(model_info.model, model_info.params, inputs,
             global_layer_options: [output_hidden_states: true]
           ),
         {:ok, hidden_states} <- extract_hidden_states(outputs),
         {:ok, hidden_state} <- extract_last_layer_hidden_state(hidden_states),
         {:ok, penultimate} <- extract_penultimate_vector(hidden_state) do
      hidden_state_shape = Nx.shape(hidden_state)

      {:ok,
       %{
         transcript: transcript,
         input_ids: input_tensor(inputs, "input_ids"),
         attention_mask: input_tensor(inputs, "attention_mask"),
         input_shapes: input_shapes(inputs),
         hidden_state_shape: hidden_state_shape,
         hidden_position: -2,
         hidden_index: penultimate_index(hidden_state_shape),
         vector_shape: Nx.shape(penultimate),
         vector: penultimate
       }}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Formats role/content maps into the coordinator transcript string.
  """
  def format_messages(messages) when is_list(messages) do
    formatted =
      Enum.reduce_while(messages, [], fn message, acc ->
        role = Map.get(message, :role, Map.get(message, "role"))
        content = Map.get(message, :content, Map.get(message, "content"))

        if is_binary(role) and is_binary(content) do
          {:cont, [{role, content} | acc]}
        else
          {:halt, {:error, {:invalid_messages, "invalid message entry: #{inspect(message)}"}}}
        end
      end)

    case formatted do
      {:error, reason} ->
        {:error, reason}

      _ ->
        formatted_strings =
          formatted
          |> Enum.reverse()
          |> Enum.map_join("\n", fn {role, content} -> "#{role}: #{content}" end)

        {:ok, formatted_strings}
    end
  end

  def format_messages(_), do: {:error, :invalid_messages}

  defp extract_penultimate_vector(hidden_state) do
    case extract_penultimate_hidden_state(hidden_state) do
      {:error, reason} -> {:error, reason}
      value -> {:ok, value}
    end
  end

  defp extract_hidden_states(outputs) when is_map(outputs) do
    cond do
      has_tensor?(outputs[:hidden_state]) -> {:ok, outputs[:hidden_state]}
      has_tensor?(outputs["hidden_state"]) -> {:ok, outputs["hidden_state"]}
      has_hidden_container?(outputs[:hidden_states]) -> {:ok, outputs[:hidden_states]}
      has_hidden_container?(outputs["hidden_states"]) -> {:ok, outputs["hidden_states"]}
      true -> {:error, :missing_hidden_state}
    end
  end

  defp extract_hidden_states(_), do: {:error, :missing_hidden_state}

  defp has_tensor?(%Axon.None{}), do: false
  defp has_tensor?(tensor) when is_struct(tensor, Nx.Tensor), do: true
  defp has_tensor?(_), do: false

  defp has_hidden_container?(%Axon.None{}), do: false
  defp has_hidden_container?(%Nx.Tensor{}), do: true

  defp has_hidden_container?(hidden_states) when is_tuple(hidden_states) do
    hidden_states
    |> Tuple.to_list()
    |> Enum.any?(&has_tensor?/1)
  end

  defp has_hidden_container?(hidden_states) when is_list(hidden_states) do
    Enum.any?(hidden_states, &has_tensor?/1)
  end

  defp has_hidden_container?(_), do: false

  defp extract_last_layer_hidden_state(hidden_states) when is_struct(hidden_states, Nx.Tensor) do
    {:ok, hidden_states}
  end

  defp extract_last_layer_hidden_state(hidden_states) when is_tuple(hidden_states) do
    hidden_states
    |> Tuple.to_list()
    |> Enum.reverse()
    |> Enum.find_value(:no_layer, fn
      %Axon.None{} -> nil
      %Nx.Tensor{} = layer -> layer
      _ -> nil
    end)
    |> case do
      nil -> {:error, :missing_hidden_state}
      :no_layer -> {:error, :missing_hidden_state}
      value -> {:ok, value}
    end
  end

  defp extract_last_layer_hidden_state(hidden_states) when is_list(hidden_states) do
    hidden_states
    |> Enum.reverse()
    |> Enum.find_value(:no_layer, fn
      %Axon.None{} -> nil
      %Nx.Tensor{} = layer -> layer
      _ -> nil
    end)
    |> case do
      nil -> {:error, :missing_hidden_state}
      :no_layer -> {:error, :missing_hidden_state}
      value -> {:ok, value}
    end
  end

  defp extract_last_layer_hidden_state(_), do: {:error, :missing_hidden_state}

  defp input_shapes(inputs) when is_map(inputs) do
    Map.new(inputs, fn {key, value} ->
      shape =
        if is_struct(value, Nx.Tensor) do
          Nx.shape(value)
        else
          :not_a_tensor
        end

      {key, shape}
    end)
  end

  defp input_shapes(_), do: :unknown

  defp input_tensor(inputs, key) when is_map(inputs) and is_binary(key) do
    case Map.fetch(inputs, key) do
      {:ok, %Nx.Tensor{} = tensor} ->
        tensor

      _ ->
        case matching_atom_key(inputs, key) do
          nil -> nil
          atom -> tensor_or_nil(Map.get(inputs, atom))
        end
    end
  end

  defp input_tensor(_inputs, _key), do: nil

  defp matching_atom_key(inputs, key) do
    Enum.find(Map.keys(inputs), fn
      atom when is_atom(atom) -> Atom.to_string(atom) == key
      _ -> false
    end)
  end

  defp tensor_or_nil(%Nx.Tensor{} = tensor), do: tensor
  defp tensor_or_nil(_), do: nil

  defp penultimate_index({_batch, seq_len, _hidden_dim}) when seq_len <= 1, do: 0
  defp penultimate_index({_batch, seq_len, _hidden_dim}), do: seq_len - 2
  defp penultimate_index(_shape), do: nil
end
