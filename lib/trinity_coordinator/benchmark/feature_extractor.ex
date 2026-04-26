defmodule TrinityCoordinator.Benchmark.FeatureExtractor do
  @moduledoc """
  Shared utilities for turning benchmark fixtures into real router vectors.

  This module intentionally uses only the real extractor path and does not mock
  hidden-state computation.
  """

  alias TrinityCoordinator.{Benchmark.Dataset, Extractor}

  @doc """
  Extract second-to-last token vectors for the given dataset cases.
  """
  @spec run(map() | {map(), map()}, [Dataset.t()] | [map()]) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def run(slm_context, cases)

  @spec run(map(), map(), [Dataset.t()] | [map()]) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def run(slm_context, cases) when is_list(cases) do
    do_run(slm_context, cases, [])
  end

  def run(model_info, tokenizer, cases)
      when is_map(model_info) and is_map(tokenizer) and is_list(cases) do
    run({model_info, tokenizer}, cases)
  end

  @spec run(map() | {map(), map()}, [Dataset.t()] | [map()], keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, term()}
  def run(slm_context, cases, opts)
      when is_list(cases) and is_list(opts) do
    do_run(slm_context, cases, opts)
  end

  defp do_run(%{model_info: model_info, tokenizer: tokenizer}, cases, opts) do
    extract_from_model_info(model_info, tokenizer, cases, opts)
  end

  defp do_run({model_info, tokenizer}, cases, opts)
       when is_map(model_info) and is_map(tokenizer) do
    extract_from_model_info(model_info, tokenizer, cases, opts)
  end

  defp do_run(_, _, _), do: {:error, :invalid_slm_context}

  defp extract_from_model_info(model_info, tokenizer, cases, opts) do
    if cases == [] do
      {:error, :invalid_inputs}
    else
      run(model_info, tokenizer, cases, :case_messages, opts)
    end
  end

  defp run(model_info, tokenizer, cases, :case_messages, opts) do
    messages = Enum.map(cases, fn %Dataset{messages: messages} -> messages end)
    batch = Keyword.get(opts, :batch, true)

    if batch do
      Extractor.extract_batch_penultimate_hidden_states(model_info, tokenizer, messages)
    else
      extract_one_by_one(model_info, tokenizer, messages)
    end
  end

  defp extract_one_by_one(model_info, tokenizer, messages) when is_list(messages) do
    vectors =
      messages
      |> Enum.reduce_while([], fn message_batch, acc ->
        case Extractor.extract_penultimate_hidden_state_with_metadata(
               model_info,
               tokenizer,
               message_batch
             ) do
          {:ok, metadata} -> {:cont, [metadata.vector | acc]}
          {:error, reason} -> {:halt, {:error, reason}}
        end
      end)

    case vectors do
      {:error, reason} ->
        {:error, reason}

      list when is_list(list) ->
        vectors = Enum.reverse(list)
        {:ok, Nx.stack(vectors)}
    end
  end
end
