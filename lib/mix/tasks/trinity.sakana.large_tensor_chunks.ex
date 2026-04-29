defmodule Mix.Tasks.Trinity.Sakana.LargeTensorChunks do
  @moduledoc """
  Replays large Sakana selected tensors through bounded row chunks.

      XLA_TARGET=cuda12 mix trinity.sakana.large_tensor_chunks \
        --components-dir tmp/sakana_parity/original_submission_svd/python_components \
        --python-report tmp/sakana_parity/large_tensor_chunks/python_large_tensor_chunks.json \
        --chunk-rows 8192 \
        --out tmp/sakana_parity/large_tensor_chunks/elixir_large_tensor_chunks.json

  The Python report may be a dedicated large-tensor chunk manifest, or the
  all-selected Python parity report that points at
  `trinity_svf_all_selected_stage_debug.safetensors`.
  """

  use Mix.Task

  alias TrinityCoordinator.Sakana.LargeTensorChunks

  @shortdoc "Replays embedding/LM-head Sakana stages in row chunks"
  @default_out "tmp/sakana_parity/large_tensor_chunks/elixir_large_tensor_chunks.json"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    opts = parse_args!(args)

    report =
      LargeTensorChunks.report!(
        components_dir: Keyword.get(opts, :components_dir),
        python_report_path: Keyword.fetch!(opts, :python_report),
        stage_dir: Keyword.get(opts, :stage_dir),
        chunk_rows: Keyword.get(opts, :chunk_rows, 1024),
        sources: sources(opts),
        require_cuda: not Keyword.get(opts, :no_cuda, false),
        progress: &print_progress/1
      )

    out = Keyword.get(opts, :out, @default_out)
    :ok = LargeTensorChunks.write_json!(out, report)

    Mix.shell().info("Wrote large-tensor chunk report: #{out}")
    print_summary(report)

    if get_in(report, ["summary", "failed_required_count"]) != 0 do
      Mix.raise("large-tensor chunk replay failed required stage checks")
    end
  end

  @doc false
  def parse_args!(args) do
    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [
          out: :string,
          components_dir: :string,
          python_report: :string,
          stage_dir: :string,
          chunk_rows: :integer,
          source: :string,
          no_cuda: :boolean
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    unless Keyword.has_key?(opts, :python_report) do
      Mix.raise("--python-report is required")
    end

    opts
  end

  defp sources(opts) do
    case Keyword.get_values(opts, :source) do
      [] -> nil
      values -> values
    end
  end

  defp print_summary(report) do
    summary = Map.fetch!(report, "summary")

    Mix.shell().info(
      "large_tensor_chunks=#{summary["chunk_count"]} sources=#{inspect(summary["sources"])} " <>
        "required_checks=#{summary["required_check_count"]} " <>
        "failed_required=#{summary["failed_required_count"]} " <>
        "functional_parity=#{inspect(summary["functional_parity_passed"])}"
    )

    report
    |> Map.fetch!("large_tensor_chunk_checks")
    |> Enum.filter(&(get_in(&1, ["stage_debug", "functional_parity_passed"]) == false))
    |> Enum.each(fn chunk ->
      Mix.shell().info(
        "failed_chunk #{chunk["source_name"]} rows=#{chunk["row_start"]}:#{chunk["row_end"]} " <>
          "required_failed=#{get_in(chunk, ["stage_debug", "required_failed_count"])}"
      )
    end)
  end

  defp print_progress(%{event: :chunk_started} = event) do
    Mix.shell().info(
      "chunk #{event.source_name} #{event.chunk_index + 1}/#{event.total_chunks} " <>
        "rows=#{event.row_start}:#{event.row_end} started"
    )
  end

  defp print_progress(%{event: :chunk_finished} = event) do
    Mix.shell().info(
      "chunk #{event.source_name} #{event.chunk_index + 1}/#{event.total_chunks} " <>
        "rows=#{event.row_start}:#{event.row_end} " <>
        "functional_parity=#{inspect(event.functional_parity_passed)} " <>
        "required_failed=#{event.required_failed_count}"
    )
  end
end
