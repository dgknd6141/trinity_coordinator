defmodule Mix.Tasks.Trinity.Sakana.ExportAdapted do
  @moduledoc """
  Export Sakana-adapted Qwen tensors and router head.

      XLA_TARGET=cuda12 mix trinity.sakana.export_adapted
      XLA_TARGET=cuda12 mix trinity.sakana.export_adapted --dry-run
  """

  use Mix.Task

  alias TrinityCoordinator.Sakana.{Exporter, ExportSpec}

  @shortdoc "Exports Sakana-adapted Qwen tensors and router head"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, rest, errors} = parse_args(args)

    unless Enum.empty?(rest) do
      Mix.raise("Unexpected arguments: #{inspect(rest)}")
    end

    unless Enum.empty?(errors) do
      Mix.raise("Invalid export options: #{inspect(errors)}")
    end

    options = normalize_opts(opts)

    validate_output_policy!(options)
    validate_only_index!(options[:only_index])

    print_summary(options)

    case Exporter.export_adapted(options) do
      {:ok, %{"status" => "dry_run"} = manifest} ->
        print_dry_run(manifest)

      {:ok, manifest} ->
        print_result(manifest)
        validate_completion_or_exit!(manifest, options)

      {:error, reason} ->
        Mix.raise("Failed to export adapted artifact: #{inspect(reason)}")
    end
  end

  defp parse_args(args) do
    OptionParser.parse(args,
      strict: [
        out: :string,
        source_vector: :string,
        tensor_name: :string,
        profile: :string,
        resume: :boolean,
        force: :boolean,
        only_index: :integer,
        skip_existing: :boolean,
        dry_run: :boolean,
        svd_compute_type: :string,
        json: :boolean
      ]
    )
  end

  defp normalize_opts(opts) do
    spec = ExportSpec.resolve!(Keyword.get(opts, :profile, "qwen3_0_6b_layer26"))

    [
      spec: spec,
      out_dir: Path.expand(Keyword.get(opts, :out, spec.out_dir)),
      source_vector_path: Keyword.get(opts, :source_vector, spec.source_vector_path),
      source_vector_tensor: Keyword.get(opts, :tensor_name, spec.source_vector_tensor),
      resume: Keyword.get(opts, :resume, false),
      force: Keyword.get(opts, :force, false),
      only_index: Keyword.get(opts, :only_index, nil),
      skip_existing: Keyword.get(opts, :skip_existing, true),
      dry_run: Keyword.get(opts, :dry_run, false),
      svd_compute_type: Keyword.get(opts, :svd_compute_type, "source"),
      progress: progress_fun(Keyword.get(opts, :json, false))
    ]
  end

  defp validate_output_policy!(opts) do
    if not Keyword.get(opts, :dry_run, false) do
      validate_output_directory_policy!(opts)
    end

    validate_source_vector_exists!(Keyword.fetch!(opts, :source_vector_path))
  end

  defp validate_output_directory_policy!(opts) do
    out_dir = Keyword.fetch!(opts, :out_dir)

    case File.stat(out_dir) do
      {:ok, %File.Stat{type: :directory}} ->
        validate_existing_output_directory!(out_dir, opts)

      {:ok, %File.Stat{type: _}} ->
        Mix.raise("Output path exists but is not a directory: #{out_dir}")

      {:error, :enoent} ->
        :ok
    end
  end

  defp validate_existing_output_directory!(out_dir, opts) do
    resume? = Keyword.get(opts, :resume, false)
    force? = Keyword.get(opts, :force, false)

    unless resume? or force? do
      Mix.raise("Output directory exists: #{out_dir}. Use --force or --resume to proceed.")
    end
  end

  defp validate_source_vector_exists!(source_path) do
    unless File.exists?(source_path) do
      Mix.raise("Source vector file does not exist: #{source_path}")
    end
  end

  defp validate_only_index!(nil), do: :ok
  defp validate_only_index!(index) when is_integer(index) and index > 0, do: :ok

  defp validate_only_index!(index) do
    Mix.raise("invalid --only-index value #{inspect(index)}; expected positive integer")
  end

  defp print_summary(opts) do
    spec = Keyword.fetch!(opts, :spec)

    IO.puts("Sakana Adapted Export")
    IO.puts("  Export spec: #{spec.name}")
    IO.puts("  Output directory: #{Keyword.fetch!(opts, :out_dir)}")
    IO.puts("  Source vector: #{Keyword.fetch!(opts, :source_vector_path)}")
    IO.puts("  Source tensor: #{Keyword.fetch!(opts, :source_vector_tensor)}")
    IO.puts("  Resume: #{Keyword.fetch!(opts, :resume)}")
    IO.puts("  Force: #{Keyword.fetch!(opts, :force)}")
    IO.puts("  Skip existing: #{Keyword.fetch!(opts, :skip_existing)}")
    IO.puts("  Dry run: #{Keyword.fetch!(opts, :dry_run)}")
    IO.puts("  SVD compute type: #{Keyword.fetch!(opts, :svd_compute_type)}")

    case Keyword.fetch!(opts, :only_index) do
      nil -> IO.puts("  Only index: (all)")
      index -> IO.puts("  Only index: #{index}")
    end
  end

  defp progress_fun(true), do: &print_json_progress/1
  defp progress_fun(false), do: &print_progress/1

  defp print_json_progress(event) do
    event
    |> normalize_for_json()
    |> Jason.encode!()
    |> IO.puts()
  end

  defp print_progress(event) do
    event_name = event[:event]

    case event_name do
      :router_head_export_complete ->
        IO.puts("router_head_export_complete: path=#{event[:path]} sha256=#{event[:sha256]}")

      :router_head_skipped ->
        IO.puts("router_head_skipped: path=#{event[:path]}")

      :tensor_skipped ->
        IO.puts("tensor_skipped: path=#{event[:path]}")

      :tensor_export_finished ->
        IO.puts("tensor_export_finished: path=#{event[:path]}")

      :tensor_export_progress ->
        IO.puts(
          "tensor_export_progress: path=#{event[:path]} decompose_ms=#{event[:decompose_ms]} reconstruct_ms=#{event[:reconstruct_ms]}"
        )

      _ ->
        :ok
    end
  end

  defp print_dry_run(manifest) do
    IO.puts("Dry run complete: no files written")
    IO.puts("  Source vector shape: #{inspect(manifest["source_vector_shape"])}")
    IO.puts("  Scale offsets shape: #{inspect(manifest["scale_offsets_shape"])}")
    IO.puts("  Router head shape: #{inspect(manifest["router_head_shape"])}")
    IO.puts("  Selected tensor count: #{manifest["selected_tensor_count"]}")
    IO.puts("  Selected singular values: #{manifest["selected_singular_value_count"]}")
    IO.puts("  Selected paths:")

    Enum.each(manifest["selected_tensors"], fn entry ->
      IO.puts(
        "    #{entry["index"]}. #{entry["path"]} shape=#{inspect(entry["shape"])} singular=#{entry["singular_values"]} backend=#{entry["backend"]}"
      )
    end)
  end

  defp print_result(manifest) do
    IO.puts("Export status: #{manifest["status"]}")
    IO.puts("Export complete flag: #{manifest["export_complete"]}")
    IO.puts("Tensor count: #{manifest["selected_tensor_count"]}")

    IO.puts(
      "Completed: #{completed_selected_tensors(manifest)} / #{manifest["selected_tensor_count"]}"
    )
  end

  defp completed_selected_tensors(manifest) do
    manifest
    |> Map.get("selected_tensors", [])
    |> Enum.count(fn entry -> Map.get(entry, "status") == "complete" end)
  end

  defp validate_completion_or_exit!(manifest, opts) do
    only_index = Keyword.get(opts, :only_index)

    if manifest["status"] != "complete" and is_nil(only_index) do
      Mix.raise("Export completed with non-canonical state: #{inspect(manifest["status"])}")
    end

    if manifest["status"] == "failed" do
      Mix.raise("Export failed for one or more tensors.")
    end
  end

  defp normalize_for_json(value) when is_map(value) do
    Map.new(value, fn {key, val} -> {to_string(key), normalize_for_json(val)} end)
  end

  defp normalize_for_json(value) when is_list(value), do: Enum.map(value, &normalize_for_json/1)
  defp normalize_for_json(value) when is_tuple(value), do: Tuple.to_list(value)
  defp normalize_for_json(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_for_json(value), do: value
end
