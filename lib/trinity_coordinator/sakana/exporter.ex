defmodule TrinityCoordinator.Sakana.Exporter do
  @moduledoc """
  Canonical Qwen/Sakana export pipeline.

  The exporter writes checkpoints incrementally, records progress in the manifest,
  and merges all checkpoints into a single adapted-weights artifact at the end.
  """

  alias TrinityCoordinator.{Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, ExportSpec, SVD}

  require Logger

  @checkpoint_name_width 4
  @complete_status "complete"
  @pending_status "pending"
  @failed_status "failed"
  @running_status "running"

  @type export_option ::
          {:out_dir, String.t()}
          | {:source_vector_path, String.t()}
          | {:source_vector_tensor, String.t()}
          | {:resume, boolean()}
          | {:force, boolean()}
          | {:only_index, nil | pos_integer()}
          | {:skip_existing, boolean()}
          | {:dry_run, boolean()}
          | {:svd_compute_type, :source | :f32}
          | {:spec, ExportSpec.t() | atom() | String.t()}
          | {:progress, (map() -> any()) | nil}

  @doc """
  Exports adapted tensors and router head artifact.
  """
  def export_adapted(opts) when is_list(opts) do
    opts =
      Keyword.validate!(
        opts,
        spec: ExportSpec.qwen3_0_6b_layer26(),
        out_dir: nil,
        source_vector_path: nil,
        source_vector_tensor: nil,
        resume: false,
        force: false,
        only_index: nil,
        skip_existing: true,
        dry_run: false,
        svd_compute_type: :source,
        progress: nil
      )
      |> normalize_options()

    opts
    |> run_export()
    |> finalize_export_result(opts)
  end

  defp run_export(opts) do
    if opts[:dry_run] do
      run_dry_run(opts)
    else
      run_full_export(opts)
    end
  end

  defp run_full_export(opts) do
    out_dir = opts[:out_dir]

    emit_progress(out_dir, opts[:progress], %{
      event: :export_started,
      options: summarize_options(opts)
    })

    do_run_full_export(opts)
  rescue
    e -> {:error, {:export_exception, Exception.message(e)}}
  end

  defp do_run_full_export(opts) do
    with :ok <- validate_only_index(opts[:only_index]),
         {:ok, out_dir, manifest_hint, profile} <- prepare_output(opts),
         {:ok, source_vector} <- load_source_vector(opts),
         {:ok, split} <- split_router_vector(source_vector, opts[:spec]),
         {:ok, model_info} <- load_profile(profile),
         {:ok, selected} <- select_tensors(model_info, opts[:spec]),
         :ok <- validate_selection(selected, split.scale_offsets, opts[:spec]),
         {:ok, manifest} <-
           build_or_resume_manifest(
             opts,
             out_dir,
             profile,
             source_vector,
             split,
             selected,
             manifest_hint
           ),
         {:ok, manifest} <- export_router_head(out_dir, split.head_weights, manifest, opts),
         {:ok, manifest} <-
           export_tensors(out_dir, split.scale_offsets, selected, manifest, opts),
         {:ok, manifest} <- finalize_manifest(manifest, opts[:only_index], out_dir) do
      finalize_merge_if_complete(out_dir, manifest)
    end
  end

  defp finalize_export_result({:ok, manifest} = ok, opts) do
    unless opts[:dry_run] do
      emit_progress(opts[:out_dir], opts[:progress], %{
        event: :export_finished,
        status: manifest["status"]
      })
    end

    ok
  end

  defp finalize_export_result({:error, reason}, opts) do
    unless opts[:dry_run] do
      emit_progress(opts[:out_dir], opts[:progress], %{event: :export_failed, reason: reason})
    end

    {:error, reason}
  end

  defp normalize_options(opts) do
    spec = ExportSpec.resolve!(opts[:spec])

    source_vector_path = opts[:source_vector_path] || spec.source_vector_path
    source_vector_tensor = opts[:source_vector_tensor] || spec.source_vector_tensor
    out_dir = opts[:out_dir] || spec.out_dir

    force = opts[:force]
    resume = opts[:resume] && !force

    opts
    |> Keyword.put(:spec, spec)
    |> Keyword.put(:source_vector_path, source_vector_path)
    |> Keyword.put(:source_vector_tensor, source_vector_tensor)
    |> Keyword.put(:out_dir, Path.expand(out_dir))
    |> Keyword.put(:force, force)
    |> Keyword.put(:resume, resume)
    |> Keyword.put(:svd_compute_type, normalize_svd_compute_type!(opts[:svd_compute_type]))
  end

  defp normalize_svd_compute_type!(:source), do: :source
  defp normalize_svd_compute_type!("source"), do: :source
  defp normalize_svd_compute_type!(:f32), do: :f32
  defp normalize_svd_compute_type!("f32"), do: :f32

  defp normalize_svd_compute_type!(other) do
    raise ArgumentError, "svd_compute_type must be :source or :f32, got #{inspect(other)}"
  end

  defp profile_for_spec(%ExportSpec{} = spec) do
    SLMProfile.qwen_coordinator()
    |> Map.put(:repo, {:hf, spec.base_model_repo})
    |> Map.put(:module, spec.bumblebee_module)
    |> Map.put(:architecture, spec.architecture)
    |> Map.put(:expected_hidden_size, spec.hidden_size)
    |> Map.put(:xla_target, spec.xla_target)
  end

  defp run_dry_run(opts) do
    profile = profile_for_spec(opts[:spec])

    with :ok <- validate_only_index(opts[:only_index]),
         {:ok, source_vector} <- load_source_vector(opts),
         {:ok, split} <- split_router_vector(source_vector, opts[:spec]),
         {:ok, model_info} <- load_profile(profile),
         {:ok, selected} <- select_tensors(model_info, opts[:spec]),
         :ok <- validate_selection(selected, split.scale_offsets, opts[:spec]) do
      {:ok, dry_run_manifest(opts, source_vector, split, selected)}
    end
  rescue
    e -> {:error, {:dry_run_exception, Exception.message(e)}}
  end

  defp dry_run_manifest(opts, source_vector, split, selected) do
    %{
      "status" => "dry_run",
      "export_complete" => false,
      "dry_run" => true,
      "export_spec" => ExportSpec.to_map(opts[:spec]),
      "source_vector_shape" => Tuple.to_list(Nx.shape(source_vector)),
      "scale_offsets_shape" => Tuple.to_list(Nx.shape(split.scale_offsets)),
      "router_head_shape" => Tuple.to_list(Nx.shape(split.head_weights)),
      "selected_tensor_count" => length(selected),
      "selected_singular_value_count" => SVD.singular_value_count(selected),
      "selected_tensors" => Enum.with_index(selected, 1) |> Enum.map(&dry_run_tensor_entry/1)
    }
  end

  defp dry_run_tensor_entry({entry, index}) do
    %{
      "index" => index,
      "path" => entry.path,
      "shape" => Tuple.to_list(Nx.shape(entry.tensor)),
      "singular_values" => entry_singular_count(entry),
      "backend" => Runtime.tensor_backend(entry.tensor)
    }
  end

  defp prepare_output(opts) do
    with :ok <- validate_output_dir(opts[:out_dir]) do
      prepare_output_dir(opts)
    end
  end

  defp validate_output_dir(out_dir) when is_binary(out_dir) and out_dir != "", do: :ok
  defp validate_output_dir(out_dir), do: {:error, {:invalid_output_dir, out_dir}}

  defp prepare_output_dir(opts) do
    cond do
      opts[:force] ->
        prepare_forced_output(opts)

      File.exists?(opts[:out_dir]) ->
        prepare_existing_output(opts)

      true ->
        prepare_new_output(opts)
    end
  end

  defp prepare_forced_output(opts) do
    File.rm_rf(opts[:out_dir])
    prepare_new_output(opts)
  end

  defp prepare_new_output(opts) do
    ensure_output_dirs!(opts[:out_dir])
    {:ok, opts[:out_dir], nil, profile_for_spec(opts[:spec])}
  end

  defp prepare_existing_output(opts) do
    if opts[:resume] do
      load_existing_manifest(opts)
    else
      {:error, :output_dir_already_exists_without_resume}
    end
  end

  defp load_existing_manifest(opts) do
    case Artifact.load_manifest(opts[:out_dir]) do
      {:ok, manifest} ->
        ensure_output_dirs!(opts[:out_dir])
        {:ok, opts[:out_dir], manifest, profile_for_spec(opts[:spec])}

      {:error, _} ->
        {:error, :missing_manifest_for_resume}
    end
  end

  defp ensure_output_dirs!(out_dir) do
    File.mkdir_p!(out_dir)
    File.mkdir_p!(Artifact.checkpoint_path(out_dir))
  end

  defp load_profile(profile) do
    Runtime.put_cuda_backend!()

    case SLMProfile.load_profile(profile) do
      {:ok, {model_info, _tokenizer}} ->
        {:ok, model_info}

      {:error, reason} ->
        {:error, {:load_profile_error, reason}}
    end
  rescue
    e ->
      {:error, {:load_profile_error, Exception.message(e)}}
  end

  defp load_source_vector(opts) do
    path = opts[:source_vector_path]
    tensor_name = opts[:source_vector_tensor]
    {:ok, SVD.load_router_vector!(path, tensor_name)}
  rescue
    e -> {:error, {:source_vector_read_error, Exception.message(e)}}
  end

  defp split_router_vector(vector, %ExportSpec{} = spec) do
    expected = ExportSpec.source_vector_size(spec)

    case Nx.shape(vector) do
      {size} when size == expected ->
        split_router_vector_with_spec(vector, spec)

      shape ->
        {:error, {:invalid_source_vector_shape, shape}}
    end
  end

  defp split_router_vector_with_spec(vector, spec) do
    {:ok,
     SVD.split_router_vector(
       vector,
       spec.scale_offset_count,
       spec.hidden_size,
       ExportSpec.output_count(spec)
     )}
  rescue
    e -> {:error, {:split_error, Exception.message(e)}}
  end

  defp select_tensors(model_info, %ExportSpec{} = spec) do
    selected =
      SVD.decomposable_tensor_entries(
        model_info.params,
        path_filter: SVD.layer_index_filter(spec.selected_layer_indices)
      )

    {:ok, selected}
  rescue
    e -> {:error, {:tensor_selection_error, Exception.message(e)}}
  end

  defp validate_selection(selected, scale_offsets, %ExportSpec{} = spec) do
    singular_total = SVD.singular_value_count(selected)
    scale_count = Nx.size(scale_offsets)

    cond do
      selected == [] ->
        {:error, :no_selected_tensors}

      singular_total != spec.scale_offset_count ->
        {:error,
         {:invalid_selection, %{expected: spec.scale_offset_count, actual: singular_total}}}

      scale_count != spec.scale_offset_count ->
        {:error,
         {:invalid_scale_count, %{expected: spec.scale_offset_count, actual: scale_count}}}

      true ->
        :ok
    end
  end

  defp build_or_resume_manifest(opts, out_dir, profile, source_vector, split, selected, existing) do
    base_manifest = manifest_seed(opts, out_dir, profile, source_vector, split, selected)

    manifest =
      if opts[:resume] && is_map(existing) do
        merge_resume_manifest(existing, base_manifest, out_dir)
      else
        base_manifest
      end

    manifest = Map.put(manifest, "only_index", opts[:only_index])
    manifest = Map.put(manifest, "skip_existing", opts[:skip_existing])

    with :ok <- validate_only_index_bounds(opts[:only_index], length(selected)),
         :ok <- Artifact.write_manifest!(out_dir, manifest) do
      {:ok, manifest}
    end
  end

  defp merge_resume_manifest(existing, base, out_dir) do
    unless Artifact.identity_matches?(base, existing) do
      raise ArgumentError,
            "existing manifest identity mismatch; use --force to rebuild artifacts"
    end

    existing_entries = Map.get(existing, "selected_tensors", [])
    existing_by_index = Map.new(existing_entries, fn entry -> {entry["index"], entry} end)

    merged_tensors =
      base["selected_tensors"]
      |> Enum.map(fn entry ->
        existing_entry = Map.get(existing_by_index, entry["index"])
        existing = verified_entry_state(existing_entry, out_dir, entry)
        Map.merge(entry, existing)
      end)

    base
    |> Map.put("selected_tensors", merged_tensors)
    |> Map.put("status", Map.get(existing, "status", @pending_status))
    |> Map.put("export_complete", Map.get(existing, "export_complete", false))
    |> Map.put("created_at", Map.get(existing, "created_at", Map.get(base, "created_at")))
    |> Map.put("updated_at", Map.get(existing, "updated_at", Map.get(base, "updated_at")))
    |> Map.put("adapted_tensors_sha256", Map.get(existing, "adapted_tensors_sha256"))
  end

  defp verified_entry_state(nil, _out_dir, entry), do: pending_entry(entry)

  defp verified_entry_state(existing, out_dir, entry) do
    case Map.get(existing, "status", @pending_status) do
      @complete_status -> verified_complete_entry(existing, out_dir, entry)
      _other -> pending_entry(entry)
    end
  end

  defp verified_complete_entry(existing, out_dir, entry) do
    checkpoint_path = Path.join(out_dir, Map.get(existing, "checkpoint_path", ""))

    if checkpoint_valid?(existing, checkpoint_path) do
      existing
      |> Map.take([
        "status",
        "checkpoint_sha256",
        "decompose_elapsed_ms",
        "reconstruct_elapsed_ms",
        "u_backend",
        "s_backend",
        "v_backend",
        "adapted_backend",
        "error"
      ])
      |> Map.put("status", @complete_status)
    else
      pending_entry(entry)
    end
  end

  defp pending_entry(entry) do
    entry
    |> Map.put("status", @pending_status)
    |> Map.put("error", nil)
    |> Map.put("checkpoint_sha256", nil)
  end

  defp manifest_seed(opts, _out_dir, _profile, source_vector, split, selected) do
    now = now_iso8601()
    source_vector_sha256 = Artifact.file_sha256!(opts[:source_vector_path])

    spec = opts[:spec]

    %{
      "artifact_version" => Artifact.manifest_version(),
      "status" => @pending_status,
      "created_at" => now,
      "updated_at" => now,
      "base_model_repo" => spec.base_model_repo,
      "bumblebee_module" => inspect(spec.bumblebee_module),
      "architecture" => Atom.to_string(spec.architecture),
      "xla_target" => spec.xla_target,
      "export_backend" => spec.export_backend,
      "export_spec" => ExportSpec.to_map(spec),
      "source_vector_path" => opts[:source_vector_path],
      "source_vector_tensor" => opts[:source_vector_tensor],
      "source_vector_shape" => Tuple.to_list(Nx.shape(source_vector)),
      "source_vector_sha256" => source_vector_sha256,
      "scale_offset_count" => spec.scale_offset_count,
      "router_head_shape" => [ExportSpec.output_count(spec), spec.hidden_size],
      "router_head_artifact" => Artifact.router_head_file(),
      "router_head_tensor_key" => spec.router_head_tensor_key,
      "adapted_tensors_artifact" => Artifact.adapted_tensors_file(),
      "artifact_layout" => Artifact.artifact_layout_checkpoint_directory(),
      "selected_tensor_count" => length(selected),
      "selected_singular_value_count" => SVD.singular_value_count(selected),
      "export_complete" => false,
      "partial_debug_only" => false,
      "selected_tensors" => build_selected_tensors(selected, opts[:svd_compute_type]),
      "source_split" => %{
        "scale_count" => spec.scale_offset_count,
        "hidden_size" => spec.hidden_size,
        "output_count" => ExportSpec.output_count(spec)
      },
      "split" => %{
        "scale_count" => split.scale_count,
        "head_count" => split.head_count
      }
    }
  end

  defp build_selected_tensors(selected, svd_compute_type) do
    selected
    |> Enum.with_index(1)
    |> Enum.map_reduce(0, fn {entry, index}, cursor ->
      count = entry_singular_count(entry)

      item =
        %{
          "index" => index,
          "path" => entry.path,
          "artifact_key" => entry.path,
          "segments" => entry.segments,
          "shape" => Tuple.to_list(Nx.shape(entry.tensor)),
          "type" => inspect(Nx.type(entry.tensor)),
          "source_type" => inspect(Nx.type(entry.tensor)),
          "svd_compute_type" => Atom.to_string(svd_compute_type),
          "status" => @pending_status,
          "offset_start" => cursor,
          "offset_end" => cursor + count,
          "singular_values" => count,
          "checkpoint_path" =>
            Path.join(Artifact.checkpoint_directory_name(), checkpoint_file(index, entry.path)),
          "backend_observed_during_export" => Runtime.tensor_backend(entry.tensor),
          "decompose_elapsed_ms" => nil,
          "reconstruct_elapsed_ms" => nil,
          "u_backend" => nil,
          "s_backend" => nil,
          "v_backend" => nil,
          "adapted_backend" => nil,
          "error" => nil,
          "checkpoint_sha256" => nil
        }

      {item, cursor + count}
    end)
    |> elem(0)
  end

  defp checkpoint_file(index, path) do
    idx = Integer.to_string(index) |> String.pad_leading(@checkpoint_name_width, "0")
    safe_path = Regex.replace(~r/[^0-9A-Za-z_.-]/, path, "_")
    "#{idx}_#{safe_path}.safetensors"
  end

  defp entry_singular_count(entry) do
    Nx.shape(entry.tensor) |> Tuple.to_list() |> Enum.min()
  end

  defp export_router_head(out_dir, head_weights, manifest, opts) do
    head_path = Path.join(out_dir, Artifact.router_head_file())
    head_key = manifest["router_head_tensor_key"] || Artifact.router_head_tensor_key()

    with :ok <- File.mkdir_p(out_dir) do
      export_or_reuse_router_head(out_dir, head_path, head_key, head_weights, manifest, opts)
    end
  end

  defp export_or_reuse_router_head(out_dir, head_path, head_key, head_weights, manifest, opts) do
    if should_try_reuse_router_head?(head_path, opts) do
      reuse_or_rewrite_router_head(out_dir, head_path, head_key, head_weights, manifest, opts)
    else
      write_router_head(out_dir, head_weights, head_key, manifest, opts)
    end
  end

  defp should_try_reuse_router_head?(head_path, opts) do
    opts[:skip_existing] and File.exists?(head_path)
  end

  defp reuse_or_rewrite_router_head(out_dir, head_path, head_key, head_weights, manifest, opts) do
    with {:ok, tensor} <- safe_read_router_head(head_path, head_key),
         :ok <- validate_router_head_shape(tensor, manifest["router_head_shape"]) do
      emit_progress(out_dir, opts[:progress], %{
        event: :router_head_skipped,
        path: head_path,
        status: manifest["status"]
      })

      {:ok, manifest}
    else
      {:error, reason} ->
        emit_progress(out_dir, opts[:progress], %{
          event: :router_head_rewrite,
          path: head_path,
          reason: reason
        })

        write_router_head(out_dir, head_weights, head_key, manifest, opts)
    end
  end

  defp safe_read_router_head(path, key) do
    case Safetensors.read!(path)[key] do
      %Nx.Tensor{} = tensor -> {:ok, tensor}
      nil -> {:error, :router_head_key_missing}
    end
  rescue
    e -> {:error, {:router_head_read_error, Exception.message(e)}}
  end

  defp validate_router_head_shape(tensor, expected_shape) when is_list(expected_shape) do
    if Nx.shape(tensor) == List.to_tuple(expected_shape),
      do: :ok,
      else: {:error, :router_head_shape_mismatch}
  end

  defp validate_router_head_shape(_tensor, _), do: {:error, :router_head_shape_mismatch}

  defp write_router_head(out_dir, head_weights, head_key, manifest, opts) do
    head_path = Path.join(out_dir, Artifact.router_head_file())
    tmp = head_path <> ".tmp"
    payload = %{head_key => Nx.backend_transfer(head_weights, Nx.BinaryBackend)}

    Safetensors.write!(tmp, payload)
    File.rename!(tmp, head_path)
    sha = Artifact.file_sha256!(head_path)
    updated = Map.put(manifest, "router_head_sha256", sha)
    Artifact.write_manifest!(out_dir, updated)

    emit_progress(
      out_dir,
      opts[:progress],
      %{event: :router_head_export_complete, path: head_path, sha256: sha}
    )

    {:ok, updated}
  rescue
    e -> {:error, {:router_head_write_error, Exception.message(e)}}
  end

  defp export_tensors(out_dir, scale_offsets, selected, manifest, opts) do
    source_tensors = Map.new(selected, fn entry -> {entry.path, entry.tensor} end)
    to_process = tensors_to_process(manifest, opts[:only_index])

    case to_process do
      [] ->
        {:error, {:invalid_only_index, opts[:only_index]}}

      [_ | _] ->
        process_export_tensors(to_process, source_tensors, out_dir, scale_offsets, manifest, opts)
    end
  end

  defp tensors_to_process(manifest, nil), do: manifest["selected_tensors"]

  defp tensors_to_process(manifest, index) do
    Enum.filter(manifest["selected_tensors"], &(&1["index"] == index))
  end

  defp process_export_tensors(to_process, source_tensors, out_dir, scale_offsets, manifest, opts) do
    context = %{
      total: length(to_process),
      source_tensors: source_tensors,
      out_dir: out_dir,
      scale_offsets: scale_offsets,
      opts: opts
    }

    to_process
    |> Enum.reduce_while(manifest, &process_export_tensor_entry(&1, &2, context))
    |> normalize_export_tensors_result()
  end

  defp process_export_tensor_entry(entry, current, context) do
    entry_path = entry["path"]
    source_tensor = Map.get(context.source_tensors, entry_path)

    cond do
      is_nil(source_tensor) ->
        {:halt, {:error, {:missing_source_tensor, entry_path}}}

      should_skip_entry?(entry, context.out_dir, context.opts[:skip_existing]) ->
        skip_tensor_entry(entry, current, context)

      true ->
        run_tensor_export(entry, source_tensor, current, context)
    end
  end

  defp skip_tensor_entry(entry, current, context) do
    emit_progress(context.out_dir, context.opts[:progress], %{
      event: :tensor_skipped,
      path: entry["path"],
      index: entry["index"],
      index_total: context.total
    })

    {:cont, current}
  end

  defp run_tensor_export(entry, source_tensor, current, context) do
    started = mark_tensor_running!(current, entry, context)

    context.out_dir
    |> export_tensor(source_tensor, entry, context.scale_offsets, context.opts)
    |> handle_tensor_export_result(entry, started, context)
  end

  defp mark_tensor_running!(current, entry, context) do
    started =
      current
      |> Map.put("updated_at", now_iso8601())
      |> update_selected_tensor(entry["index"], %{"status" => @running_status})

    :ok = Artifact.write_manifest!(context.out_dir, started)

    emit_progress(context.out_dir, context.opts[:progress], %{
      event: :tensor_export_started,
      index: entry["index"],
      total: context.total,
      path: entry["path"]
    })

    started
  end

  defp handle_tensor_export_result({:ok, entry_update}, entry, started, context) do
    updated =
      started
      |> update_selected_tensor(entry["index"], entry_update)
      |> Map.put("updated_at", now_iso8601())

    :ok = Artifact.write_manifest!(context.out_dir, updated)

    emit_progress(context.out_dir, context.opts[:progress], %{
      event: :tensor_export_finished,
      index: entry["index"],
      path: entry["path"]
    })

    {:cont, updated}
  end

  defp handle_tensor_export_result({:error, reason}, entry, started, context) do
    updated =
      started
      |> update_selected_tensor(entry["index"], %{
        "status" => @failed_status,
        "error" => inspect(reason)
      })
      |> Map.put("status", @failed_status)
      |> Map.put("updated_at", now_iso8601())

    :ok = Artifact.write_manifest!(context.out_dir, updated)

    emit_progress(context.out_dir, context.opts[:progress], %{
      event: :tensor_export_failed,
      index: entry["index"],
      path: entry["path"],
      reason: reason
    })

    {:halt, {:error, reason}}
  end

  defp normalize_export_tensors_result({:error, reason}), do: {:error, reason}
  defp normalize_export_tensors_result(final_manifest), do: {:ok, final_manifest}

  defp should_skip_entry?(entry, out_dir, true) do
    if entry["status"] == @complete_status do
      checkpoint_path = Path.join(out_dir, Map.fetch!(entry, "checkpoint_path"))
      checkpoint_valid?(entry, checkpoint_path)
    else
      false
    end
  end

  defp should_skip_entry?(_entry, _out_dir, false), do: false

  defp checkpoint_valid?(entry, path) do
    with true <- File.exists?(path),
         expected_hash <- entry["checkpoint_sha256"],
         true <- is_binary(expected_hash) and expected_hash != "",
         true <- Artifact.file_sha256!(path) == expected_hash do
      case safely_read_checkpoint_tensor(path, entry["artifact_key"]) do
        {:ok, tensor} ->
          tuple_shape = normalize_shape(entry["shape"])
          stored_type = entry["type"]
          tuple_shape == Nx.shape(tensor) && stored_type == inspect(Nx.type(tensor))

        {:error, _reason} ->
          false
      end
    else
      _ -> false
    end
  end

  defp safely_read_checkpoint_tensor(path, key) do
    case Safetensors.read!(path)[key] do
      %Nx.Tensor{} = tensor -> {:ok, tensor}
      nil -> {:error, :missing_checkpoint_tensor}
    end
  rescue
    e -> {:error, {:checkpoint_read_error, Exception.message(e)}}
  end

  defp normalize_shape(shape) when is_list(shape), do: List.to_tuple(shape)
  defp normalize_shape(shape) when is_tuple(shape), do: shape
  defp normalize_shape(_), do: nil

  defp export_tensor(out_dir, source_tensor, entry, scale_offsets, opts) do
    progress_fun = opts[:progress]

    emit_progress(out_dir, progress_fun, %{
      event: :tensor_export_started,
      index: entry["index"],
      path: entry["path"]
    })

    offset_start = entry["offset_start"]
    singular_count = entry["singular_values"]
    offsets = Nx.slice(scale_offsets, [offset_start], [singular_count])

    with {:ok, decomposition, decompose_ms, decompose_source} <-
           timed_decompose(source_tensor, opts[:svd_compute_type]),
         :ok <- ensure_cuda_backend(decomposition.u, entry["path"]),
         :ok <- ensure_cuda_backend(decomposition.s, entry["path"]),
         :ok <- ensure_cuda_backend(decomposition.v, entry["path"]),
         :ok <- ensure_cuda_backend(source_tensor, entry["path"]),
         {:ok, adapted_tensor, reconstruct_ms} <-
           timed_reconstruct(
             decomposition,
             Nx.as_type(offsets, Nx.type(decomposition.s)),
             source_tensor
           ),
         :ok <- ensure_cuda_backend(adapted_tensor, entry["path"]),
         {:ok, checksum} <-
           write_checkpoint(
             out_dir,
             entry["checkpoint_path"],
             entry["artifact_key"],
             adapted_tensor
           ) do
      emit_progress(
        out_dir,
        progress_fun,
        %{
          event: :tensor_export_progress,
          index: entry["index"],
          path: entry["path"],
          decompose_ms: decompose_ms,
          reconstruct_ms: reconstruct_ms
        }
      )

      {:ok,
       %{
         "status" => @complete_status,
         "decompose_elapsed_ms" => decompose_ms,
         "reconstruct_elapsed_ms" => reconstruct_ms,
         "checkpoint_sha256" => checksum,
         "svd_compute_type" => Atom.to_string(opts[:svd_compute_type]),
         "decompose_source_type" => inspect(Nx.type(decompose_source)),
         "reconstructed_type_before_cast" =>
           inspect(
             Nx.type(
               SVD.reconstruct(decomposition, Nx.as_type(offsets, Nx.type(decomposition.s)))
             )
           ),
         "checkpoint_type" => inspect(Nx.type(adapted_tensor)),
         "u_backend" => Runtime.tensor_backend(decomposition.u),
         "s_backend" => Runtime.tensor_backend(decomposition.s),
         "v_backend" => Runtime.tensor_backend(decomposition.v),
         "adapted_backend" => Runtime.tensor_backend(adapted_tensor),
         "error" => nil
       }}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp timed_decompose(tensor, mode) do
    start = monotonic_us()
    decompose_source = decompose_source_tensor(tensor, mode)
    decomposition = SVD.decompose_tensor(decompose_source)
    elapsed = elapsed_ms(start)
    {:ok, decomposition, elapsed, decompose_source}
  rescue
    e -> {:error, {:decompose_error, Exception.message(e)}}
  end

  defp decompose_source_tensor(tensor, :source), do: tensor
  defp decompose_source_tensor(tensor, :f32), do: Nx.as_type(tensor, :f32)

  defp timed_reconstruct(decomposition, offsets, source_tensor) do
    start = monotonic_us()
    adapted = SVD.reconstruct(decomposition, offsets) |> Nx.as_type(Nx.type(source_tensor))
    elapsed = elapsed_ms(start)
    {:ok, adapted, elapsed}
  rescue
    e -> {:error, {:reconstruct_error, Exception.message(e)}}
  end

  defp ensure_cuda_backend(tensor, path) do
    backend = Runtime.tensor_backend(tensor)

    if String.contains?(backend, "EXLA.Backend<cuda:") do
      :ok
    else
      {:error, {:non_cuda_backend, path, backend}}
    end
  end

  defp write_checkpoint(out_dir, relative_path, artifact_key, tensor) do
    full_path = Path.join(out_dir, relative_path)
    tmp = full_path <> ".tmp"
    payload = %{artifact_key => Nx.backend_transfer(tensor, Nx.BinaryBackend)}
    File.mkdir_p!(Path.dirname(full_path))

    Logger.debug("writing checkpoint #{inspect(artifact_key)} to #{full_path}")
    Safetensors.write!(tmp, payload)
    File.rename!(tmp, full_path)
    {:ok, Artifact.file_sha256!(full_path)}
  rescue
    e -> {:error, {:checkpoint_write_error, Exception.message(e)}}
  end

  defp finalize_manifest(manifest, nil, out_dir) do
    if all_tensors_complete?(manifest["selected_tensors"]) do
      finalized =
        manifest
        |> Map.put("status", "complete")
        |> Map.put("export_complete", true)
        |> Map.put("updated_at", now_iso8601())

      emit_progress(
        out_dir,
        nil,
        %{
          event: :manifest_complete,
          completed:
            length(
              Enum.filter(
                finalized["selected_tensors"],
                &(Map.get(&1, "status") == @complete_status)
              )
            )
        }
      )

      Artifact.write_manifest!(out_dir, finalized)
      {:ok, finalized}
    else
      partial =
        manifest
        |> Map.put("status", "partial")
        |> Map.put("updated_at", now_iso8601())

      emit_progress(
        out_dir,
        nil,
        %{
          event: :manifest_partial,
          completed:
            length(
              Enum.filter(
                partial["selected_tensors"],
                &(Map.get(&1, "status") == @complete_status)
              )
            ),
          total: length(partial["selected_tensors"])
        }
      )

      Artifact.write_manifest!(out_dir, partial)
      {:ok, partial}
    end
  end

  defp finalize_manifest(manifest, only_index, out_dir) do
    partial =
      manifest
      |> Map.put("status", "partial")
      |> Map.put("updated_at", now_iso8601())

    emit_progress(
      out_dir,
      nil,
      %{event: :manifest_partial_only_index, only_index: only_index}
    )

    Artifact.write_manifest!(out_dir, partial)
    {:ok, partial}
  end

  defp finalize_merge_if_complete(out_dir, manifest) do
    if manifest["status"] == "complete" and all_tensors_complete?(manifest["selected_tensors"]) do
      merge_complete_artifacts(out_dir, manifest)
    else
      {:ok, manifest}
    end
  end

  defp merge_complete_artifacts(out_dir, manifest) do
    final_path = Path.join(out_dir, Artifact.adapted_tensors_file())

    try do
      emit_progress(out_dir, nil, %{
        event: :artifact_merge_started,
        selected_tensors: length(manifest["selected_tensors"])
      })

      tmp = final_path <> ".tmp"
      tensors = merged_checkpoint_tensors(out_dir, manifest)

      Safetensors.write!(tmp, tensors)
      File.rename!(tmp, final_path)

      merged =
        manifest
        |> Map.put("artifact_layout", Artifact.artifact_layout_single_file())
        |> Map.put("adapted_tensors_sha256", Artifact.file_sha256!(final_path))
        |> Map.put("updated_at", now_iso8601())

      emit_progress(out_dir, nil, %{
        event: :artifact_merge_complete,
        path: final_path,
        sha256: merged["adapted_tensors_sha256"]
      })

      Artifact.write_manifest!(out_dir, merged)
      {:ok, merged}
    rescue
      e ->
        mark_merge_failed(out_dir, manifest, final_path, e)
    end
  end

  defp merged_checkpoint_tensors(out_dir, manifest) do
    manifest["selected_tensors"]
    |> Enum.map(fn entry ->
      path = Map.fetch!(entry, "checkpoint_path")
      key = Map.fetch!(entry, "artifact_key")
      checkpoint = Safetensors.read!(Path.join(out_dir, path))
      {key, Map.fetch!(checkpoint, key)}
    end)
    |> Map.new()
  end

  defp mark_merge_failed(out_dir, manifest, final_path, exception) do
    failed_manifest =
      manifest
      |> Map.put("status", "failed")
      |> Map.put("updated_at", now_iso8601())

    emit_progress(out_dir, nil, %{
      event: :artifact_merge_failed,
      path: final_path,
      reason: Exception.message(exception)
    })

    Artifact.write_manifest!(out_dir, failed_manifest)
    {:error, {:adapted_merge_error, Exception.message(exception)}}
  end

  defp all_tensors_complete?(entries) when is_list(entries) do
    Enum.all?(entries, fn entry -> entry["status"] == @complete_status end)
  end

  defp update_selected_tensor(manifest, index, updates) do
    updated =
      Enum.map(manifest["selected_tensors"], fn entry ->
        if entry["index"] == index do
          Map.merge(entry, updates)
        else
          entry
        end
      end)

    Map.put(manifest, "selected_tensors", updated)
  end

  defp validate_only_index_bounds(nil, _max), do: :ok

  defp validate_only_index_bounds(index, max)
       when is_integer(index) and index >= 1 and index <= max,
       do: :ok

  defp validate_only_index_bounds(_, _), do: {:error, :invalid_only_index}

  defp validate_only_index(nil), do: :ok
  defp validate_only_index(index) when is_integer(index) and index > 0, do: :ok
  defp validate_only_index(_), do: {:error, {:invalid_only_index, "must be positive integer"}}

  defp summarize_options(opts) do
    %{
      out_dir: Keyword.get(opts, :out_dir),
      source_vector_path: Keyword.get(opts, :source_vector_path),
      source_vector_tensor: Keyword.get(opts, :source_vector_tensor),
      resume: Keyword.get(opts, :resume, false),
      force: Keyword.get(opts, :force, false),
      only_index: Keyword.get(opts, :only_index),
      skip_existing: Keyword.get(opts, :skip_existing, true),
      dry_run: Keyword.get(opts, :dry_run, false),
      svd_compute_type: Keyword.get(opts, :svd_compute_type, :source),
      spec: ExportSpec.to_map(Keyword.fetch!(opts, :spec))
    }
  end

  defp emit_progress(nil, _progress_fun, _event), do: :ok

  defp emit_progress(out_dir, progress_fun, event) when is_function(progress_fun, 1) do
    normalized = timestamp_event(event)
    progress_fun.(normalized)
    log_event(out_dir, normalized)
    :ok
  end

  defp emit_progress(out_dir, _progress_fun, event) do
    log_event(out_dir, timestamp_event(event))
    :ok
  end

  defp timestamp_event(event) when is_map(event) do
    Map.put_new(event, :event_time_utc, now_iso8601())
  end

  defp log_event(nil, _event), do: :ok

  defp log_event(out_dir, event) when is_binary(out_dir) and is_map(event) do
    path = Artifact.export_log_path(out_dir)
    encoded = Jason.encode!(event)
    File.write!(path, encoded <> "\n", [:append])
    :ok
  rescue
    _ -> :ok
  end

  defp log_event(_out_dir, _event), do: :ok

  defp now_iso8601 do
    DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()
  end

  defp monotonic_us do
    System.monotonic_time(:microsecond)
  end

  defp elapsed_ms(start_us) do
    (System.monotonic_time(:microsecond) - start_us) |> div(1000)
  end
end
