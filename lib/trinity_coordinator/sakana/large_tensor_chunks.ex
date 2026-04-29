defmodule TrinityCoordinator.Sakana.LargeTensorChunks do
  @moduledoc """
  Chunked Sakana stage replay for selected large tensors.

  Embedding and LM-head matrices are too large for the all-selected semantic
  replay path to materialize safely. This module replays those tensors by row
  chunks, reading both Python baselines and SVD component rows through bounded
  safetensors slices.
  """

  alias TrinityCoordinator.Runtime
  alias TrinityCoordinator.Sakana.{SafetensorsSlice, StageCheck}

  @component_file "trinity_svf_components.safetensors"
  @scale_file "trinity_svf_scale_offsets.safetensors"
  @metadata_file "trinity_svf_debug_manifest.json"
  @all_selected_stage_file "trinity_svf_all_selected_stage_debug.safetensors"
  @default_sources ["model.embed_tokens.weight", "lm_head.weight"]
  @stage_names [
    "source_f32",
    "offsets_f32",
    "scaled_s",
    "normalization",
    "u_scaled",
    "matmul_pre_norm",
    "zero_source_f32",
    "adapted_source_f32",
    "final_f32",
    "final_bf16"
  ]

  @doc "Builds a chunked large-tensor parity report."
  def report!(opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        components_dir: nil,
        python_report_path: nil,
        stage_dir: nil,
        chunk_rows: 1024,
        sources: @default_sources,
        require_cuda: true,
        progress: nil
      )

    if opts[:require_cuda] do
      Runtime.put_cuda_backend!()
    end

    compute_backend = if opts[:require_cuda], do: {EXLA.Backend, client: :cuda}, else: nil
    chunk_rows = positive_integer!(opts[:chunk_rows], :chunk_rows)
    python_report = load_json!(opts[:python_report_path])
    components_dir = components_dir!(opts[:components_dir], python_report)
    component_metadata = load_component_metadata!(components_dir)

    components =
      components_dir
      |> Path.join(@component_file)
      |> Safetensors.read!(lazy: true)

    scales =
      components_dir
      |> Path.join(@scale_file)
      |> Safetensors.read!(lazy: true)

    baselines =
      python_report
      |> large_tensor_baselines!(components_dir, component_metadata, chunk_rows)
      |> filter_sources!(opts[:sources])

    chunk_checks =
      baselines
      |> Enum.flat_map(fn baseline ->
        source_chunk_checks(
          baseline,
          components,
          scales,
          compute_backend,
          opts[:stage_dir],
          opts[:progress]
        )
      end)

    %{
      "schema" => "trinity_sakana_large_tensor_chunks_elixir.v1",
      "generated_at_utc" =>
        DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601(),
      "paths" => %{
        "components_dir" => components_dir,
        "python_report" => opts[:python_report_path],
        "stage_dir" => opts[:stage_dir]
      },
      "selection" => %{
        "sources" => Enum.map(baselines, & &1["source_name"]),
        "chunk_rows" => chunk_rows,
        "compute_backend" => inspect(compute_backend || Nx.BinaryBackend)
      },
      "python_component_metadata" => component_metadata,
      "large_tensor_chunk_checks" => chunk_checks,
      "summary" => summary(chunk_checks)
    }
  end

  @doc "Writes a report as pretty JSON."
  def write_json!(path, report) when is_binary(path) and is_map(report) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(normalize_json(report), pretty: true))
    :ok
  end

  defp source_chunk_checks(baseline, components, scales, compute_backend, stage_dir, progress) do
    source_name = Map.fetch!(baseline, "source_name")
    safe_key = Map.fetch!(baseline, "safe_key")
    component_tensors = Map.fetch!(baseline, "component_tensors")

    keys = %{
      u: Map.get(component_tensors, "u", "svd.U.#{safe_key}"),
      s: Map.get(component_tensors, "s", "svd.S.#{safe_key}"),
      v: Map.get(component_tensors, "v", "svd.V.#{safe_key}"),
      offsets: Map.get(baseline, "scale_tensor", "svf.scale_offsets.#{safe_key}")
    }

    u_lazy = fetch_lazy!(components, keys.u)
    s_host = fetch_materialized!(components, keys.s)
    v_host = fetch_materialized!(components, keys.v)
    offsets_host = fetch_materialized!(scales, keys.offsets)

    stage_tensors =
      baseline
      |> Map.fetch!("stage_tensor_file")
      |> Safetensors.read!(lazy: true)

    context = %{
      u_lazy: u_lazy,
      s_host: s_host,
      v_host: v_host,
      offsets_host: offsets_host,
      stage_tensors: stage_tensors,
      compute_backend: compute_backend,
      stage_dir: stage_dir
    }

    chunks = Map.fetch!(baseline, "chunks")
    total_chunks = length(chunks)

    chunks
    |> Enum.map(fn chunk ->
      emit_progress(progress, %{
        event: :chunk_started,
        source_name: source_name,
        chunk_index: Map.fetch!(chunk, "chunk_index"),
        total_chunks: total_chunks,
        row_start: Map.fetch!(chunk, "row_start"),
        row_end: Map.fetch!(chunk, "row_end")
      })

      check =
        chunk_check(
          baseline,
          chunk,
          context
        )

      emit_progress(progress, %{
        event: :chunk_finished,
        source_name: source_name,
        chunk_index: check["chunk_index"],
        total_chunks: total_chunks,
        row_start: check["row_start"],
        row_end: check["row_end"],
        functional_parity_passed: get_in(check, ["stage_debug", "functional_parity_passed"]),
        required_failed_count: get_in(check, ["stage_debug", "required_failed_count"])
      })

      check
    end)
  end

  defp chunk_check(
         baseline,
         chunk,
         context
       ) do
    source_name = Map.fetch!(baseline, "source_name")
    safe_key = Map.fetch!(baseline, "safe_key")
    row_start = Map.fetch!(chunk, "row_start")
    row_end = Map.fetch!(chunk, "row_end")
    row_count = row_end - row_start

    u_chunk = SafetensorsSlice.row_slice!(context.u_lazy, row_start, row_count)

    python_chunk =
      python_stage_chunk!(
        context.stage_tensors,
        Map.fetch!(baseline, "stage_tensors"),
        safe_key,
        row_start,
        row_count
      )

    source_chunk = Map.fetch!(python_chunk, "stage.source_f32")

    stage_tensors =
      replay_stage_tensors(
        u_chunk,
        context.s_host,
        context.v_host,
        context.offsets_host,
        source_chunk,
        context.compute_backend
      )

    stage_file =
      maybe_write_stage_tensors(
        context.stage_dir,
        source_name,
        row_start,
        row_end,
        stage_tensors
      )

    checks =
      StageCheck.compare_stage_tensors(stage_tensors, python_chunk,
        include_alt_hashes: false,
        include_tensor_summaries: false,
        compute_byte_match: false
      )

    required_failed = required_failed_count(checks)

    %{
      "label" => "large_tensor_chunk_#{safe_key}_rows_#{row_start}_#{row_end}",
      "source_name" => source_name,
      "elixir_name" => Map.get(baseline, "elixir_name"),
      "safe_key" => safe_key,
      "chunk_index" => Map.fetch!(chunk, "chunk_index"),
      "row_start" => row_start,
      "row_end" => row_end,
      "row_count" => row_count,
      "compute_backend" => inspect(context.compute_backend || Nx.BinaryBackend),
      "python_stage_tensor_file" => Map.fetch!(baseline, "stage_tensor_file"),
      "stage_debug" => %{
        "schema" => "trinity_sakana_large_tensor_chunk_stage_debug.v1",
        "stage_tensor_file" => stage_file,
        "compared_to_python_stage_tensors" => true,
        "functional_parity_passed" => StageCheck.checks_passed?(checks),
        "required_failed_count" => required_failed,
        "checks" => checks
      },
      "checks" => checks
    }
  end

  defp replay_stage_tensors(u_host, s_host, v_host, offsets_host, source_host, compute_backend) do
    s_device = device_copy(s_host, compute_backend)
    offsets_device = offsets_host |> Nx.as_type(Nx.type(s_host)) |> device_copy(compute_backend)
    scaled_s = Nx.multiply(s_device, Nx.add(offsets_device, 1)) |> host_snapshot()
    normalization = Nx.divide(Nx.sum(s_device), Nx.sum(device_copy(scaled_s, compute_backend)))
    normalization_host = normalization |> Nx.reshape({1}) |> host_snapshot()

    u_scaled =
      u_host
      |> device_copy(compute_backend)
      |> Nx.multiply(
        Nx.reshape(device_copy(scaled_s, compute_backend), {1, Nx.axis_size(s_host, 0)})
      )
      |> host_snapshot()

    matmul_pre_norm =
      u_scaled
      |> device_copy(compute_backend)
      |> Nx.dot(Nx.transpose(device_copy(v_host, compute_backend)))
      |> host_snapshot()

    adapted_source_f32 =
      matmul_pre_norm
      |> device_copy(compute_backend)
      |> Nx.multiply(device_copy(normalization_host, compute_backend))
      |> host_snapshot()

    zero_source_f32 =
      u_host
      |> device_copy(compute_backend)
      |> Nx.multiply(
        Nx.reshape(device_copy(s_host, compute_backend), {1, Nx.axis_size(s_host, 0)})
      )
      |> Nx.dot(Nx.transpose(device_copy(v_host, compute_backend)))
      |> host_snapshot()

    final_f32 = adapted_source_f32 |> Nx.as_type(:f32) |> host_snapshot()

    %{
      "stage.source_f32" => source_host |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.offsets_f32" => offsets_host |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.scaled_s" => scaled_s |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.normalization" => normalization_host |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.u_scaled" => u_scaled |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.matmul_pre_norm" => matmul_pre_norm |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.zero_source_f32" => zero_source_f32 |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.adapted_source_f32" => adapted_source_f32 |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.final_f32" => final_f32,
      "stage.final_bf16" => final_f32 |> Nx.as_type(:bf16) |> host_snapshot()
    }
  end

  defp python_stage_chunk!(stage_tensors, stage_map, safe_key, row_start, row_count) do
    Map.new(@stage_names, fn stage ->
      full_key = Map.get(stage_map, stage, "tensor.#{safe_key}.#{stage}")

      {"stage.#{stage}",
       fetch_python_stage_tensor!(stage_tensors, full_key, row_start, row_count)}
    end)
  end

  defp fetch_python_stage_tensor!(stage_tensors, key, row_start, row_count) do
    value = fetch_lazy!(stage_tensors, key)

    case value do
      %Safetensors.FileTensor{shape: shape} when tuple_size(shape) == 2 ->
        SafetensorsSlice.row_slice!(value, row_start, row_count)

      _ ->
        SafetensorsSlice.materialize!(value)
    end
  end

  defp maybe_write_stage_tensors(nil, _source_name, _row_start, _row_end, _stage_tensors), do: nil
  defp maybe_write_stage_tensors("", _source_name, _row_start, _row_end, _stage_tensors), do: nil

  defp maybe_write_stage_tensors(stage_dir, source_name, row_start, row_end, stage_tensors) do
    File.mkdir_p!(stage_dir)

    file =
      "trinity_svf_large_chunk_#{sanitize_python_key(source_name)}_#{row_start}_#{row_end}.safetensors"

    path = Path.join(stage_dir, file)

    payload =
      Map.new(stage_tensors, fn {key, tensor} ->
        {key, host_snapshot(tensor)}
      end)

    Safetensors.write!(path, payload)
    path
  end

  defp large_tensor_baselines!(python_report, components_dir, component_metadata, chunk_rows) do
    case Map.get(python_report, "large_tensor_chunk_baselines") do
      baselines when is_list(baselines) and baselines != [] ->
        baselines

      _ ->
        baselines_from_component_metadata!(
          python_report,
          components_dir,
          component_metadata,
          chunk_rows
        )
    end
  end

  defp baselines_from_component_metadata!(python_report, components_dir, metadata, chunk_rows) do
    stage_file =
      get_in(python_report, ["stage_debug", "all_selected_stage_tensor_file"]) ||
        get_in(python_report, ["inputs", "all_selected_stage_tensor_file"]) ||
        Path.join(components_dir, @all_selected_stage_file)

    unless File.exists?(stage_file) do
      raise ArgumentError, "missing all-selected Python stage tensor file: #{stage_file}"
    end

    metadata
    |> Map.get("selected_tensors", [])
    |> Enum.map(fn entry ->
      source_shape = Map.get(entry, "source_shape") || Map.fetch!(entry, "shape")
      row_count = source_shape |> hd() |> positive_integer!(:source_rows)

      entry
      |> Map.take([
        "source_name",
        "elixir_name",
        "safe_key",
        "component_tensors",
        "scale_tensor",
        "source_shape",
        "offset_start",
        "offset_end",
        "singular_values",
        "stage_tensors"
      ])
      |> Map.put("stage_tensor_file", stage_file)
      |> Map.put("source_shape", source_shape)
      |> Map.put("chunks", chunks(row_count, chunk_rows))
    end)
  end

  defp chunks(row_count, chunk_rows) do
    0
    |> Stream.iterate(&(&1 + chunk_rows))
    |> Stream.take_while(&(&1 < row_count))
    |> Enum.with_index()
    |> Enum.map(fn {row_start, index} ->
      row_end = min(row_start + chunk_rows, row_count)

      %{
        "chunk_index" => index,
        "row_start" => row_start,
        "row_end" => row_end,
        "row_count" => row_end - row_start
      }
    end)
  end

  defp filter_sources!(baselines, sources) do
    source_set = MapSet.new(sources || @default_sources)

    selected =
      Enum.filter(baselines, fn baseline ->
        MapSet.member?(source_set, Map.get(baseline, "source_name"))
      end)

    if selected == [] do
      raise ArgumentError,
            "no large tensor baselines matched sources #{inspect(MapSet.to_list(source_set))}"
    end

    selected
  end

  defp summary(chunk_checks) do
    required_checks =
      chunk_checks
      |> Enum.flat_map(&Map.fetch!(&1, "checks"))
      |> Enum.filter(& &1["required_for_functional_parity"])

    failed_required = Enum.reject(required_checks, & &1["functional_passed"])
    sources = chunk_checks |> Enum.map(& &1["source_name"]) |> Enum.uniq()

    %{
      "source_count" => length(sources),
      "sources" => sources,
      "chunk_count" => length(chunk_checks),
      "required_check_count" => length(required_checks),
      "failed_required_count" => length(failed_required),
      "functional_parity_passed" => failed_required == []
    }
  end

  defp required_failed_count(checks) do
    Enum.count(checks, fn check ->
      check["required_for_functional_parity"] and not check["functional_passed"]
    end)
  end

  defp emit_progress(nil, _event), do: :ok
  defp emit_progress(progress, event) when is_function(progress, 1), do: progress.(event)

  defp components_dir!(nil, python_report) do
    get_in(python_report, ["inputs", "components_dir"]) ||
      raise ArgumentError, "components_dir is required"
  end

  defp components_dir!(components_dir, _python_report) when is_binary(components_dir),
    do: components_dir

  defp load_component_metadata!(components_dir) do
    components_dir
    |> Path.join(@metadata_file)
    |> load_json!()
  end

  defp fetch_lazy!(map, key) do
    case Map.fetch(map, key) do
      {:ok, value} ->
        value

      :error ->
        raise ArgumentError,
              "missing tensor #{inspect(key)}; available keys: #{inspect(Map.keys(map))}"
    end
  end

  defp fetch_materialized!(map, key) do
    map
    |> fetch_lazy!(key)
    |> SafetensorsSlice.materialize!()
    |> host_snapshot()
  end

  defp positive_integer!(value, _name) when is_integer(value) and value > 0, do: value

  defp positive_integer!(value, name) do
    raise ArgumentError, "#{name} must be a positive integer, got #{inspect(value)}"
  end

  defp sanitize_python_key(source_name) do
    source_name
    |> String.replace("/", "__")
    |> String.replace(~r/[^0-9A-Za-z_.-]/, "__")
  end

  defp host_snapshot(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  defp device_copy(%Nx.Tensor{} = tensor, nil), do: tensor
  defp device_copy(%Nx.Tensor{} = tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp load_json!(nil), do: raise(ArgumentError, "JSON path is required")

  defp load_json!(path) when is_binary(path) do
    path
    |> File.read!()
    |> Jason.decode!()
  end

  defp normalize_json(value) when is_map(value) do
    Map.new(value, fn {key, val} -> {to_string(key), normalize_json(val)} end)
  end

  defp normalize_json(value) when is_list(value), do: Enum.map(value, &normalize_json/1)
  defp normalize_json(value) when is_tuple(value), do: Tuple.to_list(value)
  defp normalize_json(value) when is_boolean(value) or is_nil(value), do: value
  defp normalize_json(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_json(value), do: value
end
