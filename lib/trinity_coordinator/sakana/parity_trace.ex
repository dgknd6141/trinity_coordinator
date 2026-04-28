defmodule TrinityCoordinator.Sakana.ParityTrace do
  @moduledoc """
  Incremental parity tracing for the Sakana/Python SVD sample hash.

  The native Elixir path recomputes SVD with Nx. The Python reference hash was
  produced from Python/PyTorch SVD components. For non-zero singular-value
  offsets, different valid SVD bases can reconstruct the original tensor with
  zero offsets but diverge after per-singular-value scaling. This module emits a
  compact JSON report so the native path, imported Python-component path, dtype
  choices, orientation choices, and final tensor bytes can be compared side by
  side.

  EXLA may donate or delete device buffers after compiled linear-algebra calls.
  This tracer therefore snapshots tensors needed for diagnostics to
  `Nx.BinaryBackend` before using those tensors in later dot/SVD computations.
  Expensive reconstructions receive fresh device copies from those snapshots so
  report generation can inspect intermediate values without tripping donated
  buffer reads.
  """

  alias TrinityCoordinator.{Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, SVD}

  @router_vector_path "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"
  @reference_manifest_path "priv/sakana_trinity/reference/sakana_python_reference_manifest.json"
  @scale_count 9_216
  @hidden_size 1_024
  @output_count 10
  @component_file "trinity_svf_components.safetensors"
  @scale_file "trinity_svf_scale_offsets.safetensors"

  @type report :: map()

  @doc "Builds the complete native and optional Python-component parity report."
  @spec sample_report!(keyword()) :: report()
  def sample_report!(opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        router_vector_path: @router_vector_path,
        reference_manifest_path: @reference_manifest_path,
        components_dir: nil,
        python_report_path: nil,
        stage_dir: nil,
        native?: true,
        semantic_host?: true,
        semantic_device?: true,
        semantic_layout_diagnostics?: true,
        source_from_python_stage?: false,
        require_cuda: true
      )

    if opts[:require_cuda] do
      Runtime.put_cuda_backend!()
    end

    compute_backend = if opts[:require_cuda], do: {EXLA.Backend, client: :cuda}, else: nil

    reference = load_json!(opts[:reference_manifest_path])
    sample = Map.fetch!(reference, "sample_adapted_tensor")
    python_report = maybe_load_json(opts[:python_report_path])
    python_baseline = current_python_baseline(python_report)
    python_stage_tensors = python_report |> python_stage_file() |> maybe_read_safetensors()

    vector = SVD.load_router_vector!(opts[:router_vector_path])
    split = SVD.split_router_vector(vector, @scale_count, @hidden_size, @output_count)
    offsets = sample_offsets(split.scale_offsets, sample)

    source_context =
      source_context!(
        opts[:source_from_python_stage?],
        python_stage_tensors,
        reference,
        sample
      )

    # Snapshot before variants run. SVD/reconstruction can donate device
    # buffers, so every later variant receives fresh tensors from these
    # immutable host copies.
    vector_host = host_snapshot(vector)
    offsets_host = host_snapshot(offsets)
    source_host = host_snapshot(source_context.source_tensor)
    source_backend = Runtime.tensor_backend(source_context.source_tensor)

    native_variants =
      if opts[:native?] do
        native_variants(source_host, offsets_host, sample, compute_backend, python_baseline)
      else
        []
      end

    component_metadata = component_metadata(opts[:components_dir])

    semantic_context = %{
      compute_backend: compute_backend,
      python_baseline: python_baseline,
      component_metadata: component_metadata,
      python_stage_tensors: python_stage_tensors,
      stage_dir: opts[:stage_dir],
      semantic_host?: opts[:semantic_host?],
      semantic_device?: opts[:semantic_device?],
      semantic_layout_diagnostics?: opts[:semantic_layout_diagnostics?]
    }

    semantic =
      maybe_semantic_component_report(
        opts[:components_dir],
        source_host,
        sample,
        semantic_context
      )

    %{
      "schema" => "trinity_sakana_svd_parity_trace.v2",
      "generated_at_utc" =>
        DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601(),
      "paths" => %{
        "router_vector" => opts[:router_vector_path],
        "reference_manifest" => opts[:reference_manifest_path],
        "components_dir" => opts[:components_dir],
        "python_report" => opts[:python_report_path],
        "stage_dir" => opts[:stage_dir]
      },
      "reference" => %{
        "source_name" => sample["source_name"],
        "elixir_name" => sample["elixir_name"],
        "offset_start" => sample["offset_start"],
        "offset_end" => sample["offset_end"],
        "source_shape" => sample["source_shape"],
        "sample_reconstructed_shape" => sample["sample_reconstructed_shape"],
        "expected_bf16_sha256" => sample["sample_reconstructed_bf16_sha256"],
        "expected_bf16_min" => sample["sample_reconstructed_bf16_min"],
        "expected_bf16_max" => sample["sample_reconstructed_bf16_max"]
      },
      "selection" => %{
        "selected_tensor_count" => source_context.selected_tensor_count,
        "selected_singular_value_count" => source_context.selected_singular_value_count,
        "sample_elixir_shape" => source_context.sample_elixir_shape,
        "sample_source_oriented_shape" => shape_list(source_host),
        "sample_source_type" => source_context.sample_source_type,
        "sample_source_backend" => source_context.sample_source_backend,
        "sample_source_oriented_backend" => source_backend,
        "diagnostic_snapshots_backend" => Runtime.tensor_backend(source_host),
        "compute_backend" => inspect(compute_backend || Nx.BinaryBackend),
        "native_svd_enabled" => opts[:native?],
        "source_from_python_stage" => source_context.from_python_stage?
      },
      "router_vector" => tensor_summary(vector_host, prefix_count: 8),
      "scale_offsets" => tensor_summary(offsets_host, prefix_count: 16),
      "source_tensor" =>
        tensor_summary(source_host, prefix_count: 16, backend_label: source_backend),
      "python_current_baseline" => python_baseline,
      "python_component_metadata" => component_metadata,
      "native_elixir_svd_variants" => native_variants,
      "semantic_python_component_variants" => semantic
    }
  end

  @doc "Writes a parity report as pretty JSON."
  @spec write_json!(String.t(), report()) :: :ok
  def write_json!(path, report) when is_binary(path) and is_map(report) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(normalize_json(report), pretty: true))
    :ok
  end

  defp native_variants(source_host, offsets_host, sample, compute_backend, python_baseline) do
    [
      %{
        label: "native_nx_f32_svd_offsets_singular_final_bf16",
        compute_type: :f32,
        offset_type: :singular
      },
      %{
        label: "native_nx_source_svd_offsets_singular_final_bf16",
        compute_type: :source,
        offset_type: :singular
      },
      %{label: "native_nx_f32_svd_offsets_f32_final_bf16", compute_type: :f32, offset_type: :f32},
      %{
        label: "native_nx_f32_svd_offsets_source_final_bf16",
        compute_type: :f32,
        offset_type: :source
      }
    ]
    |> Enum.map(fn config ->
      source_host
      |> native_variant(offsets_host, sample, config, compute_backend)
      |> add_python_match(python_baseline)
    end)
  end

  defp native_variant(source_host, offsets_host, sample, config, compute_backend) do
    source_device = device_copy(source_host, compute_backend)
    decomp = SVD.decompose_tensor(source_device, compute_type: config.compute_type)

    # Capture labels and host snapshots before any later reconstruction can
    # consume/donate the EXLA device buffers.
    u_backend = Runtime.tensor_backend(decomp.u)
    s_backend = Runtime.tensor_backend(decomp.s)
    v_backend = Runtime.tensor_backend(decomp.v)
    u_host = host_snapshot(decomp.u)
    s_host = host_snapshot(decomp.s)
    v_host = host_snapshot(decomp.v)

    typed_offsets_host =
      cast_offsets(offsets_host, config.offset_type, source_host, %{s: s_host}) |> host_snapshot()

    zero_offsets_host =
      Nx.broadcast(0.0, Nx.shape(s_host)) |> Nx.as_type(Nx.type(s_host)) |> host_snapshot()

    svd_source_host = svd_source_tensor(source_host, config.compute_type) |> host_snapshot()

    zero_reconstruct_host =
      %{u: u_host, s: s_host, v: v_host}
      |> reconstruct_on_backend(zero_offsets_host, compute_backend)
      |> host_snapshot()

    sample_reconstruct_host =
      %{u: u_host, s: s_host, v: v_host}
      |> reconstruct_on_backend(typed_offsets_host, compute_backend)
      |> Nx.as_type(:bf16)
      |> host_snapshot()

    sample_reconstruct_f32_host =
      %{u: u_host, s: s_host, v: v_host}
      |> reconstruct_on_backend(typed_offsets_host, compute_backend)
      |> host_snapshot()

    final_f32 =
      orient_to_shape!(
        sample_reconstruct_f32_host,
        sample["sample_reconstructed_shape"],
        "#{config.label}_f32"
      )

    final =
      orient_to_shape!(
        sample_reconstruct_host,
        sample["sample_reconstructed_shape"],
        config.label
      )

    expected = sample["sample_reconstructed_bf16_sha256"]
    observed = Artifact.tensor_sha256(final)

    %{
      "label" => config.label,
      "svd_provider" => "elixir_nx",
      "compute_type" => Atom.to_string(config.compute_type),
      "offset_type" => Atom.to_string(config.offset_type),
      "u" =>
        tensor_summary(u_host,
          prefix_count: 4,
          include_alt_hashes: false,
          backend_label: u_backend
        ),
      "s" => singular_summary(s_host, typed_offsets_host),
      "v" =>
        tensor_summary(v_host,
          prefix_count: 4,
          include_alt_hashes: false,
          backend_label: v_backend
        ),
      "component_backends" => %{"u" => u_backend, "s" => s_backend, "v" => v_backend},
      "zero_offset_max_abs_error_vs_source" =>
        max_abs_error(zero_reconstruct_host, svd_source_host),
      "final_f32_before_bf16" => tensor_summary(final_f32, prefix_count: 16),
      "final" => tensor_summary(final, prefix_count: 16),
      "observed_bf16_sha256" => observed,
      "expected_bf16_sha256" => expected,
      "matches_expected" => observed == expected
    }
  end

  defp maybe_semantic_component_report(nil, _source_host, _sample, _context), do: nil

  defp maybe_semantic_component_report("", _source_host, _sample, _context), do: nil

  defp maybe_semantic_component_report(
         components_dir,
         source_host,
         sample,
         context
       ) do
    component_path = Path.join(components_dir, @component_file)
    scale_path = Path.join(components_dir, @scale_file)

    if File.exists?(component_path) and File.exists?(scale_path) do
      components = Safetensors.read!(component_path)
      scales = Safetensors.read!(scale_path)
      safe_key = sanitize_python_key(sample["source_name"])
      u = fetch_tensor!(components, "svd.U.#{safe_key}") |> host_snapshot()
      s = fetch_tensor!(components, "svd.S.#{safe_key}") |> host_snapshot()
      v = fetch_tensor!(components, "svd.V.#{safe_key}") |> host_snapshot()
      offsets = fetch_tensor!(scales, "svf.scale_offsets.#{safe_key}") |> host_snapshot()
      decomp_host = %{u: u, s: s, v: v}
      stage_context = %{python_tensors: context.python_stage_tensors, dir: context.stage_dir}

      for compute_target <-
            semantic_compute_targets(
              context.compute_backend,
              context.semantic_host?,
              context.semantic_device?
            ),
          layout <-
            semantic_layouts(
              context.component_metadata,
              context.semantic_layout_diagnostics?
            ) do
        layout
        |> safe_semantic_variant(
          decomp_host,
          offsets,
          source_host,
          sample,
          compute_target,
          stage_context
        )
        |> add_python_match(context.python_baseline)
      end
    else
      %{
        "error" => "missing_semantic_component_files",
        "component_path" => component_path,
        "scale_path" => scale_path
      }
    end
  end

  defp safe_semantic_variant(
         layout,
         decomp_host,
         offsets,
         source_host,
         sample,
         compute_target,
         stage_context
       ) do
    semantic_variant(
      layout,
      decomp_host,
      offsets,
      source_host,
      sample,
      compute_target,
      stage_context
    )
  rescue
    e ->
      %{
        "label" => semantic_label(layout, compute_target),
        "svd_provider" => "python_components_safetensors",
        "compute_backend" => compute_target.label,
        "v_layout" => Atom.to_string(layout),
        "error" => Exception.message(e),
        "error_stacktrace" => Exception.format(:error, e, __STACKTRACE__),
        "matches_expected" => false
      }
  end

  defp semantic_variant(
         layout,
         decomp_host,
         offsets_host,
         source_host,
         sample,
         compute_target,
         stage_context
       ) do
    %{u: u_host, s: s_host, v: v_host} = decomp_host
    typed_offsets_host = Nx.as_type(offsets_host, Nx.type(s_host)) |> host_snapshot()

    zero_offsets_host =
      Nx.broadcast(0.0, Nx.shape(s_host)) |> Nx.as_type(Nx.type(s_host)) |> host_snapshot()

    zero_reconstruct_host =
      %{u: u_host, s: s_host, v: v_host}
      |> reconstruct_on_backend(zero_offsets_host, compute_target.backend, v_layout: layout)
      |> host_snapshot()

    sample_reconstruct_host =
      %{u: u_host, s: s_host, v: v_host}
      |> reconstruct_on_backend(typed_offsets_host, compute_target.backend, v_layout: layout)
      |> Nx.as_type(:bf16)
      |> host_snapshot()

    sample_reconstruct_f32_host =
      %{u: u_host, s: s_host, v: v_host}
      |> reconstruct_on_backend(typed_offsets_host, compute_target.backend, v_layout: layout)
      |> host_snapshot()

    final_f32 =
      orient_to_shape!(
        sample_reconstruct_f32_host,
        sample["sample_reconstructed_shape"],
        "semantic_#{layout}_f32"
      )

    final =
      orient_to_shape!(
        sample_reconstruct_host,
        sample["sample_reconstructed_shape"],
        "semantic_#{layout}"
      )

    expected = sample["sample_reconstructed_bf16_sha256"]
    observed = Artifact.tensor_sha256(final)

    {_stage_tensors, stage_file, stage_checks} =
      if layout == :torch_v do
        stage_tensors =
          semantic_stage_tensors(
            u_host,
            s_host,
            offsets_host,
            source_host,
            sample,
            zero_reconstruct_host,
            sample_reconstruct_f32_host
          )

        stage_checks = compare_stage_tensors(stage_tensors, stage_context.python_tensors)

        stage_file =
          maybe_write_stage_tensors(stage_context.dir, compute_target, layout, stage_tensors)

        {stage_tensors, stage_file, stage_checks}
      else
        {%{}, nil, []}
      end

    %{
      "label" => semantic_label(layout, compute_target),
      "svd_provider" => "python_components_safetensors",
      "compute_backend" => compute_target.label,
      "v_layout" => Atom.to_string(layout),
      "u" => tensor_summary(u_host, prefix_count: 4, include_alt_hashes: false),
      "s" => singular_summary(s_host, typed_offsets_host),
      "v" => tensor_summary(v_host, prefix_count: 4, include_alt_hashes: false),
      "offsets" => tensor_summary(offsets_host, prefix_count: 16),
      "zero_offset_max_abs_error_vs_source" =>
        max_abs_error(zero_reconstruct_host, Nx.as_type(source_host, :f32)),
      "final_f32_before_bf16" => tensor_summary(final_f32, prefix_count: 16),
      "final" => tensor_summary(final, prefix_count: 16),
      "stage_debug" => %{
        "schema" => "trinity_sakana_elixir_stage_debug.v1",
        "stage_tensor_file" => stage_file,
        "compared_to_python_stage_tensors" => not is_nil(stage_context.python_tensors),
        "functional_parity_passed" => stage_checks_passed?(stage_checks),
        "checks" => stage_checks
      },
      "observed_bf16_sha256" => observed,
      "expected_bf16_sha256" => expected,
      "matches_expected" => observed == expected
    }
  end

  defp reconstruct_on_backend(decomp_host, offsets_host, compute_backend, opts \\ []) do
    device_decomp = %{
      u: device_copy(decomp_host.u, compute_backend),
      s: device_copy(decomp_host.s, compute_backend),
      v: device_copy(decomp_host.v, compute_backend)
    }

    device_offsets = device_copy(offsets_host, compute_backend)
    SVD.reconstruct(device_decomp, device_offsets, opts)
  end

  defp semantic_stage_tensors(
         u,
         s,
         offsets,
         source,
         sample,
         zero_source_f32,
         adapted_source_f32
       ) do
    offsets = Nx.as_type(offsets, Nx.type(s)) |> host_snapshot()
    scaled_s = Nx.multiply(s, Nx.add(offsets, 1)) |> host_snapshot()
    normalization = Nx.divide(Nx.sum(s), Nx.sum(scaled_s)) |> host_snapshot()

    u_scaled =
      Nx.multiply(u, Nx.reshape(scaled_s, {1, Nx.axis_size(scaled_s, 0)})) |> host_snapshot()

    # Avoid a second large host-side matrix multiplication while still exposing
    # the same conceptual checkpoint Python writes.
    matmul_pre_norm = Nx.divide(adapted_source_f32, normalization) |> host_snapshot()

    final_f32 =
      orient_to_shape!(adapted_source_f32, sample["sample_reconstructed_shape"], "stage_final")
      |> Nx.as_type(:f32)
      |> host_snapshot()

    %{
      "stage.source_f32" => Nx.as_type(source, :f32) |> host_snapshot(),
      "stage.offsets_f32" => Nx.as_type(offsets, :f32) |> host_snapshot(),
      "stage.scaled_s" => Nx.as_type(scaled_s, :f32) |> host_snapshot(),
      "stage.normalization" =>
        normalization |> Nx.reshape({1}) |> Nx.as_type(:f32) |> host_snapshot(),
      "stage.u_scaled" => Nx.as_type(u_scaled, :f32) |> host_snapshot(),
      "stage.matmul_pre_norm" => Nx.as_type(matmul_pre_norm, :f32) |> host_snapshot(),
      "stage.adapted_source_f32" => Nx.as_type(adapted_source_f32, :f32) |> host_snapshot(),
      "stage.final_f32" => final_f32,
      "stage.final_bf16" => Nx.as_type(final_f32, :bf16) |> host_snapshot(),
      "stage.zero_source_f32" => Nx.as_type(zero_source_f32, :f32) |> host_snapshot()
    }
  end

  defp maybe_write_stage_tensors(nil, _compute_target, _layout, _stage_tensors), do: nil
  defp maybe_write_stage_tensors("", _compute_target, _layout, _stage_tensors), do: nil

  defp maybe_write_stage_tensors(stage_dir, compute_target, layout, stage_tensors) do
    File.mkdir_p!(stage_dir)
    file = "trinity_svf_elixir_stage_#{compute_target.label}_#{layout}.safetensors"
    path = Path.join(stage_dir, file)

    payload =
      Map.new(stage_tensors, fn {key, tensor} ->
        {key, host_snapshot(tensor)}
      end)

    Safetensors.write!(path, payload)
    path
  end

  defp compare_stage_tensors(_stage_tensors, nil), do: []

  defp compare_stage_tensors(stage_tensors, python_stage_tensors) do
    stage_tensors
    |> Enum.sort_by(fn {key, _tensor} -> key end)
    |> Enum.map(fn {key, tensor} ->
      case Map.fetch(python_stage_tensors, key) do
        {:ok, python_tensor} -> stage_check(key, tensor, python_tensor)
        :error -> missing_stage_check(key)
      end
    end)
  end

  defp stage_check(key, elixir_tensor, python_tensor) do
    tolerance = stage_tolerance(key)
    shape_match = Nx.shape(elixir_tensor) == Nx.shape(python_tensor)
    byte_match = Artifact.tensor_sha256(elixir_tensor) == Artifact.tensor_sha256(python_tensor)

    if shape_match do
      stage_value_check(key, elixir_tensor, python_tensor, tolerance, byte_match)
    else
      %{
        "stage" => key,
        "required_for_functional_parity" => tolerance.required?,
        "byte_match" => byte_match,
        "shape_match" => false,
        "elixir" => tensor_summary(elixir_tensor, prefix_count: 8),
        "python" => tensor_summary(python_tensor, prefix_count: 8),
        "max_abs_error" => nil,
        "mean_abs_error" => nil,
        "mismatched_element_count" => nil,
        "tolerance" => %{
          "max_abs_error" => tolerance.max_abs,
          "mean_abs_error" => tolerance.mean_abs
        },
        "functional_passed" => false
      }
    end
  end

  defp stage_value_check(key, elixir_tensor, python_tensor, tolerance, byte_match) do
    elixir_f32 = Nx.as_type(elixir_tensor, :f32)
    python_f32 = Nx.as_type(python_tensor, :f32)
    abs_diff = Nx.abs(Nx.subtract(elixir_f32, python_f32))
    max_abs = scalar(Nx.reduce_max(abs_diff))
    mean_abs = scalar(Nx.divide(Nx.sum(abs_diff), Nx.tensor(Nx.size(abs_diff), type: :f32)))
    mismatch_count = scalar(Nx.sum(Nx.as_type(Nx.not_equal(elixir_tensor, python_tensor), :s64)))

    %{
      "stage" => key,
      "required_for_functional_parity" => tolerance.required?,
      "byte_match" => byte_match,
      "shape_match" => true,
      "elixir" => tensor_summary(elixir_tensor, prefix_count: 8),
      "python" => tensor_summary(python_tensor, prefix_count: 8),
      "max_abs_error" => max_abs,
      "mean_abs_error" => mean_abs,
      "mismatched_element_count" => mismatch_count,
      "tolerance" => %{
        "max_abs_error" => tolerance.max_abs,
        "mean_abs_error" => tolerance.mean_abs
      },
      "functional_passed" => max_abs <= tolerance.max_abs and mean_abs <= tolerance.mean_abs
    }
  end

  defp missing_stage_check(key) do
    tolerance = stage_tolerance(key)

    %{
      "stage" => key,
      "required_for_functional_parity" => tolerance.required?,
      "byte_match" => false,
      "shape_match" => false,
      "missing_python_stage" => true,
      "functional_passed" => not tolerance.required?
    }
  end

  defp stage_checks_passed?([]), do: nil

  defp stage_checks_passed?(checks) do
    Enum.all?(checks, fn check ->
      not check["required_for_functional_parity"] or check["functional_passed"]
    end)
  end

  defp stage_tolerance("stage.source_f32"), do: %{required?: true, max_abs: 0.0, mean_abs: 0.0}
  defp stage_tolerance("stage.offsets_f32"), do: %{required?: true, max_abs: 0.0, mean_abs: 0.0}

  defp stage_tolerance("stage.scaled_s"),
    do: %{required?: true, max_abs: 1.0e-6, mean_abs: 1.0e-8}

  defp stage_tolerance("stage.normalization"),
    do: %{required?: true, max_abs: 1.0e-6, mean_abs: 1.0e-6}

  defp stage_tolerance("stage.u_scaled"),
    do: %{required?: true, max_abs: 1.0e-6, mean_abs: 1.0e-8}

  defp stage_tolerance("stage.zero_source_f32"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.matmul_pre_norm"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.adapted_source_f32"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.final_f32"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.final_bf16"),
    do: %{required?: false, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance(_key), do: %{required?: false, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp singular_summary(s_host, offsets_host) do
    offsets_host = Nx.as_type(offsets_host, Nx.type(s_host)) |> host_snapshot()
    scaled_s = Nx.multiply(s_host, Nx.add(offsets_host, 1)) |> host_snapshot()

    sum_s = Nx.sum(Nx.as_type(s_host, :f32)) |> host_snapshot()
    sum_scaled_s = Nx.sum(Nx.as_type(scaled_s, :f32)) |> host_snapshot()

    %{
      "singular_values" => tensor_summary(s_host, prefix_count: 16),
      "typed_offsets" => tensor_summary(offsets_host, prefix_count: 16),
      "scaled_s" => tensor_summary(scaled_s, prefix_count: 16),
      "sum_s" => scalar(sum_s),
      "sum_scaled_s" => scalar(sum_scaled_s),
      "normalization" => scalar(Nx.divide(sum_s, sum_scaled_s))
    }
  end

  defp add_python_match(variant, nil) when is_map(variant) do
    variant
    |> Map.put("python_current_baseline_label", nil)
    |> Map.put("python_current_bf16_sha256", nil)
    |> Map.put("matches_python_current", nil)
  end

  defp add_python_match(variant, python_baseline)
       when is_map(variant) and is_map(python_baseline) do
    python_digest = Map.get(python_baseline, "observed_bf16_sha256")

    variant
    |> Map.put("python_current_baseline_label", Map.get(python_baseline, "label"))
    |> Map.put("python_current_bf16_sha256", python_digest)
    |> Map.put(
      "matches_python_current",
      is_binary(python_digest) and Map.get(variant, "observed_bf16_sha256") == python_digest
    )
  end

  defp current_python_baseline(nil), do: nil

  defp current_python_baseline(report) when is_map(report) do
    reference = Map.get(report, "reference", %{})
    label = Map.get(reference, "current_python_baseline_label")
    digest = Map.get(reference, "current_python_baseline_bf16_sha256")

    if is_binary(label) and is_binary(digest) do
      %{
        "label" => label,
        "observed_bf16_sha256" => digest,
        "expected_hash_reproducible" => Map.get(reference, "expected_hash_reproducible"),
        "source" => "python_report_reference"
      }
    else
      report
      |> Map.get("variants", [])
      |> Enum.find(fn variant ->
        is_map(variant) and is_binary(Map.get(variant, "observed_bf16_sha256"))
      end)
      |> case do
        nil ->
          nil

        variant ->
          %{
            "label" => Map.get(variant, "label"),
            "observed_bf16_sha256" => Map.get(variant, "observed_bf16_sha256"),
            "expected_hash_reproducible" => Map.get(reference, "expected_hash_reproducible"),
            "source" => "python_report_first_variant"
          }
      end
    end
  end

  defp component_metadata(nil), do: nil
  defp component_metadata(""), do: nil

  defp component_metadata(components_dir) when is_binary(components_dir) do
    path = Path.join(components_dir, "trinity_svf_debug_manifest.json")

    if File.exists?(path) do
      load_json!(path)
    else
      nil
    end
  end

  defp python_stage_file(nil), do: nil

  defp python_stage_file(report) when is_map(report) do
    get_in(report, ["stage_debug", "stage_tensor_file"]) ||
      get_in(report, ["inputs", "stage_tensor_file"])
  end

  defp maybe_read_safetensors(nil), do: nil
  defp maybe_read_safetensors(""), do: nil

  defp maybe_read_safetensors(path) when is_binary(path) do
    if File.exists?(path) do
      path
      |> Safetensors.read!()
      |> Map.new(fn {key, tensor} -> {key, host_snapshot(tensor)} end)
    else
      nil
    end
  end

  defp semantic_compute_targets(_compute_backend, true, false) do
    [%{label: "host_binary", backend: nil}]
  end

  defp semantic_compute_targets(_compute_backend, false, false) do
    raise ArgumentError, "semantic replay requires at least one compute target"
  end

  defp semantic_compute_targets(nil, true, _semantic_device?) do
    [%{label: "host_binary", backend: nil}]
  end

  defp semantic_compute_targets(nil, false, true) do
    raise ArgumentError, "device-only semantic replay requires CUDA; remove --no-cuda"
  end

  defp semantic_compute_targets(compute_backend, false, true) do
    [%{label: "device_#{backend_label_slug(compute_backend)}", backend: compute_backend}]
  end

  defp semantic_compute_targets(compute_backend, true, true) do
    [
      %{label: "host_binary", backend: nil},
      %{label: "device_#{backend_label_slug(compute_backend)}", backend: compute_backend}
    ]
  end

  defp semantic_label(layout, compute_target) do
    "semantic_python_components_#{compute_target.label}_v_layout_#{layout}"
  end

  defp backend_label_slug(backend) do
    backend
    |> inspect()
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.trim("_")
  end

  defp semantic_layouts(metadata, false) do
    [preferred_layout(metadata)]
  end

  defp semantic_layouts(metadata, true) do
    preferred = preferred_layout(metadata)

    [preferred, :torch_v, :nx, :vh]
    |> Enum.uniq()
  end

  defp preferred_layout(metadata) do
    preferred =
      metadata
      |> layout_from_metadata()
      |> case do
        nil -> :torch_v
        layout -> layout
      end

    preferred
  end

  defp layout_from_metadata(nil), do: nil

  defp layout_from_metadata(metadata) when is_map(metadata) do
    metadata
    |> Map.get("component_v_layout")
    |> normalize_v_layout()
  end

  defp normalize_v_layout(:torch_v), do: :torch_v
  defp normalize_v_layout(:nx), do: :nx
  defp normalize_v_layout(:vh), do: :vh
  defp normalize_v_layout("torch_v"), do: :torch_v
  defp normalize_v_layout("torch-v"), do: :torch_v
  defp normalize_v_layout("torch"), do: :torch_v
  defp normalize_v_layout("nx"), do: :nx
  defp normalize_v_layout("vh"), do: :vh
  defp normalize_v_layout(_), do: nil

  defp qwen_selected_tensors(model_info) do
    SVD.decomposable_tensor_entries(
      model_info.params,
      path_filter: SVD.layer_index_filter([26])
    )
  end

  defp source_context!(true, python_stage_tensors, reference, _sample) do
    source_tensor =
      case python_stage_tensors do
        %{"stage.source_f32" => %Nx.Tensor{} = tensor} ->
          tensor

        _ ->
          raise ArgumentError,
                "--source-from-python-stage requires a Python report with stage.source_f32"
      end

    %{
      selected_tensor_count: reference_selected_tensor_count(reference),
      selected_singular_value_count: reference_selected_singular_value_count(reference),
      sample_elixir_shape: shape_list(source_tensor),
      sample_source_type: inspect(Nx.type(source_tensor)),
      sample_source_backend: "python_stage_safetensors",
      source_tensor: source_tensor,
      from_python_stage?: true
    }
  end

  defp source_context!(false, _python_stage_tensors, _reference, sample) do
    {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    selected = qwen_selected_tensors(model_info)
    sample_entry = sample_entry!(selected, sample)

    source_tensor =
      orient_to_shape!(sample_entry.tensor, sample["source_shape"], sample["elixir_name"])

    %{
      selected_tensor_count: length(selected),
      selected_singular_value_count: SVD.singular_value_count(selected),
      sample_elixir_shape: shape_list(sample_entry.tensor),
      sample_source_type: inspect(Nx.type(sample_entry.tensor)),
      sample_source_backend: Runtime.tensor_backend(sample_entry.tensor),
      source_tensor: source_tensor,
      from_python_stage?: false
    }
  end

  defp reference_selected_tensor_count(reference) do
    reference
    |> Map.get("selected_tensors", [])
    |> length()
  end

  defp reference_selected_singular_value_count(reference) do
    reference
    |> Map.get("selected_tensors", [])
    |> Enum.reduce(0, fn entry, acc -> acc + Map.fetch!(entry, "singular_values") end)
  end

  defp sample_entry!(selected, sample) do
    Enum.find(selected, &(&1.path == sample["elixir_name"])) ||
      raise ArgumentError, "sample tensor #{inspect(sample["elixir_name"])} was not selected"
  end

  defp sample_offsets(scale_offsets, sample) do
    offset_start = sample["offset_start"]
    singular_values = sample["offset_end"] - sample["offset_start"]
    Nx.slice(scale_offsets, [offset_start], [singular_values])
  end

  defp cast_offsets(offsets, :singular, _source_tensor, decomp),
    do: Nx.as_type(offsets, Nx.type(decomp.s))

  defp cast_offsets(offsets, :f32, _source_tensor, _decomp), do: Nx.as_type(offsets, :f32)

  defp cast_offsets(offsets, :source, source_tensor, _decomp),
    do: Nx.as_type(offsets, Nx.type(source_tensor))

  defp svd_source_tensor(tensor, :source), do: tensor
  defp svd_source_tensor(tensor, :f32), do: Nx.as_type(tensor, :f32)

  defp orient_to_shape!(%Nx.Tensor{} = tensor, shape_list, label) when is_list(shape_list) do
    target = List.to_tuple(shape_list)

    cond do
      Nx.shape(tensor) == target ->
        tensor

      tuple_size(Nx.shape(tensor)) == 2 and Nx.shape(Nx.transpose(tensor)) == target ->
        Nx.transpose(tensor) |> host_snapshot()

      true ->
        raise ArgumentError,
              "cannot orient #{inspect(label)} from #{inspect(Nx.shape(tensor))} to #{inspect(target)}"
    end
  end

  defp max_abs_error(left, right) do
    left = Nx.as_type(left, :f32)
    right = Nx.as_type(right, :f32)

    left
    |> Nx.subtract(right)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> host_snapshot()
    |> scalar()
  end

  defp tensor_summary(tensor, opts) do
    opts = Keyword.validate!(opts, prefix_count: 8, include_alt_hashes: true, backend_label: nil)
    tensor = host_snapshot(tensor)
    tensor_f32 = Nx.as_type(tensor, :f32) |> host_snapshot()
    size = Nx.size(tensor)
    prefix_count = min(size, opts[:prefix_count])

    base = %{
      "shape" => shape_list(tensor),
      "type" => inspect(Nx.type(tensor)),
      "backend" => opts[:backend_label] || Runtime.tensor_backend(tensor),
      "snapshot_backend" => Runtime.tensor_backend(tensor),
      "size" => size,
      "sha256" => Artifact.tensor_sha256(tensor),
      "min" => scalar(Nx.reduce_min(tensor_f32)),
      "max" => scalar(Nx.reduce_max(tensor_f32)),
      "sum" => scalar(Nx.sum(tensor_f32)),
      "prefix_f32" => prefix_f32(tensor, prefix_count)
    }

    if opts[:include_alt_hashes] do
      Map.merge(base, %{
        "sha256_as_f32" => Artifact.tensor_sha256(Nx.as_type(tensor, :f32)),
        "sha256_as_bf16" => Artifact.tensor_sha256(Nx.as_type(tensor, :bf16))
      })
    else
      base
    end
  end

  defp prefix_f32(_tensor, 0), do: []

  defp prefix_f32(tensor, count) do
    tensor
    |> host_snapshot()
    |> Nx.as_type(:f32)
    |> Nx.reshape({Nx.size(tensor)})
    |> Nx.slice([0], [count])
    |> host_snapshot()
    |> Nx.to_flat_list()
  end

  defp scalar(tensor), do: tensor |> host_snapshot() |> Nx.to_number() |> finite_float()

  defp finite_float(value) when is_float(value), do: value

  defp finite_float(value), do: value

  defp shape_list(tensor), do: Nx.shape(tensor) |> Tuple.to_list()

  defp fetch_tensor!(map, key) do
    case Map.fetch(map, key) do
      {:ok, %Nx.Tensor{} = tensor} ->
        tensor

      _ ->
        raise ArgumentError,
              "missing tensor #{inspect(key)}; available keys: #{inspect(Map.keys(map))}"
    end
  end

  defp sanitize_python_key(source_name) do
    source_name
    |> String.replace("/", "__")
    |> String.replace(~r/[^0-9A-Za-z_.-]/, "__")
  end

  defp host_snapshot(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  defp device_copy(%Nx.Tensor{} = tensor, nil), do: tensor
  defp device_copy(%Nx.Tensor{} = tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp maybe_load_json(nil), do: nil
  defp maybe_load_json(""), do: nil

  defp maybe_load_json(path) when is_binary(path) do
    if File.exists?(path) do
      load_json!(path)
    else
      raise ArgumentError, "JSON report does not exist: #{path}"
    end
  end

  defp load_json!(path) do
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
