defmodule TrinityCoordinator.Sakana.PythonImporter do
  @moduledoc """
  Converts a Python semantic TRINITY/Sakana export bundle into the canonical
  Elixir runtime artifact layout.

  Python semantic exports contain SVD components and scale offsets. The Elixir
  runtime expects reconstructed adapted tensors plus a canonical manifest. This
  importer normalizes the Python schema instead of teaching the runtime artifact
  loader to accept multiple manifest formats.
  """

  alias TrinityCoordinator.{Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, ExportSpec, SVD}

  @default_manifest "trinity_sakana_export_manifest.json"
  @default_components "trinity_svf_components.safetensors"
  @default_scales "trinity_svf_scale_offsets.safetensors"
  @default_head "trinity_router_head.safetensors"
  @python_head_key "trinity.router_head.linear.weight"
  @checkpoint_width 4

  @fallback_mapping %{
    "model.embed_tokens.weight" => "embedder.token_embedding.kernel",
    "model.layers.26.self_attn.q_proj.weight" => "decoder.blocks.26.self_attention.query.kernel",
    "model.layers.26.self_attn.k_proj.weight" => "decoder.blocks.26.self_attention.key.kernel",
    "model.layers.26.self_attn.v_proj.weight" => "decoder.blocks.26.self_attention.value.kernel",
    "model.layers.26.self_attn.o_proj.weight" => "decoder.blocks.26.self_attention.output.kernel",
    "model.layers.26.mlp.gate_proj.weight" => "decoder.blocks.26.ffn.gate.kernel",
    "model.layers.26.mlp.up_proj.weight" => "decoder.blocks.26.ffn.intermediate.kernel",
    "model.layers.26.mlp.down_proj.weight" => "decoder.blocks.26.ffn.output.kernel",
    "lm_head.weight" => "language_modeling_head.output.kernel"
  }

  @type import_opts :: [
          source_dir: String.t(),
          manifest: String.t(),
          reference_manifest: String.t() | nil,
          out_dir: String.t(),
          force: boolean(),
          resume: boolean(),
          load_qwen: boolean(),
          cast_tensors: boolean(),
          spec: ExportSpec.t() | atom() | String.t(),
          progress: (map() -> any()) | nil
        ]

  @spec import_bundle(import_opts()) :: {:ok, map()} | {:error, term()}
  def import_bundle(opts) when is_list(opts) do
    opts = normalize_opts(opts)
    result = do_import_bundle(opts)

    emit(opts, %{event: :python_import_finished, result: elem(result, 0)})
    result
  end

  def import_bundle(_opts), do: {:error, :invalid_import_options}

  defp do_import_bundle(opts) do
    with :ok <- prepare_output(opts),
         {:ok, python_manifest} <- load_json(opts.python_manifest_path),
         {:ok, reference_manifest} <- maybe_load_json(opts.reference_manifest_path),
         {:ok, paths} <- resolve_bundle_paths(opts.source_dir, python_manifest),
         {:ok, components} <- read_safetensors(paths.components, lazy: true),
         {:ok, scales} <- read_safetensors(paths.scales, lazy: true),
         {:ok, head_file} <- read_safetensors(paths.head),
         {:ok, selected_entries} <- selected_entries(python_manifest, reference_manifest),
         {:ok, targets} <- load_targets(opts),
         {:ok, head_weights} <- normalize_router_head(head_file, python_manifest, opts.spec),
         {:ok, selected_tensors} <-
           reconstruct_and_write_checkpoints(selected_entries, components, scales, targets, opts) do
      write_canonical_bundle(
        opts,
        python_manifest,
        reference_manifest,
        paths,
        head_weights,
        selected_tensors
      )
    end
  rescue
    e -> {:error, {:python_importer_exception, Exception.message(e)}}
  end

  defp normalize_opts(opts) do
    source_dir = Keyword.get(opts, :source_dir) || raise ArgumentError, "source_dir is required"
    out_dir = Keyword.get(opts, :out_dir) || raise ArgumentError, "out_dir is required"
    spec = ExportSpec.resolve!(Keyword.get(opts, :spec, :qwen3_0_6b_layer26))

    source_dir = Path.expand(source_dir)

    manifest_path =
      source_dir
      |> resolve_path(Keyword.get(opts, :manifest, @default_manifest))
      |> Path.expand()

    reference_manifest_path =
      case Keyword.get(opts, :reference_manifest) || Keyword.get(opts, :reference) do
        nil -> nil
        path -> Path.expand(path)
      end

    %{
      source_dir: source_dir,
      python_manifest_path: manifest_path,
      reference_manifest_path: reference_manifest_path,
      out_dir: Path.expand(out_dir),
      force: Keyword.get(opts, :force, false),
      resume: Keyword.get(opts, :resume, false),
      load_qwen: Keyword.get(opts, :load_qwen, true),
      cast_tensors: Keyword.get(opts, :cast_tensors, true),
      spec: spec,
      progress: Keyword.get(opts, :progress)
    }
  end

  defp prepare_output(opts) do
    cond do
      opts.force ->
        File.rm_rf(opts.out_dir)
        File.mkdir_p!(opts.out_dir)
        File.mkdir_p!(Artifact.checkpoint_path(opts.out_dir))
        :ok

      File.exists?(opts.out_dir) and not opts.resume ->
        {:error, :output_dir_already_exists_without_resume}

      true ->
        File.mkdir_p!(opts.out_dir)
        File.mkdir_p!(Artifact.checkpoint_path(opts.out_dir))
        :ok
    end
  end

  defp resolve_bundle_paths(source_dir, manifest) do
    {:ok,
     %{
       components:
         resolve_manifest_path(
           source_dir,
           manifest,
           [
             ["components_path"],
             ["component_path"],
             ["svf_components_path"],
             ["components_file"],
             ["outputs", "components"]
           ],
           @default_components
         ),
       scales:
         resolve_manifest_path(
           source_dir,
           manifest,
           [
             ["scale_offsets_path"],
             ["scales_path"],
             ["svf_scale_offsets_path"],
             ["scale_offsets_file"],
             ["outputs", "scale_offsets"]
           ],
           @default_scales
         ),
       head:
         resolve_manifest_path(
           source_dir,
           manifest,
           [
             ["router_head_path"],
             ["head_path"],
             ["router_head_file"],
             ["outputs", "head"]
           ],
           @default_head
         )
     }}
  end

  defp resolve_manifest_path(source_dir, manifest, keys, fallback) do
    value = Enum.find_value(keys, &deep_get(manifest, &1)) || fallback
    resolve_path(source_dir, value)
  end

  defp resolve_path(source_dir, path) when is_binary(path) do
    cond do
      Path.type(path) == :absolute -> path
      File.exists?(path) -> path
      true -> Path.join(source_dir, path)
    end
  end

  defp load_json(path) do
    with {:ok, body} <- File.read(path),
         {:ok, json} <- Jason.decode(body) do
      {:ok, json}
    else
      {:error, :enoent} -> {:error, {:missing_json, path}}
      {:error, reason} -> {:error, {:invalid_json, path, reason}}
    end
  end

  defp maybe_load_json(nil), do: {:ok, nil}
  defp maybe_load_json(path), do: load_json(path)

  defp read_safetensors(path, opts \\ []) do
    opts = Keyword.validate!(opts, lazy: false)
    {:ok, Safetensors.read!(path, lazy: opts[:lazy])}
  rescue
    e -> {:error, {:safetensors_read_error, path, Exception.message(e)}}
  end

  defp selected_entries(python_manifest, reference_manifest) do
    entries =
      deep_get(python_manifest, ["selected_tensors"]) ||
        deep_get(python_manifest, ["selected_entries"]) ||
        deep_get(python_manifest, ["tensors"]) ||
        deep_get(python_manifest, ["svf", "selected_tensors"]) ||
        deep_get(python_manifest, ["svf", "entries"]) ||
        deep_get(reference_manifest || %{}, ["selected_tensors"])

    if is_list(entries) and entries != [] do
      reference_by_source = reference_mapping(reference_manifest)

      normalized =
        entries
        |> Enum.with_index(1)
        |> Enum.map(fn {entry, index} ->
          normalize_selected_entry(entry, index, reference_by_source)
        end)

      {:ok, normalized}
    else
      {:error, :missing_selected_tensors}
    end
  end

  defp reference_mapping(nil), do: %{}

  defp reference_mapping(reference) when is_map(reference) do
    reference
    |> deep_get(["selected_tensors"])
    |> case do
      entries when is_list(entries) ->
        Map.new(entries, fn entry ->
          source = source_name(entry)
          {source, entry}
        end)

      _ ->
        %{}
    end
  end

  defp normalize_selected_entry(entry, index, reference_by_source) when is_map(entry) do
    source = source_name(entry)
    reference = Map.get(reference_by_source, source, %{})
    elixir_name = selected_elixir_name(entry, reference, source)

    validate_selected_names!(source, elixir_name, index)

    build_selected_entry(entry, reference, index, source, elixir_name)
  end

  defp normalize_selected_entry(other, index, _reference_by_source) do
    raise ArgumentError, "selected tensor #{index} is not a map: #{inspect(other)}"
  end

  defp selected_elixir_name(entry, reference, source) do
    deep_get(entry, ["elixir_name"]) ||
      deep_get(entry, ["elixir_path"]) ||
      deep_get(reference, ["elixir_name"]) ||
      Map.get(@fallback_mapping, source)
  end

  defp validate_selected_names!(source, elixir_name, index) do
    validate_non_empty_binary!(source, "selected tensor #{index} has no source_name")

    validate_non_empty_binary!(
      elixir_name,
      "selected tensor #{source} has no elixir_name mapping"
    )
  end

  defp validate_non_empty_binary!(value, _message) when is_binary(value) and value != "", do: :ok

  defp validate_non_empty_binary!(_value, message) do
    raise ArgumentError, message
  end

  defp build_selected_entry(entry, reference, index, source, elixir_name) do
    %{
      index: index,
      source_name: source,
      elixir_name: elixir_name,
      shape: selected_entry_shape(entry, reference),
      singular_values:
        first_metadata_value(entry, reference, ["singular_values", "num_singular_values"]),
      offset_start: first_metadata_value(entry, reference, "offset_start"),
      offset_end: first_metadata_value(entry, reference, "offset_end"),
      component_tensors: deep_get(entry, ["component_tensors"]) || %{},
      scale_tensor: deep_get(entry, ["scale_tensor"]),
      safe_key: first_metadata_value(entry, reference, ["safe_key", "safe_parameter"]),
      python_v_layout:
        first_metadata_value(entry, reference, ["python_v_layout", "v_layout"]) || "torch_v"
    }
  end

  defp selected_entry_shape(entry, reference) do
    shape = first_metadata_value(entry, reference, "shape")

    singular_values =
      first_metadata_value(entry, reference, ["singular_values", "num_singular_values"])

    if singular_value_shape?(shape, singular_values) do
      nil
    else
      normalize_shape(shape)
    end
  end

  defp singular_value_shape?([value], value), do: true
  defp singular_value_shape?({value}, value), do: true
  defp singular_value_shape?(_shape, _singular_values), do: false

  defp first_metadata_value(entry, reference, keys) when is_list(keys) do
    Enum.find_value(keys, &first_metadata_value(entry, reference, &1))
  end

  defp first_metadata_value(entry, reference, key) when is_binary(key) do
    deep_get(entry, [key]) || deep_get(reference, [key])
  end

  defp source_name(entry) when is_map(entry) do
    deep_get(entry, ["source_name"]) ||
      deep_get(entry, ["source_parameter"]) ||
      deep_get(entry, ["python_name"]) ||
      deep_get(entry, ["tensor_name"]) ||
      deep_get(entry, ["name"]) ||
      deep_get(entry, ["path"])
  end

  defp load_targets(%{load_qwen: false}), do: {:ok, %{by_path: %{}, verified?: false}}

  defp load_targets(%{load_qwen: true}) do
    Runtime.put_cuda_backend!()

    case SLMProfile.load_profile(:qwen_coordinator) do
      {:ok, {model_info, _tokenizer}} ->
        by_path = Map.new(SVD.flatten_tensor_entries(model_info.params), &{&1.path, &1})
        {:ok, %{by_path: by_path, verified?: true}}

      {:error, reason} ->
        {:error, {:qwen_target_load_error, reason}}
    end
  end

  defp normalize_router_head(head_file, manifest, spec) do
    key =
      deep_get(manifest, ["router_head_tensor"]) ||
        deep_get(manifest, ["router_head_key"]) ||
        deep_get(manifest, ["routing", "head_tensor"]) ||
        @python_head_key

    case Map.get(head_file, key) || only_tensor_if_singleton(head_file) do
      %Nx.Tensor{} = tensor ->
        expected = {ExportSpec.output_count(spec), spec.hidden_size}

        if Nx.shape(tensor) == expected or not strict_default_head?(spec) do
          {:ok, tensor}
        else
          {:error, {:router_head_shape_mismatch, expected, Nx.shape(tensor)}}
        end

      nil ->
        {:error, {:missing_router_head_tensor, key, Map.keys(head_file)}}
    end
  end

  defp strict_default_head?(%ExportSpec{name: :qwen3_0_6b_layer26}), do: true
  defp strict_default_head?(_), do: false

  defp only_tensor_if_singleton(map) when map_size(map) == 1 do
    map |> Map.values() |> hd()
  end

  defp only_tensor_if_singleton(_), do: nil

  defp reconstruct_and_write_checkpoints(entries, components, scales, targets, opts) do
    {selected_tensors, _cursor} =
      Enum.map_reduce(entries, 0, fn entry, cursor ->
        try do
          emit(opts, %{
            event: :python_import_tensor_started,
            index: entry.index,
            source_name: entry.source_name,
            elixir_name: entry.elixir_name
          })

          result = reconstruct_one(entry, components, scales, targets, opts)
          checkpoint_entry = write_checkpoint!(opts.out_dir, result, cursor)

          emit(opts, %{
            event: :python_import_tensor_checkpoint_written,
            index: entry.index,
            source_name: entry.source_name,
            checkpoint_path: checkpoint_entry["checkpoint_path"],
            checkpoint_sha256: checkpoint_entry["checkpoint_sha256"]
          })

          :erlang.garbage_collect()
          {checkpoint_entry, checkpoint_entry["offset_end"]}
        rescue
          e ->
            raise ArgumentError,
                  "failed to reconstruct #{entry.source_name}: #{Exception.message(e)}"
        end
      end)

    {:ok, selected_tensors}
  rescue
    e -> {:error, {:reconstruct_error, Exception.message(e)}}
  end

  defp reconstruct_one(entry, components, scales, targets, opts) do
    keys = component_keys(entry)

    u = fetch_tensor!(components, keys.u, :component_u)
    s = fetch_tensor!(components, keys.s, :component_s)
    v = fetch_tensor!(components, keys.v, :component_v)
    offsets = fetch_tensor!(scales, keys.scale, :scale_offsets)
    target = Map.get(targets.by_path, entry.elixir_name)
    v_layout = normalize_v_layout!(entry.python_v_layout)

    source_shape =
      normalize_shape(entry.shape) ||
        source_shape_from_components(u, v, v_layout)

    target_shape = target_shape(target, source_shape)

    u = transfer_for_target(u, target)
    s = transfer_for_target(s, target)
    v = transfer_for_target(v, target)
    offsets = transfer_for_target(offsets, target)

    reconstructed =
      SVD.reconstruct(%{u: u, s: s, v: v}, Nx.as_type(offsets, Nx.type(s)), v_layout: v_layout)

    reconstructed =
      orient_for_target!(reconstructed, target_shape, entry.elixir_name, entry.source_name)

    reconstructed =
      if (opts.cast_tensors and target) && match?(%Nx.Tensor{}, target.tensor) do
        Nx.as_type(reconstructed, Nx.type(target.tensor))
      else
        reconstructed
      end

    %{
      index: entry.index,
      path: entry.elixir_name,
      source_name: entry.source_name,
      artifact_key: entry.elixir_name,
      segments: if(target, do: target.segments, else: String.split(entry.elixir_name, ".")),
      tensor: reconstructed,
      shape: Nx.shape(reconstructed),
      type: inspect(Nx.type(reconstructed)),
      singular_values: Nx.size(offsets),
      offset_start: entry.offset_start,
      offset_end: entry.offset_end,
      component_keys: keys,
      python_v_layout: Atom.to_string(v_layout),
      python_component_zero_error: nil,
      target_verified: targets.verified? and not is_nil(target)
    }
  end

  defp normalize_v_layout!("torch_v"), do: :torch_v
  defp normalize_v_layout!("vh"), do: :vh
  defp normalize_v_layout!("nx"), do: :nx
  defp normalize_v_layout!(:torch_v), do: :torch_v
  defp normalize_v_layout!(:vh), do: :vh
  defp normalize_v_layout!(:nx), do: :nx

  defp normalize_v_layout!(other) do
    raise ArgumentError, "unsupported Python V layout #{inspect(other)}"
  end

  defp source_shape_from_components(u, v, :torch_v),
    do: {Nx.axis_size(u, 0), Nx.axis_size(v, 0)}

  defp source_shape_from_components(u, v, _layout),
    do: {Nx.axis_size(u, 0), Nx.axis_size(v, 1)}

  defp transfer_for_target(%Nx.Tensor{} = tensor, nil), do: tensor

  defp transfer_for_target(%Nx.Tensor{} = tensor, %{tensor: %Nx.Tensor{} = target}) do
    Nx.backend_transfer(tensor, backend_from_label(Runtime.tensor_backend(target)))
  end

  defp backend_from_label("EXLA.Backend<cuda" <> _), do: {EXLA.Backend, client: :cuda}
  defp backend_from_label("EXLA.Backend<host" <> _), do: {EXLA.Backend, client: :host}
  defp backend_from_label("Nx.BinaryBackend"), do: Nx.BinaryBackend
  defp backend_from_label(_), do: Nx.BinaryBackend

  defp component_keys(entry) do
    explicit = entry.component_tensors || %{}
    safe_key = entry.safe_key || sanitize_python_key(entry.source_name)

    %{
      u: component_key(explicit, "U", "u", "svd.U.#{safe_key}"),
      s: component_key(explicit, "S", "s", "svd.S.#{safe_key}"),
      v: component_key(explicit, "V", "v", "svd.V.#{safe_key}"),
      scale: entry.scale_tensor || "svf.scale_offsets.#{safe_key}"
    }
  end

  defp component_key(explicit, uppercase_key, lowercase_key, fallback) do
    deep_get(explicit, [uppercase_key]) || deep_get(explicit, [lowercase_key]) || fallback
  end

  defp fetch_tensor!(map, key, label) do
    case Map.get(map, key) do
      %Nx.Tensor{} = tensor ->
        tensor

      %{} = lazy_tensor ->
        Nx.with_default_backend(Nx.BinaryBackend, fn -> Nx.to_tensor(lazy_tensor) end)

      nil ->
        raise ArgumentError,
              "missing #{label} tensor #{inspect(key)}; available keys: #{inspect(Map.keys(map))}"
    end
  end

  defp target_shape(nil, source_shape), do: source_shape
  defp target_shape(%{tensor: %Nx.Tensor{} = tensor}, _source_shape), do: Nx.shape(tensor)

  defp orient_for_target!(tensor, nil, _path, _source_name), do: tensor

  defp orient_for_target!(tensor, target_shape, path, source_name) do
    cond do
      qwen_layer_linear_source?(source_name, path) and tuple_size(Nx.shape(tensor)) == 2 and
          Nx.shape(Nx.transpose(tensor)) == target_shape ->
        Nx.transpose(tensor)

      Nx.shape(tensor) == target_shape ->
        tensor

      tuple_size(Nx.shape(tensor)) == 2 and Nx.shape(Nx.transpose(tensor)) == target_shape ->
        Nx.transpose(tensor)

      true ->
        raise ArgumentError,
              "reconstructed tensor #{path} shape mismatch: got #{inspect(Nx.shape(tensor))}, target #{inspect(target_shape)}"
    end
  end

  defp qwen_layer_linear_source?(source_name, path)
       when is_binary(source_name) and is_binary(path) do
    String.starts_with?(source_name, "model.layers.") and
      String.ends_with?(source_name, ".weight") and
      String.ends_with?(path, ".kernel")
  end

  defp qwen_layer_linear_source?(_source_name, _path), do: false

  defp write_canonical_bundle(
         opts,
         python_manifest,
         reference_manifest,
         paths,
         head_weights,
         selected_tensors
       ) do
    emit(opts, %{event: :python_import_write_started, tensors: length(selected_tensors)})

    head_sha = write_router_head!(opts.out_dir, head_weights)

    singular_total = Enum.reduce(selected_tensors, 0, &(&1["singular_values"] + &2))
    now = DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()

    manifest = %{
      "artifact_version" => Artifact.manifest_version(),
      "status" => "complete",
      "created_at" => now,
      "updated_at" => now,
      "base_model_repo" => opts.spec.base_model_repo,
      "bumblebee_module" => inspect(opts.spec.bumblebee_module),
      "architecture" => Atom.to_string(opts.spec.architecture),
      "xla_target" => opts.spec.xla_target,
      "export_backend" => "python_semantic_importer_elixir_nx",
      "importer" => inspect(__MODULE__),
      "python_manifest_path" => opts.python_manifest_path,
      "reference_manifest_path" => opts.reference_manifest_path,
      "source_vector_path" =>
        deep_get(python_manifest, ["source_vector_path"]) ||
          deep_get(python_manifest, ["source", "router_vector"]) ||
          "python_semantic_bundle",
      "source_vector_tensor" =>
        deep_get(python_manifest, ["source_vector_tensor"]) || "python_semantic_components",
      "source_vector_shape" => source_vector_shape(python_manifest, singular_total, head_weights),
      "source_vector_sha256" =>
        deep_get(python_manifest, ["source_vector_sha256"]) ||
          deep_get(python_manifest, ["source", "router_vector_sha256"]) ||
          source_bundle_hash(paths),
      "scale_offset_count" => singular_total,
      "router_head_shape" => Tuple.to_list(Nx.shape(head_weights)),
      "router_head_artifact" => Artifact.router_head_file(),
      "router_head_tensor_key" => Artifact.router_head_tensor_key(),
      "router_head_sha256" => head_sha,
      "adapted_tensors_artifact" => Artifact.adapted_tensors_file(),
      "artifact_layout" => Artifact.artifact_layout_checkpoint_directory(),
      "selected_tensor_count" => length(selected_tensors),
      "selected_singular_value_count" => singular_total,
      "export_complete" => true,
      "partial_debug_only" => false,
      "selected_tensors" => selected_tensors,
      "source_split" => %{
        "scale_count" => singular_total,
        "hidden_size" => elem(Nx.shape(head_weights), 1),
        "output_count" => elem(Nx.shape(head_weights), 0)
      },
      "split" => %{
        "scale_count" => singular_total,
        "head_count" => Nx.size(head_weights)
      },
      "python_semantic_manifest" => compact_manifest_metadata(python_manifest),
      "python_reference_manifest_loaded" => is_map(reference_manifest)
    }

    Artifact.write_manifest!(opts.out_dir, manifest)
    emit(opts, %{event: :python_import_write_finished, out_dir: opts.out_dir})
    {:ok, manifest}
  end

  defp write_router_head!(out_dir, head_weights) do
    path = Path.join(out_dir, Artifact.router_head_file())
    tmp = path <> ".tmp"

    Safetensors.write!(tmp, %{
      Artifact.router_head_tensor_key() => Nx.backend_transfer(head_weights, Nx.BinaryBackend)
    })

    File.rename!(tmp, path)
    Artifact.file_sha256!(path)
  end

  defp write_checkpoint!(out_dir, result, cursor) do
    rel =
      Path.join(
        Artifact.checkpoint_directory_name(),
        checkpoint_file(result.index, result.path)
      )

    full = Path.join(out_dir, rel)
    tmp = full <> ".tmp"
    File.mkdir_p!(Path.dirname(full))

    host_tensor = Nx.backend_transfer(result.tensor, Nx.BinaryBackend)
    Safetensors.write!(tmp, %{result.artifact_key => host_tensor})
    File.rename!(tmp, full)
    sha = Artifact.file_sha256!(full)

    offset_start = result.offset_start || cursor
    offset_end = result.offset_end || offset_start + result.singular_values

    %{
      "index" => result.index,
      "path" => result.path,
      "source_name" => result.source_name,
      "artifact_key" => result.artifact_key,
      "segments" => result.segments,
      "shape" => Tuple.to_list(result.shape),
      "type" => result.type,
      "source_type" => result.type,
      "status" => "complete",
      "offset_start" => offset_start,
      "offset_end" => offset_end,
      "singular_values" => result.singular_values,
      "checkpoint_path" => rel,
      "checkpoint_sha256" => sha,
      "component_keys" => result.component_keys,
      "python_v_layout" => Map.get(result, :python_v_layout),
      "python_component_zero_error" => Map.get(result, :python_component_zero_error),
      "target_verified" => result.target_verified,
      "backend_observed_during_export" =>
        TrinityCoordinator.Runtime.tensor_backend(result.tensor),
      "adapted_backend" => TrinityCoordinator.Runtime.tensor_backend(result.tensor),
      "error" => nil
    }
  end

  defp checkpoint_file(index, path) do
    idx = Integer.to_string(index) |> String.pad_leading(@checkpoint_width, "0")
    safe_path = replace_non_safe_path_chars(path, "_")
    "#{idx}_#{safe_path}.safetensors"
  end

  defp source_vector_shape(python_manifest, singular_total, head_weights) do
    deep_get(python_manifest, ["source_vector_shape"]) ||
      [singular_total + Nx.size(head_weights)]
  end

  defp source_bundle_hash(paths) do
    joined = Enum.map_join(Map.values(paths), ":", &Artifact.file_sha256!/1)

    :crypto.hash(:sha256, joined)
    |> Base.encode16(case: :lower)
  end

  defp compact_manifest_metadata(manifest) when is_map(manifest) do
    manifest
    |> Map.take(["format", "version", "model", "routing", "source_vector_sha256"])
  end

  defp sanitize_python_key(source_name) do
    source_name
    |> String.replace("/", "__")
    |> replace_non_safe_path_chars("__")
  end

  defp replace_non_safe_path_chars(value, replacement) do
    value
    |> String.to_charlist()
    |> Enum.map(fn char ->
      if safe_path_char?(char), do: char, else: replacement
    end)
    |> IO.iodata_to_binary()
  end

  defp safe_path_char?(char) do
    (char >= ?0 and char <= ?9) or
      (char >= ?A and char <= ?Z) or
      (char >= ?a and char <= ?z) or
      char in [?-, ?_, ?.]
  end

  defp normalize_shape(nil), do: nil
  defp normalize_shape(shape) when is_tuple(shape), do: shape
  defp normalize_shape(shape) when is_list(shape), do: List.to_tuple(shape)
  defp normalize_shape(_), do: nil

  defp deep_get(nil, _path), do: nil
  defp deep_get(value, []), do: value

  defp deep_get(map, [key | rest]) when is_map(map) do
    value =
      case resolve_map_key(map, key) do
        nil -> nil
        resolved_key -> Map.get(map, resolved_key)
      end

    deep_get(value, rest)
  end

  defp deep_get(_value, _path), do: nil

  defp resolve_map_key(map, key) do
    cond do
      Map.has_key?(map, key) ->
        key

      Map.has_key?(map, to_string(key)) ->
        to_string(key)

      is_binary(key) ->
        Enum.find(Map.keys(map), fn
          atom when is_atom(atom) -> Atom.to_string(atom) == key
          _ -> false
        end)

      true ->
        nil
    end
  end

  defp emit(%{progress: progress}, event) when is_function(progress, 1), do: progress.(event)
  defp emit(_opts, _event), do: :ok
end
