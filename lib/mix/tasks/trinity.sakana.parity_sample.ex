defmodule Mix.Tasks.Trinity.Sakana.ParitySample do
  @moduledoc """
  Emits an incremental JSON report for the Sakana Python-reference SVD sample.

      XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
        --out tmp/sakana_parity/elixir_sample_trace.json

  To compare against Python semantic components, first run the companion Python
  script and pass the directory/report it writes:

      python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
        --out tmp/sakana_parity/python_sample_trace.json \
        --write-components-dir tmp/sakana_parity/python_components

      XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
        --semantic-only \
        --components-dir tmp/sakana_parity/python_components \
        --python-report tmp/sakana_parity/python_sample_trace.json \
        --stage-dir tmp/sakana_parity/elixir_stages \
        --out tmp/sakana_parity/elixir_sample_trace.json

  Pass `--semantic-only` while debugging Python-component parity. It skips the
  native `Nx.LinAlg.svd/2` diagnostics and avoids the expensive CUDA SVD
  compilation path. Pass `--stage-dir` to write Elixir stage tensors that can be
  compared side-by-side with Python's stage tensor bundle. Pass
  `--host-semantic-only` with `--semantic-only` for the fastest functional
  parity loop; it skips the optional CUDA semantic replay once host stage checks
  are sufficient. Add `--source-from-python-stage` when a Python report with
  stage tensors is supplied to avoid loading Qwen just to retrieve the source
  tensor for semantic-only diagnostics. Add `--preferred-layout-only` to skip
  known-wrong V-layout diagnostics, or `--device-semantic-only` to avoid the
  large CPU matrix multiply while still producing stage checks through EXLA.
  """

  use Mix.Task

  alias TrinityCoordinator.Sakana.ParityTrace

  @shortdoc "Emits Sakana SVD sample parity diagnostics"
  @default_out "tmp/sakana_parity/elixir_sample_trace.json"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    opts = parse_args!(args)

    report =
      ParityTrace.sample_report!(
        router_vector_path:
          Keyword.get(
            opts,
            :router_vector,
            "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"
          ),
        reference_manifest_path:
          Keyword.get(
            opts,
            :reference,
            "priv/sakana_trinity/reference/sakana_python_reference_manifest.json"
          ),
        components_dir: Keyword.get(opts, :components_dir),
        python_report_path: Keyword.get(opts, :python_report),
        stage_dir: Keyword.get(opts, :stage_dir),
        native?: Keyword.fetch!(opts, :native?),
        semantic_host?: not Keyword.get(opts, :device_semantic_only, false),
        semantic_device?:
          not Keyword.get(opts, :host_semantic_only, false) or
            Keyword.get(opts, :device_semantic_only, false),
        semantic_layout_diagnostics?: not Keyword.get(opts, :preferred_layout_only, false),
        source_from_python_stage?: Keyword.get(opts, :source_from_python_stage, false),
        require_cuda: not Keyword.get(opts, :no_cuda, false)
      )

    out = Keyword.get(opts, :out, @default_out)
    :ok = ParityTrace.write_json!(out, report)

    Mix.shell().info("Wrote Elixir parity report: #{out}")
    print_hash_summary(report)
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
          router_vector: :string,
          reference: :string,
          no_cuda: :boolean,
          semantic_only: :boolean,
          host_semantic_only: :boolean,
          device_semantic_only: :boolean,
          preferred_layout_only: :boolean,
          source_from_python_stage: :boolean,
          skip_native_svd: :boolean
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    native? =
      not (Keyword.get(opts, :semantic_only, false) or Keyword.get(opts, :skip_native_svd, false))

    Keyword.put(opts, :native?, native?)
  end

  defp print_hash_summary(report) do
    expected = get_in(report, ["reference", "expected_bf16_sha256"])
    python_current = get_in(report, ["python_current_baseline", "observed_bf16_sha256"])

    python_reproducible =
      get_in(report, ["python_current_baseline", "expected_hash_reproducible"])

    Mix.shell().info("Stored Python bf16 hash: #{expected}")

    if python_current do
      Mix.shell().info(
        "Current Python baseline hash: #{python_current} reproducible_stored=#{inspect(python_reproducible)}"
      )
    else
      Mix.shell().info("Current Python baseline hash: (no --python-report supplied)")
    end

    case Map.get(report, "native_elixir_svd_variants", []) do
      [] ->
        Mix.shell().info("Native Elixir SVD variants: skipped")

      variants ->
        Enum.each(variants, fn variant ->
          Mix.shell().info(
            "native #{variant["label"]}: #{variant["observed_bf16_sha256"]} match_stored=#{variant["matches_expected"]} match_python=#{variant["matches_python_current"]} zero_error=#{variant["zero_offset_max_abs_error_vs_source"]}"
          )
        end)
    end

    case Map.get(report, "semantic_python_component_variants") do
      variants when is_list(variants) ->
        Enum.each(variants, fn variant ->
          Mix.shell().info(
            "semantic #{variant["label"]}: #{variant["observed_bf16_sha256"]} match_stored=#{variant["matches_expected"]} match_python=#{variant["matches_python_current"]} zero_error=#{variant["zero_offset_max_abs_error_vs_source"]}"
          )

          print_stage_summary(variant)
        end)

      nil ->
        Mix.shell().info("No Python semantic component directory was supplied.")

      other ->
        Mix.shell().info("Semantic component status: #{inspect(other)}")
    end
  end

  defp print_stage_summary(%{"stage_debug" => %{"checks" => checks}} = variant)
       when is_list(checks) and checks != [] do
    required_failed =
      Enum.filter(checks, fn check ->
        check["required_for_functional_parity"] and not check["functional_passed"]
      end)

    first_non_byte_match =
      Enum.find(checks, fn check ->
        check["shape_match"] and not check["byte_match"]
      end)

    Mix.shell().info(
      "  stage_checks=#{length(checks)} functional_parity=#{inspect(required_failed == [])} first_non_byte_match=#{stage_label(first_non_byte_match)}"
    )

    Enum.each(required_failed, fn check ->
      Mix.shell().info(
        "  required_stage_failed #{check["stage"]}: max_abs=#{check["max_abs_error"]} mean_abs=#{check["mean_abs_error"]} tolerance=#{inspect(check["tolerance"])}"
      )
    end)

    stage_file = get_in(variant, ["stage_debug", "stage_tensor_file"])

    if stage_file do
      Mix.shell().info("  stage_tensor_file=#{stage_file}")
    end
  end

  defp print_stage_summary(_variant), do: :ok

  defp stage_label(nil), do: "(none)"
  defp stage_label(check), do: check["stage"]
end
