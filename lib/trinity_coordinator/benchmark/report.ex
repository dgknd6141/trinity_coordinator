defmodule TrinityCoordinator.Benchmark.Report do
  @moduledoc """
  Report writers for benchmark suites.
  """

  @type metric_report :: map()
  @type report :: map()

  @doc """
  Builds a common report envelope used by all suites.
  """
  @spec envelope(atom(), map(), keyword()) :: report()
  def envelope(suite, payload, opts \\ []) do
    %{
      schema_version: 1,
      suite: suite,
      generated_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      git_commit: git_ref(),
      profile: Keyword.get(opts, :profile, "unknown"),
      head: Keyword.get(opts, :head, :linear),
      head_opts: Keyword.get(opts, :head_opts, %{}),
      xla_target: Keyword.get(opts, :xla_target, ""),
      dataset_id: Keyword.get(opts, :dataset_id, ""),
      dataset_hash: Keyword.get(opts, :dataset_hash),
      platform: Keyword.get(opts, :platform, nil),
      dependencies: dependency_versions(),
      payload: payload
    }
  end

  @doc """
  Writes a report to a JSON file.
  """
  @spec write_json(String.t(), report()) :: :ok | {:error, term()}
  def write_json(path, report) when is_binary(path) do
    ensure_directory(path)
    encoded = Jason.encode!(normalize_for_json(report))
    File.write(path, encoded)
  end

  @doc """
  Writes a short human-readable summary report.
  """
  @spec write_markdown(String.t(), report()) :: :ok | {:error, term()}
  def write_markdown(path, report) when is_binary(path) do
    ensure_directory(path)

    lines = [
      "# TRINITY Benchmark Report",
      "Generated: #{report.generated_at}",
      "Suite: #{report.suite}",
      "Profile: #{report.profile}",
      "Head: #{inspect(report.head)}",
      "XLA Target: #{report.xla_target}",
      "",
      "## Summary",
      Jason.encode!(normalize_for_json(report.payload))
    ]

    File.write(path, Enum.join(lines, "\n"))
  end

  @doc """
  Convenience wrapper to write report in the format inferred from path extension.
  """
  @spec write(String.t(), report()) :: :ok | {:error, term()}
  def write(path, report) do
    case Path.extname(path) do
      ".md" -> write_markdown(path, report)
      ".jsonl" -> write_json(path, report)
      _ -> write_json(path, report)
    end
  end

  defp ensure_directory(path) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()
  end

  defp git_ref do
    case System.cmd("git", ["rev-parse", "--short", "HEAD"], stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      _ -> "unknown"
    end
  end

  defp dependency_versions do
    %{
      exla: Application.spec(:exla, :vsn) |> to_string_safe(),
      nx: Application.spec(:nx, :vsn) |> to_string_safe(),
      axon: Application.spec(:axon, :vsn) |> to_string_safe(),
      bumblebee: Application.spec(:bumblebee, :vsn) |> to_string_safe()
    }
  end

  defp to_string_safe(nil), do: "unknown"
  defp to_string_safe(value), do: to_string(value)

  defp normalize_for_json(report) when is_map(report) do
    report
    |> Enum.reduce(%{}, fn {key, value}, acc ->
      Map.put(acc, key, normalize_for_json(value))
    end)
  end

  defp normalize_for_json(value) when is_list(value) do
    if keyword_list?(value) do
      Enum.reduce(value, %{}, fn {key, val}, acc ->
        Map.put(acc, key, normalize_for_json(val))
      end)
    else
      Enum.map(value, &normalize_for_json/1)
    end
  end

  defp normalize_for_json(value) when is_tuple(value) do
    value
    |> Tuple.to_list()
    |> Enum.map(&normalize_for_json/1)
  end

  defp normalize_for_json(value), do: value

  defp keyword_list?([_ | _] = list) do
    Enum.all?(list, fn
      {key, _value} when is_atom(key) or is_binary(key) -> true
      _ -> false
    end)
  end

  defp keyword_list?([]), do: false
end
