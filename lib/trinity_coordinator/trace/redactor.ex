defmodule TrinityCoordinator.Trace.Redactor do
  @moduledoc """
  Minimal redaction helpers for trace payloads.
  """

  @sensitive_keys [
    "api_key",
    "authorization",
    "password",
    "secret",
    "token"
  ]

  @doc "Redacts sensitive keys recursively in maps and lists when mode is `:redacted`."
  def redact(value, :redacted), do: do_redact(value)
  def redact(value, _), do: value

  defp do_redact(value) when is_map(value) do
    Map.new(value, fn {k, v} ->
      key = to_key(k)

      if key in @sensitive_keys do
        {k, "<redacted>"}
      else
        {k, do_redact(v)}
      end
    end)
  end

  defp do_redact(value) when is_list(value) do
    Enum.map(value, &do_redact/1)
  end

  defp do_redact(value) when is_binary(value) do
    redacted =
      if String.contains?(String.downcase(value), "bearer") and String.contains?(value, " ") do
        "<redacted>"
      else
        value
      end

    redacted
  end

  defp do_redact(value), do: value

  defp to_key(key) when is_atom(key), do: key |> Atom.to_string() |> String.downcase()
  defp to_key(key) when is_binary(key), do: String.downcase(key)
  defp to_key(key), do: inspect(key)
end
