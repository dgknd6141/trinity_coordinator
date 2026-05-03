defmodule TrinityCoordinator.Thinker do
  @moduledoc """
  Parser for TRINITY thinker responses.

  The supplemental Python implementation allows a thinker turn to steer the
  next role by returning both:

      <suggestion>...</suggestion>
      <suggested_role>solver</suggested_role>

  Only `solver` and `verifier` are accepted suggested roles. The public Elixir
  runtime maps Python `solver` to `Worker`.
  """

  alias TrinityCoordinator.RoleInjector

  @enforce_keys [:raw]
  defstruct [:raw, :suggestion, :suggested_role, :suggested_role_id]

  @type t :: %__MODULE__{
          raw: String.t(),
          suggestion: String.t() | nil,
          suggested_role: String.t() | nil,
          suggested_role_id: non_neg_integer() | nil
        }

  @doc """
  Parses a thinker response and returns a suggested next role only when both
  required tags are present and the suggested role is valid.
  """
  @spec parse(String.t()) :: t()
  def parse(text) when is_binary(text) do
    suggestion = extract_tag(text, "suggestion")
    role = text |> extract_tag("suggested_role") |> normalize_suggested_role()

    if suggestion && role do
      %__MODULE__{
        raw: text,
        suggestion: suggestion,
        suggested_role: role,
        suggested_role_id: RoleInjector.role_id(role)
      }
    else
      %__MODULE__{raw: text, suggestion: nil, suggested_role: nil, suggested_role_id: nil}
    end
  end

  def parse(other), do: parse(to_string(other))

  defp extract_tag(text, tag) do
    open = "<" <> tag <> ">"
    close = "</" <> tag <> ">"

    case String.split(text, open, parts: 2) do
      [_before, rest] ->
        case String.split(rest, close, parts: 2) do
          [value, _after] ->
            value
            |> String.trim()
            |> blank_to_nil()

          _ ->
            nil
        end

      _ ->
        nil
    end
  end

  defp normalize_suggested_role(nil), do: nil

  defp normalize_suggested_role(role) do
    case role |> String.trim() |> String.downcase() do
      "solver" -> "Worker"
      "verifier" -> "Verifier"
      _ -> nil
    end
  end

  defp blank_to_nil(""), do: nil
  defp blank_to_nil(value), do: value
end
