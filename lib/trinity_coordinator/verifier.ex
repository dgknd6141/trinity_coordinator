defmodule TrinityCoordinator.Verifier do
  @moduledoc """
  Parser for the TRINITY verifier contract.

  The paper-level verifier contract is:

      ACCEPT | REVISE

  with an optional diagnosis after the status token.  This module keeps that
  contract explicit so orchestration, training, benchmarks, and traces do not
  have to rely on ad-hoc string-prefix checks.
  """

  @enforce_keys [:status, :raw]
  defstruct [:status, :diagnosis, :raw, :token]

  @type status :: :accepted | :revised | :unknown

  @type t :: %__MODULE__{
          status: status(),
          diagnosis: String.t() | nil,
          raw: String.t(),
          token: String.t() | nil
        }

  @default_accept_token "ACCEPT"
  @default_revise_token "REVISE"

  @doc """
  Parses a verifier response into an explicit status and diagnosis.

  Accepted examples:

      ACCEPT
      ACCEPT: final answer is correct
      revise - arithmetic error in step 2
      REVISE
      REVISE: missing edge case

  Unknown text is preserved as `diagnosis` with status `:unknown`.
  """
  @spec parse(String.t(), keyword()) :: t()
  def parse(text, opts \\ [])

  def parse(text, opts) when is_binary(text) and is_list(opts) do
    accept_token =
      opts
      |> Keyword.get(:accept_token, Keyword.get(opts, :stop_token, @default_accept_token))
      |> normalize_token()

    revise_token =
      opts
      |> Keyword.get(:revise_token, @default_revise_token)
      |> normalize_token()

    raw = String.trim(text)
    normalized = String.upcase(raw)

    cond do
      token_prefix?(normalized, accept_token) ->
        %__MODULE__{
          status: :accepted,
          raw: raw,
          token: accept_token,
          diagnosis: diagnosis_after_token(raw, accept_token)
        }

      token_prefix?(normalized, revise_token) ->
        %__MODULE__{
          status: :revised,
          raw: raw,
          token: revise_token,
          diagnosis: diagnosis_after_token(raw, revise_token)
        }

      true ->
        %__MODULE__{status: :unknown, raw: raw, token: nil, diagnosis: blank_to_nil(raw)}
    end
  end

  def parse(other, opts) when is_list(opts) do
    parse(to_string(other), opts)
  end

  @doc """
  Returns true only when the selected role is Verifier and the parsed status is accepted.
  """
  @spec accepted?(String.t() | atom(), String.t(), keyword()) :: boolean()
  def accepted?(role, text, opts \\ [])

  def accepted?(role, text, opts) when is_binary(text) and is_list(opts) do
    verifier_role?(role) and parse(text, opts).status == :accepted
  end

  def accepted?(_role, _text, _opts), do: false

  @doc """
  Returns true when the role value denotes the Verifier role.
  """
  @spec verifier_role?(String.t() | atom()) :: boolean()
  def verifier_role?(role) when is_atom(role) do
    role
    |> Atom.to_string()
    |> verifier_role?()
  end

  def verifier_role?(role) when is_binary(role) do
    role
    |> String.trim()
    |> String.downcase()
    |> case do
      "verifier" -> true
      "v" -> true
      _ -> false
    end
  end

  def verifier_role?(_), do: false

  @doc """
  Converts parsed status into the status atoms used by trace events.
  """
  @spec trace_status(t()) :: :accepted | :revised | :unknown
  def trace_status(%__MODULE__{status: status}), do: status

  defp normalize_token(token) when is_binary(token) do
    token
    |> String.trim()
    |> String.upcase()
  end

  defp normalize_token(token), do: token |> to_string() |> normalize_token()

  defp token_prefix?(_text, ""), do: false

  defp token_prefix?(text, token) do
    text == token or
      String.starts_with?(text, token <> ":") or
      String.starts_with?(text, token <> " ") or
      String.starts_with?(text, token <> "-") or
      String.starts_with?(text, token <> "—") or
      String.starts_with?(text, token <> "\n") or
      String.starts_with?(text, token <> "\r\n")
  end

  defp diagnosis_after_token(raw, token) do
    token_length = byte_size(token)

    raw
    |> binary_part_safe(token_length)
    |> clean_leading_status_punctuation()
    |> blank_to_nil()
  end

  defp binary_part_safe(raw, skip) when byte_size(raw) <= skip, do: ""

  defp binary_part_safe(raw, skip) do
    binary_part(raw, skip, byte_size(raw) - skip)
  end

  defp clean_leading_status_punctuation(text) do
    text
    |> String.trim_leading()
    |> trim_many([":", "-", "—"])
    |> String.trim()
  end

  defp trim_many(text, []), do: text

  defp trim_many(text, [prefix | rest]) do
    text
    |> String.trim_leading(prefix)
    |> String.trim_leading()
    |> trim_many(rest)
  end

  defp blank_to_nil(text) do
    if String.trim(text) == "", do: nil, else: String.trim(text)
  end
end
