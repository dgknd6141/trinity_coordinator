defmodule TrinityCoordinator.RoleInjector do
  @moduledoc """
  Injects role-specific system prompts into the conversation transcript.

  The public functions accept role ids, atoms, and paper-style role names:

    * `0`, `:thinker`, `"Thinker"`
    * `1`, `:worker`, `"Worker"`
    * `2`, `:verifier`, `"Verifier"`
  """

  @role_names %{0 => "Thinker", 1 => "Worker", 2 => "Verifier"}

  @role_aliases %{
    "0" => "Thinker",
    "1" => "Worker",
    "2" => "Verifier",
    "t" => "Thinker",
    "thinker" => "Thinker",
    "v" => "Verifier",
    "verifier" => "Verifier",
    "w" => "Worker",
    "worker" => "Worker"
  }

  @roles %{
    "Thinker" =>
      "Analyze the current state and provide high-level guidance, plans, decompositions, or critiques. Do not present unchecked final answers unless the transcript already contains enough evidence.",
    "Worker" =>
      "Execute the next concrete step of the plan. Write code, math, derivations, calculations, or concrete answer content that advances the solution.",
    "Verifier" =>
      "Check the current solution for correctness, completeness, and responsiveness. Start your response with exactly ACCEPT or REVISE. After REVISE, include a concise diagnosis."
  }

  @doc """
  Prepends a system prompt to the list of messages based on the given role.
  """
  def inject_role(messages, role) when is_list(messages) do
    role = role_name(role)
    system_prompt = Map.get(@roles, role, "You are a helpful assistant.")
    [%{role: "system", content: system_prompt}] ++ messages
  end

  @doc """
  Returns the canonical role name for ids/atoms/strings.
  """
  def role_name(role_id) when is_integer(role_id),
    do: Map.get(@role_names, role_id, "UnknownRole")

  def role_name(role) when is_atom(role) do
    role
    |> Atom.to_string()
    |> role_name()
  end

  def role_name(role) when is_binary(role) do
    normalized = role |> String.trim() |> String.downcase()
    Map.get(@role_aliases, normalized, role)
  end

  def role_name(_), do: "UnknownRole"

  @doc """
  Returns a stable atom for the canonical role.
  """
  def role_atom(role) do
    case role_name(role) do
      "Thinker" -> :thinker
      "Worker" -> :worker
      "Verifier" -> :verifier
      _ -> :unknown
    end
  end

  @doc """
  Returns the role id for known roles.
  """
  def role_id(role) do
    case role_name(role) do
      "Thinker" -> 0
      "Worker" -> 1
      "Verifier" -> 2
      _ -> nil
    end
  end
end
