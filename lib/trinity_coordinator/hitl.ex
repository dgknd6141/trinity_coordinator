defmodule TrinityCoordinator.HITL do
  @moduledoc """
  Shared helpers for human-in-the-loop CLI gates.

  The mix tasks under `Mix.Tasks.Trinity.Hitl.*` print explicit, grep-friendly
  evidence that a specific foundational milestone is working.
  """

  alias TrinityCoordinator.Runtime

  def banner(title) do
    IO.puts("")
    IO.puts("=== #{title} ===")
  end

  def kv(label, value) do
    IO.puts("#{label}: #{format_value(value)}")
  end

  def pass(label) do
    IO.puts("=== #{label}: PASS ===")
  end

  def fail!(label, reason) do
    raise "#{label}: #{inspect(reason)}"
  end

  def assert!(true, _reason), do: :ok
  def assert!(false, reason), do: raise(inspect(reason))

  def require_cuda! do
    platforms = Runtime.require_cuda!()
    kv("EXLA supported platforms", Map.keys(platforms))
    platforms
  end

  def ensure_cuda_tensor!(tensor, label) do
    backend = Runtime.tensor_backend(tensor)
    kv("#{label} backend", backend)

    unless String.contains?(backend, "EXLA.Backend<cuda:") do
      raise "#{label} is not on CUDA: #{backend}"
    end

    :ok
  end

  def ensure_shape!(tensor_or_shape, expected, label) do
    shape =
      case tensor_or_shape do
        %Nx.Tensor{} = tensor -> Nx.shape(tensor)
        shape when is_tuple(shape) -> shape
      end

    kv("#{label} shape", shape)

    unless shape == expected do
      raise "#{label} shape mismatch: expected #{inspect(expected)}, got #{inspect(shape)}"
    end

    :ok
  end

  def role_name(0), do: "Thinker"
  def role_name(1), do: "Worker"
  def role_name(2), do: "Verifier"
  def role_name(_), do: "Unknown"

  def short_logits(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(fn value ->
      value
      |> Kernel./(1.0)
      |> Float.round(4)
    end)
  end

  def parse_bool_flag(args, flag) when is_list(args) and is_binary(flag) do
    flag in args
  end

  defp format_value(value) when is_binary(value), do: value
  defp format_value(value), do: inspect(value)
end
