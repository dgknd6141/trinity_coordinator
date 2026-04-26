defmodule TrinityCoordinator.SLMProfileTest do
  use ExUnit.Case

  alias TrinityCoordinator.{Runtime, SLMProfile}

  test "tiny profile exposes runtime-ready metadata" do
    profile = SLMProfile.tiny_gpt2()

    assert profile.name == :tiny_gpt2
    assert profile.status == :ready
    assert match?({:hf, _}, profile.repo)
    assert profile.module == Bumblebee.Text.Gpt2
    assert profile.architecture == :base
    assert is_integer(profile.expected_hidden_size) and profile.expected_hidden_size > 0
  end

  test "qwen profile is explicitly tagged for production intent" do
    profile = SLMProfile.qwen_coordinator()

    assert profile.name == :qwen_coordinator
    assert profile.status in [:pending, :unsupported]
    assert profile.expected_hidden_size == 1024
    assert is_tuple(profile.repo)
    assert profile.module == nil
  end

  test "compatibility_probe reports supported modules for ready profiles" do
    {:ok, probe} = SLMProfile.compatibility_probe(:tiny_gpt2)

    assert probe.status == :compatible
    assert :"Elixir.Bumblebee.Text.Gpt2" in probe.supported_text_modules
  end

  @tag :qwen
  test "compatibility_probe reports explicit reason when profile remains unsupported" do
    {:ok, probe} = SLMProfile.compatibility_probe(:qwen_coordinator)

    assert match?({:incompatible, {:unsupported_profile_status, :pending}}, probe.status)
  end

  @tag :qwen
  test "load_profile is explicit when qwen remains unsupported" do
    assert {:error,
            {:unsupported_profile, :qwen_coordinator, {:unsupported_profile_status, :pending}}} =
             SLMProfile.load_profile(SLMProfile.qwen_coordinator())
  end

  @tag :integration
  test "loads tiny profile through load_profile/1" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:tiny_gpt2)
    assert is_map(model_info)
    assert tokenizer != nil
  end

  test "load_profile resolves only known profile names" do
    assert {:error, {:unknown_profile, :does_not_exist}} =
             SLMProfile.load_profile(:does_not_exist)
  end

  test "load_profile validates malformed profiles" do
    assert {:error, :invalid_profile} = SLMProfile.load_profile(%{name: :unknown})
  end
end
