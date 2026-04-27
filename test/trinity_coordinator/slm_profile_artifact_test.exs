defmodule TrinityCoordinator.SLMProfileArtifactTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.SLMProfile

  test "adapted qwen profile patches backbone tensors without injecting routing head into SLM params" do
    profile = SLMProfile.qwen_sakana_adapted()

    assert profile.name == :qwen_sakana_adapted
    assert profile.adapted_artifact_dir != nil
    assert profile.artifact_patch_options[:patch_router_head] == false
    assert profile.expected_hidden_size == 1024
  end
end
