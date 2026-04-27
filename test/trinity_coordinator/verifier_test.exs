defmodule TrinityCoordinator.VerifierTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Verifier

  test "parses accepted verifier output" do
    parsed = Verifier.parse("ACCEPT: final answer is correct")

    assert parsed.status == :accepted
    assert parsed.token == "ACCEPT"
    assert parsed.diagnosis == "final answer is correct"
  end

  test "parses revised verifier output" do
    parsed = Verifier.parse(" revise - arithmetic error in step 2 ")

    assert parsed.status == :revised
    assert parsed.token == "REVISE"
    assert parsed.diagnosis == "arithmetic error in step 2"
  end

  test "preserves unknown verifier text" do
    parsed = Verifier.parse("Looks plausible, but I am not sure.")

    assert parsed.status == :unknown
    assert parsed.diagnosis == "Looks plausible, but I am not sure."
  end

  test "accepted? requires verifier role" do
    assert Verifier.accepted?("Verifier", "ACCEPT")
    assert Verifier.accepted?(:verifier, "accept: ok")
    refute Verifier.accepted?("Worker", "ACCEPT")
    refute Verifier.accepted?(:thinker, "ACCEPT")
  end

  test "custom stop token is supported" do
    parsed = Verifier.parse("DONE: verified", stop_token: "DONE")

    assert parsed.status == :accepted
    assert parsed.token == "DONE"
    assert parsed.diagnosis == "verified"
  end
end
