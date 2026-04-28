defmodule Mix.Tasks.Trinity.Sakana.ParitySampleTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Sakana.ParitySample

  test "semantic-only disables native SVD diagnostics" do
    opts =
      ParitySample.parse_args!([
        "--semantic-only",
        "--components-dir",
        "tmp/sakana_parity/python_components"
      ])

    assert opts[:components_dir] == "tmp/sakana_parity/python_components"
    refute opts[:native?]
  end

  test "native SVD diagnostics stay enabled by default" do
    opts = ParitySample.parse_args!([])

    assert opts[:native?]
  end
end
