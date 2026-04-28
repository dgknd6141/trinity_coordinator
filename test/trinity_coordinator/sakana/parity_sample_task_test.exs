defmodule Mix.Tasks.Trinity.Sakana.ParitySampleTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Sakana.ParitySample
  alias TrinityCoordinator.Sakana.ParityTrace

  test "semantic-only disables native SVD diagnostics" do
    opts =
      ParitySample.parse_args!([
        "--semantic-only",
        "--components-dir",
        "tmp/sakana_parity/python_components",
        "--stage-dir",
        "tmp/sakana_parity/elixir_stages"
      ])

    assert opts[:components_dir] == "tmp/sakana_parity/python_components"
    assert opts[:stage_dir] == "tmp/sakana_parity/elixir_stages"
    refute opts[:native?]
  end

  test "native SVD diagnostics stay enabled by default" do
    opts = ParitySample.parse_args!([])

    assert opts[:native?]
  end

  @tag :tmp_dir
  test "parity JSON preserves booleans and nulls", %{tmp_dir: tmp_dir} do
    path = Path.join(tmp_dir, "report.json")

    ParityTrace.write_json!(path, %{
      "functional_passed" => true,
      "byte_match" => false,
      "stage_tensor_file" => nil,
      "atom_label" => :torch_v
    })

    assert %{
             "functional_passed" => true,
             "byte_match" => false,
             "stage_tensor_file" => nil,
             "atom_label" => "torch_v"
           } = path |> File.read!() |> Jason.decode!()
  end
end
