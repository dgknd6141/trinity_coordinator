defmodule TrinityCoordinator.SourcePolicyTest do
  use ExUnit.Case, async: true

  @text_extensions MapSet.new([
                     ".ex",
                     ".exs",
                     ".md",
                     ".sh",
                     ".py",
                     ".json",
                     ".lock",
                     ".svg"
                   ])
  @text_basenames MapSet.new([
                    ".formatter.exs",
                    ".gitignore",
                    ".tool-versions",
                    "CHANGELOG.md",
                    "LICENSE",
                    "README.md",
                    "mix.exs",
                    "mix.lock"
                  ])

  test "tracked text sources avoid dynamic atom and pattern-engine tokens" do
    hits =
      tracked_text_files()
      |> Enum.flat_map(fn path ->
        body = File.read!(path)

        forbidden_tokens()
        |> Enum.filter(&String.contains?(body, &1))
        |> Enum.map(&{path, &1})
        |> Kernel.++(dynamic_quoted_atom_hits(path, body))
      end)

    assert hits == []
  end

  test "dynamic quoted atom scanner catches prefixed interpolation" do
    dynamic =
      ":" <>
        "\"" <>
        "prefix_" <>
        "#" <>
        "{" <>
        "value" <>
        "}" <>
        "\""

    static = ":" <> "\"" <> "Elixir.Bumblebee.Text.Gpt2" <> "\""

    assert dynamic_quoted_atom_interpolation?(String.to_charlist(dynamic))
    refute dynamic_quoted_atom_interpolation?(String.to_charlist(static))
  end

  defp tracked_text_files do
    {output, 0} = System.cmd("git", ["ls-files"])

    output
    |> String.split("\n", trim: true)
    |> Enum.filter(fn path ->
      text_file?(path) and valid_text?(path)
    end)
  end

  defp text_file?(path) do
    MapSet.member?(@text_extensions, Path.extname(path)) or
      MapSet.member?(@text_basenames, Path.basename(path))
  end

  defp valid_text?(path) do
    case File.read(path) do
      {:ok, body} -> String.valid?(body)
      _ -> false
    end
  end

  defp forbidden_tokens do
    [
      "String.to_" <> "atom",
      "String.to_" <> "existing_atom",
      "binary_to_" <> "atom",
      "binary_to_" <> "existing_atom",
      "list_to_" <> "atom",
      "list_to_" <> "existing_atom",
      ":" <> "#" <> "{",
      Enum.join(["Module", ".concat"]),
      "Re" <> "gex",
      "re" <> "gex",
      "~" <> "r",
      ":" <> "re.",
      "String." <> "match",
      "Reg" <> "Exp",
      "reg" <> "exp",
      "re." <> "compile",
      "re." <> "search",
      "re." <> "match",
      "re." <> "fullmatch",
      "re." <> "sub",
      "re." <> "split",
      "re." <> "findall",
      "re." <> "finditer",
      "from " <> "re" <> " import",
      "import " <> "re"
    ]
  end

  defp dynamic_quoted_atom_hits(path, body) do
    if dynamic_quoted_atom_interpolation?(String.to_charlist(body)) do
      [{path, "dynamic quoted atom interpolation"}]
    else
      []
    end
  end

  defp dynamic_quoted_atom_interpolation?([?:, ?" | rest]) do
    quoted_atom_interpolates?(rest) or dynamic_quoted_atom_interpolation?(rest)
  end

  defp dynamic_quoted_atom_interpolation?([_char | rest]),
    do: dynamic_quoted_atom_interpolation?(rest)

  defp dynamic_quoted_atom_interpolation?([]), do: false

  defp quoted_atom_interpolates?([?" | _rest]), do: false
  defp quoted_atom_interpolates?([?#, ?{ | _rest]), do: true
  defp quoted_atom_interpolates?([?\\, _escaped | rest]), do: quoted_atom_interpolates?(rest)
  defp quoted_atom_interpolates?([_char | rest]), do: quoted_atom_interpolates?(rest)
  defp quoted_atom_interpolates?([]), do: false
end
