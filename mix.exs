defmodule TrinityCoordinator.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :trinity_coordinator,
      version: @version,
      elixir: "~> 1.18",
      description:
        "An Elixir implementation of the TRINITY multi-agent orchestration router for routing language-model calls through a compact hidden-state router.",
      source_url: "https://github.com/nshkrdotcom/trinity_coordinator",
      homepage_url: "https://github.com/nshkrdotcom/trinity_coordinator",
      start_permanent: Mix.env() == :prod,
      package: package(),
      docs: docs(),
      dialyzer: [
        plt_add_apps: [:mix],
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"}
      ],
      preferred_cli_env: [
        dialyzer: :dev,
        credo: :dev
      ],
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:bumblebee,
       github: "elixir-nx/bumblebee",
       ref: "0fd8114cf5429af9236f100f3350986e9d823c02",
       override: true},
      {:exla, "~> 0.9"},
      {:inference, github: "nshkrdotcom/inference", sparse: "apps/inference"},
      {:agent_session_manager, path: "../agent_session_manager"},
      {:gemini_cli_sdk, path: "../gemini_cli_sdk"},
      {:req, "~> 0.5"},
      {:credo, "~> 1.7", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: :dev, runtime: false},
      {:ex_doc, "~> 0.38", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      name: :trinity_coordinator,
      licenses: ["MIT"],
      maintainers: ["nshkrdotcom"],
      links: %{
        GitHub: "https://github.com/nshkrdotcom/trinity_coordinator"
      },
      files: [
        "lib",
        "mix.exs",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "assets",
        "examples",
        "guides",
        "docs"
      ]
    ]
  end

  defp docs do
    [
      main: "overview",
      source_ref: "v#{@version}",
      source_url: "https://github.com/nshkrdotcom/trinity_coordinator",
      extras: [
        {"README.md", [filename: "overview", title: "Overview"]},
        {"examples/README.md", [filename: "examples", title: "Examples"]},
        "guides/onboarding.md",
        "guides/current_direction.md",
        "guides/system_architecture.md",
        "guides/python_parity_reconstruction.md",
        "guides/stage_checks_and_tolerances.md",
        "guides/artifacts_and_export.md",
        "guides/svd_generation_runbook.md",
        "guides/service_buildout.md",
        "guides/provider_service_hardening.md",
        "guides/operations_qc.md",
        "guides/troubleshooting.md",
        "docs/sakana_svd_byte_match_rigor_plan.md",
        "docs/sakana_svd_parity_debug_checklist.md",
        "docs/elixir_svd_decomposition.md",
        "docs/production_qwen_slm_profile.md",
        "docs/coordination_head_variants.md",
        "docs/trace_persistence.md",
        "docs/configurable_provider_pools.md",
        "docs/provider_smoke_tests.md",
        "CHANGELOG.md",
        "LICENSE"
      ],
      groups_for_extras: [
        Project: ~r/^(README|CHANGELOG|LICENSE)/,
        Examples: ~r/examples\/README\.md/,
        "Start Here": ~r/guides\/(onboarding|current_direction|system_architecture)\.md/,
        "Parity Guides":
          ~r/guides\/(python_parity_reconstruction|stage_checks_and_tolerances|artifacts_and_export|svd_generation_runbook)\.md/,
        "Service Buildout":
          ~r/guides\/(service_buildout|provider_service_hardening|operations_qc|troubleshooting)\.md/,
        "Reference Notes": ~r/docs\//
      ],
      groups_for_modules: [
        Core: [
          TrinityCoordinator,
          TrinityCoordinator.Extractor,
          TrinityCoordinator.CoordinationHead,
          TrinityCoordinator.Orchestrator
        ],
        Runtime: [
          TrinityCoordinator.Runtime,
          TrinityCoordinator.StateManager,
          TrinityCoordinator.RoleInjector,
          TrinityCoordinator.Thinker,
          TrinityCoordinator.Verifier,
          TrinityCoordinator.AgentPool,
          TrinityCoordinator.AgentPool.Adapter,
          TrinityCoordinator.AgentPool.Inference,
          TrinityCoordinator.AgentPool.OpenAI
        ]
      ],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end
end
