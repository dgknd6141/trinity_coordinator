#!/usr/bin/env python3
"""Compare Python and Elixir Sakana SVD parity reports.

The comparator intentionally distinguishes the historical stored reference hash
from the current Python baseline hash.  If the current Python report cannot
reproduce the stored reference hash, exact Elixir equality against that stored
hash is not a meaningful failure signal; compare Elixir to the current Python
baseline or provide the original svd_weights.pt to the Python debug script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_report", type=Path)
    parser.add_argument("elixir_report", type=Path)
    parser.add_argument("--strict-reference", action="store_true",
                        help="Exit non-zero unless Python and Elixir both contain a variant matching the stored reference hash.")
    parser.add_argument("--strict-current-python", action="store_true",
                        help="Exit non-zero unless some Elixir variant matches the current Python baseline hash.")
    parser.add_argument("--strict-stage-tolerances", action="store_true",
                        help="Exit non-zero unless all required Elixir-vs-Python stage checks pass their declared tolerances.")
    parser.add_argument("--top-diffs", type=int, default=5,
                        help="Number of largest per-stage tensor differences to print when stage tensor files are available.")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_hashes(report: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in ["variants", "native_elixir_svd_variants", "semantic_python_component_variants"]:
        value = report.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "label" in item and "observed_bf16_sha256" in item:
                    hashes[str(item["label"])] = str(item["observed_bf16_sha256"])
    return hashes


def collect_zero_errors(report: dict[str, Any]) -> dict[str, Any]:
    errors: dict[str, Any] = {}
    for key in ["variants", "native_elixir_svd_variants", "semantic_python_component_variants"]:
        value = report.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "label" in item and "zero_offset_max_abs_error_vs_source" in item:
                    errors[str(item["label"])] = item["zero_offset_max_abs_error_vs_source"]
    return errors


def reference_hash(report: dict[str, Any]) -> str | None:
    return (
        report.get("reference", {}).get("expected_bf16_sha256")
        or report.get("reference", {}).get("expected_bf16_sha256")
    )


def current_python_baseline(py: dict[str, Any], py_hashes: dict[str, str]) -> tuple[str | None, str | None]:
    ref = py.get("reference", {})
    label = ref.get("current_python_baseline_label")
    digest = ref.get("current_python_baseline_bf16_sha256")
    if label and digest:
        return str(label), str(digest)
    if py_hashes:
        label, digest = next(iter(py_hashes.items()))
        return label, digest
    return None, None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def collect_stage_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for variant in report.get("semantic_python_component_variants", []) or []:
        if not isinstance(variant, dict):
            continue
        stage_debug = variant.get("stage_debug")
        if not isinstance(stage_debug, dict):
            continue
        for check in stage_debug.get("checks", []) or []:
            if isinstance(check, dict):
                check = dict(check)
                check["variant_label"] = variant.get("label")
                checks.append(check)
    return checks


def first_stage_file(report: dict[str, Any], *, preferred_label_contains: str | None = None) -> str | None:
    if preferred_label_contains is None:
        stage = report.get("stage_debug", {})
        if isinstance(stage, dict) and stage.get("stage_tensor_file"):
            return str(stage["stage_tensor_file"])
        inputs = report.get("inputs", {})
        if isinstance(inputs, dict) and inputs.get("stage_tensor_file"):
            return str(inputs["stage_tensor_file"])

    for variant in report.get("semantic_python_component_variants", []) or []:
        if not isinstance(variant, dict):
            continue
        label = str(variant.get("label", ""))
        if preferred_label_contains and preferred_label_contains not in label:
            continue
        stage_debug = variant.get("stage_debug", {})
        if isinstance(stage_debug, dict) and stage_debug.get("stage_tensor_file"):
            return str(stage_debug["stage_tensor_file"])
    return None


def print_stage_checks(checks: list[dict[str, Any]]) -> bool:
    if not checks:
        print("\nStage checks: (none)")
        return False

    print("\nStage checks against Python stage tensors:")
    all_required_passed = True
    for check in checks:
        required = boolish(check.get("required_for_functional_parity"))
        passed = boolish(check.get("functional_passed"))
        if required and not passed:
            all_required_passed = False
        print(
            "  "
            f"{check.get('variant_label')} {check.get('stage')}: "
            f"required={required} functional_passed={passed} "
            f"byte_match={check.get('byte_match')} shape_match={check.get('shape_match')} "
            f"max_abs={check.get('max_abs_error')} mean_abs={check.get('mean_abs_error')} "
            f"mismatches={check.get('mismatched_element_count')} "
            f"tol={check.get('tolerance')}"
        )
    return all_required_passed


def print_top_diffs(py_stage_file: str | None, ex_stage_file: str | None, count: int) -> None:
    if not py_stage_file or not ex_stage_file or count <= 0:
        return

    try:
        import torch
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"\nStage tensor top-diff details unavailable: {exc}")
        return

    py = load_file(py_stage_file, device="cpu")
    ex = load_file(ex_stage_file, device="cpu")
    stages = [
        "stage.scaled_s",
        "stage.u_scaled",
        "stage.matmul_pre_norm",
        "stage.adapted_source_f32",
        "stage.final_f32",
        "stage.final_bf16",
        "stage.zero_source_f32",
    ]

    print("\nTop tensor differences from stage bundles:")
    print(f"  Python stage file: {py_stage_file}")
    print(f"  Elixir stage file: {ex_stage_file}")
    for stage in stages:
        if stage not in py or stage not in ex:
            continue
        left = py[stage].to(torch.float32).reshape(-1)
        right = ex[stage].to(torch.float32).reshape(-1)
        diff = (right - left).abs()
        if diff.numel() == 0:
            continue
        k = min(count, diff.numel())
        values, indices = torch.topk(diff, k)
        print(f"  {stage}:")
        for value, index in zip(values.tolist(), indices.tolist()):
            print(
                "    "
                f"flat_index={index} abs_diff={float(value)} "
                f"python={float(left[index])} elixir={float(right[index])}"
            )


def main() -> None:
    args = parse_args()
    py = load(args.python_report)
    ex = load(args.elixir_report)
    expected = reference_hash(py) or reference_hash(ex)
    py_hashes = collect_hashes(py)
    ex_hashes = collect_hashes(ex)
    py_errors = collect_zero_errors(py)
    ex_errors = collect_zero_errors(ex)
    baseline_label, baseline_digest = current_python_baseline(py, py_hashes)
    reproducible = boolish(py.get("reference", {}).get("expected_hash_reproducible"))

    print(f"stored reference expected: {expected}")
    print(f"python reference hash reproducible: {reproducible}")
    print(f"current Python baseline: {baseline_label} {baseline_digest}")
    if not reproducible:
        print("note: current Python SVD did not reproduce the stored manifest hash; exact comparison to the stored hash is provenance-sensitive.")
        print("      Use --svd-weights with the original svd_weights.pt if strict historical reproduction is required.")

    print("\nPython variants:")
    for label, digest in py_hashes.items():
        print(f"  {label}: {digest} match_stored={digest == expected} zero_error={py_errors.get(label)}")

    print("\nElixir variants:")
    for label, digest in ex_hashes.items():
        print(
            f"  {label}: {digest} "
            f"match_stored={digest == expected} "
            f"match_current_python={digest == baseline_digest} "
            f"zero_error={ex_errors.get(label)}"
        )

    print("\nCross-report identical hashes:")
    any_match = False
    for py_label, py_digest in py_hashes.items():
        for ex_label, ex_digest in ex_hashes.items():
            if py_digest == ex_digest:
                any_match = True
                print(f"  {py_label} == {ex_label}: {py_digest}")
    if not any_match:
        print("  (none)")

    stage_checks = collect_stage_checks(ex)
    stage_ok = print_stage_checks(stage_checks)
    print_top_diffs(
        first_stage_file(py),
        first_stage_file(ex, preferred_label_contains="host_binary_v_layout_torch_v"),
        args.top_diffs,
    )

    if args.strict_reference:
        py_ok = any(digest == expected for digest in py_hashes.values())
        ex_ok = any(digest == expected for digest in ex_hashes.values())
        if not (py_ok and ex_ok):
            raise SystemExit("strict stored-reference comparison failed")

    if args.strict_current_python:
        if not baseline_digest or not any(digest == baseline_digest for digest in ex_hashes.values()):
            raise SystemExit("strict current-Python comparison failed")

    if args.strict_stage_tolerances and not stage_ok:
        raise SystemExit("strict stage-tolerance comparison failed")


if __name__ == "__main__":
    main()
