#!/usr/bin/env python3
"""Build a manifest for chunked Sakana large-tensor replay.

This script does not regenerate adapted tensors or re-run SVD. It points the
Elixir chunk replay task at the canonical Python component bundle and the
canonical all-selected Python stage tensor bundle, then enumerates bounded row
chunks for the embedding and LM-head tensors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

DEFAULT_COMPONENT_DIR = Path("tmp/sakana_parity/original_submission_svd/python_components")
DEFAULT_PYTHON_REPORT = Path("tmp/sakana_parity/original_submission_svd/python_sample_trace.json")
DEFAULT_OUT = Path("tmp/sakana_parity/large_tensor_chunks/python_large_tensor_chunks.json")
DEFAULT_SOURCES = ["model.embed_tokens.weight", "lm_head.weight"]
ALL_SELECTED_STAGE_FILE = "trinity_svf_all_selected_stage_debug.safetensors"
METADATA_FILE = "trinity_svf_debug_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--python-report", type=Path, default=DEFAULT_PYTHON_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage-tensor-file", type=Path, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--source", action="append", dest="sources", default=None,
                        help="Source tensor to include. May be repeated.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the pure manifest-planning self-test and exit.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def chunks(row_count: int, chunk_rows: int) -> list[dict[str, int]]:
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if row_count <= 0:
        raise ValueError(f"row_count must be positive, got {row_count}")

    result: list[dict[str, int]] = []
    row_start = 0
    while row_start < row_count:
        row_end = min(row_start + chunk_rows, row_count)
        result.append({
            "chunk_index": len(result),
            "row_start": row_start,
            "row_end": row_end,
            "row_count": row_end - row_start,
        })
        row_start = row_end
    return result


def all_selected_stage_file(args: argparse.Namespace, python_report: dict[str, Any]) -> Path:
    if args.stage_tensor_file is not None:
        return args.stage_tensor_file

    stage_debug = python_report.get("stage_debug", {})
    if isinstance(stage_debug, dict) and stage_debug.get("all_selected_stage_tensor_file"):
        return Path(stage_debug["all_selected_stage_tensor_file"])

    inputs = python_report.get("inputs", {})
    if isinstance(inputs, dict) and inputs.get("all_selected_stage_tensor_file"):
        return Path(inputs["all_selected_stage_tensor_file"])

    return args.components_dir / ALL_SELECTED_STAGE_FILE


def selected_entries(metadata: dict[str, Any], sources: list[str]) -> list[dict[str, Any]]:
    entries = metadata.get("selected_tensors")
    if not isinstance(entries, list):
        raise KeyError("component metadata has no selected_tensors list")

    source_set = set(sources)
    selected = [entry for entry in entries if entry.get("source_name") in source_set]
    if len(selected) != len(source_set):
        found = {entry.get("source_name") for entry in selected}
        missing = sorted(source_set - found)
        raise KeyError(f"missing selected tensor metadata for {missing}")
    return selected


def baseline_for_entry(entry: dict[str, Any], stage_file: Path, chunk_rows: int) -> dict[str, Any]:
    source_shape = entry.get("source_shape") or entry.get("shape")
    if not isinstance(source_shape, list) or len(source_shape) != 2:
        raise ValueError(f"expected rank-2 source shape for {entry.get('source_name')}: {source_shape}")

    row_count = int(source_shape[0])

    return {
        "source_name": entry["source_name"],
        "elixir_name": entry.get("elixir_name"),
        "safe_key": entry.get("safe_key") or sanitize_python_key(entry["source_name"]),
        "source_shape": source_shape,
        "component_source": entry.get("component_source"),
        "component_v_layout": entry.get("component_v_layout", "torch_v"),
        "component_tensors": entry.get("component_tensors", {}),
        "scale_tensor": entry.get("scale_tensor"),
        "offset_start": entry.get("offset_start"),
        "offset_end": entry.get("offset_end"),
        "singular_values": entry.get("singular_values"),
        "stage_tensor_file": str(stage_file),
        "stage_tensors": entry.get("stage_tensors", {}),
        "chunks": chunks(row_count, chunk_rows),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    sources = args.sources or DEFAULT_SOURCES
    metadata = load_json(args.components_dir / METADATA_FILE)
    python_report = load_json(args.python_report)
    stage_file = all_selected_stage_file(args, python_report)

    if not stage_file.exists():
        raise FileNotFoundError(stage_file)

    baselines = [
        baseline_for_entry(entry, stage_file, args.chunk_rows)
        for entry in selected_entries(metadata, sources)
    ]

    total_chunks = sum(len(baseline["chunks"]) for baseline in baselines)

    return {
        "schema": "trinity_sakana_large_tensor_chunks_python.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "inputs": {
            "components_dir": str(args.components_dir),
            "python_report": str(args.python_report),
            "stage_tensor_file": str(stage_file),
            "chunk_rows": args.chunk_rows,
            "sources": sources,
        },
        "large_tensor_chunk_baselines": baselines,
        "summary": {
            "source_count": len(baselines),
            "sources": [baseline["source_name"] for baseline in baselines],
            "chunk_count": total_chunks,
        },
    }


def sanitize_python_key(source_name: str) -> str:
    return "".join(
        ch if (ch.isalnum() or ch in "_.-") else "__"
        for ch in source_name.replace("/", "__")
    )


def self_test() -> None:
    assert chunks(5, 2) == [
        {"chunk_index": 0, "row_start": 0, "row_end": 2, "row_count": 2},
        {"chunk_index": 1, "row_start": 2, "row_end": 4, "row_count": 2},
        {"chunk_index": 2, "row_start": 4, "row_end": 5, "row_count": 1},
    ]

    entry = {
        "source_name": "model/embed tokens.weight",
        "source_shape": [5, 3],
        "stage_tensors": {"source_f32": "tensor.model__embed tokens.weight.source_f32"},
    }
    baseline = baseline_for_entry(entry, Path("stage.safetensors"), 4)
    assert baseline["safe_key"] == "model__embed__tokens.weight"
    assert [chunk["row_count"] for chunk in baseline["chunks"]] == [4, 1]


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return

    payload = build_manifest(args)
    write_json(args.out, payload)
    print(f"wrote large-tensor chunk Python manifest: {args.out}")
    print(
        "chunk summary: "
        f"sources={payload['summary']['sources']} chunks={payload['summary']['chunk_count']}"
    )


if __name__ == "__main__":
    main()
