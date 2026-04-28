#!/usr/bin/env python3
"""Emit Python-side checkpoints for the Sakana/TRINITY SVD sample hash.

The reference manifest contains a historical bf16 SHA-256 for one adapted tensor.
That hash is only bit-reproducible when the exact SVD components used to create
it are available. Recomputing SVD with a different PyTorch/CUDA/LAPACK version can
produce a mathematically valid but different singular-vector basis. Zero-offset
reconstruction still matches the source, but non-zero singular-value scaling can
change the adapted tensor hash.

This script therefore reports both:

* the stored reference hash from sakana_python_reference_manifest.json, and
* the current Python baseline hash produced by either an explicit svd_weights.pt
  file or by recomputing SVD in the current environment.

When --write-components-dir is enabled, the component bundle includes a small
JSON metadata file describing whether the stored reference hash was reproduced.
It also writes a stage tensor bundle by default so Elixir can compare the source
tensor, offsets, scaled singular values, reconstruction tensors, and final bf16
bytes against Python safetensors readback without relying on a single hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_REFERENCE = Path("priv/sakana_trinity/reference/sakana_python_reference_manifest.json")
DEFAULT_ROUTER_VECTOR_NPY = Path("priv/sakana_trinity/artifacts/sakana_model_iter_60.npy")
DEFAULT_OUT = Path("tmp/sakana_parity/python_sample_trace.json")
DEFAULT_COMPONENT_DIR = Path("tmp/sakana_parity/python_components")
STAGE_FILE = "trinity_svf_stage_debug.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--router-vector", type=Path, default=DEFAULT_ROUTER_VECTOR_NPY)
    parser.add_argument("--svd-weights", type=Path, default=None,
                        help="Optional original svd_weights.pt. Use this for strict historical hash reproduction; without it, SVD is recomputed in the current environment.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-components-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--readback-components-dir", type=Path, default=None,
                        help="Optional component directory to read back and reconstruct. Defaults to --write-components-dir when components are written.")
    parser.add_argument("--no-write-stage-tensors", action="store_true",
                        help="Do not write the Python stage tensor bundle used for side-by-side parity checks.")
    parser.add_argument("--no-write-components", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--vector-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--model-torch-dtype",
        default="float32",
        choices=["auto", "float32", "bfloat16"],
        help="Model weight dtype for parity diagnostics. The default uses float32; auto/bfloat16 intentionally test alternate provenance.",
    )
    parser.add_argument("--strict-reference-hash", action="store_true",
                        help="Exit non-zero when no Python variant reproduces the stored reference hash.")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    t = tensor.detach().cpu().contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


def tensor_prefix(tensor: torch.Tensor, count: int = 8) -> list[float]:
    flat = tensor.detach().to(torch.float32).cpu().reshape(-1)
    n = min(count, flat.numel())
    return [float(x) for x in flat[:n].tolist()]


def finite(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def tensor_summary(tensor: torch.Tensor, prefix_count: int = 8, alt_hashes: bool = True) -> dict[str, Any]:
    t32 = tensor.detach().to(torch.float32)
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "size": int(tensor.numel()),
        "sha256": tensor_sha256(tensor),
        "min": finite(float(t32.min().item())),
        "max": finite(float(t32.max().item())),
        "sum": finite(float(t32.sum().item())),
        "prefix_f32": tensor_prefix(tensor, prefix_count),
    }
    if alt_hashes:
        result["sha256_as_f32"] = tensor_sha256(tensor.to(torch.float32))
        result["sha256_as_bf16"] = tensor_sha256(tensor.to(torch.bfloat16))
    return result


def orient_to_shape(tensor: torch.Tensor, shape: list[int], label: str) -> torch.Tensor:
    target = tuple(int(x) for x in shape)
    if tuple(tensor.shape) == target:
        return tensor.contiguous()
    if tensor.ndim == 2 and tuple(tensor.T.shape) == target:
        return tensor.T.contiguous()
    raise ValueError(f"cannot orient {label} from {tuple(tensor.shape)} to {target}")


def sanitize_key(key: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_.-") else "__" for ch in key.replace("/", "__"))


def load_router_vector(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        vector = np.load(path)
    else:
        from safetensors.numpy import load_file

        loaded = load_file(str(path))
        if "trinity_router_es_vector" not in loaded:
            raise KeyError(f"trinity_router_es_vector not found in {path}; keys={list(loaded)}")
        vector = loaded["trinity_router_es_vector"]

    return vector.astype(np.float32 if dtype == "float32" else np.float64, copy=False)


def load_model(model_name: str, dtype_arg: Any) -> torch.nn.Module:
    """Load model weights using the current Transformers dtype keyword.

    Newer Transformers versions prefer ``dtype=``. Keep a fallback for older
    installations to avoid making the debug harness depend on one exact version.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype_arg)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype_arg)


def reconstruct_from_torch_v(
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    offsets = offsets.to(dtype=s.dtype, device=s.device)
    scaled_s = s * (1.0 + offsets)
    normalization = s.sum() / scaled_s.sum()
    reconstructed = (u * scaled_s.reshape(1, -1)) @ v.T
    reconstructed = reconstructed * normalization
    return reconstructed, singular_summary(s, offsets, scaled_s, normalization)


def reconstruct_from_vh(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    offsets = offsets.to(dtype=s.dtype, device=s.device)
    scaled_s = s * (1.0 + offsets)
    normalization = s.sum() / scaled_s.sum()
    reconstructed = (u * scaled_s.reshape(1, -1)) @ vh
    reconstructed = reconstructed * normalization
    return reconstructed, singular_summary(s, offsets, scaled_s, normalization)


def singular_summary(s: torch.Tensor, offsets: torch.Tensor, scaled_s: torch.Tensor, normalization: torch.Tensor) -> dict[str, Any]:
    return {
        "singular_values": tensor_summary(s, 16),
        "typed_offsets": tensor_summary(offsets, 16),
        "scaled_s": tensor_summary(scaled_s, 16),
        "sum_s": finite(float(s.detach().to(torch.float32).sum().item())),
        "sum_scaled_s": finite(float(scaled_s.detach().to(torch.float32).sum().item())),
        "normalization": finite(float(normalization.detach().to(torch.float32).item())),
    }


def max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().to(torch.float32) - right.detach().to(torch.float32)).abs().max().item())


def variant_report(
    label: str,
    zero_reconstructed: torch.Tensor,
    adapted_reconstructed: torch.Tensor,
    singular: dict[str, Any],
    source_f32: torch.Tensor,
    sample: dict[str, Any],
    component_source: str,
    v_layout: str,
) -> dict[str, Any]:
    final_f32_oriented = orient_to_shape(adapted_reconstructed, sample["sample_reconstructed_shape"], label)
    final_bf16 = final_f32_oriented.to(torch.bfloat16).contiguous()
    source_bf16_roundtrip = source_f32.to(torch.bfloat16).to(torch.float32)
    observed = tensor_sha256(final_bf16)
    expected = sample["sample_reconstructed_bf16_sha256"]
    return {
        "label": label,
        "component_source": component_source,
        "v_layout": v_layout,
        "zero_offset_max_abs_error_vs_source": max_abs_error(zero_reconstructed, source_f32),
        "zero_offset_max_abs_error_vs_source_bf16_roundtrip": max_abs_error(
            zero_reconstructed,
            source_bf16_roundtrip,
        ),
        "s": singular,
        "final_f32_before_bf16": tensor_summary(final_f32_oriented, 16),
        "final": tensor_summary(final_bf16, 16),
        "observed_bf16_sha256": observed,
        "expected_bf16_sha256": expected,
        "matches_expected": observed == expected,
    }


def dtype_arg_from_cli(value: str) -> Any:
    if value == "auto":
        return "auto"
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(value)


def load_svd_weight_components(path: Path, source_name: str, device: str) -> Optional[dict[str, torch.Tensor]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    loaded = torch.load(path, map_location="cpu")
    keys = {
        "u": f"{source_name}.U",
        "s": f"{source_name}.S",
        "v": f"{source_name}.V",
    }
    missing = [key for key in keys.values() if key not in loaded]
    if missing:
        raise KeyError(f"missing keys in {path}: {missing}; available sample={list(loaded.keys())[:20]}")
    return {name: loaded[key].detach().to(device) for name, key in keys.items()}


def build_variants(
    source_f32: torch.Tensor,
    offsets: torch.Tensor,
    sample: dict[str, Any],
    svd_components: Optional[dict[str, torch.Tensor]],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], str]:
    variants: list[dict[str, Any]] = []

    # Current-environment SVD. This is useful as a local baseline, but it should
    # not be treated as bit-identical to a historical reference hash.
    u_svd, s_svd, v_svd = torch.svd(source_f32)
    zeros = torch.zeros_like(s_svd)
    zero_svd, _ = reconstruct_from_torch_v(u_svd, s_svd, v_svd, zeros)
    adapted_svd, singular_svd = reconstruct_from_torch_v(u_svd, s_svd, v_svd, offsets.to(torch.float32))
    variants.append(
        variant_report(
            "python_recomputed_torch_svd_v_final_bf16",
            zero_svd,
            adapted_svd,
            singular_svd,
            source_f32,
            sample,
            "recomputed_torch_svd",
            "torch_v",
        )
    )

    u_linalg, s_linalg, vh_linalg = torch.linalg.svd(source_f32, full_matrices=False)
    zero_linalg, _ = reconstruct_from_vh(u_linalg, s_linalg, vh_linalg, torch.zeros_like(s_linalg))
    adapted_linalg, singular_linalg = reconstruct_from_vh(u_linalg, s_linalg, vh_linalg, offsets.to(torch.float32))
    variants.append(
        variant_report(
            "python_recomputed_linalg_svd_vh_final_bf16",
            zero_linalg,
            adapted_linalg,
            singular_linalg,
            source_f32,
            sample,
            "recomputed_torch_linalg_svd",
            "vh",
        )
    )

    component_bundle = {"u": u_svd, "s": s_svd, "v": v_svd}
    component_source = "recomputed_torch_svd"

    if svd_components is not None:
        u_ref = svd_components["u"].to(torch.float32)
        s_ref = svd_components["s"].to(torch.float32)
        v_ref = svd_components["v"].to(torch.float32)
        zeros_ref = torch.zeros_like(s_ref)

        zero_ref_torch_v, _ = reconstruct_from_torch_v(u_ref, s_ref, v_ref, zeros_ref)
        adapted_ref_torch_v, singular_ref_torch_v = reconstruct_from_torch_v(
            u_ref, s_ref, v_ref, offsets.to(torch.float32)
        )
        variants.append(
            variant_report(
                "python_svd_weights_torch_v_final_bf16",
                zero_ref_torch_v,
                adapted_ref_torch_v,
                singular_ref_torch_v,
                source_f32,
                sample,
                "svd_weights_pt",
                "torch_v",
            )
        )

        # Defensive diagnostic: if a supplied file was produced by linalg.svd and
        # stored Vh under .V, this variant will have much lower zero error.
        zero_ref_vh, _ = reconstruct_from_vh(u_ref, s_ref, v_ref, zeros_ref)
        adapted_ref_vh, singular_ref_vh = reconstruct_from_vh(u_ref, s_ref, v_ref, offsets.to(torch.float32))
        variants.append(
            variant_report(
                "python_svd_weights_vh_final_bf16",
                zero_ref_vh,
                adapted_ref_vh,
                singular_ref_vh,
                source_f32,
                sample,
                "svd_weights_pt",
                "vh",
            )
        )

        component_bundle = {"u": u_ref, "s": s_ref, "v": v_ref}
        component_source = "svd_weights_pt"

    return variants, component_bundle, component_source


def select_baseline_variant(
    variants: list[dict[str, Any]],
    expected: str,
    component_source: str,
) -> dict[str, Any]:
    for variant in variants:
        if variant["observed_bf16_sha256"] == expected:
            return variant

    preferred_labels = [
        "python_safetensors_readback_torch_v_final_bf16",
        "python_svd_weights_torch_v_final_bf16" if component_source == "svd_weights_pt" else "python_recomputed_torch_svd_v_final_bf16",
        "python_recomputed_torch_svd_v_final_bf16",
    ]
    for label in preferred_labels:
        for variant in variants:
            if variant["label"] == label:
                return variant
    return variants[0]


def write_component_bundle(
    out_dir: Path,
    source_name: str,
    components: dict[str, torch.Tensor],
    offsets: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = sanitize_key(source_name)
    save_file(
        {
            f"svd.U.{safe}": components["u"].detach().cpu().contiguous(),
            f"svd.S.{safe}": components["s"].detach().cpu().contiguous(),
            f"svd.V.{safe}": components["v"].detach().cpu().contiguous(),
        },
        str(out_dir / "trinity_svf_components.safetensors"),
    )
    save_file(
        {f"svf.scale_offsets.{safe}": offsets.detach().cpu().contiguous()},
        str(out_dir / "trinity_svf_scale_offsets.safetensors"),
    )
    (out_dir / "trinity_svf_debug_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_component_bundle(
    components_dir: Path,
    source_name: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    safe = sanitize_key(source_name)
    components_path = components_dir / "trinity_svf_components.safetensors"
    scales_path = components_dir / "trinity_svf_scale_offsets.safetensors"
    components = load_file(str(components_path), device="cpu")
    scales = load_file(str(scales_path), device="cpu")
    keys = {
        "u": f"svd.U.{safe}",
        "s": f"svd.S.{safe}",
        "v": f"svd.V.{safe}",
        "offsets": f"svf.scale_offsets.{safe}",
    }
    missing = [
        key
        for key in keys.values()
        if key not in components and key not in scales
    ]
    if missing:
        raise KeyError(
            f"missing readback keys in {components_dir}: {missing}; "
            f"component keys={list(components.keys())}; scale keys={list(scales.keys())}"
        )
    bundle = {
        "u": components[keys["u"]].detach().to(device),
        "s": components[keys["s"]].detach().to(device),
        "v": components[keys["v"]].detach().to(device),
    }
    offsets = scales[keys["offsets"]].detach().to(device)
    return bundle, offsets


def component_bundle_summary(
    components: dict[str, torch.Tensor],
    offsets: torch.Tensor,
) -> dict[str, Any]:
    return {
        "u": tensor_summary(components["u"], 4),
        "s": tensor_summary(components["s"], 16),
        "v": tensor_summary(components["v"], 4),
        "offsets": tensor_summary(offsets, 16),
    }


def build_readback_variants(
    components_dir: Path,
    source_name: str,
    source_f32: torch.Tensor,
    sample: dict[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
    components, offsets = load_component_bundle(components_dir, source_name, device)
    u = components["u"].to(torch.float32)
    s = components["s"].to(torch.float32)
    v = components["v"].to(torch.float32)
    offsets = offsets.to(torch.float32)
    zeros = torch.zeros_like(s)

    zero_torch_v, _ = reconstruct_from_torch_v(u, s, v, zeros)
    adapted_torch_v, singular_torch_v = reconstruct_from_torch_v(u, s, v, offsets)

    zero_vh, _ = reconstruct_from_vh(u, s, v, zeros)
    adapted_vh, singular_vh = reconstruct_from_vh(u, s, v, offsets)

    variants = [
        variant_report(
            "python_safetensors_readback_torch_v_final_bf16",
            zero_torch_v,
            adapted_torch_v,
            singular_torch_v,
            source_f32,
            sample,
            "safetensors_readback",
            "torch_v",
        ),
        variant_report(
            "python_safetensors_readback_vh_final_bf16",
            zero_vh,
            adapted_vh,
            singular_vh,
            source_f32,
            sample,
            "safetensors_readback",
            "vh",
        ),
    ]
    stage_tensors = torch_v_stage_tensors(u, s, v, offsets, source_f32, sample)
    return variants, component_bundle_summary(components, offsets), stage_tensors


def torch_v_stage_tensors(
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    source_f32: torch.Tensor,
    sample: dict[str, Any],
) -> dict[str, torch.Tensor]:
    offsets = offsets.to(dtype=s.dtype, device=s.device)
    scaled_s = s * (1.0 + offsets)
    normalization = (s.sum() / scaled_s.sum()).reshape(1)
    u_scaled = u * scaled_s.reshape(1, -1)
    matmul_pre_norm = u_scaled @ v.T
    adapted_source_f32 = matmul_pre_norm * normalization.reshape(())
    final_f32 = orient_to_shape(adapted_source_f32, sample["sample_reconstructed_shape"], "python_stage_final")
    final_bf16 = final_f32.to(torch.bfloat16).contiguous()
    zero_source_f32 = (u * s.reshape(1, -1)) @ v.T

    return {
        "stage.source_f32": source_f32.to(torch.float32).contiguous(),
        "stage.offsets_f32": offsets.to(torch.float32).contiguous(),
        "stage.scaled_s": scaled_s.to(torch.float32).contiguous(),
        "stage.normalization": normalization.to(torch.float32).contiguous(),
        "stage.u_scaled": u_scaled.to(torch.float32).contiguous(),
        "stage.matmul_pre_norm": matmul_pre_norm.to(torch.float32).contiguous(),
        "stage.adapted_source_f32": adapted_source_f32.to(torch.float32).contiguous(),
        "stage.final_f32": final_f32.to(torch.float32).contiguous(),
        "stage.final_bf16": final_bf16.contiguous(),
        "stage.zero_source_f32": zero_source_f32.to(torch.float32).contiguous(),
    }


def write_stage_bundle(out_dir: Path, stage_tensors: dict[str, torch.Tensor]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / STAGE_FILE
    save_file({key: value.detach().cpu().contiguous() for key, value in stage_tensors.items()}, str(path))
    return path


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference.read_text())
    sample = reference["sample_adapted_tensor"]
    expected = sample["sample_reconstructed_bf16_sha256"]

    vector = load_router_vector(args.router_vector, args.vector_dtype)
    offset_start = int(sample["offset_start"])
    offset_end = int(sample["offset_end"])
    offset_np = vector[offset_start:offset_end].copy()
    offsets = torch.from_numpy(offset_np).to(args.device)

    model = load_model(args.model_name, dtype_arg_from_cli(args.model_torch_dtype))
    state_dict = model.state_dict()
    source_name = sample["source_name"]
    if source_name not in state_dict:
        raise KeyError(f"{source_name!r} missing; available sample keys={list(state_dict)[:20]}")

    source = state_dict[source_name].detach().to(args.device)
    source_f32 = source.to(torch.float32)

    svd_components = load_svd_weight_components(args.svd_weights, source_name, args.device) if args.svd_weights else None
    variants, component_bundle, component_source = build_variants(source_f32, offsets, sample, svd_components)
    component_readback_summary = None
    stage_tensors: dict[str, torch.Tensor] = {}
    stage_path: Path | None = None

    in_memory_baseline = select_baseline_variant(variants, expected, component_source)
    component_metadata = {
        "schema": "trinity_sakana_sample_component_debug.v1",
        "source_name": source_name,
        "component_source": component_source,
        "component_v_layout": "torch_v",
        "baseline_label": in_memory_baseline["label"],
        "baseline_observed_bf16_sha256": in_memory_baseline["observed_bf16_sha256"],
        "stored_reference_bf16_sha256": expected,
        "stored_reference_hash_reproducible": any(v["matches_expected"] for v in variants),
        "svd_weights_path": None if args.svd_weights is None else str(args.svd_weights),
    }

    if not args.no_write_components:
        write_component_bundle(args.write_components_dir, source_name, component_bundle, offsets, component_metadata)

    readback_dir = args.readback_components_dir
    if readback_dir is None and not args.no_write_components:
        readback_dir = args.write_components_dir

    if readback_dir is not None:
        readback_variants, component_readback_summary, stage_tensors = build_readback_variants(
            readback_dir,
            source_name,
            source_f32,
            sample,
            args.device,
        )
        variants.extend(readback_variants)

        if not args.no_write_stage_tensors:
            stage_path = write_stage_bundle(readback_dir, stage_tensors)

    baseline = select_baseline_variant(variants, expected, component_source)
    reference_hash_reproducible = any(v["matches_expected"] for v in variants)

    report = {
        "schema": "trinity_sakana_python_svd_parity_trace.v2",
        "reference": {
            "path": str(args.reference),
            "source_name": sample["source_name"],
            "elixir_name": sample["elixir_name"],
            "source_shape": sample["source_shape"],
            "sample_reconstructed_shape": sample["sample_reconstructed_shape"],
            "expected_bf16_sha256": expected,
            "expected_hash_reproducible": reference_hash_reproducible,
            "current_python_baseline_label": baseline["label"],
            "current_python_baseline_bf16_sha256": baseline["observed_bf16_sha256"],
            "diagnosis": (
                "stored reference hash reproduced by this Python environment"
                if reference_hash_reproducible
                else "stored reference hash was not reproduced; use current_python_baseline_bf16_sha256 for same-run parity, or provide the original svd_weights.pt"
            ),
        },
        "inputs": {
            "model_name": args.model_name,
            "model_torch_dtype_arg": args.model_torch_dtype,
            "source_tensor_dtype": str(source.dtype).replace("torch.", ""),
            "router_vector": str(args.router_vector),
            "router_vector_sha256": sha256_file(args.router_vector),
            "router_vector_dtype_after_load": str(vector.dtype),
            "svd_weights": None if args.svd_weights is None else str(args.svd_weights),
            "write_components_dir": None if args.no_write_components else str(args.write_components_dir),
            "component_source_written": None if args.no_write_components else component_source,
            "readback_components_dir": None if readback_dir is None else str(readback_dir),
            "stage_tensor_file": None if stage_path is None else str(stage_path),
        },
        "source_tensor": tensor_summary(source, 16),
        "source_tensor_f32_svd_input": tensor_summary(source_f32, 16),
        "scale_offsets": tensor_summary(offsets, 16),
        "component_bundle_before_write": component_bundle_summary(component_bundle, offsets),
        "component_bundle_readback": component_readback_summary,
        "stage_debug": {
            "schema": "trinity_sakana_stage_debug.v1",
            "baseline_label": "python_safetensors_readback_torch_v_final_bf16",
            "stage_tensor_file": None if stage_path is None else str(stage_path),
            "stage_keys": sorted(stage_tensors.keys()),
            "interpretation": (
                "These tensors are emitted from Python safetensors readback. "
                "Use them as the stage-by-stage baseline for Elixir semantic parity. "
                "Exact final bf16 byte equality is aspirational; functional correctness is determined by required stage tolerances."
            ),
        },
        "variants": variants,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"wrote Python parity report: {args.out}")
    if not args.no_write_components:
        print(f"wrote sample Python components: {args.write_components_dir}")
    if stage_path is not None:
        print(f"wrote Python stage tensor bundle: {stage_path}")
        print("stage baseline: Python safetensors readback; compare with --strict-stage-tolerances")
    print(f"stored_reference_bf16_sha256: {expected}")
    print(f"reference_hash_reproducible: {reference_hash_reproducible}")
    print(f"current_python_baseline: {baseline['label']} {baseline['observed_bf16_sha256']}")
    for variant in variants:
        print(
            f"{variant['label']}: {variant['observed_bf16_sha256']} "
            f"match={variant['matches_expected']} zero_error={variant['zero_offset_max_abs_error_vs_source']}"
        )

    if args.strict_reference_hash and not reference_hash_reproducible:
        raise SystemExit(
            "stored reference hash was not reproduced; provide --svd-weights with the original SVD components or unset --strict-reference-hash"
        )


if __name__ == "__main__":
    main()
