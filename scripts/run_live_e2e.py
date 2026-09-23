#!/usr/bin/env python3
"""Run target-free, batch-one BindLens readout on ordinary prompts.

This command starts from a prompt and executes READ → BIND → CARRY → SCORE
online.  By default it uses the shared-KV cached tree with the vectorized BIND
policy and GPU-resident CARRY.  A single ``--prompt`` prints its readout; a
sample manifest plus ``--out-dir`` creates resumable JSONL records and a
latency summary.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import _boot  # noqa: F401
import torch

from bindlens.io import write_json
from bindlens.live import (
    LIVE_SOURCE_LAYERS,
    PROPOSAL_BACKENDS,
    TREE_POLICIES,
    live_config,
    prepare_live_readout,
    summarize_live_rows,
)
from bindlens.models import load_model
from bindlens.protocol import MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(sorted(LIVE_SOURCE_LAYERS)))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", help="one ordinary cloze prompt")
    inputs.add_argument("--samples", type=Path, help="JSON manifest with a non-empty samples list")
    parser.add_argument("--out-dir", type=Path, default=None, help="required when using --samples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default=None,
        help=(
            "optional Hugging Face placement override; use with multiple CUDA-visible GPUs "
            "to shard an otherwise single-card model"
        ),
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=int,
        default=None,
        help="per-visible-GPU placement ceiling when a device map is active",
    )
    parser.add_argument(
        "--lens-path",
        type=Path,
        default=None,
        help="optional compatible J-/R-lens artifact; leaves the released model-default lens unchanged",
    )
    parser.add_argument("--limit", type=int, default=0, help="first N manifest rows; 0 keeps all")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--carrier-batch-size",
        type=int,
        default=16,
        help="candidate microbatch inside one request (never prompt batching)",
    )
    parser.add_argument("--carrier-chunk-size", type=int, default=0)
    parser.add_argument("--top-readouts", type=int, default=3)
    parser.add_argument(
        "--proposal-backend",
        choices=PROPOSAL_BACKENDS,
        default="cached_tree",
        help="cached_tree is the default online backend; old_full_forward is the reference implementation",
    )
    parser.add_argument(
        "--tree-layer-batch-size",
        type=int,
        default=None,
        help="source layers sharing each cached-tree frontier batch; defaults to the validated model setting",
    )
    parser.add_argument(
        "--tree-policy",
        choices=TREE_POLICIES,
        default=None,
        help=(
            "cached-tree policy executor; defaults to the model serving configuration "
            "(vectorized), while reference preserves the scalar policy loop"
        ),
    )
    parser.add_argument(
        "--gpu-carry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "retain carrier vectors on GPU through SCORE; defaults to the model serving "
            "configuration (enabled)"
        ),
    )
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="synchronize around each stage and record latency (default: enabled)",
    )
    parser.add_argument(
        "--resident-jlens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep selected J matrices on GPU; default is model-specific",
    )
    parser.add_argument(
        "--resident-operators",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compatibility boolean: GPU-cache exact ridge maps, or recompute them; default is model-specific",
    )
    parser.add_argument(
        "--operator-cache-mode",
        choices=("gpu", "cpu", "none"),
        default=None,
        help=(
            "exact ridge-map cache location: gpu, cpu (stage one map per score layer), or none; "
            "defaults to the model-specific strategy"
        ),
    )
    parser.add_argument(
        "--operator-cache-dir",
        type=Path,
        default=None,
        help=(
            "durable per-layer cache for exact CPU ridge maps; only valid with "
            "--operator-cache-mode cpu"
        ),
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=None,
        help=(
            "refuse CUDA model load below this free-memory floor; defaults to 70 for Qwen3.6 "
            "and the sharded Llama3.3-70B, and 0 for smaller models"
        ),
    )
    return parser.parse_args()


def check_free_memory(
    device_name: str,
    minimum_gib: float,
    *,
    require_all_visible: bool = False,
) -> dict[str, Any] | None:
    """Optionally refuse a launch that would contend with another GPU service.

    Sharded models need the floor on *every* visible device, not just the
    device that receives input IDs.  Keeping this check before model loading
    avoids a partial two-card load followed by an avoidable OOM.
    """

    device = torch.device(device_name)
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(f"{device_name}: CUDA is not available")
    first_index = int(device.index if device.index is not None else torch.cuda.current_device())
    indices = range(torch.cuda.device_count()) if require_all_visible else (first_index,)
    snapshots: list[dict[str, int]] = []
    too_small: list[dict[str, int]] = []
    for index in indices:
        free, total = torch.cuda.mem_get_info(index)
        row = {"device_index": int(index), "free_bytes": int(free), "total_bytes": int(total)}
        snapshots.append(row)
        if float(minimum_gib) > 0.0 and free < float(minimum_gib) * (1024**3):
            too_small.append(row)
    if too_small:
        detail = ", ".join(
            f"cuda:{row['device_index']}={row['free_bytes'] / 2**30:.2f} GiB" for row in too_small
        )
        raise RuntimeError(
            f"refusing model load on {device_name}: {detail}; minimum is {float(minimum_gib):.2f} GiB"
        )
    if require_all_visible:
        return {"devices": snapshots}
    return snapshots[0]


def default_min_free_gib(model: str) -> float:
    """Return the conservative serving-memory floor for a validated model."""

    # Qwen3.6 keeps a large linear-attention state during tree expansion; the
    # Llama3.3 path is a two-card 70B model with exact score-map staging.  Both
    # need a substantial margin before loading.  Small single-card models keep
    # their historical no-floor behavior.
    return 70.0 if model in {"qwen3_6_27b", "llama3_3_70b_it"} else 0.0


def load_requests(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize an interactive prompt or a prior latency-manifest schema."""

    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be blank")
        return {"mode": "interactive"}, [
            {"trial_id": "interactive", "run_order": 0, "length_bin": "interactive", "prompt": args.prompt}
        ]

    assert args.samples is not None
    if args.out_dir is None:
        raise ValueError("--samples requires --out-dir so batch output is not lost")
    payload = json.loads(args.samples.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{args.samples}: manifest must be a JSON object")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{args.samples}: missing non-empty samples list")
    requests: list[dict[str, Any]] = []
    for index, row in enumerate(samples):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{args.samples}: malformed sample at index {index}")
        requests.append(
            {
                "trial_id": str(row.get("trial_id", index)),
                "run_order": int(row.get("run_order", index)),
                "length_bin": str(row.get("length_bin", "")),
                "prompt": str(row["prompt"]),
            }
        )
    if args.limit > len(requests):
        raise ValueError(f"--limit {args.limit} exceeds the {len(requests)} manifest samples")
    if args.limit:
        requests = requests[: int(args.limit)]
    if not requests:
        raise ValueError("--limit selected no samples")
    return payload, requests


def read_completed(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            completed.add((str(row["trial_id"]), int(row["repeat"])))
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def qwen_linear_attention_runtime(loaded: Any) -> dict[str, Any] | None:
    """Describe the optional Qwen3.6 linear-attention kernels in use.

    Transformers selects the FLA and causal-convolution kernels at import time.
    Recording their module names keeps latency reports comparable when the same
    serving command is run in a pure-PyTorch environment instead.
    """

    # ``LoadedModel.model`` is the J-lens wrapper; the optional kernels live
    # on its underlying Hugging Face module.
    layers = [
        module
        for module in loaded.hf.modules()
        if type(module).__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if not layers:
        return None
    first = layers[0]

    def module_name(attribute: str) -> str | None:
        implementation = getattr(first, attribute, None)
        return None if implementation is None else str(getattr(implementation, "__module__", ""))

    implementations = {
        "causal_conv1d_update": module_name("causal_conv1d_update"),
        "chunk_gated_delta_rule": module_name("chunk_gated_delta_rule"),
        "recurrent_gated_delta_rule": module_name("recurrent_gated_delta_rule"),
    }
    return {
        "linear_attention_layers": len(layers),
        "implementations": implementations,
        "fla_fast_path": all(
            (implementation or "").startswith(("causal_conv1d.", "fla."))
            for implementation in implementations.values()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.warmups < 0 or args.repeats < 1:
        raise ValueError("--limit and --warmups must be non-negative; --repeats must be positive")
    if (
        args.carrier_batch_size < 1
        or args.carrier_chunk_size < 0
        or args.top_readouts < 0
        or (args.tree_layer_batch_size is not None and args.tree_layer_batch_size < 1)
    ):
        raise ValueError("invalid carrier or readout size")
    minimum_free_gib = (
        default_min_free_gib(args.model)
        if args.min_free_gib is None
        else float(args.min_free_gib)
    )
    if minimum_free_gib < 0.0:
        raise ValueError("--min-free-gib must be non-negative")
    if args.max_gpu_memory_gib is not None and args.max_gpu_memory_gib < 1:
        raise ValueError("--max-gpu-memory-gib must be positive")
    if args.resident_operators is not None and args.operator_cache_mode is not None:
        raise ValueError("pass either --[no-]resident-operators or --operator-cache-mode, not both")

    manifest, requests = load_requests(args)
    config = live_config(args.model)
    requested_device_map = (
        str(args.device_map) if args.device_map is not None else MODELS[args.model].device_map
    )
    snapshot = check_free_memory(
        args.device,
        minimum_free_gib,
        require_all_visible=requested_device_map is not None,
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))

    print(f"loading {args.model} on {args.device}", flush=True)
    loaded = load_model(
        args.model,
        args.device,
        lens_path=args.lens_path,
        device_map=args.device_map,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
    )
    qwen_runtime = qwen_linear_attention_runtime(loaded)
    resolved_resident_jlens = (
        bool(config.default_resident_jlens)
        if args.resident_jlens is None
        else bool(args.resident_jlens)
    )
    resolved_operator_cache_mode = (
        (
            "gpu"
            if args.resident_operators
            else "none"
        )
        if args.resident_operators is not None
        else (
            str(args.operator_cache_mode)
            if args.operator_cache_mode is not None
        else config.default_operator_cache_mode
        )
    )
    if args.operator_cache_dir is not None and resolved_operator_cache_mode != "cpu":
        raise ValueError("--operator-cache-dir requires --operator-cache-mode cpu")
    engine = prepare_live_readout(
        loaded,
        config=config,
        tree_policy=args.tree_policy,
        gpu_carry=args.gpu_carry,
        resident_jlens=resolved_resident_jlens,
        operator_cache_mode=resolved_operator_cache_mode,
        operator_cache_dir=args.operator_cache_dir,
    )
    tree_layer_batch_size = (
        int(engine.config.default_tree_layer_batch_size)
        if args.tree_layer_batch_size is None
        else int(args.tree_layer_batch_size)
    )
    print(
        f"prepared static serving state in {engine.static_state['warm_static_state_ms']:.1f} ms; "
        f"tree_policy={engine.tree_policy}; gpu_carry={engine.gpu_carry}; warmups={args.warmups}",
        flush=True,
    )

    raw_path: Path | None = None
    summary_path: Path | None = None
    completed: set[tuple[str, int]] = set()
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = args.out_dir / "live_readouts.jsonl"
        summary_path = args.out_dir / "latency_summary.json"
        write_json(
            args.out_dir / "run_metadata.json",
            {
                **engine.metadata(),
                "device": args.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "visible_cuda_devices": torch.cuda.device_count(),
                "resolved_device_map": loaded.spec.device_map,
                "resolved_max_gpu_memory_gib": loaded.spec.max_gpu_memory_gib,
                "hf_device_map": getattr(loaded.hf, "hf_device_map", None),
                "preflight_memory": snapshot,
                "min_free_gib": minimum_free_gib,
                "min_free_gib_source": "model_default" if args.min_free_gib is None else "command_line",
                "lens_path": (
                    str(args.lens_path) if args.lens_path is not None else "model_default_released_jlens"
                ),
                "lens_n_prompts": int(loaded.lens.n_prompts),
                "lens_source_layers": [int(layer) for layer in loaded.lens.source_layers],
                "qwen_linear_attention_runtime": qwen_runtime,
                "resident_jlens": resolved_resident_jlens,
                "operator_cache_mode": resolved_operator_cache_mode,
                "operator_cache_dir": (
                    str(args.operator_cache_dir.expanduser().resolve())
                    if args.operator_cache_dir is not None
                    else None
                ),
                "carrier_batch_size": int(args.carrier_batch_size),
                "carrier_chunk_size": int(args.carrier_chunk_size),
                "top_readouts": int(args.top_readouts),
                "proposal_backend": args.proposal_backend,
                "tree_layer_batch_size": tree_layer_batch_size,
                "tree_layer_batch_size_source": (
                    "model_default" if args.tree_layer_batch_size is None else "command_line"
                ),
                "tree_policy_source": "model_default" if args.tree_policy is None else "command_line",
                "gpu_carry_source": "model_default" if args.gpu_carry is None else "command_line",
                "measure_latency": bool(args.timing),
                "warmups": int(args.warmups),
                "repeats": int(args.repeats),
                "sample_manifest": str(args.samples) if args.samples is not None else None,
                "sample_selection": manifest.get("selection"),
            },
        )
        completed = read_completed(raw_path)

    for warmup_index in range(args.warmups):
        request = requests[warmup_index % len(requests)]
        engine.read(
            request["prompt"],
            carrier_batch_size=args.carrier_batch_size,
            carrier_chunk_size=args.carrier_chunk_size,
            top_readouts=args.top_readouts,
            measure_latency=bool(args.timing),
            proposal_backend=args.proposal_backend,
            tree_layer_batch_size=tree_layer_batch_size,
        )

    for request in requests:
        for repeat in range(args.repeats):
            key = (str(request["trial_id"]), repeat)
            if key in completed:
                print(f"skip completed {request['trial_id']} repeat={repeat}", flush=True)
                continue
            print(
                f"live E2E {args.model} {request['trial_id']} bin={request['length_bin']} repeat={repeat}",
                flush=True,
            )
            try:
                result = engine.read(
                    request["prompt"],
                    carrier_batch_size=args.carrier_batch_size,
                    carrier_chunk_size=args.carrier_chunk_size,
                    top_readouts=args.top_readouts,
                    measure_latency=bool(args.timing),
                    proposal_backend=args.proposal_backend,
                    tree_layer_batch_size=tree_layer_batch_size,
                )
            except Exception as error:
                if raw_path is not None:
                    append_jsonl(
                        raw_path,
                        {
                            "status": "error",
                            "model": args.model,
                            "trial_id": str(request["trial_id"]),
                            "repeat": repeat,
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                raise

            row = {
                "status": "ok",
                "model": args.model,
                "trial_id": str(request["trial_id"]),
                "run_order": int(request["run_order"]),
                "length_bin": str(request["length_bin"]),
                "prompt_tokens": int(result["input_tokens"]),
                "repeat": repeat,
                "result": result,
            }
            if raw_path is None:
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            else:
                append_jsonl(raw_path, row)
                latency = result.get("latency_ms")
                if isinstance(latency, dict):
                    print(f"  total={float(latency['total']):.1f} ms", flush=True)

    if raw_path is not None and summary_path is not None:
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        write_json(summary_path, summarize_live_rows(rows))
        print(f"wrote {raw_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
