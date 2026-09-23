#!/usr/bin/env python3
"""Fit carrier means, pool weights, and whitening stats on a singleton list."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import _boot  # noqa: F401
import torch

from bindlens.carry import aligned_grids, capture_suffix, reconstruct_pooled_carrier
from bindlens.io import load_calibration_concepts, write_json
from bindlens.models import leading_space_ids, load_model, visible
from bindlens.paths import ACTIVE_CALIBRATION_PATH, CARRIER_DIR
from bindlens.protocol import CALIBRATION_CARRIER_PREFIX, CALIBRATION_CARRIER_SUFFIX, MODELS
from bindlens.read import token_directions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--calibration-data",
        "--data",
        dest="calibration_data",
        type=Path,
        default=None,
        help="singleton concept-list JSON (default: data/singleton_calibration_300.json)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or CARRIER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_data = args.calibration_data or ACTIVE_CALIBRATION_PATH
    concepts = load_calibration_concepts(calibration_data)
    if len(concepts) != len(set(concepts)):
        raise SystemExit(f"calibration concept list contains duplicates: {calibration_data}")
    if not concepts:
        raise SystemExit(f"calibration concept list is empty: {calibration_data}")
    loaded = load_model(args.model, args.device)
    grids, failed = aligned_grids(
        loaded.tokenizer,
        concepts,
        prefix=CALIBRATION_CARRIER_PREFIX,
        suffix=CALIBRATION_CARRIER_SUFFIX,
    )
    if [g["concept"] for g in grids] != concepts:
        raise SystemExit(f"calibration alignment failed: {failed[:5]}")
    non_single = [grid["concept"] for grid in grids if grid["n_concept_tokens"] != 1]
    if non_single:
        raise SystemExit(f"calibration requires one-token concepts: {non_single[:5]}")
    token_ids = [leading_space_ids(loaded.tokenizer, c)[0] for c in concepts]
    n_layers = int(loaded.model.n_layers)
    source_layers = list(loaded.lens.source_layers)
    capture_layers = list(range(n_layers))
    print(
        f"[{args.model}] capturing {len(grids)} singletons at {n_layers} layers "
        f"from {calibration_data}",
        flush=True,
    )
    positions = capture_suffix(
        loaded, grids, capture_layers, batch_size=args.batch_size, include_concept_final=True
    )
    n_pos = int(positions[0].shape[1])
    d_model = int(loaded.model.d_model)
    means = torch.empty((n_layers, n_pos, d_model), dtype=torch.float32)
    deviations = torch.empty((n_layers, len(grids), n_pos, d_model), dtype=torch.bfloat16)
    for layer in capture_layers:
        means[layer] = positions[layer].mean(dim=0)
        deviations[layer] = (positions[layer] - means[layer][None]).to(torch.bfloat16)

    jlens_dirs = []
    for layer in source_layers:
        jlens_dirs.append(token_directions(loaded, token_ids, layer))
    raw_jlens_stack = torch.stack(jlens_dirs, dim=0)
    # Carrier residuals are centered per calibration concept, so target
    # directions use the same centering before the pooled comparison.
    jlens_stack = raw_jlens_stack - raw_jlens_stack.mean(dim=1, keepdim=True)

    carrier_layer = loaded.spec.carrier_layer
    dirs = jlens_stack[source_layers.index(carrier_layer)].float()
    post = deviations[carrier_layer, :, 1:, :].float()
    num = (post * dirs[:, None, :]).sum(dim=-1)
    den = (post.norm(dim=-1) * dirs.norm(dim=-1)[:, None]).clamp_min(1e-12)
    weights = (num / den).mean(dim=0).clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-12)
    pooled = reconstruct_pooled_carrier(positions[carrier_layer][:, 1:, :], means[carrier_layer, 1:, :], weights)
    pooled_u = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dirs_u = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    match = (pooled_u * dirs_u).sum(dim=-1)
    ranks = 1 + ((pooled_u @ dirs_u.T) > match[:, None]).sum(dim=-1)

    ids0 = grids[0]["input_ids"]
    n_suffix = grids[0]["n_suffix_tokens"]
    start = len(ids0) - (1 + n_suffix)
    candidate_tokens = []
    for rel, abs_index in enumerate(range(start, len(ids0))):
        candidate_tokens.append(
            {
                "candidate_index": rel,
                "kind": "concept_final_control" if rel == 0 else "carrier",
                "token_id": int(ids0[abs_index]),
                "token_text": visible(loaded.tokenizer, int(ids0[abs_index])),
            }
        )

    tensor = {
        "schema_version": 1,
        "description": "Fact-bare carrier calibration on a supplied singleton concept list.",
        "concepts": concepts,
        "calibration_data": str(calibration_data),
        "calibration_data_sha256": sha256(calibration_data),
        "carrier_layer": int(carrier_layer),
        "n_suffix_tokens": int(n_suffix),
        "carrier_prefix": CALIBRATION_CARRIER_PREFIX,
        "carrier_suffix": CALIBRATION_CARRIER_SUFFIX,
        "capture_layers": capture_layers,
        "source_layers": source_layers,
        "candidate_tokens": candidate_tokens,
        "means": means,
        "deviations": deviations,
        "centered_jlens_directions": jlens_stack,
    }
    out_pt = out_dir / f"{args.model}.pt"
    torch.save(tensor, out_pt)
    meta = {
        "model": args.model,
        "calibration_data": str(calibration_data),
        "calibration_data_sha256": sha256(calibration_data),
        "n_concepts": len(concepts),
        "carrier_layer": carrier_layer,
        "carrier_prefix": CALIBRATION_CARRIER_PREFIX,
        "carrier_suffix": CALIBRATION_CARRIER_SUFFIX,
        "mean_paired_cosine": float(match.mean()),
        "top1": float((ranks == 1).float().mean()),
        "n_suffix_tokens": n_suffix,
        "pool_weights": [float(x) for x in weights.tolist()],
        "tensor_path": str(out_pt),
        "failed": failed,
    }
    write_json(out_dir / f"{args.model}.json", meta)
    print(
        f"[{args.model}] wrote {out_pt} L{carrier_layer} cos={meta['mean_paired_cosine']:.3f} "
        f"top1={meta['top1']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
