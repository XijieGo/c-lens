#!/usr/bin/env python3
"""Token-2 coverage given token-1 is in the J-lens top-10."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _boot  # noqa: F401
import torch

from bindlens.io import iter_sides, load_cloze_pairs, write_json
from bindlens.metrics import wilson
from bindlens.models import load_model
from bindlens.paths import RESULTS_DIR
from bindlens.protocol import MODELS
from bindlens.read import (
    capture_last,
    concept_token_ids,
    encode_cloze,
    jlens_logits,
    ranks_of,
)

KS = (100, 200, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (RESULTS_DIR / "coverage" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    sides = iter_sides(load_cloze_pairs())
    loaded = load_model(args.model, args.device)
    layers = list(loaded.lens.source_layers)
    encoded = []
    meta = []
    for side in sides:
        ids = encode_cloze(loaded.tokenizer, side["prompt"])
        tokens = concept_token_ids(loaded.tokenizer, side["concept"])
        encoded.append(ids)
        meta.append({**side, "token_ids": tokens, "n_tokens": len(tokens)})
    print(f"[{args.model}] capturing last-token residuals, n={len(encoded)} layers={layers[0]}-{layers[-1]}", flush=True)
    states = capture_last(loaded, encoded, layers, batch_size=args.batch_size)

    records = []
    two_token = []
    for index, row in enumerate(meta):
        layer_ranks = []
        for layer in layers:
            logits = jlens_logits(loaded, states[layer][index], layer)
            ranks = ranks_of(logits, row["token_ids"])
            layer_ranks.append({"layer": layer, "ranks": ranks})
        rec = {
            "trial_id": row["trial_id"],
            "concept": row["concept"],
            "n_tokens": row["n_tokens"],
            "layers": layer_ranks,
        }
        records.append(rec)
        if row["n_tokens"] == 2:
            two_token.append(rec)

    gated = []
    for rec in two_token:
        eligible = [item for item in rec["layers"] if item["ranks"] and item["ranks"][0] <= 10]
        if not eligible:
            continue
        best_t2 = min(item["ranks"][1] for item in eligible)
        gated.append(best_t2)

    coverage = {}
    n = len(gated)
    for k in KS:
        hits = sum(1 for rank in gated if rank <= k)
        lo, hi = wilson(hits, n)
        coverage[f"@{k}"] = {"n": n, "hits": hits, "rate": hits / n if n else None, "wilson": [lo, hi]}

    payload = {
        "model": args.model,
        "n_sides": len(sides),
        "n_two_token": len(two_token),
        "n_gated": n,
        "coverage": coverage,
    }
    write_json(out_dir / "metrics.json", payload)
    lines = [
        f"# Coverage {MODELS[args.model].display_name}",
        "",
        f"Two-token sides: {len(two_token)}. Gated (token-1 ≤ 10 at some layer): {n}.",
        "",
        "| k | Coverage | n |",
        "|---:|---:|---:|",
    ]
    for k in KS:
        cell = coverage[f"@{k}"]
        lines.append(f"| {k} | {100 * cell['rate']:.1f}% | {cell['hits']}/{cell['n']} |")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
