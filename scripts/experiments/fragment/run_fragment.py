#!/usr/bin/env python3
"""Forced-split oracle for held-out singleton reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import _boot  # noqa: F401
import torch

from bindlens.carry import load_calibration, pool_weights, reconstruct_pooled_carrier
from bindlens.fragments import build_model_split_rows, validate_fragment_rows
from bindlens.io import (
    load_calibration_concepts,
    load_json,
    load_reconstruction_test_concepts,
    write_json,
)
from bindlens.metrics import rank_against
from bindlens.models import load_model
from bindlens.paths import (
    ACTIVE_CALIBRATION_PATH,
    RECONSTRUCTION_TEST_CONCEPTS_PATH,
    RESULTS_DIR,
    carrier_path,
)
from bindlens.protocol import (
    CALIBRATION_CARRIER_PREFIX,
    CALIBRATION_CARRIER_SUFFIX,
    MODELS,
    RANDOM_DIRECTION_SEED,
)
from bindlens.read import token_directions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--test-data",
        "--data",
        dest="test_data",
        type=Path,
        default=RECONSTRUCTION_TEST_CONCEPTS_PATH,
        help="singleton reconstruction-test JSON (default: data/singleton_reconstruction_test_500.json)",
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=None,
        help="concept list used for the calibration split (for held-out accounting)",
    )
    parser.add_argument(
        "--calibration-artifact",
        "--calibration",
        dest="calibration_artifact",
        type=Path,
        default=None,
        help="calibration tensor to use (defaults to weight/carrier/<model>.pt)",
    )
    parser.add_argument(
        "--split-ledger",
        type=Path,
        default=None,
        help="prebuilt common forced-split ledger; otherwise derive it from --test-data",
    )
    parser.add_argument(
        "--min-piece-length",
        type=int,
        default=1,
        help="minimum characters per forced piece when deriving a singleton ledger",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_DIRECTION_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_from_ledger(path: Path, model: str) -> tuple[list[dict], int]:
    payload = load_json(path)
    if isinstance(payload, dict):
        values = payload.get("rows")
        if values is None and payload.get("concepts") and isinstance(payload["concepts"][0], dict):
            values = payload["concepts"]
        requested = int(payload.get("requested_n") or len(values or []))
    else:
        values = payload
        requested = len(values) if isinstance(values, list) else 0
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: expected a non-empty forced-split row list")
    rows = [dict(row) for row in values]
    validate_fragment_rows(rows, model)
    return rows, requested


def prepare_fragments(
    *,
    args: argparse.Namespace,
    tokenizer: object,
    out_dir: Path,
) -> tuple[list[dict], int, list[dict[str, str]], Path | None]:
    """Load a prebuilt ledger or derive forced-split rows from public surfaces."""
    if args.split_ledger is not None:
        rows, requested = _rows_from_ledger(args.split_ledger, args.model)
        return rows, requested, [], args.split_ledger
    concepts = load_reconstruction_test_concepts(args.test_data)
    rows, failed = build_model_split_rows(
        tokenizer,
        args.model,
        concepts,
        min_piece_length=args.min_piece_length,
    )
    validate_fragment_rows(rows, args.model)
    ledger_path = out_dir / "forced_split_ledger.json"
    write_json(
        ledger_path,
        {
            "schema_version": 1,
            "model": args.model,
            "source": str(args.test_data),
            "source_sha256": sha256(args.test_data),
            "requested_n": len(concepts),
            "min_piece_length": args.min_piece_length,
            "rows": rows,
            "failed": failed,
        },
    )
    return rows, len(concepts), failed, ledger_path


def forced_input_ids(tokenizer: object, piece_ids: list[int]) -> list[int]:
    prefix = [int(x) for x in tokenizer(CALIBRATION_CARRIER_PREFIX, add_special_tokens=True).input_ids]
    suffix = [int(x) for x in tokenizer(CALIBRATION_CARRIER_SUFFIX, add_special_tokens=False).input_ids]
    return [*prefix, *piece_ids, *suffix]


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (RESULTS_DIR / "fragment" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_model(args.model, args.device)
    triples, requested_n, split_failures, ledger_path = prepare_fragments(
        args=args,
        tokenizer=loaded.tokenizer,
        out_dir=out_dir,
    )
    validate_fragment_rows(triples, args.model, loaded.tokenizer)
    calibration_data = args.calibration_data or ACTIVE_CALIBRATION_PATH
    calib_words = {
        str(word).strip().casefold() for word in load_calibration_concepts(calibration_data)
    }
    layer = loaded.spec.carrier_layer
    calibration_artifact = args.calibration_artifact or carrier_path(args.model)
    calibration = load_calibration(args.model, calibration_artifact)
    artifact_concepts = calibration.get("concepts")
    calibration_concepts = load_calibration_concepts(calibration_data)
    if artifact_concepts is not None and [str(x) for x in artifact_concepts] != calibration_concepts:
        raise RuntimeError(
            "calibration artifact concepts do not match --calibration-data: "
            f"{calibration_artifact} vs {calibration_data}"
        )
    tokenizer = loaded.tokenizer
    artifact_layer = calibration.get("carrier_layer")
    if artifact_layer is not None and int(artifact_layer) != int(layer):
        raise RuntimeError(
            f"calibration carrier layer {artifact_layer} != model protocol layer {layer}"
        )

    grids = []
    for row in triples:
        piece_ids = [int(x) for x in row["tokens"][args.model]["piece_ids"]]
        ids = forced_input_ids(tokenizer, piece_ids)
        grids.append(
            {
                "input_ids": ids,
                "n_suffix": len(tokenizer(CALIBRATION_CARRIER_SUFFIX, add_special_tokens=False).input_ids),
            }
        )

    if not grids:
        raise RuntimeError(f"no valid forced splits for {args.test_data}")
    n_suffix = grids[0]["n_suffix"]
    artifact_suffix = calibration.get("n_suffix_tokens")
    if artifact_suffix is not None and int(artifact_suffix) != int(n_suffix):
        raise RuntimeError(
            f"calibration suffix length {artifact_suffix} != forced carrier suffix {n_suffix}"
        )
    print(
        f"[{args.model}] capturing n={len(grids)}/{requested_n} at L{layer} "
        f"calibration={calibration_data}",
        flush=True,
    )
    # Capture the post-concept carrier window used by the reconstruction rule.
    from bindlens.carry import capture_suffix

    cap_grids = [{"input_ids": g["input_ids"], "n_suffix_tokens": n_suffix} for g in grids]
    positions = capture_suffix(loaded, cap_grids, [layer], batch_size=args.batch_size, include_concept_final=True)[layer]
    capture_layers = [int(x) for x in calibration["capture_layers"]]
    means = calibration["means"][capture_layers.index(layer)].float()
    weights = pool_weights(calibration, layer)
    reconstructed = reconstruct_pooled_carrier(positions[:, 1:, :], means[1:, :], weights)

    canonical = [int(row["tokens"][args.model]["canonical_id"]) for row in triples]
    first_ids = [int(row["tokens"][args.model]["piece_ids"][0]) for row in triples]
    second_ids = [int(row["tokens"][args.model]["piece_ids"][1]) for row in triples]
    targets = token_directions(loaded, canonical, layer)
    first = token_directions(loaded, first_ids, layer)
    second = token_directions(loaded, second_ids, layer)
    gen = torch.Generator().manual_seed(int(args.random_seed))
    rand = torch.randn(len(triples), int(loaded.model.d_model), generator=gen)
    vectors = {
        "v_C": reconstructed,
        "v_first": first,
        "v_mean": (first + second) / 2,
        "v_rand": rand,
    }
    arms = {name: rank_against(vec, targets) for name, vec in vectors.items()}
    for metric in arms.values():
        metric["requested_n"] = requested_n
        metric["requested_top1"] = metric["top1_count"] / requested_n if requested_n else None
    held = [
        i
        for i, row in enumerate(triples)
        if str(row["concept"]).strip().casefold() not in calib_words
    ]
    held_idx = torch.tensor(held, dtype=torch.long)
    held_arms = {
        name: rank_against(vec[held_idx], targets[held_idx]) if held else None
        for name, vec in vectors.items()
    }
    payload = {
        "model": args.model,
        "n": len(triples),
        "n_evaluated": len(triples),
        "n_requested": requested_n,
        "n_split_failed": len(split_failures),
        "split_failures": split_failures,
        "test_data": str(args.test_data),
        "test_data_sha256": sha256(args.test_data),
        "split_ledger": str(ledger_path) if ledger_path else None,
        "min_piece_length": args.min_piece_length,
        "calibration_data": str(calibration_data),
        "calibration_data_sha256": sha256(calibration_data),
        "calibration_artifact": str(calibration_artifact),
        "layer": layer,
        "arms": arms,
        "heldout_vs_calib": held_arms,
        "n_heldout": len(held),
    }
    write_json(out_dir / "metrics.json", payload)
    lines = [
        f"# Fragment oracle {MODELS[args.model].display_name} n={len(triples)}/{requested_n} L{layer}",
        "",
        f"- evaluated exact forced splits: {len(triples)}/{requested_n} (failed: {len(split_failures)})",
        f"- calibration concepts: {len(calibration_concepts)}; held-out evaluated rows: {len(held)}",
        "| direction | cos_wrong | Top-1 |",
        "|---|---:|---:|",
    ]
    for name in ("v_C", "v_first", "v_mean", "v_rand"):
        a = arms[name]
        lines.append(
            f"| {name} | {a['cos_wrong']:.3f} | "
            f"{100 * a['top1']:.1f}% ({a['top1_count']}/{a['n']}; "
            f"requested {100 * a['requested_top1']:.1f}%) |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
