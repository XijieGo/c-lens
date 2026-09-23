#!/usr/bin/env python3
"""Causal install (parameter-free swap) or directional ablation.

E2E and Swap load the same stored centered-positive reader.  The install
path uses exact Cholesky transport, no trailing space, delete last 12 tokens,
model-specific write bands/windows and format bans, over every directed side
of the selected paired-cloze bank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _boot  # noqa: F401
import torch

from bindlens.carry import (
    construct_directions_from_reader,
    load_positive_pool_artifact,
    validate_positive_pool_artifact,
)
from bindlens.intervene import (
    LiveProjectOut,
    delete_start,
    format_ban_directions,
    live_swap_logits,
    occupancy,
    swap_arm_spec,
)
from bindlens.io import (
    iter_sides,
    load_cloze_pairs,
    pair_bank_provenance,
    side_ids_digest,
    write_json,
)
from bindlens.metrics import wilson
from bindlens.models import last_residual_logits, leading_space_ids, load_model, token_rank
from bindlens.paths import CLOZES_PATH, RESULTS_DIR, positive_pool_reader_path
from bindlens.protocol import (
    CARRIER_PREFIX,
    CARRIER_SUFFIX,
    MODELS,
    RANDOM_DIRECTION_SEED,
    SWAP_ARMS,
    SWAP_DELETE_TOKENS,
    SWAP_TRANSPORT,
    swap_protocol,
)
from bindlens.read import capture_last, encode_swap_prompt, token_directions
from bindlens.score import transport_cholesky


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--mode", choices=("install", "ablation"), default="install")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=None,
        help="paired-cloze JSON list; default: data/clozes.json",
    )
    parser.add_argument(
        "--expected-sides",
        type=int,
        default=2400,
        help="assert the full directed-side denominator before applying --limit; 0 disables",
    )
    parser.add_argument("--band", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--gpu-mem-fraction", type=float, default=None)
    parser.add_argument(
        "--reader-artifact",
        type=Path,
        default=None,
        help="override the stored centered-positive reader shared with E2E",
    )
    parser.add_argument("--carrier-prefix", default=CARRIER_PREFIX, help="expected reader carrier prefix")
    parser.add_argument("--carrier-suffix", default=CARRIER_SUFFIX, help="expected reader carrier suffix")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=SWAP_ARMS,
        default=None,
        help="install arms to evaluate together; default: full",
    )
    parser.add_argument("--direction-chunk-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def answer_id(tokenizer, answer: str) -> int:
    return leading_space_ids(tokenizer, answer)[0]


def checked_logit_summary(values: torch.Tensor, *, label: str) -> dict[str, float]:
    """Fail loudly if a model produces degenerate readout logits."""
    logits = values.detach().float().reshape(-1)
    if logits.numel() < 2 or not torch.isfinite(logits).all():
        raise RuntimeError(f"{label}: non-finite or malformed logits")
    span = float((logits.max() - logits.min()).item())
    std = float(logits.std().item())
    if span <= 1e-7 or std <= 1e-8:
        raise RuntimeError(f"{label}: degenerate logits (span={span:.3g}, std={std:.3g})")
    return {"span": span, "std": std}


def requested_arms(args: argparse.Namespace) -> tuple[str, ...]:
    if args.mode == "ablation":
        if args.arms is not None:
            raise ValueError("--arms is only valid with --mode install")
        return ()
    arms = tuple(dict.fromkeys(args.arms or ("full",)))
    if not arms:
        raise ValueError("install requires at least one arm")
    return arms


def resolve_input_sides(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Resolve the public paired-cloze bank into an ordered directed denominator."""
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.expected_sides < 0:
        raise ValueError("--expected-sides must be non-negative")
    source = args.pairs_path or CLOZES_PATH
    pairs = load_cloze_pairs(source)
    provenance = pair_bank_provenance(pairs, path=source)
    provenance["input_schema"] = "concept_lens_paired_clozes"
    full_sides = iter_sides(pairs)
    if args.expected_sides and len(full_sides) != int(args.expected_sides):
        raise RuntimeError(
            f"input bank has {len(full_sides)} directed sides, expected {args.expected_sides}"
        )
    if args.limit > len(full_sides):
        raise ValueError(f"--limit {args.limit} exceeds input size {len(full_sides)}")
    sides = full_sides[: int(args.limit)] if args.limit else full_sides
    evaluation_ids = [str(side["trial_id"]) for side in full_sides]
    provenance.update(
        {
            "full_n_sides": len(full_sides),
            "run_n_sides": len(sides),
            "run_side_ids_sha256": side_ids_digest(sides),
            "expected_sides": int(args.expected_sides) if args.expected_sides else None,
        }
    )
    return sides, evaluation_ids, provenance


def main() -> None:
    args = parse_args()
    arms = requested_arms(args)
    spec = MODELS[args.model]
    reader_artifact = args.reader_artifact or positive_pool_reader_path(args.model)
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.mode == "install" and arms != ("full",):
        out_dir = RESULTS_DIR / "swap" / "install_arms" / "-".join(arms) / args.model
    else:
        out_dir = RESULTS_DIR / "swap" / args.mode / args.model
    if args.pairs_path is not None and args.out_dir is None:
        raise ValueError("an alternate --pairs-path requires --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    sides, evaluation_ids, input_provenance = resolve_input_sides(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.gpu_mem_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), device=device)
    loaded = load_model(args.model, device)
    carrier_layer = spec.carrier_layer
    band = tuple(args.band) if args.band else spec.swap_band
    layers = list(range(int(band[0]), int(band[1]) + 1))
    n_layers = int(loaded.model.n_layers)
    if carrier_layer >= n_layers or not layers or min(layers) < 0 or max(layers) >= n_layers:
        raise RuntimeError(
            f"invalid carrier/write layers for {args.model}: carrier={carrier_layer}, band={layers}, n_layers={n_layers}"
        )
    if loaded.unembed.ndim != 2 or int(loaded.unembed.shape[1]) != int(loaded.model.d_model):
        raise RuntimeError(
            f"unembed shape {tuple(loaded.unembed.shape)} does not match d_model={loaded.model.d_model}"
        )
    architecture = {
        "configured_model_class": spec.model_class,
        "hf_class": type(loaded.hf).__name__,
        "wrapped_class": type(loaded.model).__name__,
        "n_layers": n_layers,
        "d_model": int(loaded.model.d_model),
        "unembed_shape": [int(value) for value in loaded.unembed.shape],
        "lens_source_layers": [int(layer) for layer in loaded.lens.source_layers],
    }
    if spec.model_class == "image_text_to_text" and not any(
        name in architecture["hf_class"].casefold() for name in ("gemma", "qwen")
    ):
        raise RuntimeError(
            f"conditional-generation architecture mismatch: loaded {architecture['hf_class']}"
        )
    protocol = swap_protocol(spec)
    protocol["band"] = list(band)
    # Primary Swap evaluates every directed side in the selected pair bank.
    protocol["primary_n"] = len(evaluation_ids)
    protocol["arms"] = list(arms) if arms else ["ablation_projection"]
    concepts = sorted({s["concept"] for s in sides} | {s["partner_concept"] for s in sides})
    reader, reader_metadata = load_positive_pool_artifact(reader_artifact)
    validate_positive_pool_artifact(
        loaded,
        reader_metadata,
        reader,
        prefix=args.carrier_prefix,
        suffix=args.carrier_suffix,
        layer=carrier_layer,
    )
    vectors, failed = construct_directions_from_reader(
        loaded,
        concepts,
        reader,
        layer=carrier_layer,
        batch_size=args.batch_size,
        reference_suffix=[int(token) for token in reader_metadata["suffix_ids"]],
        prefix=str(reader_metadata["carrier_prefix"]),
        suffix=str(reader_metadata["carrier_suffix"]),
        concept_chunk_size=args.direction_chunk_size,
    )
    reader_source = str(reader_artifact)
    protocol["carrier_prefix"] = str(reader_metadata["carrier_prefix"])
    protocol["suffix"] = str(reader_metadata["carrier_suffix"])
    protocol["pooling"] = str(reader_metadata["pooling"])
    missing = [name for name in concepts if name not in vectors]
    if missing and not args.limit:
        raise RuntimeError(f"missing swap directions for {missing[:5]}")
    print(
        f"[{args.model} {args.mode}] n={len(sides)}/{len(evaluation_ids)} arms={list(arms)} "
        f"band={band[0]}-{band[1]} add_last_n={spec.add_last_n} "
        f"delete={SWAP_DELETE_TOKENS} transport={SWAP_TRANSPORT} reader={reader_source}",
        flush=True,
    )

    names = [name for name in concepts if name in vectors]
    stacked = torch.stack([vectors[name] for name in names], dim=0)
    moved: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        transported = transport_cholesky(
            stacked, loaded, carrier_layer=carrier_layer, source_layer=layer
        )
        moved[layer] = {names[i]: transported[i] for i in range(len(names))}
        del transported
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    first_moved: dict[int, dict[str, torch.Tensor]] = {}
    if "first" in arms:
        first_ids = [leading_space_ids(loaded.tokenizer, name)[0] for name in names]
        first_stacked = token_directions(loaded, first_ids, carrier_layer)
        for layer in layers:
            transported = transport_cholesky(
                first_stacked, loaded, carrier_layer=carrier_layer, source_layer=layer
            )
            first_moved[layer] = {names[i]: transported[i] for i in range(len(names))}
            del transported
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del first_stacked

    source_ids = [encode_swap_prompt(loaded.tokenizer, s["prompt"]) for s in sides]
    partner_ids = [encode_swap_prompt(loaded.tokenizer, s["partner_prompt"]) for s in sides]
    partner_hidden = capture_last(loaded, partner_ids, layers, batch_size=args.batch_size)
    ban_directions, ban_ids = format_ban_directions(loaded, spec.ban_surfaces)
    print(f"  format ban {list(spec.ban_surfaces)} -> {ban_ids}", flush=True)

    n_ok = 0
    hits1 = {arm: 0 for arm in arms}
    hits10 = {arm: 0 for arm in arms}
    source_hits10 = {arm: 0 for arm in arms}
    complete_swaps10 = {arm: 0 for arm in arms}
    rows = []
    clean_logit_check: dict[str, float] | None = None
    edited_logit_check: dict[str, dict[str, float]] = {}
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(RANDOM_DIRECTION_SEED)
    for index, side in enumerate(sides):
        if side["concept"] not in vectors or side["partner_concept"] not in vectors:
            continue
        src_ans = answer_id(loaded.tokenizer, side["answer"])
        tgt_ans = answer_id(loaded.tokenizer, side["partner_answer"])
        clean = last_residual_logits(loaded, source_ids[index])
        if clean_logit_check is None:
            clean_logit_check = checked_logit_summary(clean, label="clean swap logits")
        src_rank = token_rank(clean, src_ans)
        tgt_rank = token_rank(clean, tgt_ans)
        n_ok += 1
        carrier_deletes = {layer: moved[layer][side["concept"]] for layer in layers}
        carrier_adds = {layer: moved[layer][side["partner_concept"]] for layer in layers}
        seq_len = len(source_ids[index])

        if args.mode == "ablation":
            hook = LiveProjectOut(loaded.model.layers, carrier_deletes, pos_start=1)
            with hook:
                edited = last_residual_logits(loaded, source_ids[index])
            if "ablation" not in edited_logit_check:
                edited_logit_check["ablation"] = checked_logit_summary(
                    edited, label="ablation swap logits"
                )
            e_rank = token_rank(edited, src_ans)
            rows.append(
                {
                    "trial_id": side["trial_id"],
                    "clean_src": src_rank,
                    "clean_tgt": tgt_rank,
                    "edited_rank": e_rank,
                }
            )
            print(f"  [{n_ok}] {side['concept']} -> edited_rank={e_rank}", flush=True)
            continue

        random_add = None
        if "random" in arms:
            random_add = torch.randn(int(loaded.model.d_model), generator=random_generator)
        arm_rows: dict[str, dict[str, object]] = {}
        for arm in arms:
            arm_spec = swap_arm_spec(arm)
            if arm_spec["family"] == "first":
                deletes = {layer: first_moved[layer][side["concept"]] for layer in layers}
                adds = {layer: first_moved[layer][side["partner_concept"]] for layer in layers}
                betas = {
                    layer: occupancy(partner_hidden[layer][index], adds[layer]) for layer in layers
                }
            else:
                deletes = carrier_deletes
                adds = (
                    {layer: random_add for layer in layers}
                    if arm_spec["add"] == "random" and random_add is not None
                    else carrier_adds
                )
                betas = {
                    layer: occupancy(partner_hidden[layer][index], carrier_adds[layer]) for layer in layers
                }
            edited = live_swap_logits(
                loaded,
                source_ids[index],
                deletes=deletes,
                adds=adds,
                betas=betas,
                pos_start=delete_start(seq_len),
                do_delete=bool(arm_spec["do_delete"]),
                do_add=bool(arm_spec["do_add"]),
                add_last_n=spec.add_last_n,
                ban_directions=ban_directions,
            )
            if arm not in edited_logit_check:
                edited_logit_check[arm] = checked_logit_summary(
                    edited, label=f"{arm} edited swap logits"
                )
            e_rank = token_rank(edited, tgt_ans)
            edited_src_rank = token_rank(edited, src_ans)
            hits1[arm] += int(e_rank == 1)
            hits10[arm] += int(e_rank <= 10)
            source_hits10[arm] += int(edited_src_rank <= 10)
            complete_swaps10[arm] += int(e_rank <= 10 and edited_src_rank > 10)
            beta_values = sorted(float(value) for value in betas.values())
            arm_rows[arm] = {
                "edited_rank": e_rank,
                "edited_src_rank": edited_src_rank,
                "median_beta": beta_values[len(beta_values) // 2],
            }
        row = {
            "trial_id": side["trial_id"],
            "clean_src": src_rank,
            "clean_tgt": tgt_rank,
            "arms": arm_rows,
        }
        if "full" in arm_rows:
            row["edited_rank"] = arm_rows["full"]["edited_rank"]
            row["edited_src_rank"] = arm_rows["full"]["edited_src_rank"]
        rows.append(row)
        ranks = ", ".join(f"{arm}={arm_rows[arm]['edited_rank']}" for arm in arms)
        print(f"  [{n_ok}] {side['concept']} -> {ranks}", flush=True)

    if not args.limit and n_ok != len(evaluation_ids):
        raise RuntimeError(
            f"full Swap denominator changed: evaluated {n_ok}, active pair-bank sides {len(evaluation_ids)}"
        )

    arm_metrics = {
        arm: {
            "succ1": hits1[arm] / n_ok if n_ok else None,
            "succ10": hits10[arm] / n_ok if n_ok else None,
            "source10": source_hits10[arm] / n_ok if n_ok else None,
            "swap10": complete_swaps10[arm] / n_ok if n_ok else None,
            "wilson1": list(wilson(hits1[arm], n_ok)) if n_ok else None,
            "wilson10": list(wilson(hits10[arm], n_ok)) if n_ok else None,
            "wilson_source10": list(wilson(source_hits10[arm], n_ok)) if n_ok else None,
            "wilson_swap10": list(wilson(complete_swaps10[arm], n_ok)) if n_ok else None,
        }
        for arm in arms
    }
    full_metrics = arm_metrics.get("full")
    payload = {
        "model": args.model,
        "mode": args.mode,
        "protocol": {
            **protocol,
            "reader_source": reader_source,
            "ban_token_ids": ban_ids,
            "primary_subset": "all_input_pair_bank_directed_sides",
            "n_input": len(evaluation_ids),
            "input": input_provenance,
            "logits": "last_residual_unembed",
            "random_direction_seed": RANDOM_DIRECTION_SEED if "random" in arms else None,
            "reader_metadata": reader_metadata,
        },
        "band": list(band),
        "n": n_ok,
        "arms": arm_metrics if args.mode == "install" else None,
        "succ1": None if full_metrics is None else full_metrics["succ1"],
        "succ10": None if full_metrics is None else full_metrics["succ10"],
        "wilson1": None if full_metrics is None else full_metrics["wilson1"],
        "failed_concepts": failed,
        "architecture": architecture,
        "logit_sanity": {"clean": clean_logit_check, "edited": edited_logit_check},
    }
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "rows.json", rows)
    if args.mode == "install":
        labels = {
            "full": "Full (v_C)",
            "delete": "Delete only",
            "add": "Add only",
            "first": "First token",
            "random": "Random add",
        }
        lines = [
            f"# Swap {spec.display_name} L{band[0]}-{band[1]}",
            "",
            f"all pair-bank sides n={n_ok}/{len(evaluation_ids)}; arms={', '.join(arms)}",
            "",
            "| arm | partner@1 | partner@10 | source retained@10 | complete swap@10 |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in arms:
            metric = arm_metrics[arm]
            lines.append(
                f"| {labels[arm]} | {100 * float(metric['succ1'] or 0):.1f}% | "
                f"{100 * float(metric['succ10'] or 0):.1f}% | "
                f"{100 * float(metric['source10'] or 0):.1f}% | "
                f"{100 * float(metric['swap10'] or 0):.1f}% |"
            )
        lines.extend(
            [
                "",
                f"add_last_n={spec.add_last_n}  trailing_space={protocol['trailing_space']}  "
                f"delete={SWAP_DELETE_TOKENS}  ban={list(spec.ban_surfaces)}",
                f"transport={SWAP_TRANSPORT}  reader={reader_source}",
            ]
        )
        report = "\n".join(lines) + "\n"
    else:
        report = (
            f"# Ablation {spec.display_name} L{band[0]}-{band[1]}\n\n"
            f"all pair-bank sides n={n_ok}/{len(evaluation_ids)}\n"
        )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    sys.exit(main())
