#!/usr/bin/env python3
"""Rank an explicitly gold-injected, fixed candidate set.

By default this is a conditional *oracle-candidate-set* evaluation for the
public paired clozes.  It retains the ordinary J-lens gate and fixed top-k proposal events,
but appends the annotated concept whenever that event did not propose it.
``--all-input-sides`` instead replays every fixed layer-local bank row,
including rows whose ordinary proposal list is empty, and injects the annotated
concept in every missing case.  Thus every evaluated event exposes precisely
the same candidate list to ``v_C`` and to the seeded random-direction control;
candidate availability is no longer a confound of the scoring comparison.

The default denominator is the set of directed sides that passed the fixed
first-token gate at at least one layer.  With ``--all-input-sides`` it is all
2,400 directed sides, and the gate is retained only as a diagnostic statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _boot  # noqa: F401
import torch

from bindlens.bind import Candidate, normalise, select_candidates
from bindlens.carry import (
    construct_directions_from_reader,
    load_positive_pool_artifact,
    validate_positive_pool_artifact,
)
from bindlens.io import (
    iter_sides,
    load_cloze_pairs,
    pair_bank_provenance,
    side_ids_digest,
    write_json,
)
from bindlens.metrics import rate, wilson
from bindlens.models import leading_space_ids, load_model
from bindlens.paths import CLOZES_PATH
from bindlens.protocol import (
    CARRIER_PREFIX,
    CARRIER_SUFFIX,
    GATE,
    MODELS,
    RANDOM_DIRECTION_SEED,
    RANKING_METHODS,
    SELECT_TOP_K,
)
from bindlens.read import (
    SOURCE_RESIDUAL_PROTOCOL,
    encode_cloze,
    input_ids_digest,
    load_candidate_vector_workspace,
    load_last_token_workspace,
    save_candidate_vector_workspace,
)
from bindlens.score import score_vocab_union_layer, unembed_row_norms


@dataclass(frozen=True)
class Event:
    """One oracle-scored fixed proposal event."""

    layer: int
    trial_id: str
    trial_index: int
    gold_key: str
    candidates: tuple[Candidate, ...]
    gold_was_proposed: bool
    gold_was_injected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--direction-chunk-size", type=int, default=256)
    parser.add_argument("--vocab-event-batch", type=int, default=16)
    parser.add_argument(
        "--clozes-path",
        type=Path,
        default=CLOZES_PATH,
        help="paired-cloze JSON list; default: data/clozes.json",
    )
    parser.add_argument(
        "--expected-sides",
        type=int,
        default=2400,
        help="assert the number of directed sides in the selected pair bank",
    )
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--reader-artifact", type=Path, required=True)
    parser.add_argument(
        "--candidate-vector-workspace",
        type=Path,
        default=None,
        help="Optional compatible v_C table; missing injected concepts are constructed.",
    )
    parser.add_argument(
        "--write-candidate-vectors",
        type=Path,
        default=None,
        help="Persist the complete oracle vector table for a later exact replay.",
    )
    parser.add_argument("--candidate-vector-max-gib", type=float, default=4.0)
    parser.add_argument(
        "--all-input-sides",
        action="store_true",
        help="Score every in-scope fixed bank event regardless of its ordinary first-token gate.",
    )
    parser.add_argument("--carrier-prefix", default=CARRIER_PREFIX)
    parser.add_argument("--carrier-suffix", default=CARRIER_SUFFIX)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _candidate_key_check(candidates: tuple[Candidate, ...], *, label: str) -> None:
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{label}: duplicate normalized candidate keys")


def _gold_candidate(tokenizer: Any, concept: str) -> Candidate:
    ids = tuple(int(token) for token in leading_space_ids(tokenizer, concept))
    key = normalise(concept)
    if not ids or not key:
        raise RuntimeError(f"cannot construct a gold candidate for {concept!r}")
    return Candidate(ids=ids, key=key, surface=str(concept), score=float("inf"))


def load_gold_injected_events(
    path: Path,
    sides: list[dict[str, Any]],
    *,
    model: str,
    tokenizer: Any,
    gate: int,
    top_k: int,
    all_input_sides: bool,
) -> tuple[dict[int, list[Event]], set[str], set[str], set[str], dict[str, Any]]:
    """Replay a fixed bank and append gold exactly when it is absent.

    We do not regenerate candidates or change their proposer order.  Appending
    a gold vector is the sole oracle operation, and is recorded per event.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    side_index: dict[str, int] = {}
    for index, side in enumerate(sides):
        trial_id = str(side["trial_id"])
        if trial_id in side_index:
            raise RuntimeError(f"duplicate trial id {trial_id}")
        side_index[trial_id] = index

    events: dict[int, list[Event]] = {}
    eligible_trials: set[str] = set()
    gate_eligible_trials: set[str] = set()
    originally_proposed_trials: set[str] = set()
    surfaces: set[str] = set()
    seen: set[tuple[int, str]] = set()
    digest = hashlib.sha256()
    stats: dict[str, Any] = {
        "path": str(path),
        "model": str(model),
        "gate": int(gate),
        "select_top_k": int(top_k),
        "all_input_sides": bool(all_input_sides),
        "rows": 0,
        "rows_in_scope": 0,
        "rows_outside_scope": 0,
        "eligible_rows": 0,
        "scored_rows": 0,
        "oracle_events": 0,
        "candidate_instances_before_injection": 0,
        "candidate_instances_after_injection": 0,
        "gold_proposed_events": 0,
        "gold_injected_events": 0,
    }
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            stats["rows"] += 1
            if str(row.get("model", "")) != str(model):
                raise RuntimeError(f"{path}: model mismatch in candidate bank")
            trial_id = str(row.get("trial_id", ""))
            index = side_index.get(trial_id)
            if index is None:
                stats["rows_outside_scope"] += 1
                continue
            stats["rows_in_scope"] += 1
            side = sides[index]
            gold_key = normalise(str(side["concept"]))
            if normalise(str(row.get("gold", ""))) != gold_key:
                raise RuntimeError(f"{path}: gold mismatch for {trial_id}")
            layer = int(row["layer"])
            event_key = (layer, trial_id)
            if event_key in seen:
                raise RuntimeError(f"{path}: duplicate event L{layer} {trial_id}")
            seen.add(event_key)
            gate_pass = int(row.get("first_rank", 10**9)) <= int(gate)
            if gate_pass:
                gate_eligible_trials.add(trial_id)
                stats["eligible_rows"] += 1
            if not all_input_sides and not gate_pass:
                continue
            eligible_trials.add(trial_id)
            selected = select_candidates(list(row.get("candidates") or []), top_k=int(top_k))
            _candidate_key_check(tuple(selected), label=f"{trial_id} L{layer}")
            gold_was_proposed = any(candidate.key == gold_key for candidate in selected)
            if gold_was_proposed:
                candidates = tuple(selected)
                originally_proposed_trials.add(trial_id)
                stats["gold_proposed_events"] += 1
            else:
                candidates = tuple([*selected, _gold_candidate(tokenizer, str(side["concept"]))])
                stats["gold_injected_events"] += 1
            _candidate_key_check(candidates, label=f"oracle {trial_id} L{layer}")
            stats["candidate_instances_before_injection"] += len(selected)
            stats["candidate_instances_after_injection"] += len(candidates)
            surfaces.update(candidate.surface for candidate in candidates)
            events.setdefault(layer, []).append(
                Event(
                    layer=layer,
                    trial_id=trial_id,
                    trial_index=index,
                    gold_key=gold_key,
                    candidates=candidates,
                    gold_was_proposed=gold_was_proposed,
                    gold_was_injected=not gold_was_proposed,
                )
            )
            stats["oracle_events"] += 1
            stats["scored_rows"] += 1
    stats["sha256"] = digest.hexdigest()
    stats["eligible_trial_ids"] = len(eligible_trials)
    stats["gate_eligible_trial_ids"] = len(gate_eligible_trials)
    stats["gold_proposed_trial_ids"] = len(originally_proposed_trials)
    stats["unique_surfaces_after_injection"] = len(surfaces)
    if stats["rows"] and not stats["rows_in_scope"]:
        raise RuntimeError(
            f"{path}: no candidate-bank rows match this paired-cloze trial-ID scheme; "
            "regenerate the bank for data/clozes.json"
        )
    if not events or not eligible_trials:
        raise RuntimeError("no oracle events in scope")
    if all_input_sides:
        missing_trials = sorted(set(side_index).difference(eligible_trials))
        if missing_trials:
            raise RuntimeError(f"full-input oracle lacks {len(missing_trials)} directed sides, e.g. {missing_trials[:3]}")
    return events, eligible_trials, originally_proposed_trials, surfaces, stats


def random_directions(keys: list[str], dimension: int, *, model: str, seed: int) -> torch.Tensor:
    effective = int(hashlib.sha256(f"{model}:{seed}".encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(effective)
    values = torch.randn((len(keys), int(dimension)), generator=generator, dtype=torch.float32)
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def best_rank(layer_ranks: dict[str, dict[str, dict[str, int | None]]], trial_id: str, method: str) -> int | None:
    values = [
        int(row[method])
        for row in layer_ranks.get(trial_id, {}).values()
        if row.get(method) is not None
    ]
    return min(values) if values else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if int(args.expected_sides) <= 0:
        raise ValueError("--expected-sides must be positive")
    if int(args.batch_size) <= 0 or int(args.direction_chunk_size) <= 0 or int(args.vocab_event_batch) <= 0:
        raise ValueError("batch and chunk sizes must be positive")
    if float(args.candidate_vector_max_gib) <= 0.0:
        raise ValueError("--candidate-vector-max-gib must be positive")
    for path in (args.clozes_path, args.candidate_bank, args.source_workspace, args.reader_artifact):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.candidate_vector_workspace is not None and not args.candidate_vector_workspace.is_file():
        raise FileNotFoundError(args.candidate_vector_workspace)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.out_dir}")
    if args.write_candidate_vectors is not None and args.write_candidate_vectors.exists():
        raise FileExistsError(f"refusing to overwrite {args.write_candidate_vectors}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_cloze_pairs(args.clozes_path)
    input_provenance = pair_bank_provenance(pairs, path=args.clozes_path)
    input_provenance["input_schema"] = "concept_lens_paired_clozes"
    sides = iter_sides(pairs)
    if len(sides) != int(args.expected_sides):
        raise RuntimeError(f"expected {args.expected_sides} directed sides, found {len(sides)}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    loaded = load_model(args.model, device)
    encoded = [encode_cloze(loaded.tokenizer, str(side["prompt"])) for side in sides]
    events, eligible_trials, originally_proposed_trials, surfaces, bank_stats = load_gold_injected_events(
        args.candidate_bank,
        sides,
        model=args.model,
        tokenizer=loaded.tokenizer,
        gate=int(GATE),
        top_k=int(SELECT_TOP_K),
        all_input_sides=bool(args.all_input_sides),
    )
    layers = sorted(events)
    lens_layers = {int(layer) for layer in loaded.lens.source_layers}
    missing_lens_layers = [layer for layer in layers if layer not in lens_layers]
    if missing_lens_layers:
        raise RuntimeError(f"candidate bank has layers outside J-lens: {missing_lens_layers}")
    scope_label = "all input sides" if args.all_input_sides else "eligible sides"
    print(
        f"[{args.model} oracle] n={len(sides)}, {scope_label}={len(eligible_trials)}, "
        f"events={bank_stats['oracle_events']}, injected events={bank_stats['gold_injected_events']}, "
        f"surfaces={len(surfaces)}",
        flush=True,
    )

    reader, reader_metadata = load_positive_pool_artifact(args.reader_artifact)
    validate_positive_pool_artifact(
        loaded,
        reader_metadata,
        reader,
        prefix=args.carrier_prefix,
        suffix=args.carrier_suffix,
        layer=int(reader_metadata["carrier_layer"]),
    )
    states = load_last_token_workspace(
        args.source_workspace,
        input_ids=encoded,
        required_layers=layers,
        expected_metadata={
            "model": args.model,
            "source_residual_protocol": SOURCE_RESIDUAL_PROTOCOL,
            "n_layers": int(loaded.model.n_layers),
            "d_model": int(loaded.model.d_model),
        },
    )

    vectors_by_surface: dict[str, torch.Tensor] = {}
    workspace_provenance: dict[str, Any] = {"mode": "constructed"}
    if args.candidate_vector_workspace is not None:
        workspace = load_candidate_vector_workspace(
            args.candidate_vector_workspace,
            input_ids=encoded,
            expected_model=args.model,
        )
        if Path(workspace["candidate_bank"]).resolve() != args.candidate_bank.resolve():
            raise RuntimeError("candidate-vector workspace points to a different candidate bank")
        if dict(workspace["reader_metadata"]) != dict(reader_metadata):
            raise RuntimeError("candidate-vector workspace has incompatible reader metadata")
        vectors_by_surface = {
            str(surface): workspace["vectors"][index]
            for index, surface in enumerate(workspace["surfaces"])
            if str(surface) in surfaces
        }
        workspace_provenance = {
            "mode": "reused_plus_constructed_missing",
            "path": str(args.candidate_vector_workspace),
            "stored_surfaces": len(workspace["surfaces"]),
            "reused_surfaces": len(vectors_by_surface),
        }
    missing_surfaces = sorted(surfaces.difference(vectors_by_surface))
    if missing_surfaces:
        print(f"  constructing {len(missing_surfaces)} missing v_C vectors", flush=True)
        constructed, failed = construct_directions_from_reader(
            loaded,
            missing_surfaces,
            reader,
            layer=int(reader_metadata["carrier_layer"]),
            batch_size=int(args.batch_size),
            reference_suffix=[int(token) for token in reader_metadata["suffix_ids"]],
            prefix=str(reader_metadata["carrier_prefix"]),
            suffix=str(reader_metadata["carrier_suffix"]),
            concept_chunk_size=int(args.direction_chunk_size),
        )
        if failed or set(constructed) != set(missing_surfaces):
            raise RuntimeError(f"failed to construct {len(failed)} oracle candidate vectors, e.g. {failed[:3]}")
        vectors_by_surface.update(constructed)
    ordered_surfaces = sorted(surfaces)
    if set(vectors_by_surface) != set(ordered_surfaces):
        extra = sorted(set(vectors_by_surface).difference(ordered_surfaces))
        if extra:
            vectors_by_surface = {surface: vectors_by_surface[surface] for surface in ordered_surfaces}
        if set(vectors_by_surface) != set(ordered_surfaces):
            raise RuntimeError("oracle vector table does not cover its candidates")
    vc_vectors = torch.stack([vectors_by_surface[surface] for surface in ordered_surfaces], dim=0).float()
    vc_index = {surface: index for index, surface in enumerate(ordered_surfaces)}
    workspace_provenance["constructed_surfaces"] = len(missing_surfaces)
    workspace_provenance["total_surfaces"] = len(ordered_surfaces)

    if args.write_candidate_vectors is not None:
        saved_vectors = save_candidate_vector_workspace(
            args.write_candidate_vectors,
            model=args.model,
            candidate_bank=args.candidate_bank,
            candidate_bank_stats=bank_stats,
            reader_metadata=reader_metadata,
            input_ids=encoded,
            surfaces=ordered_surfaces,
            vectors=vc_vectors,
            storage_dtype=torch.float16,
            max_bytes=int(float(args.candidate_vector_max_gib) * 2**30),
        )
        workspace_provenance["written_oracle_workspace"] = saved_vectors

    keys = sorted({candidate.key for rows in events.values() for event in rows for candidate in event.candidates})
    random_vectors = random_directions(
        keys,
        int(loaded.model.d_model),
        model=args.model,
        seed=int(RANDOM_DIRECTION_SEED),
    )
    random_key_index = {key: index for index, key in enumerate(keys)}
    wu_norms = unembed_row_norms(loaded.unembed)
    ranks: dict[str, dict[str, dict[str, dict[str, int | None]]]] = {"strict": {}, "worst": {}}
    event_rows: list[dict[str, Any]] = []
    for layer in layers:
        layer_ranks = score_vocab_union_layer(
            loaded=loaded,
            layer=int(layer),
            carrier_layer=int(reader_metadata["carrier_layer"]),
            records=events[layer],
            source_states=states[layer],
            vc_vectors=vc_vectors,
            vc_index=vc_index,
            random_vectors=random_vectors,
            random_key_index=random_key_index,
            wu_norms=wu_norms,
            vocab_event_batch=int(args.vocab_event_batch),
        )
        for tie in ("strict", "worst"):
            for method in RANKING_METHODS:
                for trial_id, value in layer_ranks[tie][method].items():
                    ranks[tie].setdefault(trial_id, {}).setdefault(str(layer), {})[method] = value
        for event in events[layer]:
            event_rows.append(
                {
                    "trial_id": event.trial_id,
                    "layer": int(layer),
                    "gold_key": event.gold_key,
                    "candidate_count": len(event.candidates),
                    "gold_was_proposed": event.gold_was_proposed,
                    "gold_was_injected": event.gold_was_injected,
                    "strict_ranks": {method: layer_ranks["strict"][method][event.trial_id] for method in RANKING_METHODS},
                    "worst_ranks": {method: layer_ranks["worst"][method][event.trial_id] for method in RANKING_METHODS},
                }
            )
        print(f"  scored L{layer}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    side_by_id = {str(side["trial_id"]): side for side in sides}
    directions: dict[str, Any] = {}
    per_side: list[dict[str, Any]] = []
    for tie in ("strict", "worst"):
        directions[tie] = {}
        for method in RANKING_METHODS:
            best = {trial_id: best_rank(ranks[tie], trial_id, method) for trial_id in eligible_trials}
            unresolved = [trial_id for trial_id, value in best.items() if value is None]
            if unresolved:
                raise RuntimeError(f"{tie}/{method}: {len(unresolved)} eligible sides lack ranks")
            values = [int(value) for value in best.values() if value is not None]
            directions[tie][method] = {
                "n_ranked": len(values),
                "pass_at_1": rate(sum(value <= 1 for value in values), len(eligible_trials)),
                "pass_at_10": rate(sum(value <= 10 for value in values), len(eligible_trials)),
                "pass_at_100": rate(sum(value <= 100 for value in values), len(eligible_trials)),
                "pass_at_1000": rate(sum(value <= 1000 for value in values), len(eligible_trials)),
                "wilson10": list(wilson(sum(value <= 10 for value in values), len(eligible_trials))),
            }
    for trial_id in sorted(eligible_trials):
        side = side_by_id[trial_id]
        per_side.append(
            {
                "trial_id": trial_id,
                "pair_id": str(side["pair_id"]),
                "source_side": int(side["source_side"]),
                "concept": str(side["concept"]),
                "answer": str(side["answer"]),
                "strict": {method: best_rank(ranks["strict"], trial_id, method) for method in RANKING_METHODS},
                "worst": {method: best_rank(ranks["worst"], trial_id, method) for method in RANKING_METHODS},
            }
        )

    oracle_n = len(eligible_trials)
    payload = {
        "schema_version": 1,
        "kind": (
            "concept_lens_full_input_gold_injected_candidate_ranking"
            if args.all_input_sides
            else "concept_lens_oracle_gold_injected_candidate_ranking"
        ),
        "model": args.model,
        "input": input_provenance,
        "input_side_ids_sha256": side_ids_digest(sides),
        "protocol": {
            "oracle_candidate_set": (
                "every fixed layer-local top-25 candidate list (including empty ungated rows) plus explicit gold injection iff absent"
                if args.all_input_sides
                else "fixed gate-10/top-25 layer-local candidates plus explicit gold injection iff absent"
            ),
            "candidate_set_shared_across_methods": True,
            "gold_injection": True,
            "all_input_sides": bool(args.all_input_sides),
            "gate_applied_to_evaluation": not bool(args.all_input_sides),
            "evaluation_denominator": (
                "all directed trial IDs, each counted once at its best evaluated layer"
                if args.all_input_sides
                else "gate-eligible directed trial IDs, each counted once at its best evaluated layer"
            ),
            "primary_denominator": oracle_n,
            "input_sides": len(sides),
            "gate": int(GATE),
            "select_top_k": int(SELECT_TOP_K),
            "score": "unit(d) · final_norm(J h) * median||W_U|| for every method and every vocab row",
            "competitor_pool": "full_vocabulary_union_layer_local_gold_injected_candidates",
            "candidate_bank": bank_stats,
            "reader_artifact": str(args.reader_artifact),
            "reader_metadata": reader_metadata,
            "source_workspace": str(args.source_workspace),
            "candidate_vectors": workspace_provenance,
            "random_direction_seed": int(RANDOM_DIRECTION_SEED),
        },
        "funnel": {
            "gate_eligible": rate(int(bank_stats["gate_eligible_trial_ids"]), len(sides)),
            "target_available_before_injection": rate(len(originally_proposed_trials), oracle_n),
            "target_available_after_injection": rate(oracle_n, oracle_n),
            "gold_injected_event_fraction": rate(bank_stats["gold_injected_events"], bank_stats["oracle_events"]),
        },
        "directions": directions,
    }
    write_json(args.out_dir / "metrics.json", payload)
    write_jsonl(args.out_dir / "per_event.jsonl", event_rows)
    write_json(args.out_dir / "per_side.json", per_side)

    def show(metric: dict[str, Any]) -> str:
        return f"{100.0 * float(metric['rate']):.2f}% ({metric['k']}/{metric['n']})"

    report = [
        f"# Gold-injected Oracle candidate ranking: {args.model}",
        "",
        (
            f"- all-input denominator: {oracle_n}/{len(sides)} directed sides"
            if args.all_input_sides
            else f"- eligible denominator: {oracle_n}/{len(sides)} directed sides"
        ),
        (
            f"- gate-eligible diagnostic: {int(bank_stats['gate_eligible_trial_ids'])}/{len(sides)} directed sides"
            if args.all_input_sides
            else ""
        ),
        f"- target available before injection: {len(originally_proposed_trials)}/{oracle_n}",
        "- target available after injection: 100% by construction",
        "",
        "| method | Rank@1 | Rank@10 |",
        "|---|---:|---:|",
    ]
    for method in RANKING_METHODS:
        row = directions["strict"][method]
        report.append(f"| {method} | {show(row['pass_at_1'])} | {show(row['pass_at_10'])} |")
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    sys.exit(main())
