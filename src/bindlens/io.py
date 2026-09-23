"""Load Concept Lens datasets and write compact JSON provenance artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bindlens.paths import (
    ACTIVE_CALIBRATION_PATH,
    CLOZES_PATH,
    RECONSTRUCTION_TEST_CONCEPTS_PATH,
)


def load_json(path: Path | str) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, value: Any) -> None:
    """Atomically replace a small, human-readable JSON artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_cloze_pairs(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the public paired-cloze bank in its stored order."""
    source = Path(path) if path is not None else CLOZES_PATH
    rows = load_json(source)
    if not isinstance(rows, list):
        raise ValueError(f"{source}: expected a JSON list of paired clozes")
    return rows


def _required_text(row: dict[str, Any], field: str, *, label: str) -> str:
    """Return one required surface without changing its tokenizer-relevant text."""
    value = row.get(field)
    text = str(value) if value is not None else ""
    if not text.strip():
        raise ValueError(f"{label}: missing {field}")
    return text


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def side_ids_digest(sides: list[dict[str, Any]]) -> str:
    """Return a stable fingerprint of an ordered directed-side denominator."""
    return _json_sha256([str(side["trial_id"]) for side in sides])


def iter_sides(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand an ordered paired-cloze bank into two directed sides per pair."""
    sides: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"pair {index}: expected an object")
        pair_id = _required_text(pair, "id", label=f"pair {index}")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair ID: {pair_id}")
        pair_ids.add(pair_id)
        for source, target in ((1, 2), (2, 1)):
            label = f"pair {pair_id} side {source}"
            sides.append(
                {
                    "trial_id": f"{pair_id}:p{source}->p{target}",
                    "pair_id": pair_id,
                    "source_side": source,
                    "target_side": target,
                    "prompt": _required_text(pair, f"prompt{source}", label=label),
                    "partner_prompt": _required_text(pair, f"prompt{target}", label=label),
                    "concept": _required_text(pair, f"concept{source}", label=label),
                    "partner_concept": _required_text(pair, f"concept{target}", label=label),
                    "answer": _required_text(pair, f"answer{source}", label=label),
                    "partner_answer": _required_text(pair, f"answer{target}", label=label),
                }
            )
    trial_ids = [str(side["trial_id"]) for side in sides]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("paired-cloze bank yields duplicate directed-side IDs")
    return sides


def pair_bank_provenance(
    pairs: list[dict[str, Any]], *, path: Path | str | None = None
) -> dict[str, Any]:
    """Describe a pair bank without duplicating its prompts in result files."""
    sides = iter_sides(pairs)
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "concept_lens_paired_cloze_input",
        "n_pairs": len(pairs),
        "n_sides": len(sides),
        "side_ids_sha256": side_ids_digest(sides),
        "pairs_content_sha256": _json_sha256(pairs),
    }
    if path is not None:
        source = Path(path)
        record["path"] = str(source)
        if source.is_file():
            record["file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return record


def load_singleton_concepts(path: Path | str) -> list[str]:
    """Load an ordered, duplicate-free singleton concept list."""
    source = Path(path)
    payload = load_json(source)
    values = payload.get("concepts") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"{source}: expected a concept list or an object with concepts")
    concepts = [str(item) for item in values]
    if any(not concept.strip() for concept in concepts):
        raise ValueError(f"{source}: concept list contains an empty surface")
    if len(concepts) != len(set(concepts)):
        raise ValueError(f"{source}: concept list contains duplicates")
    return concepts


def load_calibration_concepts(path: Path | str | None = None) -> list[str]:
    """Load the singleton calibration split used to fit readout artifacts."""
    return load_singleton_concepts(path or ACTIVE_CALIBRATION_PATH)


def load_reconstruction_test_concepts(path: Path | str | None = None) -> list[str]:
    """Load the held-out singleton split used by the fragment reconstruction test."""
    return load_singleton_concepts(path or RECONSTRUCTION_TEST_CONCEPTS_PATH)


def ranking_fit_concepts() -> list[str]:
    """Return the public calibration split used for positive-pool fitting."""
    concepts = load_calibration_concepts()
    from bindlens.bind import normalise

    pair_gold = {
        normalise(str(pair[f"concept{side}"]))
        for pair in load_cloze_pairs()
        for side in (1, 2)
    }
    overlap = [concept for concept in concepts if normalise(concept) in pair_gold]
    if overlap:
        raise ValueError(
            "positive-pool calibration split overlaps paired-cloze concepts: "
            f"{overlap[:5]}"
        )
    return concepts
