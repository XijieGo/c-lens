"""Utilities for turning singleton concepts into forced two-token inputs.

The public singleton files intentionally store only surfaces.  This module
derives a deterministic, tokenizer-checked two-token ledger at run time, so
the reconstruction test remains tied to the tokenizer actually being used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(value) for value in encoded.input_ids]


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            [int(value) for value in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def exact_token_id(tokenizer: Any, text: str) -> int | None:
    """Return the id of an exact, non-special one-token surface, if any."""
    ids = _encode(tokenizer, text)
    if len(ids) != 1:
        return None
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    if ids[0] in special or _decode(tokenizer, ids) != text:
        return None
    return ids[0]


def _split_quality(concept: str, boundary: int) -> tuple[float, int, int]:
    """Quality key used to choose one split reproducibly.

    Prefer the most balanced character split, then the largest shorter piece,
    then the earliest boundary.  The final tie-break makes the selection
    independent of dictionary or set iteration order.
    """
    left = boundary
    right = len(concept) - boundary
    shorter, longer = sorted((left, right))
    return (-shorter / longer, -shorter, boundary)


def _candidate_splits(
    tokenizer: Any,
    concept: str,
    *,
    min_piece_length: int,
) -> tuple[int | None, list[dict[str, Any]], str | None]:
    surface = f" {concept}"
    canonical_id = exact_token_id(tokenizer, surface)
    if canonical_id is None:
        return None, [], "canonical surface is not one exact non-special token"
    candidates: list[dict[str, Any]] = []
    for boundary in range(min_piece_length, len(concept) - min_piece_length + 1):
        left_text = f" {concept[:boundary]}"
        right_text = concept[boundary:]
        left_id = exact_token_id(tokenizer, left_text)
        right_id = exact_token_id(tokenizer, right_text)
        if left_id is None or right_id is None:
            continue
        piece_ids = [left_id, right_id]
        if _decode(tokenizer, piece_ids) != surface:
            continue
        candidates.append(
            {
                "boundary": boundary,
                "split": [left_text, right_text],
                "piece_ids": piece_ids,
                "pieces": [_decode(tokenizer, [left_id]), _decode(tokenizer, [right_id])],
            }
        )
    if not candidates:
        return canonical_id, [], "no exact two-token split at the requested minimum piece length"
    candidates.sort(key=lambda row: _split_quality(concept, int(row["boundary"])))
    return canonical_id, candidates, None


def build_forced_split_ledger(
    tokenizers: Mapping[str, Any],
    concepts: Sequence[str],
    *,
    min_piece_length: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build a common forced-bigram ledger for all supplied tokenizers.

    A row is retained only when the *same character boundary* is valid for
    every tokenizer.  This keeps the surface split comparable across models;
    token ids are allowed to differ.  ``min_piece_length=1`` is intentional
    for the public 500-concept reconstruction split, which contains valid
    three-letter singletons.
    """
    if min_piece_length < 1:
        raise ValueError("min_piece_length must be at least one")
    names = list(tokenizers)
    if not names:
        raise ValueError("at least one tokenizer is required")
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for raw_concept in concepts:
        concept = str(raw_concept)
        if not concept or concept != concept.strip() or not concept.isascii():
            failed.append({"concept": concept, "reason": "concept must be non-empty ASCII without surrounding whitespace"})
            continue
        per_model: dict[str, tuple[int | None, list[dict[str, Any]], str | None]] = {
            name: _candidate_splits(tokenizer, concept, min_piece_length=min_piece_length)
            for name, tokenizer in tokenizers.items()
        }
        bad = [
            f"{name}: {result[2]}"
            for name, result in per_model.items()
            if result[0] is None or not result[1]
        ]
        if bad:
            failed.append({"concept": concept, "reason": "; ".join(bad)})
            continue
        common_boundaries = set(int(row["boundary"]) for row in per_model[names[0]][1])
        for name in names[1:]:
            common_boundaries &= {int(row["boundary"]) for row in per_model[name][1]}
        if not common_boundaries:
            failed.append({"concept": concept, "reason": "no common character boundary across tokenizers"})
            continue
        boundary = min(common_boundaries, key=lambda value: _split_quality(concept, value))
        tokens: dict[str, Any] = {}
        for name in names:
            canonical_id, candidates, _ = per_model[name]
            selected = next(row for row in candidates if int(row["boundary"]) == boundary)
            tokens[name] = {
                "canonical_id": int(canonical_id),
                "canonical": f" {concept}",
                "piece_ids": [int(value) for value in selected["piece_ids"]],
                "pieces": list(selected["pieces"]),
            }
        rows.append(
            {
                "concept": concept,
                "canonical_surface": f" {concept}",
                "split": [f" {concept[:boundary]}", concept[boundary:]],
                "tokens": tokens,
            }
        )
    return rows, failed


def build_model_split_rows(
    tokenizer: Any,
    model: str,
    concepts: Sequence[str],
    *,
    min_piece_length: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build the same row shape for one model (convenient CLI fallback)."""
    rows, failed = build_forced_split_ledger(
        {str(model): tokenizer}, concepts, min_piece_length=min_piece_length
    )
    return rows, failed


def validate_fragment_rows(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    tokenizer: Any | None = None,
) -> None:
    """Validate the minimum ledger contract before a model forward pass."""
    for index, row in enumerate(rows):
        if not isinstance(row.get("concept"), str):
            raise ValueError(f"forced-split row {index}: missing concept")
        info = row.get("tokens", {}).get(model)
        if not isinstance(info, Mapping):
            raise ValueError(f"forced-split row {index}: missing tokens[{model!r}]")
        piece_ids = info.get("piece_ids")
        if not isinstance(piece_ids, Sequence) or len(piece_ids) != 2:
            raise ValueError(f"forced-split row {index}: expected two piece ids")
        if info.get("canonical_id") is None:
            raise ValueError(f"forced-split row {index}: missing canonical id")
        if tokenizer is not None:
            concept = str(row["concept"])
            canonical_id = exact_token_id(tokenizer, f" {concept}")
            if canonical_id is None or int(info["canonical_id"]) != canonical_id:
                raise ValueError(f"forced-split row {index}: canonical token does not match tokenizer")
            decoded = _decode(tokenizer, [int(value) for value in piece_ids])
            if decoded != f" {concept}":
                raise ValueError(
                    f"forced-split row {index}: forced pieces decode to {decoded!r}, not {concept!r}"
                )
