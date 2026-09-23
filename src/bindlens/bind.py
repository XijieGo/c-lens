"""BIND: cloze-conditioned phrase search over the J-lens token set."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from bindlens.models import LoadedModel, one_token_id, visible
from bindlens.protocol import (
    EXPAND_K,
    MAX_ACTIVE,
    MAX_TOKENS,
    ROOT_BANDS,
    S_K,
    SEED_K,
    SELECT_TOP_K,
    TOP_P,
    decode_prompt,
)
from bindlens.read import topk_ids


_FRAGMENT_IDS: dict[tuple[int, str], torch.Tensor] = {}
_LEXICAL_IDS: dict[tuple[int, int, str], torch.Tensor] = {}


@dataclass(frozen=True)
class Candidate:
    ids: tuple[int, ...]
    key: str
    surface: str
    score: float


def normalise(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip().strip("\"'`")
    return text.rstrip(". ").casefold()


def decode_surface(tokenizer: Any, ids: list[int] | tuple[int, ...]) -> str:
    return (
        tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        .replace("\n", " ")
        .strip()
    )


def is_lexical_token(tokenizer: Any, token_id: int) -> bool:
    piece = decode_surface(tokenizer, [token_id]).replace(" ", "")
    return bool(piece) and piece[0].isalpha() and all(char.isalpha() or char in "'-" for char in piece)


def is_word_initial_lexical_token(tokenizer: Any, token_id: int) -> bool:
    """Whether ``token_id`` can start ``concept: <seed>`` naturally.

    Source READ represents a cloze answer with a leading-space token.  The
    scaffold must use the same convention: it ends at ``concept:``, and a
    word-initial seed supplies the one intervening space.  Bare BPE fragments
    such as ``ert`` are valid only after a word has begun, never as a concept
    seed.
    """
    piece = tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not piece or not piece[:1].isspace():
        return False
    surface = piece.strip()
    return bool(surface) and surface[0].isalpha() and all(
        char.isalpha() or char in "'-" for char in surface
    )


def is_intra_word_fragment(tokenizer: Any, token_id: int) -> bool:
    piece = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return bool(piece) and not piece[:1].isspace() and piece[0].isalpha() and all(
        char.isalpha() or char in "'-" for char in piece
    )


def valid_phrase(text: str) -> bool:
    return bool(text) and text[0].isalpha() and all(char.isalpha() or char in " '-" for char in text)


def control_ids(tokenizer: Any) -> dict[str, Any]:
    ends: dict[str, int] = {}
    for label, text in (
        ("period", "."),
        ("space_period", " ."),
        ("newline", "\n"),
        ("dot_newline", ".\n"),
    ):
        token_id = one_token_id(tokenizer, text)
        if token_id is not None:
            ends[label] = token_id
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ends["eos"] = int(eos)
    return {"ends": ends, "end_ids": set(ends.values())}


def decode_prefix_ids(tokenizer: Any, cloze: str) -> list[int]:
    prompt = decode_prompt(cloze)
    return [int(token_id) for token_id in tokenizer(prompt, add_special_tokens=True).input_ids]


def fragment_ids(tokenizer: Any, vocab_size: int, device: torch.device) -> torch.Tensor:
    key = (id(tokenizer), str(device))
    cached = _FRAGMENT_IDS.get(key)
    if cached is not None:
        return cached
    values = [
        token_id
        for token_id in range(int(vocab_size))
        if is_intra_word_fragment(tokenizer, token_id)
    ]
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    _FRAGMENT_IDS[key] = tensor
    return tensor


def lexical_ids(tokenizer: Any, vocab_size: int, device: torch.device) -> torch.Tensor:
    """All lexical token IDs, cached on the requested device.

    This is deliberately a format filter rather than a source-token filter:
    callers use it when a decoder may choose any vocabulary item, but must
    still emit a phrase that can be carried and scored as a concept.
    """
    key = (id(tokenizer), int(vocab_size), str(device))
    cached = _LEXICAL_IDS.get(key)
    if cached is not None:
        return cached
    values = [
        token_id
        for token_id in range(int(vocab_size))
        if is_lexical_token(tokenizer, token_id)
    ]
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    _LEXICAL_IDS[key] = tensor
    return tensor


def restricted_logprobs(
    logits: torch.Tensor, allowed_ids: list[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    allowed = (
        allowed_ids.to(device=logits.device, dtype=torch.long)
        if isinstance(allowed_ids, torch.Tensor)
        else torch.tensor(allowed_ids, dtype=torch.long, device=logits.device)
    )
    values = logits.index_select(0, allowed)
    return allowed, torch.log_softmax(values.float(), dim=-1)


def select_nucleus(logprobs: torch.Tensor, top_p: float, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if logprobs.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=logprobs.device)
        return empty, logprobs
    k = max(1, min(int(top_k), int(logprobs.numel())))
    values, indices = torch.topk(logprobs, k=k)
    csum = torch.cumsum(values.exp(), dim=-1)
    keep = int((csum < float(top_p)).sum().item()) + 1
    keep = max(1, min(keep, k))
    return indices[:keep], values[:keep]


def next_logits_batch(
    hf: Any,
    sequences: list[list[int]],
    pad_id: int,
    input_device: torch.device,
) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=input_device)
    attn = torch.zeros_like(batch)
    for row, seq in enumerate(sequences):
        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=input_device)
        attn[row, : len(seq)] = 1
    with torch.inference_mode():
        out = hf(input_ids=batch, attention_mask=attn, use_cache=False)
    last = attn.sum(dim=1) - 1
    rows = torch.arange(len(sequences), device=out.logits.device)
    return out.logits[rows, last].float()


def _unique(ids: list[int]) -> list[int]:
    return list(dict.fromkeys(int(token_id) for token_id in ids))


def _terminal_logprobs(
    *,
    row_logits: torch.Tensor,
    main_content_ids: list[int] | torch.Tensor,
    end_ids: list[int],
    top_p: float,
    expand_k: int,
    fragments: torch.Tensor,
) -> list[float]:
    if isinstance(main_content_ids, torch.Tensor):
        if not main_content_ids.numel() or not end_ids:
            return []
        end_tensor = torch.tensor(end_ids, dtype=torch.long, device=row_logits.device)
        terminal_ids: list[int] | torch.Tensor = torch.cat(
            [main_content_ids.to(row_logits.device), end_tensor]
        )
    else:
        terminal_ids = _unique([*main_content_ids, *end_ids])
    if (isinstance(terminal_ids, torch.Tensor) and not terminal_ids.numel()) or (
        not isinstance(terminal_ids, torch.Tensor) and not terminal_ids
    ) or not end_ids:
        return []
    terminal_tensor, terminal_lps = restricted_logprobs(row_logits, terminal_ids)
    selected_locs, selected_lps = select_nucleus(terminal_lps, top_p=top_p, top_k=expand_k)
    end_set = set(end_ids)
    selected_end = [
        float(lp.item())
        for loc, lp in zip(selected_locs, selected_lps)
        if int(terminal_tensor[int(loc)].item()) in end_set
    ]
    if not selected_end:
        return []
    end_tensor = torch.tensor(end_ids, dtype=torch.long, device=row_logits.device)
    best_end = row_logits.index_select(0, end_tensor).max()
    if fragments.numel():
        best_fragment = row_logits.index_select(0, fragments).max()
        if bool(best_fragment > best_end):
            return []
    return selected_end


def _main_actions(
    row_logits: torch.Tensor,
    main_content_ids: list[int] | torch.Tensor,
    *,
    top_p: float,
    expand_k: int,
) -> list[tuple[int, float]]:
    if isinstance(main_content_ids, torch.Tensor):
        empty = not main_content_ids.numel()
    else:
        empty = not main_content_ids
    if empty:
        return []
    ids, lps = restricted_logprobs(row_logits, main_content_ids)
    selected_locs, _ = select_nucleus(lps, top_p=top_p, top_k=expand_k)
    return [(int(ids[int(loc)].item()), float(lps[int(loc)].item())) for loc in selected_locs]


def _decode_group(
    *,
    tokenizer: Any,
    logits_for_prefixes: Callable[[list[tuple[int, ...]]], list[torch.Tensor]],
    main_source_ids: list[int] | torch.Tensor,
    root_ids: list[int],
    seed_k: int,
    top_p: float,
    expand_k: int,
    max_active: int,
    max_tokens: int,
    end_ids: list[int],
    fragments: torch.Tensor,
) -> list[dict[str, Any]]:
    # A sharded HF model can return continuation logits on its input device
    # even when the J-lens/unembedding device is different.  Keep a tiny
    # per-device token-index cache so terminal checks never mix GPU indices.
    fragments_by_device: dict[str, torch.Tensor] = {str(fragments.device): fragments}

    if isinstance(main_source_ids, torch.Tensor):
        # ``lexical_ids()`` has already applied the only format constraint.
        # Do not materialize this full-vocabulary tensor as a Python list for
        # every beam event.
        main_content: list[int] | torch.Tensor = main_source_ids
        roots = [
            token_id
            for token_id in _unique(root_ids)
            if is_word_initial_lexical_token(tokenizer, token_id)
        ]
        roots = roots[: int(seed_k)]
    else:
        end_set = set(int(token_id) for token_id in end_ids)
        main_content = [
            int(token_id)
            for token_id in _unique(main_source_ids)
            if token_id not in end_set and is_lexical_token(tokenizer, token_id)
        ]
        main_set = set(main_content)
        # A source J-lens list can contain an intra-word BPE fragment.  It is
        # lexical once a word is already under way, but it cannot follow the
        # terminal ``concept:`` scaffold prefix: that would produce
        # ``concept:fragment`` instead of ``concept: fragment``.  Apply the
        # same word-initial boundary filter used by the free-tail branch.
        roots = [
            int(token_id)
            for token_id in root_ids
            if int(token_id) in main_set and is_word_initial_lexical_token(tokenizer, int(token_id))
        ][: int(seed_k)]
    if not roots:
        return []

    first_logits = logits_for_prefixes([()])[0]
    seed_tensor, seed_lps = restricted_logprobs(first_logits, main_content)
    if isinstance(main_content, torch.Tensor):
        # The cached lexical vocabulary is sorted by token ID, so lookup of a
        # few roots is O(log |V|) each.  Avoid constructing a 100k-entry
        # Python dictionary for every full-vocabulary candidate event.
        root_tensor = torch.tensor(roots, dtype=torch.long, device=seed_tensor.device)
        locations = torch.searchsorted(seed_tensor, root_tensor)
        if bool((locations >= seed_tensor.numel()).any()) or not bool(
            torch.equal(seed_tensor.index_select(0, locations), root_tensor)
        ):
            raise RuntimeError("unconstrained root is not in the lexical vocabulary")
        root_scores = seed_lps.index_select(0, locations).tolist()
        active = [
            {"ids": (token_id,), "score": float(score)}
            for token_id, score in zip(roots, root_scores)
        ]
    else:
        seed_score = {
            int(token_id.item()): float(lp.item())
            for token_id, lp in zip(seed_tensor, seed_lps)
        }
        active = [{"ids": (token_id,), "score": seed_score[token_id]} for token_id in roots]
    completed: dict[str, dict[str, Any]] = {}

    def terminal(row: dict[str, Any], row_logits: torch.Tensor) -> None:
        if len(row["ids"]) < 2:
            return
        fragment_key = str(row_logits.device)
        local_fragments = fragments_by_device.get(fragment_key)
        if local_fragments is None:
            local_fragments = fragments.to(device=row_logits.device)
            fragments_by_device[fragment_key] = local_fragments
        selected = _terminal_logprobs(
            row_logits=row_logits,
            main_content_ids=main_content,
            end_ids=end_ids,
            top_p=top_p,
            expand_k=expand_k,
            fragments=local_fragments,
        )
        for terminal_lp in selected:
            ids = tuple(int(token_id) for token_id in row["ids"])
            text = decode_surface(tokenizer, ids)
            key = normalise(text)
            if not key or not valid_phrase(text):
                continue
            score = float(row["score"]) + float(terminal_lp)
            prev = completed.get(key)
            if prev is None or score > float(prev["score"]):
                completed[key] = {"ids": ids, "text": text, "score": score}

    for _ in range(1, int(max_tokens)):
        if not active:
            break
        row_logits_batch = logits_for_prefixes([row["ids"] for row in active])
        nxt: list[dict[str, Any]] = []
        for row, row_logits in zip(active, row_logits_batch):
            terminal(row, row_logits)
            for token_id, token_lp in _main_actions(
                row_logits, main_content, top_p=top_p, expand_k=expand_k
            ):
                new_ids = tuple([*row["ids"], int(token_id)])
                if len(new_ids) != len(set(new_ids)):
                    continue
                text = decode_surface(tokenizer, new_ids)
                if not valid_phrase(text):
                    continue
                nxt.append({"ids": new_ids, "score": float(row["score"]) + float(token_lp)})
        nxt.sort(key=lambda row: float(row["score"]), reverse=True)
        active = nxt[: int(max_active)]

    if active:
        cap_logits = logits_for_prefixes([row["ids"] for row in active])
        for row, row_logits in zip(active, cap_logits):
            terminal(row, row_logits)
    return sorted(completed.values(), key=lambda row: float(row["score"]), reverse=True)


def prefix_dominance(candidates: list[Candidate]) -> list[Candidate]:
    """Drop a longer token-prefix when a shorter prefix scores no lower."""
    kept: list[Candidate] = []
    for candidate in candidates:
        dominated = any(
            len(shorter.ids) < len(candidate.ids)
            and candidate.ids[: len(shorter.ids)] == shorter.ids
            and shorter.score >= candidate.score
            for shorter in candidates
            if shorter is not candidate
        )
        if not dominated:
            kept.append(candidate)
    return kept


def select_candidates(
    rows: list[dict[str, Any]] | list[Candidate],
    *,
    top_k: int = SELECT_TOP_K,
) -> list[Candidate]:
    """Per-layer top-k by proposer score, then token-prefix score-dominance."""
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for row in rows:
        if isinstance(row, Candidate):
            candidate = row
        else:
            ids = tuple(int(token) for token in row.get("ids") or [])
            surface = re.sub(r"\s+", " ", str(row.get("text", "")).replace("\n", " ")).strip()
            key = normalise(surface)
            if not ids or not key:
                continue
            candidate = Candidate(
                ids=ids,
                key=key,
                surface=surface,
                score=float(row.get("score", float("-inf"))),
            )
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        candidates.append(candidate)
    candidates.sort(key=lambda item: item.score, reverse=True)
    top = candidates[: int(top_k)] if top_k > 0 else candidates
    return prefix_dominance(top)


def propose(
    loaded: LoadedModel,
    jlens_logits: torch.Tensor,
    *,
    cloze_prompt: str,
    s_k: int = S_K,
    seed_k: int = SEED_K,
    top_p: float = TOP_P,
    expand_k: int = EXPAND_K,
    max_tokens: int = MAX_TOKENS,
    root_bands: tuple[tuple[int, int, int], ...] = ROOT_BANDS,
    independent_top_n: int = 0,
    independent_root_active: int | None = None,
) -> list[dict[str, Any]]:
    """Rank-banded lexical decode. Gold is never used.

    ``independent_top_n`` keeps each of the first N J-lens roots in its own
    beam so a high-probability collocation cannot prune a top-10 first token.
    Remaining ``root_bands`` whose range starts after N still share one budget.
    """
    tokenizer = loaded.tokenizer
    controls = control_ids(tokenizer)
    end_ids = sorted(int(token_id) for token_id in controls["end_ids"])
    source_ids = topk_ids(jlens_logits, s_k)
    context_ids = decode_prefix_ids(tokenizer, cloze_prompt)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id or 0)
    vocab_size = int(
        getattr(loaded.hf.config, "vocab_size", None)
        or getattr(getattr(loaded.hf.config, "text_config", None), "vocab_size", None)
        or len(tokenizer)
    )
    # The J-lens list supplies *only* possible first tokens.  Every later
    # token must be drawn from the frozen model's ordinary next-token
    # distribution (subject only to the lexical/termination format filter).
    # Passing ``source_ids`` here would incorrectly constrain a completion to
    # the source J-lens top-1000 tokens, contrary to the PROPOSE method.
    # Candidate continuations are model forwards, not J-lens operations.  A
    # sharded model returns those logits at its input device, so cache lexical
    # IDs there; READ/J-lens remains on ``loaded.device`` (lm-head device).
    decoder_device = loaded.model.input_device
    content_ids = lexical_ids(tokenizer, vocab_size, decoder_device)
    fragments = fragment_ids(tokenizer, vocab_size, decoder_device)
    cache: dict[tuple[int, ...], torch.Tensor] = {}

    def logits_for_prefixes(prefixes: list[tuple[int, ...]]) -> list[torch.Tensor]:
        missing = [prefix for prefix in dict.fromkeys(prefixes) if prefix not in cache]
        if missing:
            sequences = [[*context_ids, *prefix] for prefix in missing]
            logits = next_logits_batch(loaded.hf, sequences, pad_id, loaded.model.input_device)
            for prefix, row in zip(missing, logits.unbind(dim=0)):
                cache[prefix] = row
        return [cache[prefix] for prefix in prefixes]

    merged: dict[str, dict[str, Any]] = {}

    def add_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            key = normalise(row["text"])
            prev = merged.get(key)
            if prev is None or float(row["score"]) > float(prev["score"]):
                merged[key] = row

    independent_n = max(0, int(independent_top_n))
    per_root_active = int(independent_root_active or max(int(expand_k), 8))
    if independent_n:
        for token_id in source_ids[:independent_n]:
            add_rows(
                _decode_group(
                    tokenizer=tokenizer,
                    logits_for_prefixes=logits_for_prefixes,
                    main_source_ids=content_ids,
                    root_ids=[int(token_id)],
                    seed_k=1,
                    top_p=top_p,
                    expand_k=expand_k,
                    max_active=per_root_active,
                    max_tokens=max_tokens,
                    end_ids=end_ids,
                    fragments=fragments,
                )
            )

    for start, stop, active_width in root_bands:
        if independent_n and int(stop) <= independent_n:
            continue
        band_start = max(int(start), independent_n + 1) if independent_n else int(start)
        roots = source_ids[band_start - 1 : int(stop)]
        add_rows(
            _decode_group(
                tokenizer=tokenizer,
                logits_for_prefixes=logits_for_prefixes,
                main_source_ids=content_ids,
                root_ids=roots,
                seed_k=seed_k,
                top_p=top_p,
                expand_k=expand_k,
                max_active=int(active_width),
                max_tokens=max_tokens,
                end_ids=end_ids,
                fragments=fragments,
            )
        )
    out = []
    for row in sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True):
        out.append(
            {
                "ids": list(row["ids"]),
                "text": row["text"],
                "score": float(row["score"]),
                "tokens": [visible(tokenizer, token_id) for token_id in row["ids"]],
            }
        )
    return out


# Retained reference implementation for compatibility and controlled
# comparisons.  ``propose`` keeps its longstanding public name because the
# paper evaluators and several modules import it directly; online serving uses
# ``CachedTreeProposer`` from ``bind_tree.py`` by default instead.
old_full_forward_propose = propose


def propose_unconstrained(
    loaded: LoadedModel,
    *,
    cloze_prompt: str,
    first_token_groups: tuple[tuple[list[int], int], ...] | None = None,
    seed_k: int = SEED_K,
    top_p: float = TOP_P,
    expand_k: int = EXPAND_K,
    max_tokens: int = MAX_TOKENS,
    root_bands: tuple[tuple[int, int, int], ...] = ROOT_BANDS,
) -> list[dict[str, Any]]:
    """Decode a concept from the full lexical vocabulary.

    No continuation token is drawn from a J-lens token set.  When
    ``first_token_groups`` is omitted, the language-model distribution over
    the cloze prompt chooses every first-token root.  A caller may instead
    provide rank-banded J-lens roots to isolate the effect of preserving only
    the first token from the standard BIND proposal policy.  In both cases,
    phrase grammar, search width, termination, and selection score remain the
    same as :func:`propose`.
    """
    tokenizer = loaded.tokenizer
    controls = control_ids(tokenizer)
    end_ids = sorted(int(token_id) for token_id in controls["end_ids"])
    context_ids = decode_prefix_ids(tokenizer, cloze_prompt)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id or 0)
    vocab_size = int(
        getattr(loaded.hf.config, "vocab_size", None)
        or getattr(getattr(loaded.hf.config, "text_config", None), "vocab_size", None)
        or len(tokenizer)
    )
    decoder_device = loaded.model.input_device
    content_ids = lexical_ids(tokenizer, vocab_size, decoder_device)
    fragments = fragment_ids(tokenizer, vocab_size, decoder_device)
    cache: dict[tuple[int, ...], torch.Tensor] = {}

    def logits_for_prefixes(prefixes: list[tuple[int, ...]]) -> list[torch.Tensor]:
        missing = [prefix for prefix in dict.fromkeys(prefixes) if prefix not in cache]
        if missing:
            sequences = [[*context_ids, *prefix] for prefix in missing]
            logits = next_logits_batch(loaded.hf, sequences, pad_id, loaded.model.input_device)
            for prefix, row in zip(missing, logits.unbind(dim=0)):
                cache[prefix] = row
        return [cache[prefix] for prefix in prefixes]

    groups: list[tuple[list[int], int]]
    if first_token_groups is None:
        # Match the standard rank locations, per-band seed cap, and 12/6/6
        # continuation allocation, but rank roots under the LM's
        # cloze-conditioned distribution rather than a J-lens distribution.
        # In particular, this preserves the original 1-20 / 21-50 / 51-100
        # allocation even though all roots are now free LM choices.
        total_roots = min(int(content_ids.numel()), max(int(stop) for _start, stop, _width in root_bands))
        first_logits = logits_for_prefixes([()])[0]
        local = torch.topk(first_logits.index_select(0, content_ids), k=total_roots).indices
        free_roots = [int(token_id) for token_id in content_ids.index_select(0, local).tolist()]
        groups = []
        for start, stop, active_width in root_bands:
            roots = free_roots[int(start) - 1 : int(stop)][: int(seed_k)]
            groups.append((roots, int(active_width)))
    else:
        groups = [
            ([int(token_id) for token_id in roots], int(active_width))
            for roots, active_width in first_token_groups
        ]

    merged: dict[str, dict[str, Any]] = {}
    for roots, active_width in groups:
        rows = _decode_group(
            tokenizer=tokenizer,
            logits_for_prefixes=logits_for_prefixes,
            main_source_ids=content_ids,
            root_ids=roots,
            seed_k=seed_k,
            top_p=top_p,
            expand_k=expand_k,
            max_active=active_width,
            max_tokens=max_tokens,
            end_ids=end_ids,
            fragments=fragments,
        )
        for row in rows:
            key = normalise(row["text"])
            prev = merged.get(key)
            if prev is None or float(row["score"]) > float(prev["score"]):
                merged[key] = row

    return [
        {
            "ids": list(row["ids"]),
            "text": row["text"],
            "score": float(row["score"]),
            "tokens": [visible(tokenizer, token_id) for token_id in row["ids"]],
        }
        for row in sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
    ]
