"""Cache-aware tree execution for BIND's online proposer.

The proposal *policy* remains in :mod:`bindlens.bind`: rank-banded J-lens
roots, lexical continuations, nucleus expansion, and prefix dominance are
unchanged.  This module changes only how that policy is executed for a single
online request:

* prefill the cloze scaffold once and retain its ``past_key_values``;
* reuse that short prefix state for every source layer;
* decode all rank bands in one batched frontier at each tree depth;
* decode duplicate child prefixes only once; and
* reuse the final frontier logits for the terminal check.

It intentionally uses ordinary Hugging Face cache APIs rather than a custom
kernel.  That keeps the backend portable across the supported Llama, Qwen,
and Gemma model families.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from bindlens.bind import (
    Candidate,
    _main_actions,
    _terminal_logprobs,
    control_ids,
    decode_prefix_ids,
    decode_surface,
    fragment_ids,
    is_word_initial_lexical_token,
    lexical_ids,
    normalise,
    restricted_logprobs,
    select_candidates,
    valid_phrase,
)
from bindlens.models import LoadedModel, visible
from bindlens.protocol import EXPAND_K, MAX_TOKENS, ROOT_BANDS, S_K, SEED_K, TOP_P
from bindlens.read import topk_ids


@dataclass
class ContextState:
    """Reusable cloze-prefill cache and its final-position logits."""

    cache: Any
    logits: torch.Tensor


@dataclass
class TreeDecodeStats:
    """Per-request execution counters for observability and latency reports."""

    context_forwards: int = 0
    incremental_forwards: int = 0
    incremental_rows: int = 0
    deduplicated_rows: int = 0
    layers: int = 0
    layer_batches: int = 0
    max_frontier_rows: int = 0


@dataclass
class _Node:
    ids: tuple[int, ...]
    score: float
    group: int


@dataclass
class _Group:
    roots: list[int]
    max_active: int
    completed: dict[str, dict[str, Any]]


def _restricted_logprobs_rows(
    logits: torch.Tensor,
    allowed_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute lexical log-probabilities for one or more tree rows."""

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    allowed = allowed_ids.to(device=logits.device, dtype=torch.long)
    values = logits.index_select(-1, allowed)
    return allowed, torch.log_softmax(values.float(), dim=-1)


def _select_nucleus_batched(
    logprobs: torch.Tensor,
    top_p: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the standard BIND nucleus rule independently to every row.

    The returned mask retains the same per-row truncation as
    :func:`bindlens.bind.select_nucleus`; vectorizing it avoids synchronizing
    GPU scalars for every frontier row.
    """

    if logprobs.ndim == 1:
        logprobs = logprobs.unsqueeze(0)
    batch, width = logprobs.shape
    if width == 0:
        empty_indices = torch.empty(batch, 0, dtype=torch.long, device=logprobs.device)
        empty_values = torch.empty(batch, 0, dtype=logprobs.dtype, device=logprobs.device)
        empty_mask = torch.empty(batch, 0, dtype=torch.bool, device=logprobs.device)
        return empty_indices, empty_values, empty_mask
    width = max(1, min(int(top_k), width))
    values, indices = torch.topk(logprobs, k=width, dim=-1)
    keep = (torch.cumsum(values.exp(), dim=-1) < float(top_p)).sum(dim=-1) + 1
    keep = keep.clamp(min=1, max=width)
    columns = torch.arange(width, device=logprobs.device)
    mask = columns.unsqueeze(0) < keep.unsqueeze(1)
    return indices, values, mask


def _main_actions_batched(
    logits: torch.Tensor,
    main_content_ids: torch.Tensor,
    *,
    top_p: float,
    expand_k: int,
) -> list[list[tuple[int, float]]]:
    """Reference BIND main actions for a complete GPU frontier."""

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    batch = int(logits.shape[0])
    if not main_content_ids.numel():
        return [[] for _ in range(batch)]
    allowed, logprobs = _restricted_logprobs_rows(logits, main_content_ids)
    locations, _values, mask = _select_nucleus_batched(logprobs, top_p, expand_k)
    if locations.numel() == 0:
        return [[] for _ in range(batch)]
    token_ids = allowed.index_select(0, locations.reshape(-1)).reshape(locations.shape)
    token_logprobs = logprobs.gather(1, locations)
    token_ids_cpu = token_ids.detach().cpu()
    token_logprobs_cpu = token_logprobs.detach().cpu()
    mask_cpu = mask.detach().cpu()
    rows: list[list[tuple[int, float]]] = []
    for row_index in range(batch):
        actions: list[tuple[int, float]] = []
        for column in range(int(mask_cpu.shape[1])):
            if bool(mask_cpu[row_index, column]):
                actions.append(
                    (
                        int(token_ids_cpu[row_index, column]),
                        float(token_logprobs_cpu[row_index, column]),
                    )
                )
        rows.append(actions)
    return rows


def _terminal_logprobs_batched(
    logits: torch.Tensor,
    main_content_ids: torch.Tensor,
    end_ids: list[int],
    fragments: torch.Tensor,
    *,
    top_p: float,
    expand_k: int,
    eligible: list[bool],
) -> list[list[float]]:
    """Reference BIND terminal checks for a complete GPU frontier."""

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    batch = int(logits.shape[0])
    if batch != len(eligible):
        raise ValueError("eligible mask must match the logit batch")
    if not end_ids or not main_content_ids.numel():
        return [[] for _ in range(batch)]

    end_tensor = torch.tensor(
        [int(token_id) for token_id in end_ids],
        dtype=torch.long,
        device=logits.device,
    )
    terminal_ids = torch.cat([main_content_ids.to(device=logits.device), end_tensor])
    allowed, logprobs = _restricted_logprobs_rows(logits, terminal_ids)
    locations, _values, mask = _select_nucleus_batched(logprobs, top_p, expand_k)
    if locations.numel() == 0:
        return [[] for _ in range(batch)]
    selected_tokens = allowed.index_select(0, locations.reshape(-1)).reshape(locations.shape)
    selected_logprobs = logprobs.gather(1, locations)
    end_max = logits.index_select(-1, end_tensor).max(dim=-1).values
    if fragments.numel():
        fragment_max = logits.index_select(-1, fragments.to(device=logits.device)).max(dim=-1).values
        veto = fragment_max > end_max
    else:
        veto = torch.zeros(batch, dtype=torch.bool, device=logits.device)

    selected_tokens_cpu = selected_tokens.detach().cpu()
    selected_logprobs_cpu = selected_logprobs.detach().cpu()
    mask_cpu = mask.detach().cpu()
    veto_cpu = veto.detach().cpu()
    end_set = {int(token_id) for token_id in end_ids}
    rows: list[list[float]] = []
    for row_index in range(batch):
        if not eligible[row_index]:
            rows.append([])
            continue
        selected_end = [
            float(selected_logprobs_cpu[row_index, column])
            for column in range(int(mask_cpu.shape[1]))
            if bool(mask_cpu[row_index, column])
            and int(selected_tokens_cpu[row_index, column]) in end_set
        ]
        rows.append([] if not selected_end or bool(veto_cpu[row_index]) else selected_end)
    return rows


def _clone_cache(cache: Any) -> Any:
    """Clone a Hugging Face cache without mutating the shared cloze prefix."""

    if cache is None:
        raise RuntimeError("model did not return a past_key_values cache")
    # ``deepcopy`` supports current DynamicCache objects and tuple-style
    # caches.  The shared prefix is short, and cloning it per source layer is
    # much cheaper than recomputing the entire cloze scaffold.
    return copy.deepcopy(cache)


def _cache_tensor_device(cache: Any) -> torch.device | None:
    """Find a device in modern or tuple-style cache representations."""

    def visit(value: Any) -> torch.device | None:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            for item in value.values():
                found = visit(item)
                if found is not None:
                    return found
        if isinstance(value, (tuple, list)):
            for item in value:
                found = visit(item)
                if found is not None:
                    return found
        return None

    layers = getattr(cache, "layers", None)
    if layers is not None:
        found = visit(layers)
        if found is not None:
            return found
    return visit(cache)


def _cache_supports_batch_operation(cache: Any, method_name: str) -> bool:
    """Return whether every layer implements a cache batch helper.

    ``Cache`` exposes generic ``batch_*`` methods even when one of its
    constituent layers does not.  Qwen3.6's linear-attention cache is one
    such case: calling the outer helper would fail after partially traversing
    the cache.  Check the concrete layers before choosing that fast path.
    """

    if not callable(getattr(cache, method_name, None)):
        return False
    layers = getattr(cache, "layers", None)
    if layers is None:
        return True
    return all(callable(getattr(layer, method_name, None)) for layer in layers)


def _reorder_cache_rows(
    cache: Any,
    indices: torch.Tensor,
    *,
    index_device: torch.device | None = None,
) -> Any | None:
    """Use the standard beam-search cache API to duplicate or select rows.

    For a single-device decoder, placing the index on that device once avoids
    one CPU--GPU copy per cache layer.  Sharded decoders retain CPU indices:
    Hugging Face then moves them to each layer's own device.  The API mutates
    the cache in place, matching the native ``batch_*`` methods above.
    """

    method = getattr(cache, "reorder_cache", None)
    if not callable(method):
        return None
    target = {"dtype": torch.long}
    if index_device is not None:
        target["device"] = index_device
    method(indices.to(**target))
    return cache


def _repeat_cache(
    cache: Any,
    repeats: int,
    *,
    index_device: torch.device | None = None,
) -> Any:
    """Repeat a one-row cache for a frontier of ``repeats`` roots."""

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if repeats == 1:
        return cache
    method = getattr(cache, "batch_repeat_interleave", None)
    if callable(method) and _cache_supports_batch_operation(cache, "batch_repeat_interleave"):
        method(repeats)
        return cache
    # The cached tree always expands a one-row cloze prefill here.  Repeating
    # row zero through ``reorder_cache`` is exactly repeat-interleave for that
    # input and covers recurrent/linear-attention caches without bespoke code.
    repeated = _reorder_cache_rows(
        cache,
        torch.zeros(repeats, dtype=torch.long),
        index_device=index_device,
    )
    if repeated is not None:
        return repeated
    if isinstance(cache, (tuple, list)):
        layers = []
        for layer in cache:
            if isinstance(layer, (tuple, list)):
                layers.append(
                    tuple(
                        value.repeat_interleave(repeats, dim=0)
                        if isinstance(value, torch.Tensor)
                        else value
                        for value in layer
                    )
                )
            else:
                layers.append(layer)
        return type(cache)(layers)
    raise TypeError(f"unsupported HF cache type for repeat: {type(cache)!r}")


def _select_cache(
    cache: Any,
    indices: list[int] | torch.Tensor,
    *,
    index_device: torch.device | None = None,
) -> Any:
    """Select cache rows in the same order as a new tree frontier."""

    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.long)
    else:
        indices = indices.to(dtype=torch.long)
    method = getattr(cache, "batch_select_indices", None)
    if callable(method) and _cache_supports_batch_operation(cache, "batch_select_indices"):
        # DynamicCache layers of a device-mapped 70B live on different GPUs.
        # Hugging Face accepts CPU indices for every layer; moving the index
        # tensor onto the first layer's device (cuda:0) makes later layers on
        # cuda:1 raise `indices should be either on cpu or on the same device`.
        indices = indices.detach().to(device="cpu")
        method(indices)
        return cache
    selected = _reorder_cache_rows(cache, indices, index_device=index_device)
    if selected is not None:
        return selected
    if isinstance(cache, (tuple, list)):
        layers = []
        for layer in cache:
            if isinstance(layer, (tuple, list)):
                layers.append(
                    tuple(
                        value.index_select(0, indices.to(value.device))
                        if isinstance(value, torch.Tensor)
                        else value
                        for value in layer
                    )
                )
            else:
                layers.append(layer)
        return type(cache)(layers)
    raise TypeError(f"unsupported HF cache type for select: {type(cache)!r}")


def _run_hf(hf: Any, kwargs: dict[str, Any]) -> Any:
    """Call a cache-capable HF model with graceful older-wrapper fallback."""

    with torch.inference_mode():
        try:
            return hf(**kwargs)
        except TypeError:
            # Some older wrappers do not expose one or both optional output
            # controls.  Retry without them, but never change cache state or
            # the actual token inputs.
            retry = dict(kwargs)
            retry.pop("logits_to_keep", None)
            try:
                return hf(**retry)
            except TypeError:
                retry.pop("return_dict", None)
                return hf(**retry)


def _output_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("model output did not contain logits")
    return logits


def _output_cache(output: Any, fallback: Any) -> Any:
    cache = getattr(output, "past_key_values", None)
    if cache is None and isinstance(output, (tuple, list)) and len(output) > 1:
        cache = output[1]
    return fallback if cache is None else cache


def _model_forward(
    loaded: LoadedModel,
    token_ids: list[int],
    cache: Any,
) -> tuple[torch.Tensor, Any]:
    """Advance a batched cache by one token and return next-token logits."""

    if not token_ids:
        raise ValueError("cannot run an empty incremental batch")
    batch = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=loaded.model.input_device,
    ).reshape(-1, 1)
    output = _run_hf(
        loaded.hf,
        {
            "input_ids": batch,
            "past_key_values": cache,
            "use_cache": True,
            # Only the final position is consumed by BIND.  Avoid materializing
            # unnecessary vocabulary logits when the model supports this API.
            "logits_to_keep": 1,
            "return_dict": True,
        },
    )
    logits = _output_logits(output)[:, -1, :].float()
    return logits, _output_cache(output, cache)


def _context_forward(loaded: LoadedModel, context_ids: list[int]) -> ContextState:
    """Prefill the cloze scaffold once and retain its cache."""

    if not context_ids:
        raise ValueError("empty cloze context")
    batch = torch.tensor([context_ids], dtype=torch.long, device=loaded.model.input_device)
    output = _run_hf(
        loaded.hf,
        {
            "input_ids": batch,
            "attention_mask": torch.ones_like(batch),
            "use_cache": True,
            "logits_to_keep": 1,
            "return_dict": True,
        },
    )
    cache = _output_cache(output, None)
    if cache is None:
        raise RuntimeError("context forward did not return past_key_values")
    return ContextState(cache=cache, logits=_output_logits(output)[0, -1, :].float().detach())


class CachedTreeProposer:
    """BIND proposer with a request-level shared cloze cache.

    The class implements the standard rank-banded BIND policy only.  The
    non-default ``independent_top_n`` research mode of :func:`bindlens.bind.propose`
    remains available through that baseline function.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        *,
        s_k: int = S_K,
        seed_k: int = SEED_K,
        top_p: float = TOP_P,
        expand_k: int = EXPAND_K,
        max_tokens: int = MAX_TOKENS,
        root_bands: tuple[tuple[int, int, int], ...] = ROOT_BANDS,
        layer_batch_size: int = 8,
    ) -> None:
        if (
            int(s_k) < 1
            or int(seed_k) < 1
            or int(expand_k) < 1
            or int(max_tokens) < 1
            or int(layer_batch_size) < 1
        ):
            raise ValueError("s_k, seed_k, expand_k, max_tokens, and layer_batch_size must be positive")
        if not root_bands:
            raise ValueError("root_bands must not be empty")
        self.loaded = loaded
        self.s_k = int(s_k)
        self.seed_k = int(seed_k)
        self.top_p = float(top_p)
        self.expand_k = int(expand_k)
        self.max_tokens = int(max_tokens)
        # A request's source layers share the same cloze-conditioned decoder
        # tree.  Keep the first implementation conservative enough for an
        # 80-GiB card; the entry point remains tunable for larger cards.
        self.layer_batch_size = int(layer_batch_size)
        # A single-device model can reuse one GPU index tensor across all
        # cache layers.  Sharded models need CPU indices because different
        # cache layers reside on different devices.
        self._cache_index_device = (
            torch.device(loaded.model.input_device) if loaded.spec.device_map is None else None
        )
        self.root_bands = tuple(
            (int(start), int(stop), int(width)) for start, stop, width in root_bands
        )
        self.stats = TreeDecodeStats()

    def prepare_context(self, cloze_prompt: str) -> ContextState:
        """Build the one reusable cloze cache for an online request."""

        state = _context_forward(
            self.loaded,
            decode_prefix_ids(self.loaded.tokenizer, cloze_prompt),
        )
        self.stats.context_forwards += 1
        return state

    def _groups(self, source_ids: list[int]) -> list[_Group]:
        tokenizer = self.loaded.tokenizer
        groups: list[_Group] = []
        for start, stop, active_width in self.root_bands:
            roots = [
                int(token_id)
                for token_id in source_ids[start - 1 : stop]
                if is_word_initial_lexical_token(tokenizer, int(token_id))
            ][: self.seed_k]
            groups.append(_Group(roots=roots, max_active=active_width, completed={}))
        return groups

    def _propose_layer_batch(
        self,
        context: ContextState,
        layer_groups: list[tuple[int, list[_Group]]],
        main_content: torch.Tensor,
        fragments: torch.Tensor,
        end_ids: list[int],
    ) -> dict[int, list[dict[str, Any]]]:
        """Decode several source-layer trees in one frontier per depth.

        The BIND policy remains independent for every layer and rank band.
        Only model execution is shared: all rows at a given tree depth have
        the same context length, so they can advance through one cache and one
        batched model call.  Identical token prefixes are deduplicated because
        their continuation logits do not depend on the source layer.
        """

        if not layer_groups:
            return {}
        self.stats.layers += len(layer_groups)
        self.stats.layer_batches += 1

        groups: list[_Group] = []
        groups_by_layer: dict[int, list[int]] = {}
        for layer, layer_value in layer_groups:
            indices: list[int] = []
            for group in layer_value:
                indices.append(len(groups))
                groups.append(group)
            groups_by_layer[int(layer)] = indices

        root_pairs = [
            (group_index, root)
            for group_index, group in enumerate(groups)
            for root in group.roots
        ]
        if not root_pairs:
            return {int(layer): [] for layer, _groups in layer_groups}

        allowed, logprobs = restricted_logprobs(context.logits, main_content)
        root_tensor = torch.tensor(
            [root for _group, root in root_pairs],
            dtype=torch.long,
            device=allowed.device,
        )
        locations = torch.searchsorted(allowed, root_tensor)
        if bool((locations >= allowed.numel()).any()) or not bool(
            torch.equal(allowed.index_select(0, locations), root_tensor)
        ):
            raise RuntimeError("a filtered root is missing from the lexical vocabulary")
        scores = logprobs.index_select(0, locations).tolist()
        nodes = [
            _Node(ids=(root,), score=float(score), group=group_index)
            for (group_index, root), score in zip(root_pairs, scores)
        ]

        cache = _clone_cache(context.cache)
        _repeat_cache(cache, len(nodes), index_device=self._cache_index_device)
        current_logits, cache = _model_forward(
            self.loaded,
            [node.ids[-1] for node in nodes],
            cache,
        )
        self.stats.incremental_forwards += 1
        self.stats.incremental_rows += len(nodes)
        self.stats.max_frontier_rows = max(self.stats.max_frontier_rows, len(nodes))

        active = nodes
        for _depth in range(1, self.max_tokens):
            if not active:
                break
            children_by_group: dict[int, list[tuple[_Node, int]]] = {
                index: [] for index in range(len(groups))
            }
            for row_index, (node, row_logits) in enumerate(
                zip(active, current_logits.unbind(0))
            ):
                group = groups[node.group]
                self._record_terminal(
                    group,
                    node,
                    row_logits,
                    main_content=main_content,
                    end_ids=end_ids,
                    fragments=fragments,
                )
                for token_id, token_logprob in _main_actions(
                    row_logits,
                    main_content,
                    top_p=self.top_p,
                    expand_k=self.expand_k,
                ):
                    new_ids = (*node.ids, int(token_id))
                    if len(new_ids) != len(set(new_ids)):
                        continue
                    if not valid_phrase(decode_surface(self.loaded.tokenizer, new_ids)):
                        continue
                    children_by_group[node.group].append(
                        (
                            _Node(
                                ids=new_ids,
                                score=float(node.score) + float(token_logprob),
                                group=node.group,
                            ),
                            row_index,
                        )
                    )

            selected_children: list[tuple[_Node, int]] = []
            for group_index, group in enumerate(groups):
                rows = children_by_group[group_index]
                rows.sort(key=lambda pair: float(pair[0].score), reverse=True)
                selected_children.extend(rows[: group.max_active])
            if not selected_children:
                active = []
                break
            cache, current_logits = self._advance_unique(cache, selected_children)
            active = [node for node, _parent_index in selected_children]
            self.stats.max_frontier_rows = max(self.stats.max_frontier_rows, len(active))

        if active:
            for node, row_logits in zip(active, current_logits.unbind(0)):
                self._record_terminal(
                    groups[node.group],
                    node,
                    row_logits,
                    main_content=main_content,
                    end_ids=end_ids,
                    fragments=fragments,
                )

        output: dict[int, list[dict[str, Any]]] = {}
        for layer, group_indices in groups_by_layer.items():
            merged: dict[str, dict[str, Any]] = {}
            for group_index in group_indices:
                for row in groups[group_index].completed.values():
                    key = normalise(row["text"])
                    previous = merged.get(key)
                    if previous is None or float(row["score"]) > float(previous["score"]):
                        merged[key] = row
            output[int(layer)] = [
                {
                    "ids": list(row["ids"]),
                    "text": row["text"],
                    "score": float(row["score"]),
                    "tokens": [
                        visible(self.loaded.tokenizer, token_id) for token_id in row["ids"]
                    ],
                }
                for row in sorted(
                    merged.values(), key=lambda item: float(item["score"]), reverse=True
                )
            ]
        return output

    def _root_nodes(
        self,
        context: ContextState,
        groups: list[_Group],
        content_ids: torch.Tensor,
    ) -> tuple[list[_Node], Any, torch.Tensor]:
        """Score every band root and advance its first-token cache row."""

        root_pairs = [
            (group_index, root)
            for group_index, group in enumerate(groups)
            for root in group.roots
        ]
        if not root_pairs:
            return [], None, torch.empty((0, 0), device=context.logits.device)

        allowed, logprobs = restricted_logprobs(context.logits, content_ids)
        root_tensor = torch.tensor(
            [root for _group, root in root_pairs],
            dtype=torch.long,
            device=allowed.device,
        )
        locations = torch.searchsorted(allowed, root_tensor)
        if bool((locations >= allowed.numel()).any()) or not bool(
            torch.equal(allowed.index_select(0, locations), root_tensor)
        ):
            raise RuntimeError("a filtered root is missing from the lexical vocabulary")
        scores = logprobs.index_select(0, locations).tolist()
        nodes = [
            _Node(ids=(root,), score=float(score), group=group_index)
            for (group_index, root), score in zip(root_pairs, scores)
        ]

        cache = _clone_cache(context.cache)
        _repeat_cache(cache, len(nodes), index_device=self._cache_index_device)
        logits, cache = _model_forward(self.loaded, [node.ids[-1] for node in nodes], cache)
        self.stats.incremental_forwards += 1
        self.stats.incremental_rows += len(nodes)
        return nodes, cache, logits

    def _advance_unique(
        self,
        cache: Any,
        children: list[tuple[_Node, int]],
    ) -> tuple[Any, torch.Tensor]:
        """Advance each distinct child prefix once, then restore duplicates."""

        unique: list[tuple[_Node, int]] = []
        first_for_prefix: dict[tuple[int, ...], int] = {}
        inverse: list[int] = []
        for node, parent_index in children:
            unique_index = first_for_prefix.get(node.ids)
            if unique_index is None:
                unique_index = len(unique)
                first_for_prefix[node.ids] = unique_index
                unique.append((node, parent_index))
            inverse.append(unique_index)

        self.stats.deduplicated_rows += len(children) - len(unique)
        parent_indices = [parent_index for _node, parent_index in unique]
        child_tokens = [node.ids[-1] for node, _parent_index in unique]
        next_cache = _select_cache(
            cache,
            parent_indices,
            index_device=self._cache_index_device,
        )
        logits, next_cache = _model_forward(self.loaded, child_tokens, next_cache)
        self.stats.incremental_forwards += 1
        self.stats.incremental_rows += len(unique)
        if len(inverse) != len(unique):
            inverse_tensor = torch.tensor(inverse, dtype=torch.long, device=logits.device)
            next_cache = _select_cache(
                next_cache,
                inverse_tensor,
                index_device=self._cache_index_device,
            )
            logits = logits.index_select(0, inverse_tensor)
        return next_cache, logits

    def _record_terminal(
        self,
        group: _Group,
        node: _Node,
        row_logits: torch.Tensor,
        *,
        main_content: torch.Tensor,
        end_ids: list[int],
        fragments: torch.Tensor,
    ) -> None:
        if len(node.ids) < 2:
            return
        selected = _terminal_logprobs(
            row_logits=row_logits,
            main_content_ids=main_content,
            end_ids=end_ids,
            top_p=self.top_p,
            expand_k=self.expand_k,
            fragments=fragments.to(device=row_logits.device),
        )
        for terminal_logprob in selected:
            text = decode_surface(self.loaded.tokenizer, node.ids)
            key = normalise(text)
            if not key or not valid_phrase(text):
                continue
            score = float(node.score) + float(terminal_logprob)
            previous = group.completed.get(key)
            if previous is None or score > float(previous["score"]):
                group.completed[key] = {"ids": node.ids, "text": text, "score": score}

    def propose_layer(
        self,
        context: ContextState,
        jlens_logits: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Propose raw BIND candidates for one source layer."""

        self.stats.layers += 1
        tokenizer = self.loaded.tokenizer
        end_ids = sorted(int(token_id) for token_id in control_ids(tokenizer)["end_ids"])
        source_ids = topk_ids(jlens_logits, self.s_k)
        vocab_size = int(
            getattr(self.loaded.hf.config, "vocab_size", None)
            or getattr(getattr(self.loaded.hf.config, "text_config", None), "vocab_size", None)
            or len(tokenizer)
        )
        decoder_device = self.loaded.model.input_device
        main_content = lexical_ids(tokenizer, vocab_size, decoder_device)
        fragments = fragment_ids(tokenizer, vocab_size, decoder_device)
        groups = self._groups(source_ids)
        nodes, cache, current_logits = self._root_nodes(context, groups, main_content)
        if not nodes:
            return []

        active = nodes
        # The baseline performs a cap forward after the final expansion solely
        # to score terminals.  Here ``current_logits`` already holds exactly
        # those next-token logits, so the cap pass is unnecessary.
        for _depth in range(1, self.max_tokens):
            if not active:
                break
            children_by_group: dict[int, list[tuple[_Node, int]]] = {
                index: [] for index in range(len(groups))
            }
            for row_index, (node, row_logits) in enumerate(zip(active, current_logits.unbind(0))):
                group = groups[node.group]
                self._record_terminal(
                    group,
                    node,
                    row_logits,
                    main_content=main_content,
                    end_ids=end_ids,
                    fragments=fragments,
                )
                for token_id, token_logprob in _main_actions(
                    row_logits,
                    main_content,
                    top_p=self.top_p,
                    expand_k=self.expand_k,
                ):
                    new_ids = (*node.ids, int(token_id))
                    if len(new_ids) != len(set(new_ids)):
                        continue
                    if not valid_phrase(decode_surface(tokenizer, new_ids)):
                        continue
                    children_by_group[node.group].append(
                        (
                            _Node(
                                ids=new_ids,
                                score=float(node.score) + float(token_logprob),
                                group=node.group,
                            ),
                            row_index,
                        )
                    )

            selected_children: list[tuple[_Node, int]] = []
            for group_index, group in enumerate(groups):
                rows = children_by_group[group_index]
                rows.sort(key=lambda pair: float(pair[0].score), reverse=True)
                selected_children.extend(rows[: group.max_active])
            if not selected_children:
                active = []
                break
            cache, current_logits = self._advance_unique(cache, selected_children)
            active = [node for node, _parent_index in selected_children]

        if active:
            for node, row_logits in zip(active, current_logits.unbind(0)):
                self._record_terminal(
                    groups[node.group],
                    node,
                    row_logits,
                    main_content=main_content,
                    end_ids=end_ids,
                    fragments=fragments,
                )

        merged: dict[str, dict[str, Any]] = {}
        for group in groups:
            for row in group.completed.values():
                key = normalise(row["text"])
                previous = merged.get(key)
                if previous is None or float(row["score"]) > float(previous["score"]):
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

    def propose_all(
        self,
        cloze_prompt: str,
        lens_logits_by_layer: dict[int, torch.Tensor],
    ) -> dict[int, list[Candidate]]:
        """Propose selected candidates for every layer of one request."""

        raw = self.propose_all_raw(cloze_prompt, lens_logits_by_layer)
        return {int(layer): select_candidates(rows) for layer, rows in raw.items()}

    def propose_all_raw(
        self,
        cloze_prompt: str,
        lens_logits_by_layer: dict[int, torch.Tensor],
    ) -> dict[int, list[dict[str, Any]]]:
        """Return raw proposals with cross-layer frontier batching enabled."""

        context = self.prepare_context(cloze_prompt)
        tokenizer = self.loaded.tokenizer
        end_ids = sorted(int(token_id) for token_id in control_ids(tokenizer)["end_ids"])
        vocab_size = int(
            getattr(self.loaded.hf.config, "vocab_size", None)
            or getattr(getattr(self.loaded.hf.config, "text_config", None), "vocab_size", None)
            or len(tokenizer)
        )
        decoder_device = self.loaded.model.input_device
        main_content = lexical_ids(tokenizer, vocab_size, decoder_device)
        fragments = fragment_ids(tokenizer, vocab_size, decoder_device)
        items = [
            (int(layer), self._groups(topk_ids(logits, self.s_k)))
            for layer, logits in lens_logits_by_layer.items()
        ]
        output: dict[int, list[dict[str, Any]]] = {}
        batch_size = max(1, int(self.layer_batch_size))
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            output.update(
                self._propose_layer_batch(
                    context,
                    batch,
                    main_content,
                    fragments,
                    end_ids,
                )
            )
        return output


class VectorizedCachedTreeProposer(CachedTreeProposer):
    """Shared-KV tree proposer with a batched execution of the BIND policy.

    Candidate roots, nucleus selection, terminal vetoes, and group pruning are
    unchanged from :class:`CachedTreeProposer`.  This variant computes those
    policy decisions for a full frontier at once and transfers only the small
    selected tables back to Python.
    """

    def propose_layer(
        self,
        context: ContextState,
        jlens_logits: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Propose one layer through the same vectorized frontier path."""

        tokenizer = self.loaded.tokenizer
        end_ids = sorted(int(token_id) for token_id in control_ids(tokenizer)["end_ids"])
        vocab_size = int(
            getattr(self.loaded.hf.config, "vocab_size", None)
            or getattr(getattr(self.loaded.hf.config, "text_config", None), "vocab_size", None)
            or len(tokenizer)
        )
        decoder_device = self.loaded.model.input_device
        main_content = lexical_ids(tokenizer, vocab_size, decoder_device)
        fragments = fragment_ids(tokenizer, vocab_size, decoder_device)
        groups = self._groups(topk_ids(jlens_logits, self.s_k))
        return self._propose_layer_batch(
            context,
            [(-1, groups)],
            main_content,
            fragments,
            end_ids,
        )[-1]

    def _propose_layer_batch(
        self,
        context: ContextState,
        layer_groups: list[tuple[int, list[_Group]]],
        main_content: torch.Tensor,
        fragments: torch.Tensor,
        end_ids: list[int],
    ) -> dict[int, list[dict[str, Any]]]:
        """Vectorize the reference per-row policy loop across a frontier."""

        if not layer_groups:
            return {}
        self.stats.layers += len(layer_groups)
        self.stats.layer_batches += 1

        groups: list[_Group] = []
        groups_by_layer: dict[int, list[int]] = {}
        for layer, layer_value in layer_groups:
            indices: list[int] = []
            for group in layer_value:
                indices.append(len(groups))
                groups.append(group)
            groups_by_layer[int(layer)] = indices

        root_pairs = [
            (group_index, root)
            for group_index, group in enumerate(groups)
            for root in group.roots
        ]
        if not root_pairs:
            return {int(layer): [] for layer, _groups in layer_groups}

        allowed, logprobs = restricted_logprobs(context.logits, main_content)
        root_tensor = torch.tensor(
            [root for _group, root in root_pairs],
            dtype=torch.long,
            device=allowed.device,
        )
        locations = torch.searchsorted(allowed, root_tensor)
        if bool((locations >= allowed.numel()).any()) or not bool(
            torch.equal(allowed.index_select(0, locations), root_tensor)
        ):
            raise RuntimeError("a filtered root is missing from the lexical vocabulary")
        scores = logprobs.index_select(0, locations).tolist()
        nodes = [
            _Node(ids=(root,), score=float(score), group=group_index)
            for (group_index, root), score in zip(root_pairs, scores)
        ]

        cache = _clone_cache(context.cache)
        _repeat_cache(cache, len(nodes), index_device=self._cache_index_device)
        current_logits, cache = _model_forward(
            self.loaded,
            [node.ids[-1] for node in nodes],
            cache,
        )
        self.stats.incremental_forwards += 1
        self.stats.incremental_rows += len(nodes)
        self.stats.max_frontier_rows = max(self.stats.max_frontier_rows, len(nodes))

        active = nodes
        for _depth in range(1, self.max_tokens):
            if not active:
                break
            selected_children = self._select_children_from_logits(
                active,
                current_logits,
                groups,
                main_content=main_content,
                fragments=fragments,
                end_ids=end_ids,
            )
            if not selected_children:
                active = []
                break
            cache, current_logits = self._advance_unique(cache, selected_children)
            active = [node for node, _parent_index in selected_children]
            self.stats.max_frontier_rows = max(self.stats.max_frontier_rows, len(active))

        if active:
            self._apply_terminals_from_logits(
                active,
                current_logits,
                groups,
                main_content=main_content,
                fragments=fragments,
                end_ids=end_ids,
            )

        output: dict[int, list[dict[str, Any]]] = {}
        for layer, group_indices in groups_by_layer.items():
            merged: dict[str, dict[str, Any]] = {}
            for group_index in group_indices:
                for row in groups[group_index].completed.values():
                    key = normalise(row["text"])
                    previous = merged.get(key)
                    if previous is None or float(row["score"]) > float(previous["score"]):
                        merged[key] = row
            output[int(layer)] = [
                {
                    "ids": list(row["ids"]),
                    "text": row["text"],
                    "score": float(row["score"]),
                    "tokens": [visible(self.loaded.tokenizer, token_id) for token_id in row["ids"]],
                }
                for row in sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
            ]
        return output

    def _apply_terminals_from_logits(
        self,
        active: list[_Node],
        current_logits: torch.Tensor,
        groups: list[_Group],
        *,
        main_content: torch.Tensor,
        fragments: torch.Tensor,
        end_ids: list[int],
    ) -> None:
        eligible = [len(node.ids) >= 2 for node in active]
        terminals = _terminal_logprobs_batched(
            current_logits,
            main_content,
            end_ids,
            fragments,
            top_p=self.top_p,
            expand_k=self.expand_k,
            eligible=eligible,
        )
        tokenizer = self.loaded.tokenizer
        for node, selected_end in zip(active, terminals):
            if not selected_end:
                continue
            text = decode_surface(tokenizer, node.ids)
            key = normalise(text)
            if not key or not valid_phrase(text):
                continue
            group = groups[node.group]
            for terminal_logprob in selected_end:
                score = float(node.score) + float(terminal_logprob)
                previous = group.completed.get(key)
                if previous is None or score > float(previous["score"]):
                    group.completed[key] = {
                        "ids": node.ids,
                        "text": text,
                        "score": score,
                    }

    def _select_children_from_logits(
        self,
        active: list[_Node],
        current_logits: torch.Tensor,
        groups: list[_Group],
        *,
        main_content: torch.Tensor,
        fragments: torch.Tensor,
        end_ids: list[int],
    ) -> list[tuple[_Node, int]]:
        self._apply_terminals_from_logits(
            active,
            current_logits,
            groups,
            main_content=main_content,
            fragments=fragments,
            end_ids=end_ids,
        )
        actions = _main_actions_batched(
            current_logits,
            main_content,
            top_p=self.top_p,
            expand_k=self.expand_k,
        )
        children_by_group: dict[int, list[tuple[_Node, int]]] = {
            index: [] for index in range(len(groups))
        }
        tokenizer = self.loaded.tokenizer
        for row_index, node in enumerate(active):
            for token_id, token_logprob in actions[row_index]:
                new_ids = (*node.ids, int(token_id))
                if len(new_ids) != len(set(new_ids)):
                    continue
                if not valid_phrase(decode_surface(tokenizer, new_ids)):
                    continue
                children_by_group[node.group].append(
                    (
                        _Node(
                            ids=new_ids,
                            score=float(node.score) + float(token_logprob),
                            group=node.group,
                        ),
                        row_index,
                    )
                )
        selected_children: list[tuple[_Node, int]] = []
        for group_index, group in enumerate(groups):
            rows = children_by_group[group_index]
            rows.sort(key=lambda pair: float(pair[0].score), reverse=True)
            selected_children.extend(rows[: group.max_active])
        return selected_children


__all__ = ["CachedTreeProposer", "ContextState", "TreeDecodeStats", "VectorizedCachedTreeProposer"]
