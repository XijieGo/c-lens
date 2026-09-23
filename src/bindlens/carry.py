"""CARRY: build a concept direction from post-concept carrier residuals."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from jlens.hooks import ActivationRecorder

from bindlens.models import LoadedModel
from bindlens.paths import carrier_path
from bindlens.protocol import CARRIER_PREFIX, CARRIER_SUFFIX, POOL_MODE


def load_calibration(model: str, path: Path | None = None) -> dict[str, Any]:
    return torch.load(str(path or carrier_path(model)), map_location="cpu", weights_only=True)


def concept_grid(
    tokenizer: Any,
    concept: str,
    *,
    prefix: str = CARRIER_PREFIX,
    suffix: str = CARRIER_SUFFIX,
    allow_empty_suffix: bool = False,
) -> dict[str, Any]:
    """Tokenization ledger for a carrier with a prompt-final suffix.

    ``prefix`` and ``suffix`` default to the frozen protocol text.  Keeping
    them as explicit optional arguments lets exploratory carrier-content
    trials reuse the exact boundary and alignment checks without changing the
    protocol constants.  ``allow_empty_suffix`` is only for experiments that
    deliberately read the concept-final residual rather than a post-concept
    carrier state.
    """
    prompt = f"{prefix}{concept}{suffix}"
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = [int(value) for value in encoded.input_ids]
    offsets = [(int(start), int(end)) for start, end in encoded.offset_mapping]
    concept_start = len(prefix)
    concept_end = concept_start + len(concept)
    concept_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and end > concept_start and start < concept_end
    ]
    if not concept_indices:
        raise ValueError(f"{concept!r}: no concept tokens")
    if any(offsets[index][1] > concept_end for index in concept_indices):
        raise ValueError(f"{concept!r}: concept token crosses into suffix")
    suffix_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start >= concept_end
    ]
    if not suffix_indices and not allow_empty_suffix:
        raise ValueError(f"{concept!r}: empty suffix")
    if suffix_indices and suffix_indices != list(range(suffix_indices[0], suffix_indices[-1] + 1)):
        raise ValueError(f"{concept!r}: suffix tokens are not contiguous")
    if suffix_indices and suffix_indices[-1] != len(input_ids) - 1:
        raise ValueError(f"{concept!r}: suffix is not prompt-final")
    if not suffix_indices and concept_indices[-1] != len(input_ids) - 1:
        raise ValueError(f"{concept!r}: concept-final state is not prompt-final")
    return {
        "concept": concept,
        "prompt": prompt,
        "input_ids": input_ids,
        "n_concept_tokens": len(concept_indices),
        "concept_ids": [input_ids[index] for index in concept_indices],
        "n_suffix_tokens": len(suffix_indices),
        "suffix_ids": [input_ids[index] for index in suffix_indices],
    }


def aligned_grids(
    tokenizer: Any,
    concepts: list[str],
    *,
    prefix: str = CARRIER_PREFIX,
    suffix: str = CARRIER_SUFFIX,
    allow_empty_suffix: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    grids: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    reference: list[int] | None = None
    for concept in concepts:
        try:
            grid = concept_grid(
                tokenizer,
                concept,
                prefix=prefix,
                suffix=suffix,
                allow_empty_suffix=allow_empty_suffix,
            )
        except ValueError as error:
            failed.append({"concept": concept, "reason": str(error)})
            continue
        if reference is None:
            reference = grid["suffix_ids"]
        elif grid["suffix_ids"] != reference:
            failed.append({"concept": concept, "reason": "suffix tokenization drifted"})
            continue
        grids.append(grid)
    return grids, failed


def reconstruct_pooled_carrier(
    positions: torch.Tensor,
    means: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    centered = positions.float() - means.float()[None, :, :]
    return (centered * weights.float()[None, :, None]).sum(dim=1)


def pool_weights(calibration: dict[str, Any], layer: int, mode: str = POOL_MODE) -> torch.Tensor:
    capture_layers = [int(item) for item in calibration["capture_layers"]]
    source_layers = [int(item) for item in calibration["source_layers"]]
    capture_index = capture_layers.index(layer)
    if layer in source_layers:
        jlens_index = source_layers.index(layer)
    else:
        nearest = min(source_layers, key=lambda value: abs(value - layer))
        jlens_index = source_layers.index(nearest)
    deviations = calibration["deviations"][capture_index, :, 1:, :].float()
    directions = calibration["centered_jlens_directions"][jlens_index].float()
    concept_norm = deviations.norm(dim=-1).clamp_min(1e-8)
    direction_norm = directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (deviations * directions[:, None, :]).sum(dim=-1) / (concept_norm * direction_norm)
    scores = cosine.mean(dim=0).clamp_min(0.0)
    if mode == "uniform":
        weights = torch.ones_like(scores)
    elif mode == "top3":
        weights = torch.zeros_like(scores)
        count = min(3, int(scores.numel()))
        weights[torch.topk(scores, k=count).indices] = 1.0
    else:
        weights = scores.clone()
        if float(weights.sum().item()) <= 0:
            weights = torch.ones_like(scores)
    return weights / weights.sum().clamp_min(1e-12)


@torch.inference_mode()
def capture_suffix(
    loaded: LoadedModel,
    grids: list[dict[str, Any]],
    layers: list[int],
    *,
    batch_size: int = 8,
    include_concept_final: bool = False,
) -> dict[int, torch.Tensor]:
    n_suffix = int(grids[0]["n_suffix_tokens"])
    n_pos = n_suffix + (1 if include_concept_final else 0)
    if n_pos <= 0:
        raise ValueError("capture requires a concept-final token or at least one suffix token")
    result = {
        layer: torch.empty((len(grids), n_pos, int(loaded.model.d_model)), dtype=torch.float32)
        for layer in layers
    }
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, grid in enumerate(grids):
        by_length[len(grid["input_ids"])].append(index)
    for length, group in sorted(by_length.items()):
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batch = torch.tensor([grids[i]["input_ids"] for i in chunk], dtype=torch.long)
            with ActivationRecorder(loaded.model.layers, at=layers) as recorder:
                loaded.model.forward(batch.to(loaded.model.input_device))
                for layer in layers:
                    hidden = recorder.activations[layer][:, -n_pos:, :].detach().float().cpu()
                    for local, original in enumerate(chunk):
                        result[layer][original] = hidden[local]
    return result


@torch.inference_mode()
def capture_suffix_on_device(
    loaded: LoadedModel,
    grids: list[dict[str, Any]],
    layers: list[int],
    *,
    batch_size: int = 8,
    include_concept_final: bool = False,
) -> dict[int, torch.Tensor]:
    """Capture suffix states without staging every carrier batch through CPU.

    This is numerically the same carrier capture as :func:`capture_suffix`,
    but retains the activation buffer on the model input device.  The live
    serving path can therefore pool candidate directions on GPU and pass them
    directly to SCORE.  The regular function above remains the conservative
    CPU-returning API used by offline experiment scripts.
    """

    n_suffix = int(grids[0]["n_suffix_tokens"])
    n_pos = n_suffix + (1 if include_concept_final else 0)
    if n_pos <= 0:
        raise ValueError("capture requires a concept-final token or at least one suffix token")
    device = loaded.model.input_device
    result = {
        int(layer): torch.empty(
            (len(grids), n_pos, int(loaded.model.d_model)),
            dtype=torch.float32,
            device=device,
        )
        for layer in layers
    }
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, grid in enumerate(grids):
        by_length[len(grid["input_ids"])].append(index)
    for _length, group in sorted(by_length.items()):
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batch = torch.tensor(
                [grids[index]["input_ids"] for index in chunk],
                dtype=torch.long,
                device=device,
            )
            with ActivationRecorder(loaded.model.layers, at=list(layers)) as recorder:
                loaded.model.forward(batch)
                indices = torch.tensor(chunk, dtype=torch.long, device=device)
                for layer in layers:
                    hidden = recorder.activations[int(layer)][:, -n_pos:, :].detach().to(
                        device=device,
                        dtype=torch.float32,
                    )
                    result[int(layer)].index_copy_(0, indices, hidden)
    return result


def construct_directions(
    loaded: LoadedModel,
    concepts: list[str],
    calibration: dict[str, Any] | None = None,
    *,
    layer: int | None = None,
    batch_size: int = 8,
    fit_concepts: list[str] | None = None,
    prefix: str = CARRIER_PREFIX,
    suffix: str = CARRIER_SUFFIX,
) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
    """Fit-and-apply a centered-positive reader for a non-primary control.

    The reader is fit on the independent corpus supplied by ``fit_concepts``
    (or the public calibration split when it is omitted), then applied to
    ``concepts``.  Primary E2E and Swap instead load the same immutable
    artifact through :func:`construct_directions_from_reader`.
    ``calibration`` is accepted so existing call sites stay unchanged; pooling
    no longer reads that tensor.
    """
    _ = calibration
    reader, metadata = fit_positive_pool_reader(
        loaded,
        prefix=prefix,
        suffix=suffix,
        layer=layer,
        batch_size=batch_size,
        fit_concepts=fit_concepts,
    )
    return construct_directions_from_reader(
        loaded,
        concepts,
        reader,
        layer=int(metadata["carrier_layer"]),
        batch_size=batch_size,
        reference_suffix=[int(token) for token in metadata["suffix_ids"]],
        prefix=prefix,
        suffix=suffix,
    )


def fit_positive_pool(positions: torch.Tensor, directions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fit a positive suffix pool in centered target coordinates.

    The fitted pool predicts ``directions - direction_mean`` from
    ``positions - means``.  It is therefore a *residual* predictor.  A reader
    built from this fit must restore ``direction_mean`` before its output is
    compared to ordinary (uncentered) J-lens or vocabulary directions.
    """
    means = positions.float().mean(dim=0)
    direction_mean = directions.float().mean(dim=0)
    centered = positions.float() - means[None, :, :]
    targets = directions.float() - direction_mean[None, :]
    cosine = (centered * targets[:, None, :]).sum(dim=-1)
    cosine = cosine / (centered.norm(dim=-1) * targets.norm(dim=-1)[:, None]).clamp_min(1e-12)
    scores = cosine.mean(dim=0).clamp_min(0.0)
    weights = scores / scores.sum().clamp_min(1e-12)
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = torch.full_like(scores, 1.0 / max(int(scores.numel()), 1))
    return {"means": means, "direction_mean": direction_mean, "weights": weights}


def apply_positive_pool(positions: torch.Tensor, fit: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the centered-target residual predicted by a fitted pool.

    This intentionally does not add ``fit['direction_mean']``: callers using
    it during fitting evaluate residual alignment.  Frozen readers restore
    that intercept in :func:`apply_frozen_positive_pool`.
    """
    centered = positions.float() - fit["means"].float()[None, :, :]
    weights = fit["weights"].float()
    return (centered * weights[None, :, None]).sum(dim=1)


def fit_positive_pool_reader(
    loaded: LoadedModel,
    *,
    prefix: str,
    suffix: str,
    layer: int | None = None,
    batch_size: int = 8,
    fit_concepts: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a centered, non-negative carrier-position reader once.

    The target remains the J-lens direction of the exact concept token.  If a
    carrier prompt changes token boundaries, only concepts that are still one
    exact token are retained for fitting; using a token-average target would
    silently change the reconstruction method.  Since fitting centers those
    target directions, the artifact stores their mean as ``output_bias``.
    Hence its inference output is in the same absolute coordinate system as
    SCORE's uncentered query and vocabulary directions:

    ``pool(suffix_state - fit_state_mean) + fit_target_direction_mean``.
    """
    from bindlens.io import ranking_fit_concepts
    from bindlens.read import token_directions

    carrier_layer = int(layer if layer is not None else loaded.spec.carrier_layer)
    corpus = list(fit_concepts) if fit_concepts is not None else ranking_fit_concepts()
    if len(corpus) != len(set(corpus)):
        raise ValueError("positive-pool fit corpus contains duplicate concepts")
    grids, failed = aligned_grids(loaded.tokenizer, corpus, prefix=prefix, suffix=suffix)
    if failed or [grid["concept"] for grid in grids] != corpus:
        raise ValueError(f"positive-pool fit corpus failed alignment: {failed[:3]}")
    singleton_grids = [grid for grid in grids if int(grid["n_concept_tokens"]) == 1]
    if not singleton_grids:
        raise ValueError("positive-pool fit has no exact singleton concept tokens")
    positions = capture_suffix(loaded, singleton_grids, [carrier_layer], batch_size=batch_size)[carrier_layer]
    token_ids = [int(grid["concept_ids"][0]) for grid in singleton_grids]
    targets = token_directions(loaded, token_ids, carrier_layer)
    fit = fit_positive_pool(positions, targets)
    pooled = apply_positive_pool(positions, fit)
    centered_targets = targets.float() - fit["direction_mean"].float()[None, :]
    paired = (pooled * centered_targets).sum(dim=-1) / (
        pooled.norm(dim=-1) * centered_targets.norm(dim=-1)
    ).clamp_min(1e-12)
    target_mean = fit["direction_mean"].detach().float().cpu()
    reader = {
        "kind": "positive_pool",
        "means": fit["means"].detach().float().cpu(),
        "direction_mean": target_mean,
        "weights": fit["weights"].detach().float().cpu(),
        "active_positions": list(range(int(fit["means"].shape[0]))),
        # ``fit_positive_pool`` regresses the centered target, so preserve
        # the target intercept explicitly rather than returning a residual in
        # an absolute SCORE coordinate system.
        "output_bias": target_mean.clone(),
    }
    metadata = {
        "schema_version": 3,
        "model": loaded.name,
        "carrier_prefix": str(prefix),
        "carrier_suffix": str(suffix),
        "carrier_layer": carrier_layer,
        "suffix_ids": [int(token) for token in grids[0]["suffix_ids"]],
        "pooling": "centered_positive_weight_pool_with_target_mean_intercept",
        "fit_target": "exact_singleton_concept_token_jlens_direction",
        "inference_target_reconstruction": "pool(centered_suffix_states)+fit_target_direction_mean",
        "output_bias_kind": "fit_target_direction_mean",
        "output_bias_coefficient": 1.0,
        "fit_corpus_requested": len(corpus),
        "fit_corpus_aligned": len(grids),
        "fit_singleton_used": len(singleton_grids),
        "fit_multitoken_excluded": len(grids) - len(singleton_grids),
        "fit_mean_paired_cosine": float(paired.mean().item()),
        "fit_mean_paired_cosine_centered_residual": float(paired.mean().item()),
    }
    return reader, metadata


def load_positive_pool_reader(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reader = dict((payload.get("readers") or {}).get("positive_pool") or {})
    if reader.get("kind") != "positive_pool":
        raise ValueError(f"{path}: missing positive_pool reader")
    if not isinstance(reader.get("means"), torch.Tensor) or not isinstance(reader.get("weights"), torch.Tensor):
        raise ValueError(f"{path}: positive_pool is missing means or weights")
    return reader


def load_positive_pool_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the immutable reader payload written by the trial fit CLI."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata") or {})
    reader = dict((payload.get("readers") or {}).get("positive_pool") or {})
    if reader.get("kind") != "positive_pool":
        raise ValueError(f"{path}: missing positive_pool reader")
    if not isinstance(reader.get("means"), torch.Tensor) or not isinstance(reader.get("weights"), torch.Tensor):
        raise ValueError(f"{path}: positive_pool is missing means or weights")
    return reader, metadata


def validate_positive_pool_artifact(
    loaded: LoadedModel,
    metadata: dict[str, Any],
    reader: dict[str, Any],
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    layer: int | None = None,
) -> None:
    """Reject a reader if its model, carrier text, or tensor shape differs."""
    expected = {
        "model": loaded.name,
        "carrier_layer": int(layer if layer is not None else loaded.spec.carrier_layer),
    }
    if prefix is not None:
        expected["carrier_prefix"] = str(prefix)
    if suffix is not None:
        expected["carrier_suffix"] = str(suffix)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"reader metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}"
            )
    suffix_ids = metadata.get("suffix_ids")
    if not isinstance(suffix_ids, list) or not suffix_ids:
        raise RuntimeError("reader metadata has no fixed suffix_ids")
    means = reader["means"]
    weights = reader["weights"]
    direction_mean = reader.get("direction_mean")
    if means.ndim != 2 or int(means.shape[1]) != int(loaded.model.d_model):
        raise RuntimeError(f"reader mean shape {tuple(means.shape)} does not match model d_model")
    if weights.ndim != 1 or int(weights.shape[0]) != int(means.shape[0]):
        raise RuntimeError(f"reader weight shape {tuple(weights.shape)} does not match reader means")
    if not isinstance(direction_mean, torch.Tensor) or direction_mean.ndim != 1:
        raise RuntimeError("reader is missing its one-dimensional target-direction mean")
    if int(direction_mean.shape[0]) != int(loaded.model.d_model):
        raise RuntimeError(
            f"reader target-direction mean shape {tuple(direction_mean.shape)} "
            f"does not match model d_model"
        )
    if not torch.isfinite(means).all() or not torch.isfinite(weights).all() or not torch.isfinite(direction_mean).all():
        raise RuntimeError("reader has non-finite fitted parameters")
    if abs(float(weights.sum().item()) - 1.0) > 1e-4 or bool((weights < 0).any()):
        raise RuntimeError("reader is not a normalized non-negative pool")
    bias = reader.get("output_bias")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
            raise RuntimeError("reader output_bias must be a one-dimensional tensor")
        if int(bias.shape[0]) != int(loaded.model.d_model):
            raise RuntimeError(
                f"reader output_bias shape {tuple(bias.shape)} does not match model d_model"
            )
        if not torch.isfinite(bias).all():
            raise RuntimeError("reader output_bias has non-finite values")
    bias_kind = metadata.get("output_bias_kind")
    if bias_kind == "fit_target_direction_mean":
        if float(metadata.get("output_bias_coefficient", 1.0)) != 1.0:
            raise RuntimeError("target-mean reader must use an output_bias coefficient of exactly one")
        if not isinstance(bias, torch.Tensor) or not torch.equal(bias, direction_mean):
            raise RuntimeError("target-mean reader output_bias must exactly equal direction_mean")


def apply_frozen_positive_pool(positions: torch.Tensor, reader: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Apply a reader in the coordinate system recorded by its artifact.

    Readers can store the fitted target mean as ``output_bias`` because
    their positive pool predicts centered target residuals.  Artifacts that
    omit that field retain their explicitly encoded zero-intercept behavior.
    """
    active = [int(value) for value in reader.get("active_positions") or range(int(positions.shape[1]))]
    means = reader["means"].to(device=device, dtype=torch.float32)
    weights = reader["weights"].to(device=device, dtype=torch.float32)
    bias = reader.get("output_bias")
    if bias is None:
        bias = torch.zeros(means.shape[-1], dtype=torch.float32)
    bias = bias.to(device=device, dtype=torch.float32)
    centered = positions[:, active, :].to(device=device, dtype=torch.float32) - means[None, :, :]
    if weights.ndim == 1:
        result = (centered * weights[None, :, None]).sum(dim=1)
    else:
        result = (centered * weights[None, :, :]).sum(dim=1)
    return result + bias[None, :]


def construct_directions_from_reader(
    loaded: LoadedModel,
    concepts: list[str],
    reader: dict[str, Any],
    *,
    layer: int | None = None,
    batch_size: int = 8,
    reference_suffix: list[int] | None = None,
    prefix: str = CARRIER_PREFIX,
    suffix: str = CARRIER_SUFFIX,
    concept_chunk_size: int = 0,
) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
    layer = int(layer if layer is not None else loaded.spec.carrier_layer)
    grids, failed = aligned_grids(loaded.tokenizer, concepts, prefix=prefix, suffix=suffix)
    if not grids:
        return {}, failed
    suffix_ids = (
        [int(token) for token in reference_suffix]
        if reference_suffix is not None
        else [int(token) for token in grids[0]["suffix_ids"]]
    )
    kept: list[dict[str, Any]] = []
    for grid in grids:
        if [int(token) for token in grid["suffix_ids"]] != suffix_ids:
            failed.append({"concept": grid["concept"], "reason": "suffix tokenization drifted"})
            continue
        kept.append(grid)
    if not kept:
        return {}, failed
    n_means = int(reader["means"].shape[0])
    chunk_size = int(concept_chunk_size) if concept_chunk_size else len(kept)
    if chunk_size <= 0:
        raise ValueError("concept_chunk_size must be positive")
    vectors: dict[str, torch.Tensor] = {}
    for start in range(0, len(kept), chunk_size):
        chunk = kept[start : start + chunk_size]
        positions = capture_suffix(loaded, chunk, [layer], batch_size=batch_size)[layer]
        if int(positions.shape[1]) != n_means:
            raise RuntimeError(
                f"suffix length {int(positions.shape[1])} != reader means {n_means}"
            )
        pooled = apply_frozen_positive_pool(positions, reader, loaded.device).detach().cpu()
        vectors.update({grid["concept"]: pooled[index] for index, grid in enumerate(chunk)})
        del positions, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vectors, failed


def construct_directions_from_reader_gpu(
    loaded: LoadedModel,
    concepts: list[str],
    reader: dict[str, Any],
    *,
    layer: int | None = None,
    batch_size: int = 8,
    reference_suffix: list[int] | None = None,
    prefix: str = CARRIER_PREFIX,
    suffix: str = CARRIER_SUFFIX,
    concept_chunk_size: int = 0,
) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
    """GPU-resident equivalent of :func:`construct_directions_from_reader`.

    The alignment checks and stored-reader arithmetic are intentionally the
    same as the reference path.  Only temporary carrier activations and the
    resulting candidate vectors stay on device, avoiding a device--host--
    device round trip between CARRY and SCORE.
    """

    layer = int(layer if layer is not None else loaded.spec.carrier_layer)
    grids, failed = aligned_grids(loaded.tokenizer, concepts, prefix=prefix, suffix=suffix)
    if not grids:
        return {}, failed
    suffix_ids = (
        [int(token) for token in reference_suffix]
        if reference_suffix is not None
        else [int(token) for token in grids[0]["suffix_ids"]]
    )
    kept: list[dict[str, Any]] = []
    for grid in grids:
        if [int(token) for token in grid["suffix_ids"]] != suffix_ids:
            failed.append({"concept": grid["concept"], "reason": "suffix tokenization drifted"})
            continue
        kept.append(grid)
    if not kept:
        return {}, failed
    n_means = int(reader["means"].shape[0])
    chunk_size = int(concept_chunk_size) if concept_chunk_size else len(kept)
    if chunk_size <= 0:
        raise ValueError("concept_chunk_size must be positive")
    vectors: dict[str, torch.Tensor] = {}
    for start in range(0, len(kept), chunk_size):
        chunk = kept[start : start + chunk_size]
        positions = capture_suffix_on_device(loaded, chunk, [layer], batch_size=batch_size)[layer]
        if int(positions.shape[1]) != n_means:
            raise RuntimeError(
                f"suffix length {int(positions.shape[1])} != reader means {n_means}"
            )
        pooled = apply_frozen_positive_pool(positions, reader, loaded.device).detach()
        for index, grid in enumerate(chunk):
            vectors[grid["concept"]] = pooled[index].contiguous()
        del positions, pooled
    return vectors, failed


PREFIX = CARRIER_PREFIX
SUFFIX = CARRIER_SUFFIX
