"""READ: J-lens scores and ranks from a residual."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from jlens.hooks import ActivationRecorder

from bindlens.models import LoadedModel, leading_space_ids
from bindlens.protocol import with_swap_trailing_space, with_trailing_space


# Kept in artifact metadata so CPU replay cannot mix an incompatible
# standalone-space source encoding with the natural BPE boundary used here.
SOURCE_RESIDUAL_PROTOCOL = "prompt.rstrip()+leading_space_ids(concept)"
_LAST_TOKEN_WORKSPACE_KIND = "concept_lens_last_token_residual_workspace"
_CANDIDATE_VECTOR_WORKSPACE_KIND = "concept_lens_carrier_candidate_vectors"


def token_direction(loaded: LoadedModel, token_id: int, layer: int) -> torch.Tensor:
    """Residual-space J-lens direction d_ℓ(w) = (W_U J_ℓ)_w."""
    jacobian = loaded.lens.jacobians[layer].to(device=loaded.device, dtype=torch.float32)
    row = loaded.unembed[int(token_id)].to(device=loaded.device, dtype=torch.float32)
    return (row @ jacobian).cpu()


def token_directions(loaded: LoadedModel, token_ids: list[int], layer: int) -> torch.Tensor:
    if not token_ids:
        return torch.zeros((0, loaded.model.d_model), dtype=torch.float32)
    ids = torch.tensor(token_ids, dtype=torch.long, device=loaded.device)
    jacobian = loaded.lens.jacobians[layer].to(device=loaded.device, dtype=torch.float32)
    rows = loaded.unembed[ids].to(device=loaded.device, dtype=torch.float32)
    return (rows @ jacobian).cpu()


def jlens_logits(loaded: LoadedModel, residual: torch.Tensor, layer: int) -> torch.Tensor:
    """Vocabulary logits of J-lens at one residual, shape [vocab]."""
    hidden = residual.float().to(loaded.device)
    transported = loaded.lens.transport(hidden, layer)
    return loaded.model.unembed(transported).float().reshape(-1)


def ranks_of(logits: torch.Tensor, token_ids: list[int]) -> list[int]:
    if not token_ids:
        return []
    ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    values = logits.index_select(0, ids)
    return [int(x) + 1 for x in (logits.unsqueeze(1) > values).sum(dim=0).tolist()]


def topk_ids(logits: torch.Tensor, k: int) -> list[int]:
    k = min(int(k), int(logits.numel()))
    return [int(x) for x in torch.topk(logits, k=k).indices.tolist()]


@torch.inference_mode()
def capture_last(
    loaded: LoadedModel,
    input_ids: list[list[int]],
    layers: list[int],
    *,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """Last-token residual for each prompt, per layer. Shape [N, D]."""
    n = len(input_ids)
    out = {
        layer: torch.empty((n, int(loaded.model.d_model)), dtype=torch.float32)
        for layer in layers
    }
    by_length: dict[int, list[int]] = {}
    for index, ids in enumerate(input_ids):
        by_length.setdefault(len(ids), []).append(index)
    for length, group in sorted(by_length.items()):
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            batch = torch.tensor([input_ids[i] for i in chunk], dtype=torch.long)
            with ActivationRecorder(loaded.model.layers, at=layers) as recorder:
                loaded.model.forward(batch.to(loaded.model.input_device))
                for layer in layers:
                    hidden = recorder.activations[layer][:, -1, :].detach().float().cpu()
                    for local, original in enumerate(chunk):
                        out[layer][original] = hidden[local]
    return out


def workspace_layers(n_layers: int, *, start_fraction: float = 0.10) -> list[int]:
    """Return post-block layers from approximately ``start_fraction`` to the end.

    Layer indices are zero based while a captured residual is the output after
    that block.  For example, a 32-layer model with ``start_fraction=0.10``
    starts at layer 3 (the state after the fourth block, 12.5% depth).  This
    keeps the workspace small enough for CPU-side analysis while retaining all
    later source layers where a concept may become readable.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if not 0.0 < float(start_fraction) <= 1.0:
        raise ValueError(f"start_fraction must lie in (0, 1], got {start_fraction}")
    first = max(0, int(math.ceil(float(n_layers) * float(start_fraction))) - 1)
    return list(range(first, int(n_layers)))


def last_token_workspace_bytes(*, n_sides: int, d_model: int, layers: list[int]) -> int:
    """Exact tensor payload size for an FP32 last-token workspace.

    This excludes the small JSON metadata and the serialized input IDs.  It is
    used before a capture so an exploratory run cannot accidentally retain a
    full-activation cache in place of the bounded CPU-analysis artifact.
    """
    if n_sides < 0 or d_model <= 0:
        raise ValueError(f"invalid workspace shape n={n_sides}, d={d_model}")
    unique_layers = {int(layer) for layer in layers}
    if any(layer < 0 for layer in unique_layers):
        raise ValueError(f"invalid workspace layers: {sorted(unique_layers)}")
    return int(n_sides) * int(d_model) * len(unique_layers) * torch.empty(
        (), dtype=torch.float32
    ).element_size()


def input_ids_digest(input_ids: list[list[int]]) -> str:
    """Stable fingerprint for a ragged token-ID batch used by a workspace."""
    normalized = [[int(token) for token in row] for row in input_ids]
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(target: Path, payload: dict[str, Any]) -> int:
    """Atomically persist a tensor artifact and return its final byte size."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return int(target.stat().st_size)


def _workspace_payload(
    *,
    states: Mapping[int, torch.Tensor],
    input_ids: list[list[int]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not states:
        raise ValueError("cannot write an empty source workspace")
    normalized_ids = [[int(token) for token in row] for row in input_ids]
    n_sides = len(normalized_ids)
    if any(not row for row in normalized_ids):
        raise ValueError("workspace input_ids contains an empty sequence")
    tensors: dict[str, torch.Tensor] = {}
    width: int | None = None
    for layer, state in sorted(states.items()):
        if int(layer) < 0:
            raise ValueError(f"invalid workspace layer {layer}")
        value = state.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if value.ndim != 2 or int(value.shape[0]) != n_sides:
            raise ValueError(
                f"workspace L{layer} must have shape [{n_sides}, d], got {tuple(value.shape)}"
            )
        if width is None:
            width = int(value.shape[1])
        elif int(value.shape[1]) != width:
            raise ValueError(f"workspace L{layer} has d={value.shape[1]}, expected {width}")
        tensors[str(int(layer))] = value
    assert width is not None
    return {
        "schema_version": 1,
        "kind": _LAST_TOKEN_WORKSPACE_KIND,
        "metadata": dict(metadata),
        "n_sides": n_sides,
        "d_model": width,
        "layers": [int(layer) for layer in sorted(states)],
        "input_ids": normalized_ids,
        "input_ids_sha256": input_ids_digest(normalized_ids),
        "states": tensors,
    }


def save_last_token_workspace(
    path: str | Path,
    *,
    states: Mapping[int, torch.Tensor],
    input_ids: list[list[int]],
    metadata: Mapping[str, Any],
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Atomically persist reusable source residuals for later CPU analysis.

    We deliberately store only last-position residuals, not activations at all
    positions, vocabulary logits, KV caches, or per-candidate transports.  The
    resulting tensor package can reconstruct J-lens/logit scores and candidate
    comparisons on CPU without rerunning the source cloze forward pass.
    """
    target = Path(path)
    payload = _workspace_payload(states=states, input_ids=input_ids, metadata=metadata)
    tensor_bytes = sum(int(value.numel()) * int(value.element_size()) for value in payload["states"].values())
    if max_bytes is not None:
        if int(max_bytes) <= 0:
            raise ValueError(f"workspace max_bytes must be positive, got {max_bytes}")
        if tensor_bytes > int(max_bytes):
            raise RuntimeError(
                f"source workspace tensor payload {tensor_bytes / 2**30:.2f} GiB exceeds "
                f"the configured cap {int(max_bytes) / 2**30:.2f} GiB"
            )
    return {
        "path": str(target),
        "bytes": _atomic_torch_save(target, payload),
        "n_sides": int(payload["n_sides"]),
        "d_model": int(payload["d_model"]),
        "layers": list(payload["layers"]),
        "tensor_bytes": tensor_bytes,
        "input_ids_sha256": str(payload["input_ids_sha256"]),
        "metadata": dict(payload["metadata"]),
    }


def save_candidate_vector_workspace(
    path: str | Path,
    *,
    model: str,
    candidate_bank: str | Path,
    candidate_bank_stats: Mapping[str, Any],
    reader_metadata: Mapping[str, Any],
    input_ids: list[list[int]],
    surfaces: list[str],
    vectors: torch.Tensor,
    storage_dtype: torch.dtype,
    max_bytes: int,
) -> dict[str, Any]:
    """Persist only reusable carrier-layer ``v_C`` vectors for CPU replay.

    This deliberately excludes source residuals (stored in the separate
    last-token workspace), logits, KV caches, and layer-specific transport
    matrices.  The exact source input-ID digest ties the vector table to its
    candidate bank's experiment without retaining prompts a second time.
    """
    if vectors.ndim != 2 or int(vectors.shape[0]) != len(surfaces):
        raise ValueError(f"candidate-vector shape {tuple(vectors.shape)} does not match {len(surfaces)} surfaces")
    if not surfaces or any(not str(surface).strip() for surface in surfaces):
        raise ValueError("candidate-vector workspace has an empty surface list")
    if len(surfaces) != len(set(surfaces)):
        raise ValueError("candidate-vector workspace has duplicate surfaces")
    if storage_dtype not in (torch.float16, torch.float32):
        raise ValueError(f"unsupported candidate-vector storage dtype {storage_dtype}")
    if int(max_bytes) <= 0:
        raise ValueError(f"candidate-vector max_bytes must be positive, got {max_bytes}")
    estimated_bytes = int(vectors.numel()) * torch.empty((), dtype=storage_dtype).element_size()
    if estimated_bytes > int(max_bytes):
        raise RuntimeError(
            "candidate-vector workspace would exceed its per-model storage cap: "
            f"{estimated_bytes / 2**30:.2f} GiB > {int(max_bytes) / 2**30:.2f} GiB; "
            "reduce retained candidates or raise the explicitly configured cap"
        )
    value = vectors.detach().to(device="cpu", dtype=storage_dtype).contiguous()
    payload = {
        "schema_version": 1,
        "kind": _CANDIDATE_VECTOR_WORKSPACE_KIND,
        "model": str(model),
        "carrier_layer": int(reader_metadata["carrier_layer"]),
        "source_residual_protocol": SOURCE_RESIDUAL_PROTOCOL,
        "input_ids_sha256": input_ids_digest(input_ids),
        "candidate_bank": str(candidate_bank),
        "candidate_bank_stats": dict(candidate_bank_stats),
        "reader_metadata": dict(reader_metadata),
        "surfaces": [str(surface) for surface in surfaces],
        "vectors": value,
    }
    target = Path(path)
    return {
        "path": str(target),
        "bytes": _atomic_torch_save(target, payload),
        "n_surfaces": len(surfaces),
        "d_model": int(value.shape[1]),
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
        "estimated_bytes": estimated_bytes,
        "carrier_layer": int(reader_metadata["carrier_layer"]),
    }


def load_candidate_vector_workspace(
    path: str | Path,
    *,
    input_ids: list[list[int]],
    expected_model: str | None = None,
) -> dict[str, Any]:
    """Load a candidate-vector table for CPU analysis with exact input checks."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a candidate-vector dictionary")
    if payload.get("schema_version") != 1 or payload.get("kind") != _CANDIDATE_VECTOR_WORKSPACE_KIND:
        raise ValueError(f"{source}: unsupported candidate-vector workspace schema")
    if expected_model is not None and str(payload.get("model")) != str(expected_model):
        raise ValueError(f"{source}: model {payload.get('model')!r} != {expected_model!r}")
    if payload.get("input_ids_sha256") != input_ids_digest(input_ids):
        raise ValueError(f"{source}: input token IDs differ from this run")
    surfaces = payload.get("surfaces")
    vectors = payload.get("vectors")
    if not isinstance(surfaces, list) or not all(isinstance(surface, str) and surface for surface in surfaces):
        raise ValueError(f"{source}: malformed candidate surfaces")
    if len(surfaces) != len(set(surfaces)):
        raise ValueError(f"{source}: duplicate candidate surfaces")
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2 or int(vectors.shape[0]) != len(surfaces):
        raise ValueError(f"{source}: malformed candidate vectors")
    if vectors.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"{source}: unsupported candidate-vector dtype {vectors.dtype}")
    reader_metadata = payload.get("reader_metadata")
    if not isinstance(reader_metadata, dict):
        raise ValueError(f"{source}: missing reader metadata")
    return {
        "model": str(payload["model"]),
        "carrier_layer": int(payload["carrier_layer"]),
        "candidate_bank": str(payload["candidate_bank"]),
        "candidate_bank_stats": dict(payload.get("candidate_bank_stats") or {}),
        "reader_metadata": reader_metadata,
        "surfaces": list(surfaces),
        "vectors": vectors.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "input_ids_sha256": str(payload["input_ids_sha256"]),
    }


def load_last_token_workspace(
    path: str | Path,
    *,
    input_ids: list[list[int]],
    required_layers: list[int],
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[int, torch.Tensor]:
    """Load and strictly validate a saved last-token residual workspace.

    Returned tensors are CPU FP32.  Validation against the exact source token
    IDs prevents accidentally reusing a workspace across tokenizer/boundary
    protocol changes or a different input split.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a workspace dictionary")
    if payload.get("schema_version") != 1 or payload.get("kind") != _LAST_TOKEN_WORKSPACE_KIND:
        raise ValueError(f"{source}: unsupported workspace schema")
    normalized_ids = [[int(token) for token in row] for row in input_ids]
    expected_digest = input_ids_digest(normalized_ids)
    if payload.get("input_ids_sha256") != expected_digest:
        raise ValueError(
            f"{source}: input token IDs differ from this run; do not reuse the source workspace"
        )
    if int(payload.get("n_sides", -1)) != len(normalized_ids):
        raise ValueError(f"{source}: n_sides does not match this run")
    stored_ids = payload.get("input_ids")
    if stored_ids != normalized_ids:
        raise ValueError(f"{source}: stored input IDs fail exact validation")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: missing metadata")
    for key, value in (expected_metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{source}: metadata mismatch for {key!r}: "
                f"expected {value!r}, found {metadata.get(key)!r}"
            )
    raw_states = payload.get("states")
    if not isinstance(raw_states, dict):
        raise ValueError(f"{source}: missing state tensors")
    d_model = int(payload.get("d_model", -1))
    result: dict[int, torch.Tensor] = {}
    for layer in sorted({int(layer) for layer in required_layers}):
        value = raw_states.get(str(layer))
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{source}: missing state tensor for L{layer}")
        value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if value.ndim != 2 or tuple(value.shape) != (len(normalized_ids), d_model):
            raise ValueError(f"{source}: malformed L{layer} tensor {tuple(value.shape)}")
        result[layer] = value
    return result


def encode_cloze(tokenizer: Any, prompt: str) -> list[int]:
    text = with_trailing_space(prompt)
    ids = tokenizer(text, add_special_tokens=True).input_ids
    if not ids:
        raise ValueError(f"empty prompt tokenization: {prompt!r}")
    return [int(x) for x in ids]


def encode_swap_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Swap source/partner encoding: rstrip, no trailing space (26-16-39)."""
    ids = tokenizer(with_swap_trailing_space(prompt), add_special_tokens=True).input_ids
    if not ids:
        raise ValueError(f"empty prompt tokenization: {prompt!r}")
    return [int(x) for x in ids]


def concept_token_ids(tokenizer: Any, concept: str) -> list[int]:
    return leading_space_ids(tokenizer, concept)
