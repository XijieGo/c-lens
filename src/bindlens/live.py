"""Online, target-free READ → BIND → CARRY → SCORE readout.

This module is the serving path for one ordinary prompt.  It deliberately does
not use an offline source workspace, a precomputed candidate bank, gold labels, or
precomputed candidate vectors.  The serving default uses a vectorized
shared-KV BIND policy and GPU-resident CARRY; reference implementations remain
selectable for agreement checks.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import torch
from jlens.hooks import ActivationRecorder

from bindlens.bind import Candidate, old_full_forward_propose, select_candidates
from bindlens.bind_tree import CachedTreeProposer, VectorizedCachedTreeProposer
from bindlens.carry import (
    construct_directions_from_reader,
    construct_directions_from_reader_gpu,
    load_positive_pool_artifact,
    validate_positive_pool_artifact,
)
from bindlens.models import LoadedModel
from bindlens.paths import llama33_l60_intercept_reader_path, positive_pool_reader_path
from bindlens.protocol import CARRIER_PREFIX, CARRIER_SUFFIX, MODELS, ROOT_BANDS
from bindlens.read import encode_cloze, jlens_logits
from bindlens.score import (
    official_pre_unembed,
    ridge_transport_operator,
    score_in_unembed,
    to_unembed_space,
    unembed_row_norms,
    vocab_unembed_scores,
)


# These are the source-layer grids used by the validated live E2E path.  The
# Qwen3.6 grid and root allocation match its completed ordinary E2E protocol.
# Llama-3.3's grid is the full fitted R-lens source interval; L78 is the
# identity *target* of that fit and is deliberately not treated as a source.
LIVE_SOURCE_LAYERS: dict[str, tuple[int, ...]] = {
    "llama3_1_8b": tuple(range(10, 31)),
    "qwen3_14b": tuple(range(11, 39)),
    "gemma3_12b_it": (11, *range(17, 47)),
    "qwen3_6_27b": tuple(range(28, 59)),
    "llama3_3_70b_it": tuple(range(30, 76)),
}
PROPOSAL_BACKENDS = ("cached_tree", "old_full_forward")
TREE_POLICIES = ("reference", "vectorized")

# Qwen3.6's ordinary E2E run used only the top-20 J-lens roots, with the
# normal total active budget split as 16 + 8.  This is a serving configuration,
# not a change to the BIND candidate policy.
QWEN36_ROOT_BANDS: tuple[tuple[int, int, int], ...] = ((1, 10, 16), (11, 20, 8))

# The Llama-3.3 online configuration uses this frozen reader.  Its natural-space
# carrier boundary is intentionally ``Concept: `` (one trailing space), unlike
# the newer no-space default carrier used by the other live configurations.
LLAMA33_RLENS_CARRIER_LAYER = 60
LLAMA33_RLENS_PREFIX = CARRIER_PREFIX + " "


@dataclass(frozen=True)
class LiveReadoutConfig:
    """Fixed serving configuration for one validated model family."""

    source_layers: tuple[int, ...]
    carrier_layer: int
    reader_path: Path
    prefix: str = CARRIER_PREFIX
    suffix: str = CARRIER_SUFFIX
    root_bands: tuple[tuple[int, int, int], ...] = ROOT_BANDS
    default_tree_layer_batch_size: int = 8
    # The vectorized policy keeps the BIND decision rule unchanged while
    # evaluating a whole frontier without per-row GPU scalar synchronizations.
    default_tree_policy: str = "vectorized"
    # Retain reconstructed candidate vectors on device until SCORE consumes
    # them, rather than staging every carrier microbatch through host RAM.
    default_gpu_carry: bool = True
    default_resident_jlens: bool = True
    # ``gpu`` retains all exact maps on the scoring GPU; ``cpu`` retains them
    # in host RAM and stages one map at a time; ``none`` recomputes each map.
    default_operator_cache_mode: str = "gpu"


def live_config(model: str) -> LiveReadoutConfig:
    """Return the validated online configuration for ``model``."""

    if model not in LIVE_SOURCE_LAYERS:
        supported = ", ".join(sorted(LIVE_SOURCE_LAYERS))
        raise KeyError(f"{model!r} has no validated live E2E configuration; choose from {supported}")
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}")
    if model == "llama3_3_70b_it":
        return LiveReadoutConfig(
            source_layers=LIVE_SOURCE_LAYERS[model],
            carrier_layer=LLAMA33_RLENS_CARRIER_LAYER,
            reader_path=llama33_l60_intercept_reader_path(),
            prefix=LLAMA33_RLENS_PREFIX,
            # The two-A800 70B model leaves roughly 12 GiB/card after model
            # placement.  Two simultaneous layer trees preserve cache reuse
            # while leaving enough margin for the exact scoring workspace.
            default_tree_layer_batch_size=2,
            default_resident_jlens=False,
            # 46 d=8192 FP64 operators would take about 23 GiB on the lm-head
            # GPU.  Build them once, retain them in abundant host RAM, and
            # transfer one exact map only for its score pass.
            default_operator_cache_mode="cpu",
        )
    return LiveReadoutConfig(
        source_layers=LIVE_SOURCE_LAYERS[model],
        carrier_layer=int(MODELS[model].carrier_layer),
        reader_path=positive_pool_reader_path(model),
        root_bands=QWEN36_ROOT_BANDS if model == "qwen3_6_27b" else ROOT_BANDS,
        # Qwen3.6's 48 linear-attention layers retain FP32 recurrent state.
        # On an otherwise idle A800-80GB, two independent source-layer trees
        # fit alongside that state and cut its online proposal latency sharply.
        # ``--tree-layer-batch-size 1`` remains the lower-memory single-tree
        # setting for cross-layer agreement checks or smaller devices.
        default_tree_layer_batch_size=2 if model == "qwen3_6_27b" else 8,
    )


def synchronize_cuda() -> None:
    """Synchronize every visible device before a measured latency boundary."""

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


T = TypeVar("T")


def _timed(call: Callable[[], T], *, enabled: bool) -> tuple[T, float | None]:
    if not enabled:
        return call(), None
    synchronize_cuda()
    started = time.perf_counter_ns()
    value = call()
    synchronize_cuda()
    return value, (time.perf_counter_ns() - started) / 1_000_000.0


def _operator_cache_expectation(
    loaded: LoadedModel,
    config: LiveReadoutConfig,
    source_layer: int,
) -> dict[str, Any]:
    """Return the immutable identity recorded next to one cached ridge map."""

    lens_path = Path(loaded.lens_path).resolve()
    lens_stat = lens_path.stat()
    return {
        "schema_version": 1,
        "kind": "concept_lens_live_exact_ridge_operator",
        "model": loaded.name,
        "lens_path": str(lens_path),
        "lens_size_bytes": int(lens_stat.st_size),
        "lens_mtime_ns": int(lens_stat.st_mtime_ns),
        "lens_n_prompts": int(loaded.lens.n_prompts),
        "lens_source_layers": [int(layer) for layer in loaded.lens.source_layers],
        "source_layer": int(source_layer),
        "carrier_layer": int(config.carrier_layer),
        "d_model": int(loaded.lens.d_model),
        "dtype": "float64",
    }


def _operator_cache_path(directory: Path, source_layer: int) -> Path:
    return directory / f"ridge_source_L{int(source_layer):02d}.pt"


def _load_cached_operator(path: Path, expected: dict[str, Any]) -> torch.Tensor | None:
    """Load one validated CPU FP64 operator, or return ``None`` if absent."""

    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an operator-cache dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing operator-cache metadata")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"{path}: operator-cache {key}={metadata.get(key)!r}, expected {value!r}; "
                "use a fresh cache directory rather than mixing lens artifacts"
            )
    operator = payload.get("operator")
    width = int(expected["d_model"])
    if (
        not isinstance(operator, torch.Tensor)
        or tuple(operator.shape) != (width, width)
        or operator.dtype != torch.float64
        or not bool(torch.isfinite(operator).all())
    ):
        raise ValueError(f"{path}: malformed FP64 operator tensor")
    return operator.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _save_cached_operator(path: Path, metadata: dict[str, Any], operator: torch.Tensor) -> None:
    """Durably save one CPU exact map without exposing a partial checkpoint."""

    if operator.device.type != "cpu" or operator.dtype != torch.float64:
        raise ValueError("operator cache expects a CPU float64 tensor")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save({"metadata": metadata, "operator": operator.contiguous()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@torch.inference_mode()
def capture_live_residuals(
    loaded: LoadedModel,
    input_ids: list[int],
    layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Capture this request's last-position residual at every source layer."""

    batch = torch.tensor([input_ids], dtype=torch.long, device=loaded.model.input_device)
    with ActivationRecorder(loaded.model.layers, at=list(layers)) as recorder:
        loaded.model.forward(batch)
        return {
            int(layer): recorder.activations[int(layer)][:, -1, :].detach()
            for layer in layers
        }


def _prepare_static_state(
    loaded: LoadedModel,
    config: LiveReadoutConfig,
    *,
    resident_jlens: bool,
    operator_cache_mode: str,
    operator_cache_dir: Path | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor | None], dict[str, Any]]:
    """Prepare immutable serving state outside individual request time.

    Exact dense ridge maps are substantial for d=8192 (512 MiB each in
    FP64).  ``cpu`` mode therefore pays their Cholesky cost once at startup,
    keeps the results in host RAM, and lets :class:`LiveReadoutEngine` stage
    one map to the scoring GPU at a time.  It has the same math as ``gpu``
    mode without trying to reserve all maps beside a 70B checkpoint.
    """

    if operator_cache_mode not in {"gpu", "cpu", "none"}:
        raise ValueError(
            f"unknown operator_cache_mode {operator_cache_mode!r}; choose from gpu, cpu, none"
        )
    if operator_cache_dir is not None and operator_cache_mode != "cpu":
        raise ValueError("operator_cache_dir is supported only with operator_cache_mode='cpu'")
    resolved_operator_cache_dir = (
        Path(operator_cache_dir).expanduser().resolve() if operator_cache_dir is not None else None
    )
    started = time.perf_counter_ns()
    if resident_jlens:
        for layer in sorted(set(config.source_layers) | {config.carrier_layer}):
            loaded.lens.jacobians[int(layer)] = loaded.lens.jacobians[int(layer)].to(
                device=loaded.device,
                dtype=torch.float32,
            )
    synchronize_cuda()
    norms = unembed_row_norms(loaded.unembed)
    synchronize_cuda()
    operators: dict[int, torch.Tensor | None] = {}
    cache_hits = 0
    cache_misses = 0
    if operator_cache_mode != "none":
        for layer in config.source_layers:
            source_layer = int(layer)
            operator: torch.Tensor | None = None
            expected: dict[str, Any] | None = None
            cache_path: Path | None = None
            if source_layer != int(config.carrier_layer):
                if resolved_operator_cache_dir is not None:
                    expected = _operator_cache_expectation(loaded, config, source_layer)
                    cache_path = _operator_cache_path(resolved_operator_cache_dir, source_layer)
                    operator = _load_cached_operator(cache_path, expected)
                if operator is None:
                    cache_misses += 1
                    operator = ridge_transport_operator(
                        loaded,
                        source_layer,
                        carrier_layer=config.carrier_layer,
                    )
                    if operator is None:
                        raise RuntimeError(f"L{source_layer}: unexpected identity ridge map")
                    if operator_cache_mode == "cpu":
                        operator = operator.cpu()
                        # ``ridge_transport_operator`` has several transient
                        # d×d workspaces.  Return those blocks before building
                        # the next map rather than relying on allocator reuse
                        # near a 70B model.
                        torch.cuda.empty_cache()
                        if cache_path is not None and expected is not None:
                            _save_cached_operator(cache_path, expected, operator)
                else:
                    cache_hits += 1
            operators[int(layer)] = operator
            synchronize_cuda()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return norms, operators, {
        "warm_static_state_ms": elapsed_ms,
        "resident_jlens": bool(resident_jlens),
        "resident_operators": operator_cache_mode == "gpu",
        "operator_cache_mode": operator_cache_mode,
        "operator_cache_dir": (
            str(resolved_operator_cache_dir) if resolved_operator_cache_dir is not None else None
        ),
        "operator_cache_hits": cache_hits,
        "operator_cache_misses": cache_misses,
        "operators_retained": len(operators),
        "operator_dtype": "float64_exact_cholesky" if operator_cache_mode != "none" else None,
    }


@torch.inference_mode()
def score_live_candidates(
    *,
    loaded: LoadedModel,
    layer: int,
    residual: torch.Tensor,
    candidates: list[Candidate],
    vectors_by_surface: dict[str, torch.Tensor],
    wu_norms: torch.Tensor,
    operator: torch.Tensor | None,
    top_readouts: int,
) -> list[dict[str, Any]]:
    """Rank a layer's retained candidates against the complete vocabulary."""

    if not candidates:
        return []
    candidate_vectors = torch.stack([vectors_by_surface[item.surface] for item in candidates])
    hidden = residual.to(device=loaded.device, dtype=torch.float32)
    query = official_pre_unembed(loaded, loaded.lens.transport(hidden, int(layer))).float()
    if operator is None:
        source_vectors = candidate_vectors.to(device=loaded.device, dtype=torch.float32)
    else:
        source_vectors = (
            candidate_vectors.to(device=operator.device, dtype=operator.dtype) @ operator.T
        ).float()
    directions = to_unembed_space(loaded, source_vectors, int(layer))
    median_wu = float(wu_norms.median().item())
    candidate_scores = score_in_unembed(
        directions,
        query.expand(len(candidates), -1),
        median_wu,
    )
    vocabulary_scores = vocab_unembed_scores(
        loaded.unembed,
        query,
        wu_norms,
        median_wu,
    ).reshape(-1)
    vocab_above = (vocabulary_scores[:, None] > candidate_scores[None, :]).sum(dim=0)
    candidate_above = (candidate_scores[None, :] > candidate_scores[:, None]).sum(dim=1)
    ranks = 1 + vocab_above + candidate_above
    order = torch.argsort(ranks)[: max(0, int(top_readouts))].detach().cpu().tolist()
    scores = candidate_scores.detach().float().cpu().tolist()
    ranks_cpu = ranks.detach().cpu().tolist()
    return [
        {
            "surface": candidates[index].surface,
            "rank": int(ranks_cpu[index]),
            "score": float(scores[index]),
        }
        for index in order
    ]


@dataclass
class LiveReadoutEngine:
    """Prepared, batch-one online BindLens readout service.

    Construction loads and validates the fixed reader and optionally makes
    J-lens matrices and ridge operators resident.  :meth:`read` then executes
    exactly one user prompt without any target-derived information.
    """

    loaded: LoadedModel
    config: LiveReadoutConfig
    reader: dict[str, Any]
    reader_metadata: dict[str, Any]
    wu_norms: torch.Tensor
    # Values are either GPU-resident maps or CPU-resident maps that are staged
    # per score layer.  ``None`` marks the carrier layer's identity map.
    cached_operators: dict[int, torch.Tensor | None]
    static_state: dict[str, Any]
    tree_policy: str = "vectorized"
    gpu_carry: bool = True

    def metadata(self) -> dict[str, Any]:
        """Return JSON-ready serving provenance independent of a prompt."""

        return {
            "schema_version": 2,
            "purpose": "target_free_live_e2e_readout",
            "model": self.loaded.name,
            "dtype": str(self.loaded.unembed.dtype).removeprefix("torch."),
            "source_layers": list(self.config.source_layers),
            "carrier_layer": self.config.carrier_layer,
            "root_bands": [list(band) for band in self.config.root_bands],
            "reader_artifact": str(self.config.reader_path),
            "reader_metadata": self.reader_metadata,
            "request_batch_size": 1,
            "default_proposal_backend": "cached_tree",
            "available_proposal_backends": list(PROPOSAL_BACKENDS),
            "tree_policy": self.tree_policy,
            "vectorized_policy": self.tree_policy == "vectorized",
            "gpu_carry": bool(self.gpu_carry),
            "context_kv_reused_across_layers": True,
            "tree_frontier_batched_within_layer": True,
            "tree_frontier_batched_across_source_layers": True,
            "tree_layer_batch_size": self.config.default_tree_layer_batch_size,
            "terminal_cap_forward_reused": True,
            "static_runtime_state": self.static_state,
        }

    def read(
        self,
        prompt: str,
        *,
        carrier_batch_size: int = 16,
        carrier_chunk_size: int = 0,
        top_readouts: int = 3,
        measure_latency: bool = True,
        proposal_backend: str = "cached_tree",
        tree_layer_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Read one prompt end to end.

        Timings begin after prompt tokenization, matching the prior live E2E
        measurement protocol.  ``carrier_batch_size`` batches candidate
        carrier reconstructions inside this one request; it never batches
        multiple user prompts together.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if int(carrier_batch_size) < 1:
            raise ValueError("carrier_batch_size must be positive")
        if int(carrier_chunk_size) < 0:
            raise ValueError("carrier_chunk_size must be non-negative")
        if int(top_readouts) < 0:
            raise ValueError("top_readouts must be non-negative")
        resolved_tree_layer_batch_size = (
            int(self.config.default_tree_layer_batch_size)
            if tree_layer_batch_size is None
            else int(tree_layer_batch_size)
        )
        if resolved_tree_layer_batch_size < 1:
            raise ValueError("tree_layer_batch_size must be positive")
        if proposal_backend not in PROPOSAL_BACKENDS:
            raise ValueError(
                f"unknown proposal_backend {proposal_backend!r}; choose from {PROPOSAL_BACKENDS}"
            )
        if self.tree_policy not in TREE_POLICIES:
            raise ValueError(
                f"unknown tree_policy {self.tree_policy!r}; choose from {TREE_POLICIES}"
            )

        input_ids = encode_cloze(self.loaded.tokenizer, prompt)
        if measure_latency:
            synchronize_cuda()
            total_started = time.perf_counter_ns()

        residuals, source_forward_ms = _timed(
            lambda: capture_live_residuals(self.loaded, input_ids, self.config.source_layers),
            enabled=measure_latency,
        )

        def lens_step() -> dict[int, torch.Tensor]:
            return {
                int(layer): jlens_logits(self.loaded, residuals[int(layer)], int(layer))
                for layer in self.config.source_layers
            }

        lens_logits_by_layer, lens_ms = _timed(lens_step, enabled=measure_latency)

        def proposal_step() -> tuple[dict[int, list[Candidate]], dict[int, int], dict[str, Any]]:
            selected: dict[int, list[Candidate]] = {}
            raw_counts: dict[int, int] = {}
            if proposal_backend == "old_full_forward":
                for layer in self.config.source_layers:
                    raw = old_full_forward_propose(
                        self.loaded,
                        lens_logits_by_layer[int(layer)],
                        cloze_prompt=prompt,
                        root_bands=self.config.root_bands,
                    )
                    raw_counts[int(layer)] = len(raw)
                    selected[int(layer)] = select_candidates(raw)
                return selected, raw_counts, {
                    "backend": "old_full_forward",
                    "source_layers": len(self.config.source_layers),
                }

            proposer_class = (
                VectorizedCachedTreeProposer
                if self.tree_policy == "vectorized"
                else CachedTreeProposer
            )
            proposer = proposer_class(
                self.loaded,
                root_bands=self.config.root_bands,
                layer_batch_size=resolved_tree_layer_batch_size,
            )
            raw_by_layer = proposer.propose_all_raw(prompt, lens_logits_by_layer)
            for layer in self.config.source_layers:
                raw = raw_by_layer[int(layer)]
                raw_counts[int(layer)] = len(raw)
                selected[int(layer)] = select_candidates(raw)
            stats = {
                "backend": "cached_tree",
                "tree_policy": self.tree_policy,
                "layer_batch_size": resolved_tree_layer_batch_size,
                **asdict(proposer.stats),
            }
            del raw_by_layer, proposer
            return selected, raw_counts, stats

        (candidates_by_layer, raw_counts, backend_stats), proposal_ms = _timed(
            proposal_step,
            enabled=measure_latency,
        )
        if set(candidates_by_layer) != set(self.config.source_layers):
            raise RuntimeError("live proposal did not cover the complete source grid")

        surfaces: list[str] = []
        seen_surfaces: set[str] = set()
        for layer in self.config.source_layers:
            for candidate in candidates_by_layer[int(layer)]:
                if candidate.surface not in seen_surfaces:
                    seen_surfaces.add(candidate.surface)
                    surfaces.append(candidate.surface)

        def recovery_step() -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
            construct = (
                construct_directions_from_reader_gpu
                if self.gpu_carry
                else construct_directions_from_reader
            )
            return construct(
                self.loaded,
                surfaces,
                self.reader,
                layer=self.config.carrier_layer,
                batch_size=int(carrier_batch_size),
                concept_chunk_size=int(carrier_chunk_size),
                reference_suffix=None,
                prefix=self.config.prefix,
                suffix=self.config.suffix,
            )

        (vectors_by_surface, recovery_failures), recovery_ms = _timed(
            recovery_step,
            enabled=measure_latency,
        )
        candidates_by_layer = {
            int(layer): [
                candidate
                for candidate in candidates_by_layer[int(layer)]
                if candidate.surface in vectors_by_surface
            ]
            for layer in self.config.source_layers
        }

        def score_step() -> dict[int, list[dict[str, Any]]]:
            readouts: dict[int, list[dict[str, Any]]] = {}
            for layer in self.config.source_layers:
                cached_operator = self.cached_operators.get(int(layer))
                operator = cached_operator
                temporary_operator: torch.Tensor | None = None
                if cached_operator is not None and cached_operator.device != self.loaded.device:
                    # CPU-cached exact maps are deliberately staged only for
                    # their own layer.  Keeping all 46 on a 70B lm-head GPU
                    # would consume roughly 23 GiB.
                    temporary_operator = cached_operator.to(
                        device=self.loaded.device,
                        dtype=torch.float64,
                    )
                    operator = temporary_operator
                elif int(layer) != self.config.carrier_layer and int(layer) not in self.cached_operators:
                    temporary_operator = ridge_transport_operator(
                        self.loaded,
                        int(layer),
                        carrier_layer=self.config.carrier_layer,
                    )
                    operator = temporary_operator
                readouts[int(layer)] = score_live_candidates(
                    loaded=self.loaded,
                    layer=int(layer),
                    residual=residuals[int(layer)],
                    candidates=candidates_by_layer[int(layer)],
                    vectors_by_surface=vectors_by_surface,
                    wu_norms=self.wu_norms,
                    operator=operator,
                    top_readouts=int(top_readouts),
                )
                if temporary_operator is not None:
                    del temporary_operator
            return readouts

        readouts_by_layer, score_ms = _timed(score_step, enabled=measure_latency)
        total_ms: float | None = None
        if measure_latency:
            synchronize_cuda()
            total_ms = (time.perf_counter_ns() - total_started) / 1_000_000.0

        result = {
            "input_tokens": len(input_ids),
            "latency_ms": None
            if not measure_latency
            else {
                "source_forward": source_forward_ms,
                "read_plus_jlens": float(source_forward_ms) + float(lens_ms),
                "jlens_transform": lens_ms,
                "proposal": proposal_ms,
                "recovery": recovery_ms,
                "score": score_ms,
                "total": total_ms,
            },
            "candidate_counts": {
                "raw_by_layer": {str(layer): int(raw_counts[int(layer)]) for layer in self.config.source_layers},
                "retained_by_layer": {
                    str(layer): len(candidates_by_layer[int(layer)]) for layer in self.config.source_layers
                },
                "unique_surfaces_before_recovery": len(surfaces),
                "recovered_surfaces": len(vectors_by_surface),
                "recovery_failures": len(recovery_failures),
            },
            "proposal_backend": proposal_backend,
            "tree_policy": self.tree_policy if proposal_backend == "cached_tree" else None,
            "gpu_carry": bool(self.gpu_carry),
            "backend_stats": backend_stats,
            "recovery_failures": recovery_failures,
            "readouts_by_layer": {str(layer): rows for layer, rows in readouts_by_layer.items()},
        }
        del residuals, lens_logits_by_layer, candidates_by_layer, vectors_by_surface
        return result


def prepare_live_readout(
    loaded: LoadedModel,
    *,
    config: LiveReadoutConfig | None = None,
    tree_policy: str | None = None,
    gpu_carry: bool | None = None,
    resident_jlens: bool | None = None,
    resident_operators: bool | None = None,
    operator_cache_mode: str | None = None,
    operator_cache_dir: Path | None = None,
) -> LiveReadoutEngine:
    """Load the fixed reader and prepare an online engine outside query time.

    ``tree_policy`` selects the cached-tree policy executor (the vectorized
    serving path is the configuration default); ``gpu_carry`` retains
    candidate directions on device through SCORE.  ``resident_operators``
    remains a backwards-compatible boolean spelling:
    true selects the GPU cache and false selects per-layer recomputation.  New
    callers should use ``operator_cache_mode`` so large sharded models can use
    the exact CPU cache mode.  With that mode, ``operator_cache_dir`` provides
    a per-layer durable cache so a resumed full E2E run does not recompute all
    d×d Cholesky maps.
    """

    config = config or live_config(loaded.name)
    if resident_operators is not None and operator_cache_mode is not None:
        raise ValueError("pass either resident_operators or operator_cache_mode, not both")
    resolved_tree_policy = config.default_tree_policy if tree_policy is None else str(tree_policy)
    if resolved_tree_policy not in TREE_POLICIES:
        raise ValueError(
            f"unknown tree_policy {resolved_tree_policy!r}; choose from {TREE_POLICIES}"
        )
    resolved_gpu_carry = bool(config.default_gpu_carry) if gpu_carry is None else bool(gpu_carry)
    resolved_resident_jlens = (
        bool(config.default_resident_jlens) if resident_jlens is None else bool(resident_jlens)
    )
    if operator_cache_mode is None:
        resolved_operator_cache_mode = (
            ("gpu" if resident_operators else "none")
            if resident_operators is not None
            else config.default_operator_cache_mode
        )
    else:
        resolved_operator_cache_mode = str(operator_cache_mode)
    available_layers = {int(layer) for layer in loaded.lens.source_layers}
    missing = [layer for layer in config.source_layers if layer not in available_layers]
    if missing:
        raise RuntimeError(f"{loaded.name}: live source layers are unavailable in this lens: {missing}")
    if int(config.carrier_layer) not in available_layers:
        raise RuntimeError(
            f"{loaded.name}: carrier layer L{config.carrier_layer} is unavailable in this lens"
        )
    if not config.reader_path.is_file():
        raise FileNotFoundError(config.reader_path)
    reader, reader_metadata = load_positive_pool_artifact(config.reader_path)
    validate_positive_pool_artifact(
        loaded,
        reader_metadata,
        reader,
        prefix=config.prefix,
        suffix=config.suffix,
        layer=config.carrier_layer,
    )
    wu_norms, operators, static_state = _prepare_static_state(
        loaded,
        config,
        resident_jlens=resolved_resident_jlens,
        operator_cache_mode=resolved_operator_cache_mode,
        operator_cache_dir=operator_cache_dir,
    )
    return LiveReadoutEngine(
        loaded=loaded,
        config=config,
        reader=reader,
        reader_metadata=reader_metadata,
        wu_norms=wu_norms,
        cached_operators=operators,
        static_state=static_state,
        tree_policy=resolved_tree_policy,
        gpu_carry=resolved_gpu_carry,
    )


def summarize_live_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize completed, timed live-readout rows for a latency report."""

    completed = [
        row
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("result", {}).get("latency_ms"), dict)
    ]
    stages = ("read_plus_jlens", "proposal", "recovery", "score", "total")

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for stage in stages:
            timings = [float(row["result"]["latency_ms"][stage]) for row in group]
            if timings:
                values[stage] = {
                    "median_ms": statistics.median(timings),
                    "p25_ms": statistics.quantiles(timings, n=4, method="inclusive")[0]
                    if len(timings) > 1
                    else timings[0],
                    "p75_ms": statistics.quantiles(timings, n=4, method="inclusive")[2]
                    if len(timings) > 1
                    else timings[0],
                }
        return values

    by_length_bin: dict[str, Any] = {}
    for label in sorted({str(row.get("length_bin", "")) for row in completed}):
        group = [row for row in completed if str(row.get("length_bin", "")) == label]
        by_length_bin[label] = {
            "n": len(group),
            "median_prompt_tokens": statistics.median(
                [int(row.get("prompt_tokens", row["result"]["input_tokens"])) for row in group]
            )
            if group
            else None,
            "latency_ms": {
                stage: statistics.median([float(row["result"]["latency_ms"][stage]) for row in group])
                for stage in stages
            }
            if group
            else {},
        }
    return {
        "n_samples": len(completed),
        "overall": describe(completed),
        "by_length_bin": by_length_bin,
    }


__all__ = [
    "LIVE_SOURCE_LAYERS",
    "PROPOSAL_BACKENDS",
    "TREE_POLICIES",
    "LiveReadoutConfig",
    "LiveReadoutEngine",
    "capture_live_residuals",
    "live_config",
    "prepare_live_readout",
    "score_live_candidates",
    "summarize_live_rows",
    "synchronize_cuda",
]
