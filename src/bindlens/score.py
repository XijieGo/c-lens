"""SCORE: transport a concept direction, or rank in the official unembed space."""

from __future__ import annotations

from typing import Any

import torch

from bindlens.models import LoadedModel
from bindlens.protocol import RIDGE_RELATIVE_LAMBDA, TIE_EPS, WHITEN_EPS_FRACTION


def ridge_preimage(
    jacobian: torch.Tensor,
    target_effect: torch.Tensor,
    *,
    relative_lambda: float = RIDGE_RELATIVE_LAMBDA,
    max_iterations: int = 40,
    tolerance: float = 1e-4,
) -> torch.Tensor:
    """Solve min_v ||J v - q||^2 + λ||v||^2 for row vectors via PCG."""
    d_model = jacobian.shape[0]
    normal_scale = jacobian.square().sum() / d_model
    ridge_lambda = normal_scale * relative_lambda
    diagonal = jacobian.square().sum(dim=0) + ridge_lambda
    rhs = target_effect @ jacobian

    def normal(values: torch.Tensor) -> torch.Tensor:
        return (values @ jacobian.T) @ jacobian + ridge_lambda * values

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual / diagonal[None, :]
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum(dim=-1)
    rhs_norm = rhs.norm(dim=-1).clamp_min(1e-12)
    for _ in range(max_iterations):
        normal_direction = normal(direction)
        denominator = (direction * normal_direction).sum(dim=-1).clamp_min(1e-30)
        step = rz / denominator
        solution = solution + step[:, None] * direction
        residual = residual - step[:, None] * normal_direction
        relative_residual = residual.norm(dim=-1) / rhs_norm
        if bool((relative_residual <= tolerance).all()):
            break
        next_preconditioned = residual / diagonal[None, :]
        next_rz = (residual * next_preconditioned).sum(dim=-1)
        beta = next_rz / rz.clamp_min(1e-30)
        direction = next_preconditioned + beta[:, None] * direction
        preconditioned, rz = next_preconditioned, next_rz
    return solution


def transport(
    vectors: torch.Tensor,
    loaded: LoadedModel,
    *,
    carrier_layer: int,
    source_layer: int,
    relative_lambda: float = RIDGE_RELATIVE_LAMBDA,
    batch_size: int = 32,
) -> torch.Tensor:
    if int(carrier_layer) == int(source_layer):
        return vectors.float()
    device = loaded.device
    jacobian_carrier = loaded.lens.jacobians[carrier_layer].to(device=device, dtype=torch.float32)
    jacobian_source = loaded.lens.jacobians[source_layer].to(device=device, dtype=torch.float32)
    chunks = []
    try:
        for start in range(0, len(vectors), max(1, batch_size)):
            part = vectors[start : start + max(1, batch_size)].to(device=device, dtype=torch.float32)
            effect = part @ jacobian_carrier.T
            moved = ridge_preimage(jacobian_source, effect, relative_lambda=relative_lambda)
            chunks.append(moved.cpu())
    finally:
        del jacobian_carrier, jacobian_source
    return torch.cat(chunks, dim=0) if chunks else vectors.float()


def whitening_scale(calibration: dict[str, Any], layer: int) -> torch.Tensor:
    """Diagonal whitening A = diag(σ² + ε)^{-1/2} from calibration residuals."""
    from bindlens.carry import pool_weights

    weights = pool_weights(calibration, layer)
    capture_layers = [int(item) for item in calibration["capture_layers"]]
    index = capture_layers.index(layer)
    pooled = (calibration["deviations"][index, :, 1:, :].float() * weights[None, :, None]).sum(dim=1)
    variance = pooled.square().mean(dim=0)
    epsilon = float(torch.median(variance).item()) * WHITEN_EPS_FRACTION
    return torch.rsqrt(variance + epsilon)


def concept_score(
    direction: torch.Tensor,
    hidden: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> float:
    left = direction.float().reshape(-1)
    right = hidden.float().reshape(-1)
    if scale is not None:
        scale = scale.float().reshape(-1)
        left = left * scale
        right = right * scale
    return float(torch.dot(left, right))


def nnls_direction(token_dirs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    token = token_dirs.float()
    token = token / token.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    hidden = hidden.float()
    gram = token @ token.T
    rhs = token @ hidden
    weight = torch.clamp(rhs, min=0.0)
    for _ in range(40):
        weight = torch.clamp(weight - 0.25 * (gram @ weight - rhs), min=0.0)
    return weight @ token


def nnls_batch(token_dirs: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token = token_dirs.float()
    token = token / token.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    token = token * mask[:, :, None]
    gram = token @ token.transpose(1, 2)
    rhs = (token * hidden[:, None, :]).sum(dim=-1)
    weight = torch.clamp(rhs, min=0.0)
    for _ in range(40):
        weight = torch.clamp(weight - 0.25 * (torch.bmm(gram, weight[:, :, None]).squeeze(-1) - rhs), min=0.0)
    return (weight[:, :, None] * token).sum(dim=1)


def unit_rows(matrix: torch.Tensor) -> torch.Tensor:
    values = matrix.float()
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def official_pre_unembed(loaded: LoadedModel, residual: torch.Tensor) -> torch.Tensor:
    """`final_norm` of a final-layer residual: the space where `W_U` rows live."""
    model = loaded.model
    target_device = model._lm_head.weight.device
    target_dtype = model._lm_head.weight.dtype
    return model._final_norm(residual.to(device=target_device, dtype=target_dtype))


def unembed_row_norms(weight: torch.Tensor, *, chunk: int = 4096) -> torch.Tensor:
    vocab = int(weight.shape[0])
    norms = torch.empty(vocab, dtype=torch.float32, device=weight.device)
    for start in range(0, vocab, max(1, chunk)):
        stop = min(start + max(1, chunk), vocab)
        norms[start:stop] = weight[start:stop].float().norm(dim=-1)
    return norms.clamp_min(1e-12)


def ridge_transport_operator(
    loaded: LoadedModel,
    source_layer: int,
    *,
    carrier_layer: int | None = None,
) -> torch.Tensor | None:
    """Dense ridge map: `v_source = v_carrier @ operator.T`."""
    carrier_layer = int(carrier_layer if carrier_layer is not None else loaded.spec.carrier_layer)
    if int(source_layer) == carrier_layer:
        return None
    dtype = torch.float64
    jacobian_source = loaded.lens.jacobians[int(source_layer)].to(device=loaded.device, dtype=dtype)
    jacobian_carrier = loaded.lens.jacobians[carrier_layer].to(device=loaded.device, dtype=dtype)
    d_model = int(jacobian_source.shape[0])
    ridge_lambda = (jacobian_source.square().sum() / d_model) * RIDGE_RELATIVE_LAMBDA
    normal = jacobian_source.T @ jacobian_source
    normal.diagonal().add_(ridge_lambda)
    factor, info = torch.linalg.cholesky_ex(normal, check_errors=False)
    if int(info.max().item()) != 0:
        raise RuntimeError(f"ridge normal was not PD at L{source_layer}: info={int(info.max().item())}")
    operator = torch.cholesky_solve(jacobian_source.T @ jacobian_carrier, factor)
    del jacobian_source, jacobian_carrier, normal, factor
    return operator


def transport_cholesky(
    vectors: torch.Tensor,
    loaded: LoadedModel,
    *,
    source_layer: int,
    carrier_layer: int | None = None,
) -> torch.Tensor:
    """Exact fp64 dense Cholesky transport: `v_source = v_carrier @ operator.T`."""
    operator = ridge_transport_operator(loaded, source_layer, carrier_layer=carrier_layer)
    if operator is None:
        return vectors.float()
    try:
        moved = vectors.to(device=operator.device, dtype=operator.dtype) @ operator.T
        return moved.float().cpu()
    finally:
        del operator


def to_unembed_space(loaded: LoadedModel, residual: torch.Tensor, layer: int) -> torch.Tensor:
    """Map a residual-space direction through `J` then `final_norm`."""
    return official_pre_unembed(loaded, loaded.lens.transport(residual.float(), layer)).float()


def score_in_unembed(directions: torch.Tensor, query: torch.Tensor, median_wu_norm: float) -> torch.Tensor:
    """One formula for every competitor: `unit(d) · q * median||W_U||`."""
    return (unit_rows(directions) * query.float()).sum(dim=-1) * float(median_wu_norm)


def vocab_unembed_scores(
    unembed: torch.Tensor,
    query: torch.Tensor,
    wu_norms: torch.Tensor,
    median_wu_norm: float,
) -> torch.Tensor:
    """Same formula on every unembed row. ``query`` is `[B, D]`; result `[V, B]`."""
    raw = torch.matmul(unembed, query.to(dtype=unembed.dtype).T).float()
    return raw / wu_norms[:, None] * float(median_wu_norm)


def union_ranks(
    *,
    token_logits: torch.Tensor,
    candidate_scores: torch.Tensor,
    gold_index: int | None,
    tie_eps: float = TIE_EPS,
) -> tuple[int | None, int | None]:
    """Strict and worst-tie ranks of gold in vocab ∪ other candidates."""
    if gold_index is None:
        return None, None
    gold = float(candidate_scores[gold_index].item())
    if gold != gold:
        return None, None
    n_vocab_strict = int((token_logits > gold).sum().item())
    n_vocab_worst = int((token_logits >= gold - tie_eps * max(1.0, abs(gold))).sum().item())
    others = torch.cat([candidate_scores[:gold_index], candidate_scores[gold_index + 1 :]])
    if others.numel():
        n_cand_strict = int((others > gold).sum().item())
        n_cand_worst = int((others >= gold - tie_eps * max(1.0, abs(gold))).sum().item())
    else:
        n_cand_strict = n_cand_worst = 0
    return 1 + n_vocab_strict + n_cand_strict, 1 + n_vocab_worst + n_cand_worst


def score_vocab_union_layer(
    *,
    loaded: LoadedModel,
    layer: int,
    carrier_layer: int | None = None,
    records: list[Any],
    source_states: torch.Tensor,
    vc_vectors: torch.Tensor,
    vc_index: dict[str, int],
    random_vectors: torch.Tensor,
    random_key_index: dict[str, int],
    wu_norms: torch.Tensor,
    direction_batch_size: int = 256,
    vocab_event_batch: int = 16,
) -> dict[str, dict[str, dict[str, int | None]]]:
    """Rank gold against vocab ∪ candidates with one unembed-space score.

    Every method is a vector in the official pre-unembed space
    (`final_norm(J ·)`), then `unit(d) · q * median||W_U||`. Token
    competitors use the same formula on the unembed rows.
    """
    from bindlens.protocol import RANKING_METHODS

    methods = RANKING_METHODS
    flat: list[tuple[int, Any]] = []
    offsets: list[tuple[int, int]] = []
    gold_pos: list[int | None] = []
    vc_ids: list[int] = []
    for record_index, record in enumerate(records):
        begin = len(flat)
        found = None
        for candidate in record.candidates:
            if candidate.surface not in vc_index:
                raise RuntimeError(f"missing v_C for {candidate.surface!r}")
            if candidate.key == record.gold_key:
                found = len(flat) - begin
            vc_ids.append(vc_index[candidate.surface])
            flat.append((record_index, candidate))
        offsets.append((begin, len(flat)))
        gold_pos.append(found)
    empty = {method: {} for method in methods}
    if not flat:
        return {"strict": empty, "worst": empty}

    unique_ids = sorted({int(token) for _, cand in flat for token in cand.ids})
    id_index = {token: i for i, token in enumerate(unique_ids)}
    max_tokens = max(len(cand.ids) for _, cand in flat)
    padded = torch.zeros((len(flat), max_tokens), dtype=torch.long)
    mask = torch.zeros((len(flat), max_tokens), dtype=torch.float32)
    random_indices = torch.empty(len(flat), dtype=torch.long)
    owners = torch.empty(len(flat), dtype=torch.long)
    for row, (record_index, candidate) in enumerate(flat):
        padded[row, : len(candidate.ids)] = torch.tensor([id_index[int(t)] for t in candidate.ids])
        mask[row, : len(candidate.ids)] = 1.0
        random_indices[row] = int(random_key_index[candidate.key])
        owners[row] = int(record_index)

    device = loaded.device
    source_indices = torch.tensor([record.trial_index for record in records], dtype=torch.long)
    hidden = source_states.index_select(0, source_indices).to(device=device, dtype=torch.float32)
    query = official_pre_unembed(loaded, loaded.lens.transport(hidden, layer))
    query_f = query.float()
    median_wu = float(wu_norms.median().item())
    cand_score = {method: torch.empty(len(flat), dtype=torch.float32) for method in methods}
    # A trial reader may be fitted at a layer other than the model's default
    # carrier layer.  Its vectors must be transported from that recorded
    # layer, not silently from ``loaded.spec.carrier_layer``.
    operator = ridge_transport_operator(loaded, layer, carrier_layer=carrier_layer)
    unique_rows = loaded.unembed.index_select(
        0, torch.tensor(unique_ids, dtype=torch.long, device=loaded.unembed.device)
    ).float()
    vc_cpu = torch.tensor(vc_ids, dtype=torch.long)

    for start in range(0, len(flat), max(1, direction_batch_size)):
        stop = min(start + max(1, direction_batch_size), len(flat))
        pos = torch.arange(start, stop, dtype=torch.long)
        token_pos = padded.index_select(0, pos).to(device=unique_rows.device)
        token_mask = mask.index_select(0, pos).to(device=unique_rows.device)
        token_rows = unique_rows.index_select(0, token_pos.reshape(-1)).reshape(stop - start, max_tokens, -1)
        owner = owners.index_select(0, pos).to(device=query_f.device)
        q_owner = query_f.index_select(0, owner)
        first = token_rows[:, 0, :]
        mean = (token_rows * token_mask[:, :, None]).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        nnls = nnls_batch(token_rows, q_owner, token_mask)
        vc = vc_vectors.index_select(0, vc_cpu.index_select(0, pos))
        if operator is None:
            vc_source = vc.to(device=device, dtype=torch.float32)
        else:
            vc_source = (vc.to(device=operator.device, dtype=operator.dtype) @ operator.T).float()
        vc_dir = to_unembed_space(loaded, vc_source, layer)
        rand = random_vectors.index_select(0, random_indices.index_select(0, pos)).to(device=device, dtype=torch.float32)
        rand_dir = to_unembed_space(loaded, rand, layer)
        built = {
            "v_first": first,
            "v_mean": mean,
            "v_nnls": nnls,
            "v_C": vc_dir,
            "v_rand": rand_dir,
        }
        for method, direction in built.items():
            cand_score[method][start:stop] = score_in_unembed(direction, q_owner, median_wu).detach().float().cpu()
        del token_rows, first, mean, nnls, vc, vc_source, vc_dir, rand, rand_dir, q_owner, owner, token_pos, token_mask
    del operator, unique_rows

    strict = {method: {} for method in methods}
    worst = {method: {} for method in methods}
    for batch_start in range(0, len(records), max(1, vocab_event_batch)):
        batch_stop = min(batch_start + max(1, vocab_event_batch), len(records))
        token_scores = vocab_unembed_scores(
            loaded.unembed, query[batch_start:batch_stop], wu_norms, median_wu
        )
        for local, record_index in enumerate(range(batch_start, batch_stop)):
            begin, end = offsets[record_index]
            trial_id = records[record_index].trial_id
            tok = token_scores[:, local]
            gold_index = gold_pos[record_index]
            for method in methods:
                s, w = union_ranks(
                    token_logits=tok,
                    candidate_scores=cand_score[method][begin:end],
                    gold_index=gold_index,
                )
                strict[method][trial_id] = s
                worst[method][trial_id] = w
        del token_scores
    return {"strict": strict, "worst": worst}
