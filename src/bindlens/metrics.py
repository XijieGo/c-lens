"""Small metric helpers used by every experiment."""

from __future__ import annotations

import math
from typing import Any

import torch


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (max(0.0, center - margin), min(1.0, center + margin))


def cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_u = left.float() / left.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    right_u = right.float() / right.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return left_u @ right_u.T


def rank_against(values: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    sim = cosine_matrix(values, targets)
    diag = sim.diag()
    ranks = 1 + (sim > diag[:, None]).sum(dim=-1)
    n = int(sim.shape[0])
    wrong = (sim.sum(dim=-1) - diag) / max(n - 1, 1)
    top1 = int((ranks == 1).sum().item())
    return {
        "n": n,
        "cos_match": float(diag.mean()),
        "cos_wrong": float(wrong.mean()),
        "gap": float((diag - wrong).mean()),
        "top1": top1 / n,
        "top1_count": top1,
        "mrr": float((1.0 / ranks.float()).mean()),
        "chance_top1": 1.0 / n,
    }


def unit(vector: torch.Tensor) -> torch.Tensor:
    flat = vector.float().reshape(-1)
    return flat / flat.norm().clamp_min(1e-12)


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": (float(k) / n if n else None)}
