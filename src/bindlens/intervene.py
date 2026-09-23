"""Causal edits: parameter-free swap and directional projection."""

from __future__ import annotations

from typing import Any

import torch

from bindlens.metrics import unit
from bindlens.models import LoadedModel
from bindlens.protocol import SWAP_DELETE_TOKENS


def swap_arm_spec(arm: str) -> dict[str, bool | str]:
    """One fixed causal control, all using the same live Swap operator.

    ``random`` keeps the source delete and replaces only the partner add with
    a random direction.  Its magnitude is still the partner-carrier
    occupancy, so it controls direction identity rather than edit scale.
    """
    specs: dict[str, dict[str, bool | str]] = {
        "full": {"family": "carrier", "do_delete": True, "do_add": True, "add": "partner"},
        "delete": {"family": "carrier", "do_delete": True, "do_add": False, "add": "partner"},
        "add": {"family": "carrier", "do_delete": False, "do_add": True, "add": "partner"},
        "first": {"family": "first", "do_delete": True, "do_add": True, "add": "partner"},
        "random": {"family": "carrier", "do_delete": True, "do_add": True, "add": "random"},
    }
    try:
        return specs[arm]
    except KeyError as error:
        raise ValueError(f"unknown swap arm: {arm}") from error


def occupancy(hidden: torch.Tensor, direction: torch.Tensor) -> float:
    return abs(float(torch.dot(hidden.float().reshape(-1), unit(direction))))


def paramfree_edit(
    hidden: torch.Tensor,
    delete_vec: torch.Tensor,
    add_vec: torch.Tensor,
    beta: float,
    *,
    pos_start: int,
    do_delete: bool,
    do_add: bool,
    add_last_n: int = 1,
) -> torch.Tensor:
    """h ← h − (h·v̂_C) v̂_C + β v̂_D on a recorded [T, D] residual."""
    out = hidden.float().clone()
    start = min(max(int(pos_start), 0), out.shape[0])
    if start >= out.shape[0]:
        return out
    if do_delete:
        delete_u = unit(delete_vec)
        coeff = out[start:] @ delete_u
        out[start:] = out[start:] - coeff.unsqueeze(-1) * delete_u
    if do_add:
        add_u = unit(add_vec)
        n = min(max(int(add_last_n), 1), out.shape[0])
        out[-n:] = out[-n:] + float(beta) * add_u
    return out


class LiveParamFree:
    """Apply delete+add on the running residual of selected layers."""

    def __init__(
        self,
        blocks: Any,
        deletes: dict[int, torch.Tensor],
        adds: dict[int, torch.Tensor],
        betas: dict[int, float],
        *,
        pos_start: int,
        do_delete: bool,
        do_add: bool,
        add_last_n: int = 1,
    ) -> None:
        self.blocks = blocks
        self.deletes = {int(layer): unit(vector) for layer, vector in deletes.items()}
        self.adds = {int(layer): unit(vector) for layer, vector in adds.items()}
        self.betas = {int(layer): float(value) for layer, value in betas.items()}
        self.pos_start = int(pos_start)
        self.do_delete = bool(do_delete)
        self.do_add = bool(do_add)
        self.add_last_n = int(add_last_n)
        self._handles: list[Any] = []

    def _hook(self, layer: int):
        pos_start = self.pos_start
        do_delete = self.do_delete
        do_add = self.do_add
        add_last_n = self.add_last_n
        delete = self.deletes[layer]
        add = self.adds[layer]
        beta = self.betas[layer]

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            seq_len = int(tensor.shape[1])
            start = min(max(pos_start, 0), seq_len)
            if start >= seq_len:
                return output
            patched = tensor.clone()
            if do_delete:
                unit_del = delete.to(device=tensor.device, dtype=tensor.dtype).reshape(1, 1, -1)
                window = patched[:, start:, :]
                coeff = (window * unit_del).sum(dim=-1, keepdim=True)
                patched[:, start:, :] = window - coeff * unit_del
            if do_add:
                unit_add = add.to(device=tensor.device, dtype=tensor.dtype).reshape(1, 1, -1)
                n = min(max(add_last_n, 1), seq_len)
                patched[:, -n:, :] = patched[:, -n:, :] + beta * unit_add
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        return hook

    def __enter__(self) -> "LiveParamFree":
        for layer in self.deletes:
            self._handles.append(self.blocks[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class LiveProjectOut:
    """h ← h − (h·v̂) v̂ on every token after pos_start."""

    def __init__(self, blocks: Any, directions: dict[int, torch.Tensor], *, pos_start: int) -> None:
        self.blocks = blocks
        self.directions = {int(layer): unit(vector) for layer, vector in directions.items()}
        self.pos_start = int(pos_start)
        self._handles: list[Any] = []

    def _hook(self, direction: torch.Tensor):
        pos_start = self.pos_start

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            seq_len = int(tensor.shape[1])
            start = min(max(pos_start, 0), seq_len)
            if start >= seq_len:
                return output
            unit_dir = direction.to(device=tensor.device, dtype=tensor.dtype)
            patched = tensor.clone()
            window = patched[:, start:, :]
            coeff = (window * unit_dir.reshape(1, 1, -1)).sum(dim=-1, keepdim=True)
            patched[:, start:, :] = window - coeff * unit_dir
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        return hook

    def __enter__(self) -> "LiveProjectOut":
        for layer, direction in self.directions.items():
            self._handles.append(self.blocks[layer].register_forward_hook(self._hook(direction)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def delete_start(seq_len: int, suffix_tokens: int = SWAP_DELETE_TOKENS) -> int:
    if seq_len <= 1:
        return 0
    return max(1, seq_len - max(1, int(suffix_tokens)))


class LastTokenBan:
    """Project formatting directions out of the last residual token after the edit."""

    def __init__(self, block: Any, directions: torch.Tensor, scale: float = 1.0) -> None:
        self.block = block
        self.directions = directions.float().cpu()
        self.scale = float(scale)
        self.handle: Any | None = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.directions.numel() == 0:
            return output
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        window = patched[:, -1:, :]
        for direction in self.directions.to(device=tensor.device, dtype=tensor.dtype):
            vector = direction.reshape(1, 1, -1)
            coefficient = (window * vector).sum(dim=-1, keepdim=True)
            window = window - self.scale * coefficient * vector
        patched[:, -1:, :] = window
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    def __enter__(self) -> "LastTokenBan":
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def format_ban_directions(loaded: LoadedModel, surfaces: tuple[str, ...]) -> tuple[torch.Tensor, list[int]]:
    ids: list[int] = []
    for surface in surfaces:
        pieces = loaded.tokenizer.encode(surface, add_special_tokens=False)
        if len(pieces) == 1 and int(pieces[0]) not in ids:
            ids.append(int(pieces[0]))
    if not ids:
        return torch.zeros((0, int(loaded.model.d_model)), dtype=torch.float32), ids
    rows = loaded.unembed[torch.tensor(ids, dtype=torch.long)].detach().float().cpu()
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12), ids


@torch.inference_mode()
def live_swap_logits(
    loaded: LoadedModel,
    input_ids: list[int],
    *,
    deletes: dict[int, torch.Tensor],
    adds: dict[int, torch.Tensor],
    betas: dict[int, float],
    pos_start: int,
    do_delete: bool,
    do_add: bool,
    add_last_n: int,
    ban_directions: torch.Tensor,
) -> torch.Tensor:
    """One live edited forward; logits from the last residual through the official unembed."""
    from jlens.hooks import ActivationRecorder

    last_layer = int(loaded.model.n_layers) - 1
    batch = torch.tensor([input_ids], dtype=torch.long, device=loaded.model.input_device)
    with LiveParamFree(
        loaded.model.layers,
        deletes,
        adds,
        betas,
        pos_start=pos_start,
        do_delete=do_delete,
        do_add=do_add,
        add_last_n=add_last_n,
    ), LastTokenBan(loaded.model.layers[last_layer], ban_directions):
        with ActivationRecorder(loaded.model.layers, at=[last_layer]) as recorder:
            loaded.model.forward(batch)
            final = recorder.activations[last_layer][:, -1, :]
            return loaded.model.unembed(final)[0].float().cpu()
