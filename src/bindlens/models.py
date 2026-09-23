"""Load a Hugging Face model together with its configured Jacobian lens."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import transformers

import jlens
from jlens.lens import JacobianLens

from bindlens.paths import jlens_path, model_dir
from bindlens.protocol import MODELS, ModelSpec


@dataclass
class LoadedModel:
    name: str
    spec: ModelSpec
    model: Any
    hf: Any
    tokenizer: Any
    lens: JacobianLens
    # Provenance for optional persistent exact-operator caches.
    lens_path: Path
    device: torch.device

    @property
    def unembed(self) -> torch.Tensor:
        return self.model._lm_head.weight.detach()


def prepare_tokenizer(spec: ModelSpec) -> Any:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_dir(spec.local_dir)),
        local_files_only=True,
        use_fast=True,
    )
    if getattr(tokenizer, "bos_token_id", None) is not None and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = True
    return tokenizer


def load_model(
    name: str,
    device: str | torch.device = "cuda",
    *,
    lens_path: str | Path | None = None,
    device_map: str | None = None,
    max_gpu_memory_gib: int | None = None,
) -> LoadedModel:
    """Load ``name`` and its configured lens or an explicit compatible override.

    ``lens_path`` is deliberately an opt-in override.  It lets the online
    runner use a separately fitted R-lens without replacing the default
    J-lens under ``weight/jlens`` or changing any other command's default.
    """

    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    base_spec = MODELS[name]
    if max_gpu_memory_gib is not None and int(max_gpu_memory_gib) < 1:
        raise ValueError("max_gpu_memory_gib must be positive when provided")
    # Resolve the lens before touching a multi-gigabyte checkpoint.  A missing
    # external artifact should fail immediately, rather than after a model has
    # already consumed CPU/GPU memory.
    resolved_lens_path = Path(lens_path) if lens_path is not None else jlens_path(name)
    if not resolved_lens_path.is_file():
        raise FileNotFoundError(
            f"J-/R-lens artifact is missing: {resolved_lens_path}. "
            "Pass --lens-path or set CONCEPT_LENS_JLENS_DIR to a directory "
            "containing the released lens file."
        )
    checkpoint_dir = model_dir(base_spec.local_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"model checkpoint directory is missing: {checkpoint_dir}. "
            "Set CONCEPT_LENS_MODELS_DIR to the local Hugging Face checkpoint root."
        )
    # A serving command may opt a normally single-card checkpoint into an
    # Accelerate device map for a latency experiment.  Keep that resolved
    # placement in ``LoadedModel.spec`` so cache utilities know not to reuse
    # one GPU's row-index tensor for layers placed on another device.
    spec = replace(
        base_spec,
        device_map=base_spec.device_map if device_map is None else str(device_map),
        max_gpu_memory_gib=(
            base_spec.max_gpu_memory_gib
            if max_gpu_memory_gib is None
            else int(max_gpu_memory_gib)
        ),
    )
    device = torch.device(device)
    tokenizer = prepare_tokenizer(spec)
    if spec.model_class == "image_text_to_text":
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is None:
            raise RuntimeError(
                "this model configuration needs a recent Transformers release "
                "with AutoModelForImageTextToText"
            )
    else:
        cls = transformers.AutoModelForCausalLM
    load_kwargs: dict[str, Any] = {
        # ``torch_dtype`` remains supported across the Transformers versions
        # covered by this project; newer releases also accept ``dtype``.
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if spec.device_map is None:
        hf = cls.from_pretrained(str(model_dir(spec.local_dir)), **load_kwargs)
        hf.eval().to(device)
    else:
        if device.type != "cuda":
            raise ValueError(f"{name}: {spec.device_map=} requires CUDA")
        visible = torch.cuda.device_count()
        if visible < 2:
            raise RuntimeError(
                f"{name}: requires at least two visible GPUs for {spec.device_map=}; found {visible}"
            )
        ceiling = int(spec.max_gpu_memory_gib or 75)
        load_kwargs.update(
            {
                "device_map": spec.device_map,
                "max_memory": {**{index: f"{ceiling}GiB" for index in range(visible)}, "cpu": "80GiB"},
            }
        )
        hf = cls.from_pretrained(str(model_dir(spec.local_dir)), **load_kwargs)
        hf.eval()
    wrapped = jlens.from_hf(hf, tokenizer, compile=False, force_bos=True)
    lens = JacobianLens.load(str(resolved_lens_path))
    # With a device map, the input embedding and lm-head normally live on
    # different GPUs.  J-lens transport and vocabulary operations must follow
    # the lm-head; forward helpers explicitly use ``model.input_device``.
    compute_device = wrapped._lm_head.weight.device
    return LoadedModel(
        name=name,
        spec=spec,
        model=wrapped,
        hf=hf,
        tokenizer=tokenizer,
        lens=lens,
        lens_path=resolved_lens_path.resolve(),
        device=compute_device,
    )


def visible(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def encode_text(tokenizer: Any, text: str, *, special: bool) -> list[int]:
    ids = tokenizer(text, add_special_tokens=special).input_ids
    return [int(x) for x in ids]


def leading_space_ids(tokenizer: Any, text: str) -> list[int]:
    surface = text if text[:1].isspace() else " " + text
    ids = encode_text(tokenizer, surface, special=False)
    if not ids:
        raise ValueError(f"empty tokenization for {text!r}")
    return ids


def one_token_id(tokenizer: Any, text: str) -> int | None:
    ids = encode_text(tokenizer, text, special=False)
    if len(ids) != 1:
        return None
    return ids[0]


@torch.inference_mode()
def last_token_logits(loaded: LoadedModel, input_ids: list[int]) -> torch.Tensor:
    batch = torch.tensor([input_ids], dtype=torch.long, device=loaded.model.input_device)
    output = loaded.hf(input_ids=batch, use_cache=False)
    return output.logits[0, -1].detach().float().cpu()


@torch.inference_mode()
def last_residual_logits(loaded: LoadedModel, input_ids: list[int]) -> torch.Tensor:
    """Last-token logits via the jlens residual and official unembed (swap readout)."""
    from jlens.hooks import ActivationRecorder

    last_layer = int(loaded.model.n_layers) - 1
    batch = torch.tensor([input_ids], dtype=torch.long, device=loaded.model.input_device)
    with ActivationRecorder(loaded.model.layers, at=[last_layer]) as recorder:
        loaded.model.forward(batch)
        final = recorder.activations[last_layer][:, -1, :]
        return loaded.model.unembed(final)[0].float().cpu()


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[int(token_id)]).sum().item()) + 1
