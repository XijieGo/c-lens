#!/usr/bin/env python3
"""Fit one centered-positive carrier reader with its target-mean intercept."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import _boot  # noqa: F401
import torch

from bindlens.carry import fit_positive_pool_reader
from bindlens.io import load_calibration_concepts, write_json
from bindlens.models import load_model
from bindlens.paths import ACTIVE_CALIBRATION_PATH, positive_pool_reader_path
from bindlens.protocol import CARRIER_PREFIX, CARRIER_SUFFIX, MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--carrier-prefix", default=CARRIER_PREFIX)
    parser.add_argument("--carrier-suffix", default=CARRIER_SUFFIX)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument(
        "--fit-data",
        "--calibration-data",
        dest="fit_data",
        type=Path,
        default=None,
        help="singleton concept-list JSON for the reader fit (defaults to the active calibration split)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-mem-fraction", type=float, default=None)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="carrier layer to fit; default is the model spec carrier_layer",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    out_file = args.out_file or positive_pool_reader_path(args.model)
    if out_file.exists():
        raise FileExistsError(f"refusing to overwrite reader artifact: {out_file}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.gpu_mem_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), device=device)
    loaded = load_model(args.model, device)
    fit_data = args.fit_data or ACTIVE_CALIBRATION_PATH
    fit_concepts = load_calibration_concepts(fit_data)
    reader, metadata = fit_positive_pool_reader(
        loaded,
        prefix=args.carrier_prefix,
        suffix=args.carrier_suffix,
        layer=args.layer if args.layer is not None else loaded.spec.carrier_layer,
        batch_size=args.batch_size,
        fit_concepts=fit_concepts,
    )
    metadata["fit_data"] = str(fit_data)
    metadata["fit_data_sha256"] = sha256(fit_data)
    payload = {
        "schema_version": int(metadata["schema_version"]),
        "metadata": metadata,
        "readers": {"positive_pool": reader},
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_file)
    meta_path = out_file.with_suffix(".json")
    write_json(meta_path, metadata)
    print(
        f"[{args.model}] wrote {out_file} L{metadata['carrier_layer']} "
        f"pool={metadata['pooling']} used={metadata['fit_singleton_used']}/"
        f"{metadata['fit_corpus_requested']} residual_cos="
        f"{metadata['fit_mean_paired_cosine_centered_residual']:.4f} "
        f"bias={metadata['output_bias_kind']}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
