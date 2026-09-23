#!/usr/bin/env python3
"""Validate the source checkout without loading a model or using a GPU.

This is deliberately a small release sanity check.  It checks the frozen data
schema, expected v4 counts, required launchers, and Python syntax.  Large
external artifacts are reported as warnings rather than failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SRC / "jlens") not in sys.path:
    sys.path.insert(0, str(SRC / "jlens"))

from bindlens.io import iter_sides, load_cloze_pairs, load_singleton_concepts  # noqa: E402
from bindlens.paths import (  # noqa: E402
    CALIBRATION_CONCEPTS_PATH,
    carrier_path,
    CLOZES_PATH,
    JLENS_DIR,
    POSITIVE_POOL_READER_DIR,
    RECONSTRUCTION_TEST_CONCEPTS_PATH,
    RLENS_DIR,
    llama33_l60_intercept_reader_path,
    model_dir,
    jlens_path,
    rlens_path,
)
from bindlens.protocol import MODELS  # noqa: E402


REQUIRED_SCRIPTS = (
    "scripts/run_live_e2e.py",
    "scripts/fit_positive_pool.py",
    "scripts/experiments/calibration/run_calibration.py",
    "scripts/experiments/coverage/run_coverage.py",
    "scripts/experiments/fragment/run_fragment.py",
    "scripts/experiments/oracle_ranking/run_oracle_candidate_ranking.py",
    "scripts/experiments/swap/run_swap.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-pairs", type=int, default=1200)
    parser.add_argument("--expected-sides", type=int, default=2400)
    parser.add_argument("--expected-calibration", type=int, default=300)
    parser.add_argument("--expected-reconstruction", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        CLOZES_PATH,
        CALIBRATION_CONCEPTS_PATH,
        RECONSTRUCTION_TEST_CONCEPTS_PATH,
    ]
    for path in required_files:
        if not path.is_file():
            errors.append(f"missing required file: {path}")

    if not errors:
        try:
            pairs = load_cloze_pairs(CLOZES_PATH)
            sides = iter_sides(pairs)
            if args.expected_pairs and len(pairs) != args.expected_pairs:
                errors.append(f"cloze pair count is {len(pairs)}, expected {args.expected_pairs}")
            if args.expected_sides and len(sides) != args.expected_sides:
                errors.append(f"directed-side count is {len(sides)}, expected {args.expected_sides}")
            calibration = load_singleton_concepts(CALIBRATION_CONCEPTS_PATH)
            reconstruction = load_singleton_concepts(RECONSTRUCTION_TEST_CONCEPTS_PATH)
            if args.expected_calibration and len(calibration) != args.expected_calibration:
                errors.append(
                    f"calibration count is {len(calibration)}, expected {args.expected_calibration}"
                )
            if args.expected_reconstruction and len(reconstruction) != args.expected_reconstruction:
                errors.append(
                    "reconstruction count is "
                    f"{len(reconstruction)}, expected {args.expected_reconstruction}"
                )
            overlap = sorted(set(calibration).intersection(reconstruction))
            if overlap:
                errors.append(f"singleton splits overlap ({len(overlap)} concepts)")
            print(f"data: {len(pairs)} pairs / {len(sides)} directed sides")
            print(f"singletons: {len(calibration)} calibration / {len(reconstruction)} reconstruction")
        except Exception as exc:  # pragma: no cover - the message is the check's output
            errors.append(f"data validation failed: {exc}")

    for relative in REQUIRED_SCRIPTS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing launcher: {ROOT / relative}")

    # Compile source and launchers in memory.  This deliberately avoids
    # imports, model construction, GPU access, and even writing __pycache__.
    for directory in (ROOT / "src", ROOT / "scripts"):
        if not directory.is_dir():
            continue
        for source in sorted(directory.rglob("*.py")):
            try:
                compile(source.read_text(encoding="utf-8"), str(source), "exec")
            except (OSError, SyntaxError, UnicodeError) as exc:
                errors.append(f"Python syntax check failed for {source}: {exc}")

    for model in MODELS:
        checkpoint = model_dir(MODELS[model].local_dir)
        if not checkpoint.is_dir():
            warnings.append(f"{model}: missing model checkpoint directory: {checkpoint}")
        for label, path in (
            ("J-lens", jlens_path(model)),
            ("R-lens", rlens_path(model)),
            ("carrier calibration", carrier_path(model)),
            ("positive-pool reader", POSITIVE_POOL_READER_DIR / f"{model}.pt"),
        ):
            if not path.is_file():
                warnings.append(f"{model}: missing external {label}: {path}")
    l60_reader = llama33_l60_intercept_reader_path()
    if not l60_reader.is_file():
        warnings.append(f"llama3_3_70b_it: missing L60 intercept reader: {l60_reader}")

    print(f"artifact directories: J={JLENS_DIR} R={RLENS_DIR}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("check_project: OK (external artifact warnings do not affect the source check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
