# Concept Lens

This repository contains the code and frozen input data for the Concept Lens
pipeline: READ, rank-banded BIND, carrier-based CARRY, SCORE, and the causal
Swap evaluation.  It is a source checkout; model checkpoints and fitted lens
artifacts are deliberately kept outside the repository.

## Layout

```text
data/                         frozen v4 cloze and singleton inputs
src/bindlens/                 Concept Lens implementation
src/jlens/                    Jacobian-lens dependency used by the code
scripts/run_live_e2e.py       target-free online readout
scripts/experiments/          calibration, coverage, fragment, oracle, swap
weight/                       local links/placeholders for large artifacts
results/                      local experiment output (ignored by git)
```

## Install and check

Use Python 3.10 or newer.  The `accelerate` dependency is included because the
Llama-3.3-70B configuration uses Hugging Face device mapping.

```bash
python -m pip install -e .
python scripts/check_project.py
```

`check_project.py` is a model-free check: it validates the checked-in data,
script inventory, and Python syntax.  It does not load a checkpoint or start a
GPU job.

## Local artifacts

See [`weight/README.md`](weight/README.md) for the expected checkpoint and lens
names.  By default they are looked up below `weight/`; on a shared machine it
is usually easier to set paths explicitly:

```bash
export CONCEPT_LENS_MODELS_DIR=/path/to/huggingface/checkpoints
export CONCEPT_LENS_JLENS_DIR=/path/to/j_lenses
export CONCEPT_LENS_RLENS_DIR=/path/to/r_lenses
export CONCEPT_LENS_CARRIER_DIR=/path/to/carrier_artifacts
export CONCEPT_LENS_POSITIVE_POOL_DIR=/path/to/positive_pool_readers
```

`CONCEPT_LENS_ROOT` can point to the checkout when commands are launched from
another directory.  No large model or fitted tensor is required by the static
check, and none is bundled in this source tree.

## Entry points

After the required local artifacts are installed, representative commands are:

```bash
python scripts/run_live_e2e.py --model llama3_1_8b \
  --prompt "The capital of France is"
python scripts/experiments/coverage/run_coverage.py --model llama3_1_8b
python scripts/experiments/fragment/run_fragment.py --model llama3_1_8b
python scripts/experiments/swap/run_swap.py --model llama3_1_8b --mode install
```

The experiment commands write to `results/` by default.  They are intentionally
separate from the model-free repository check above.

## Data and protocol

The checked-in `data/clozes.json` is the v4 paired-cloze bank (1,200 pairs,
2,400 directed sides).  The singleton files contain the independent 300-item
calibration list and 500-item reconstruction list used by the fitting and
fragment commands.  Protocol constants and model-specific settings live in
`src/bindlens/protocol.py`.

The included `src/jlens` code is Apache-2.0 licensed; its license is retained
in `src/jlens/LICENSE`.
