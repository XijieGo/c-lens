"""Project-local paths for the Concept Lens repository.

Every default is resolved from this checkout.  An installed package can find
the checkout from its current directory, or use CONCEPT_LENS_ROOT explicitly.
The remaining CONCEPT_LENS_* variables intentionally place large local
artifacts outside the checkout.
"""

from __future__ import annotations

import os
from pathlib import Path


def _discover_repo_root() -> Path:
    """Find the checkout that owns data and artifacts for this import."""
    configured = os.environ.get("CONCEPT_LENS_ROOT")
    if configured:
        return Path(configured).resolve()

    module_root = Path(__file__).resolve().parents[2]
    if (module_root / "pyproject.toml").is_file() and (module_root / "data").is_dir():
        return module_root

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    # When the package is installed (rather than imported from ``src/``), the
    # module's parents are inside site-packages and cannot identify this
    # checkout.  Falling back to the current directory is predictable and
    # makes the required ``CONCEPT_LENS_ROOT`` override useful from any
    # other working directory; returning a site-packages parent would silently
    # point all data/artifact paths at the wrong place.
    return current


REPO_ROOT = _discover_repo_root()
DATA_DIR = Path(os.environ.get("CONCEPT_LENS_DATA_DIR", str(REPO_ROOT / "data")))
WEIGHT_DIR = Path(os.environ.get("CONCEPT_LENS_WEIGHT_DIR", str(REPO_ROOT / "weight")))
RESULTS_DIR = Path(os.environ.get("CONCEPT_LENS_RESULTS_DIR", str(REPO_ROOT / "results")))
LOGS_DIR = Path(os.environ.get("CONCEPT_LENS_LOGS_DIR", str(REPO_ROOT / "logs")))

MODELS_DIR = Path(os.environ.get("CONCEPT_LENS_MODELS_DIR", str(WEIGHT_DIR / "models")))
JLENS_DIR = Path(os.environ.get("CONCEPT_LENS_JLENS_DIR", str(WEIGHT_DIR / "jlens")))
RLENS_DIR = Path(os.environ.get("CONCEPT_LENS_RLENS_DIR", str(WEIGHT_DIR / "rlens")))
CARRIER_DIR = Path(os.environ.get("CONCEPT_LENS_CARRIER_DIR", str(WEIGHT_DIR / "carrier")))
POSITIVE_POOL_READER_DIR = Path(
    os.environ.get("CONCEPT_LENS_POSITIVE_POOL_DIR", str(CARRIER_DIR / "positive_pool"))
)

CLOZES_PATH = DATA_DIR / "clozes.json"
CALIBRATION_CONCEPTS_PATH = DATA_DIR / "singleton_calibration_300.json"
RECONSTRUCTION_TEST_CONCEPTS_PATH = DATA_DIR / "singleton_reconstruction_test_500.json"
ACTIVE_CALIBRATION_PATH = Path(
    os.environ.get("CONCEPT_LENS_CALIBRATION_DATA", str(CALIBRATION_CONCEPTS_PATH))
)


def model_dir(local_dir: str) -> Path:
    """Return the project-local directory for one Hugging Face checkpoint."""
    return MODELS_DIR / local_dir


def jlens_path(model: str) -> Path:
    """Return the default Jacobian-lens artifact for one model."""
    return JLENS_DIR / f"{model}.pt"


def rlens_path(model: str) -> Path:
    """Return the optional readout-lens artifact for one model."""
    # Released R-lens files use the explicit ``_r_lens`` suffix.  Keep a
    # compatibility fallback for older local workspaces that used the short
    # name, while always returning the released spelling for a new checkout.
    released = RLENS_DIR / f"{model}_r_lens.pt"
    legacy = RLENS_DIR / f"{model}.pt"
    return legacy if legacy.is_file() and not released.exists() else released


def carrier_path(model: str) -> Path:
    """Return the calibration carrier artifact for one model."""
    return CARRIER_DIR / f"{model}.pt"


def positive_pool_reader_path(model: str) -> Path:
    """Return the stored centered-positive reader shared by E2E and Swap."""
    return POSITIVE_POOL_READER_DIR / f"{model}.pt"


def llama33_l60_intercept_reader_path() -> Path:
    """Return the Llama-3.3 L60 reader with its fitted target-mean intercept.

    This reader intentionally remains separate from the standard L72
    calibration artifact because its validated online configuration uses a
    different carrier layer and natural-space prefix.
    """

    return POSITIVE_POOL_READER_DIR / "llama3_3_70b_it_l60_intercept.pt"
