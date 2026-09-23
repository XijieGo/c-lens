"""Locked method constants.

BIND proposes phrases from a cloze-conditioned decode over the J-lens
token set (rank-banded roots, lexical-only, no generic bridges).
CARRY builds `v_C` with one Fact carrier and one centered, non-negative
pool shared by E2E and Swap. SCORE ranks against the official J-lens
vocabulary and the layer-local candidates.
"""

from __future__ import annotations

from dataclasses import dataclass


# Primary E2E / Swap carrier.  There is deliberately no space after ``:``:
# it is part of the frozen prompt and therefore part of the token boundary.
CARRIER_PREFIX = "Remember the following concept in your mind. Concept:"
CARRIER_SUFFIX = ". Fact: The capital of France is Paris."

# The fragment reconstruction control uses its own carrier prompt.  Keeping it
# explicit prevents that control from mixing its fitted means with the online
# serving carrier.
CALIBRATION_CARRIER_PREFIX = "The following text names a concept.\nConcept: "
CALIBRATION_CARRIER_SUFFIX = ".\nFact: The capital of France is Paris."

DECODE_TEMPLATE = (
    "when model is answering the cloze below:\n"
    "{cloze} .\n"
    "it will think a concept:"
)

BRIDGE_TEXTS = (" is", " of", " the", " and", " a", " an")

S_K = 1000
SEED_K = 20
TOP_P = 0.85
EXPAND_K = 10
MAX_ACTIVE = 24
MAX_TOKENS = 4
RIDGE_RELATIVE_LAMBDA = 1e-2
WHITEN_EPS_FRACTION = 1e-3
# READ treats a cloze as the prefix of ``prompt + " " + concept``.  The
# word-initial space therefore belongs to the next concept token (for example
# ``ĠCentral``), rather than being supplied as a standalone input token.
# Keep the public constant for callers that record the protocol, but do not
# use it to append a token in ``with_trailing_space`` below.
TRAILING_SPACE = False
# Primary Swap: shared Fact-carrier reader, exact Cholesky transport, no
# trailing space, delete last 12, model-specific add windows, and format bans.
SWAP_TRAILING_SPACE = False
SWAP_DELETE_TOKENS = 12
SWAP_TRANSPORT = "exact_fp64_dense_cholesky"
SWAP_ARMS = ("full", "delete", "add", "first", "random")
POOL_MODE = "weighted"

# Rank-banded first-token allocation: (start, stop, active_width), 1-indexed inclusive.
# seed_k truncates each band. Rankband100_s20 uses 1-20:12 / 21-50:6 / 51-100:6.
ROOT_BANDS: tuple[tuple[int, int, int], ...] = (
    (1, 20, 12),
    (21, 50, 6),
    (51, 100, 6),
)

GATE = 10
SELECT_TOP_K = 25
MIN_SOURCE_LAYER = 10
TIE_EPS = 1e-6
RANDOM_DIRECTION_SEED = 20260828
RANKING_METHODS = ("v_C", "v_first", "v_mean", "v_nnls", "v_rand")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    hf_id: str
    local_dir: str
    model_class: str
    carrier_layer: int
    swap_band: tuple[int, int] | None
    add_last_n: int
    ban_surfaces: tuple[str, ...]
    # Large models may be sharded by Hugging Face across the visible GPUs.
    # ``None`` retains the ordinary single-device loading path.
    device_map: str | None = None
    max_gpu_memory_gib: int | None = None


MODELS: dict[str, ModelSpec] = {
    "gemma3_12b_it": ModelSpec(
        name="gemma3_12b_it",
        display_name="Gemma-3-12B-IT",
        hf_id="google/gemma-3-12b-it",
        local_dir="gemma-3-12b-it",
        model_class="image_text_to_text",
        carrier_layer=43,
        swap_band=(28, 33),
        add_last_n=8,
        ban_surfaces=(" the", "the", " The", "The"),
    ),
    "llama3_1_8b": ModelSpec(
        name="llama3_1_8b",
        display_name="Llama-3.1-8B",
        hf_id="meta-llama/Llama-3.1-8B",
        local_dir="Meta-Llama-3.1-8B",
        model_class="causal_lm",
        carrier_layer=27,
        swap_band=(16, 21),
        add_last_n=1,
        ban_surfaces=(' "', '"', " '", "'", " ‘", "’"),
    ),
    "qwen3_14b": ModelSpec(
        name="qwen3_14b",
        display_name="Qwen3-14B",
        hf_id="Qwen/Qwen3-14B",
        local_dir="Qwen3-14B",
        model_class="causal_lm",
        carrier_layer=37,
        swap_band=(31, 33),
        add_last_n=4,
        ban_surfaces=(' "', '"', " '", "'", " ‘", "’"),
    ),
    "qwen3_6_27b": ModelSpec(
        name="qwen3_6_27b",
        display_name="Qwen3.6-27B",
        hf_id="Qwen/Qwen3.6-27B",
        local_dir="Qwen3.6-27B",
        # This checkpoint is a text-capable conditional-generation wrapper;
        # the J-lens HF adapter resolves its embedded language-model stack.
        model_class="image_text_to_text",
        # L58 is 91% depth in the 64-block model, matching the late carrier
        # placement used for the other models and within the released lens
        # source-layer range L0--L62.
        carrier_layer=58,
        # Scale Qwen3-14B's L31--L33 write band by block depth: in the
        # 64-block Qwen3.6 this is L50--L53.  Keep the same late-token write
        # window and quote/format directions used by the other Qwen run.
        swap_band=(50, 53),
        add_last_n=4,
        ban_surfaces=(' "', '"', " '", "'", " ‘", "’"),
    ),
    "llama3_3_70b_it": ModelSpec(
        name="llama3_3_70b_it",
        display_name="Llama-3.3-70B-Instruct",
        hf_id="meta-llama/Llama-3.3-70B-Instruct",
        local_dir="Llama-3.3-70B-Instruct",
        model_class="causal_lm",
        # The 70B reader is fitted on the independent 300-singleton
        # calibration split.  L72 is the upper endpoint of the validated
        # serving workspace and is the carrier layer for this run.
        carrier_layer=72,
        # Scale Llama-3.1-8B's L16--L21 write band by block depth: in the
        # 80-block 70B this is L40--L53.  Keep the same late-token write
        # window and quote/format directions used by the 8B Llama run.
        swap_band=(40, 53),
        add_last_n=1,
        ban_surfaces=(' "', '"', " '", "'", " ‘", "’"),
        device_map="auto",
        # The live R-lens path stages exact fp64 transport maps and cached-tree
        # frontiers beside the 70B checkpoint.  Cap model placement at 68 GiB
        # per A800-80GB so that this serving workspace has a real margin.
        max_gpu_memory_gib=68,
    ),
}


class Protocol:
    prefix = CARRIER_PREFIX
    suffix = CARRIER_SUFFIX
    decode_template = DECODE_TEMPLATE
    s_k = S_K
    seed_k = SEED_K
    top_p = TOP_P
    expand_k = EXPAND_K
    max_active = MAX_ACTIVE
    max_tokens = MAX_TOKENS
    ridge = RIDGE_RELATIVE_LAMBDA
    pool_mode = POOL_MODE
    root_bands = ROOT_BANDS
    gate = GATE
    select_top_k = SELECT_TOP_K
    min_source_layer = MIN_SOURCE_LAYER
    swap_trailing_space = SWAP_TRAILING_SPACE
    swap_delete_tokens = SWAP_DELETE_TOKENS
    swap_transport = SWAP_TRANSPORT


def carrier_text(concept: str) -> str:
    return f"{CARRIER_PREFIX}{concept}{CARRIER_SUFFIX}"


def decode_prompt(cloze: str) -> str:
    """Paper scaffold with exactly one seed boundary supplied by the seed.

    Data clozes conventionally end in a blank answer slot.  Stripping it
    prevents a second space before the scaffold period, and leaving the colon
    terminal lets a word-initial J-lens seed supply the single space in
    ``concept: <seed>``.
    """
    return DECODE_TEMPLATE.format(cloze=str(cloze).rstrip())


def with_trailing_space(prompt: str) -> str:
    """Return the natural-BPE source prefix (historical public name).

    ``leading_space_ids(concept)`` is the exact continuation of this prefix.
    Adding a standalone final-space token here would instead score/generate a
    double-space sequence for the Llama, Gemma, and Qwen tokenizers.
    """
    return str(prompt).rstrip()


def with_swap_trailing_space(prompt: str) -> str:
    text = with_trailing_space(prompt)
    return text + " " if SWAP_TRAILING_SPACE else text


def swap_protocol(spec: ModelSpec) -> dict[str, object]:
    """Locked primary Swap knobs for one model.

    The denominator is deliberately supplied by the runner, rather than this
    model spec: primary Swap evaluates every directed side of the selected
    pair bank and does not use model-specific ID subsets.
    """
    if spec.swap_band is None:
        raise ValueError(f"{spec.name}: Swap has no configured write band")
    return {
        "carrier_layer": spec.carrier_layer,
        "band": list(spec.swap_band),
        "add_last_n": spec.add_last_n,
        "ban_surfaces": list(spec.ban_surfaces),
        "delete_tokens": SWAP_DELETE_TOKENS,
        "trailing_space": SWAP_TRAILING_SPACE,
        "transport": SWAP_TRANSPORT,
        "primary_n": None,
        "carrier_prefix": CARRIER_PREFIX,
        "suffix": CARRIER_SUFFIX,
        "reader": "frozen_centered_positive_pool",
        "pooling": "centered_positive_weight_pool",
    }
