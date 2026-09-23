# External artifacts

Large checkpoints and fitted tensors are not part of the source repository.
Place them under `weight/`, create local symlinks, or set the corresponding
`CONCEPT_LENS_*` environment variables.  The current code expects these names:

| Artifact | Relative path |
|---|---|
| Hugging Face checkpoints | `weight/models/<checkpoint-directory>/` |
| J-lens | `weight/jlens/<model>.pt` |
| R-lens (optional live override) | `weight/rlens/<model>_r_lens.pt` |
| Legacy carrier calibration | `weight/carrier/<model>.pt` |
| Shared positive-pool reader | `weight/carrier/positive_pool/<model>.pt` |
| Llama-3.3 L60 reader | `weight/carrier/positive_pool/llama3_3_70b_it_l60_intercept.pt` |

Model keys are `gemma3_12b_it`, `llama3_1_8b`, `qwen3_14b`,
`qwen3_6_27b`, and `llama3_3_70b_it`.  Their local checkpoint directory names
are defined in `src/bindlens/protocol.py`.

For a machine-specific layout, prefer explicit overrides, for example:

```bash
export CONCEPT_LENS_MODELS_DIR=/data/models
export CONCEPT_LENS_JLENS_DIR=/data/concept-lens/jlens
export CONCEPT_LENS_RLENS_DIR=/data/concept-lens/rlens
export CONCEPT_LENS_CARRIER_DIR=/data/concept-lens/carrier
export CONCEPT_LENS_POSITIVE_POOL_DIR=/data/concept-lens/positive_pool
```

The repository check reports missing artifacts as warnings because a clean code
checkout is useful without multi-gigabyte files.  Model commands fail early
with an actionable path message when a required checkpoint or lens is absent.
