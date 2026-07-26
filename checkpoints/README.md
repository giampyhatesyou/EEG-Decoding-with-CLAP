# Checkpoints

Paper-published model checkpoints from Akama et al., "Decoding Selective Auditory
Attention to Musical Elements in Ecologically Valid Music Listening" (Sony CSL, 2025).

| File | Setup | Notes |
|------|-------|-------|
| `model-all0.ckpt` | within-subject, all attention levels, 0ms delay | The headline model ("Model: all-0 ms" in the paper). Achieves 86.5% global accuracy. |
| `model-sub2.ckpt` | per-subject 2 (lowest attention) | For per-subject ablations |
| `model-sub3.ckpt` | per-subject 3 (best attention) | Highest within-subject accuracy in the paper (97.8%) |
| `model-sub7.ckpt` | per-subject 7 (middle attention) | Mid-range comparison |

These files are **not tracked by git** (~28 MB total). Extract them from the source
archive with:

```bash
bash scripts/setup_checkpoints.sh
```

Source archive: `archive/proposed_model_checkpoints.7z` (kept tracked, ~25 MB compressed).

## Notes on architecture

The checkpoints contain weights for an extra `projector1` layer (a 4-class classification
head) which is not present in the current model code. This was an auxiliary supervised
head used during training in the earlier version of the paper. When loading the checkpoint
the code uses `strict=False` to ignore these unused weights (and prints the missing /
unexpected key diff so a real mismatch is not masked). See the `load_state_dict` call
in `src/checkpoint_test.py`.
