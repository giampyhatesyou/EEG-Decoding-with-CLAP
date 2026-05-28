# Baseline: paper reproduction (Model: all-0 ms, within-subject)

**Date**: 2026-05-25 (test executed 13:18-13:30 on edu02 CPU)
**Checkpoint**: `proposed_model_checkpoints/model-all0.ckpt` (paper authors, epoch 48/82, best val_loss 2.25)
**Dataset**: `dataset/eeg_within_sub/` test split (806 windows / ~62 trials)

## Verifies that

The current code pipeline + paper-provided checkpoint reproduce the published numbers exactly. Any future modification can be compared against this fixed reference.

## Global metrics

| Metric                          | Paper  | Reproduced |
|---------------------------------|--------|------------|
| Global Accuracy (All data)      | 86.50% | **86.50%** |
| Global Accuracy (High attention)| 85.23% | **85.23%** |

## Per-task accuracy

| Task   | All data | High attention | Paper note                        |
|--------|----------|----------------|-----------------------------------|
| Vocal  | 84.08%   | 83.76%         | >80% expected                     |
| Drum   | 98.72%   | 98.46%         | >80% expected                     |
| Bass   | 85.88%   | **65.38%**     | Paper: "reached 65%" in attn eval |
| Others | 81.32%   | 80.42%         | >80% expected                     |

## Pairwise matrix (All data)

```
        Vocal   Drum    Bass    Others
Vocal   nan     0.9125  0.9284  0.8488
Drum    0.9872  nan     0.9936  0.9936
Bass    0.8588  0.9765  nan     0.9412
Others  0.9011  0.9011  0.9890  nan
```

## Pairwise matrix (High attention)

```
        Vocal   Drum    Bass    Others
Vocal   nan     0.9088  0.9259  0.8462
Drum    0.9846  nan     0.9923  0.9923
Bass    0.6538  0.9231  nan     0.8077
Others  0.9161  0.8741  0.9860  nan
```

## How to reproduce

```bash
cd codes_attention/attention
bash sequential_test.sh
```

(`sequential_test.sh` points to `../../proposed_model_checkpoints/model-all0.ckpt`)

## Files in this directory

- `test_pairwise_matrix.png` — heatmap delle matrici 4x4 (All vs High-attention)
- `test_per_task_accuracy.png` — bar chart per-task con linea di chance
- `hparams.yaml` — iperparametri usati a inference
- `events.out.tfevents.*` — log TensorBoard
- `RESULTS.md` — questo file
