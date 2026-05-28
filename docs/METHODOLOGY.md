# Methodology (working draft)

This document is a working draft of the methodology section. It is **not**
intended as a finished thesis chapter — it's a structured set of paragraphs
that you can port to Overleaf / Word with minor stylistic adjustments. Each
section is small enough to be moved or reordered without losing context.

Conventions:
- — paragraphs ready to lift into the thesis with minor copy-editing.
- — placeholders that need a number, citation, or decision before the
  section can be considered complete.
- — design choices that the thesis must explicitly defend.

---

## 1. Task and dataset

The task addressed in this thesis is **selective auditory attention
decoding from EEG during naturalistic music listening**: given a short
multi-channel EEG segment, recorded while the subject listens to a
polyphonic music mixture, the model must infer which musical element
(vocal, drum, bass, or *others*) the subject was instructed to attend.

The dataset is the one released by Akama *et al.* (Sony CSL, 2025):
24 subjects, 4-channel consumer-grade EEG (Muse 2) at 256 Hz, paired with
the corresponding music mixtures and per-stem audio at 44.1 kHz. Each
trial is associated with a task label in `{0, 1, 2, 3}` mapping to
`{vocal, drum, bass, others}` and with a behavioural *attention score*
recorded during the experiment. The within-subject split used in this
work yields 3454 train, 1144 validation, and 806 test segments after the
sliding-window construction (window size 1280, stride 256, EEG clip
length 768).

*Add: figure of the experimental paradigm (can reuse from Akama paper
with citation).*

---

## 2. Baseline model (reproduction)

### 2.1 Architecture

As a baseline we reproduce the contrastive learning model proposed by
Akama *et al.* The model consists of:

- **One EEG encoder** (`SampleCNN2DEEG`, a 3-block 2D convolutional
  network with batch normalization and a 100-d projection head), which
  maps a `(4, 768)` EEG segment into a 100-dimensional latent vector
  `z_eeg`.
- **Four parallel audio encoders**, one per musical stem (vocal, drum,
  bass, others). Each is an independent copy of the same architecture
  used for the EEG encoder, applied to the corresponding audio stem
  segment, and produces a 100-dimensional latent vector `z_stem`.
- **A CLIP-style contrastive loss** (InfoNCE variant) which encourages
  `z_eeg` to be more similar to the `z_stem` of the attended stream than
  to any other stem in the batch.

The total parameter count is ~580K trainable parameters, of which ~464K
come from the four audio encoders.

### 2.2 Training protocol

Training uses Adam, learning rate 3 × 10⁻³, batch size 8, gradient
accumulation × 6, EarlyStopping on validation loss with patience 10 and a
minimum of 50 epochs. The model is trained on a single NVIDIA L40S GPU.
Splits and seeds are fixed (split_seed = 42, seed = 42).

### 2.3 Result

Test-set accuracy reaches **0.8650** *global accuracy (all data)* and
**0.8523** *high-attention accuracy*, matching the published baseline of
Akama *et al.* This reproduction is used as the reference point for all
subsequent experiments.

*Add: a small table per-class with [vocal, drum, bass, others] =
[0.84, 0.99, 0.86, 0.81] on all-data and [0.84, 0.98, 0.65, 0.80] on
high-attention. Discuss the asymmetry on `bass` vs `drum`.*

---

## 3. Proposed extension: CLAP-based audio representation

### 3.1 Motivation

🔬 The baseline learns each stem's audio encoder end-to-end on the same
small dataset that supervises the contrastive task. This couples *audio
representation learning* and *EEG-audio alignment* into a single
optimization problem, with the audio side bearing almost 80 % of the
trainable parameters. Two consequences follow:

1. The audio representation is necessarily narrow: 30 songs × 4 stems is
   a tiny audio corpus compared to what self-supervised audio models
   currently see.
2. Any signal the EEG side could exploit *beyond* what those small
   audio encoders capture is, by construction, inaccessible.

We therefore replace the four end-to-end audio encoders with a single
**frozen LAION-CLAP** backbone (Wu *et al.*, 2023), a contrastive audio-
language model pre-trained on millions of audio clips. The hypothesis is
that a representation pre-trained at scale on naturalistic audio
captures more transferable structure for music stems than a 116K-parameter
CNN trained on a few thousand clips.

### 3.2 Architecture

The CLAP variant keeps the EEG side and the loss identical to the
baseline. On the audio side:

- The four parallel audio encoders are replaced by **one shared
  `CLAPEncoder` module**, applied to each of the four stems in turn.
- `CLAPEncoder` wraps:
  1. an on-the-fly resampler from 44.1 kHz (the dataset's sample rate)
     to 48 kHz (what LAION-CLAP expects);
  2. the LAION-CLAP audio encoder (HTSAT-tiny variant), **frozen** —
     `requires_grad = False`, in `.eval()` mode — yielding a 512-d
     audio embedding;
  3. a small trainable projection head, `Linear(512 → 256) → GELU →
     Linear(256 → 100)`, mapping the CLAP embedding into the 100-d
     contrastive space used by the EEG encoder.

The CLAP backbone contains ~158M frozen parameters; the projection head
adds ~157K trainable parameters. Together with the (unchanged) EEG
encoder, the total trainable budget drops from ~580K (baseline) to
~273K, roughly halving it.

**Sharing the projection head across stems.** In our first design
the same `CLAPEncoder` instance — and therefore the same projection
head — is used for all four stems. The model discriminates between
vocal/drum/bass/others purely on the basis of how separable LAION-CLAP's
native representation already makes them. This isolates the contribution
of the pre-trained representation from any stem-specific adaptation.
A second design with four independent projection heads is a natural
follow-up if the shared-head variant under-performs.

### 3.3 Training protocol (comparison)

Hyperparameters, splits, seed, optimizer, batch size, EarlyStopping
configuration, and the CLIP loss are kept **identical** to the baseline.
The only change is the model architecture on the audio side, controlled
by a single flag `audio_repr ∈ {raw, clap}` in the configuration file.
This ensures the comparison is apples-to-apples on a single axis.

### 3.4 Known methodological limitations

🔬 Two limitations of the current CLAP integration are worth recording
explicitly:

1. **Out-of-distribution clip length.** LAION-CLAP was pre-trained on
   audio clips of 10–30 seconds. The Akama dataset pipeline pads each
   3-second clip to 3¹¹ samples at 44.1 kHz (~4.016 s), introducing
   roughly 1 s of silence on each side of the music. CLAP receives this
   padded segment as-is. Removing the padding would require modifying
   the dataset code, which is held fixed to keep the baseline
   reproducible. The first set of CLAP runs therefore measures
   "CLAP-frozen evaluated slightly out of its native input regime", and
   results should be read with that caveat.
2. **Frozen backbone.** Only the projection head is updated. If the
   first comparison indicates that the bottleneck is the rigidity of
   the representation rather than the head, a natural follow-up is
   parameter-efficient fine-tuning (e.g. LoRA adapters on the last
   transformer block of CLAP).

---

## 4. Comparison and analysis

🟡 *To be filled after the CLAP run finishes.*

Expected structure:

- **Headline number.** Test-set accuracy with `audio_repr=clap` vs the
  baseline 0.8650.
- **Per-class breakdown.** Whether the gap (if any) is uniform across
  vocal/drum/bass/others or concentrated on a specific stem.
- **High-attention vs all-data.** Whether CLAP's gain (or loss) is
  larger on the cleaner high-attention subset.
- **Parameter count vs accuracy.** ~273K trainable vs ~580K, useful to
  argue about sample efficiency regardless of the absolute accuracy.

🟡 *Optional: training curves (validation loss vs epoch) overlaid for
the two variants, to show convergence speed.*

---

## 5. Reproducibility

✏️ All code, configuration, and seeds required to reproduce the
experiments are available at
`https://github.com/giampyhatesyou/EEG-Attention-decoding-with-CLAP`.
The baseline run is launched with `bash scripts/train.sh` and the CLAP
variant with `bash scripts/train_clap.sh`. Each script writes its
checkpoints and TensorBoard logs into a separate `results/<training_date>`
subdirectory so the two runs cannot overwrite each other.

✏️ Dependencies are pinned in `requirements.txt`; the only versions
that diverge from a vanilla LAION-CLAP install are `transformers<4.40`
(LAION-CLAP 1.1.6 imports several classes from transformers in a way
that broke at version 4.40) and `numpy==1.26.4` (the LAION-CLAP pin on
1.23.5 is over-strict; the rest of the stack requires ≥1.24).

🟡 *Add: a short paragraph on the cluster setup once you settle on
which node ran the final runs (edu02 with NVIDIA L40S for training,
SLURM allocation 8 CPUs × 1 GPU; meg-server-3 for CPU-only baselines
when relevant).*

---

## 6. Things to add later

- Citation block for: Akama *et al.* 2025, Wu *et al.* 2023 (LAION-CLAP),
  PyTorch Lightning, the dataset license.
- A short paragraph on the *attention filter* / behavioural attention
  score, and on the distinction between "all-data" and "high-attention"
  metrics — this is currently implicit in the result tables.
- If you decide to pursue music-specific CLAP, LoRA, or a non-shared
  projection head, those become their own sub-sections under §3 and §4.
