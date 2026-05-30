# Audit Findings — EEG Attention Decoding with CLAP

This document records the full audit and the extension experiments performed
on top of Akama et al. (2025), "Decoding Selective Auditory Attention to
Musical Elements in Ecologically Valid Music Listening" (Sony CSL).

All numbers in this document come from real run outputs; the path to each
`test_breakdown_summary.txt` is listed in §5. Only the baseline and its
negative-control sweep are committed under `results/` in this repo; the CLAP,
LOSO, and LSO run directories live on the training cluster (paths in §5).

---

## 0. TL;DR

I replaced the paper's four raw-CNN audio encoders with a single shared,
**frozen LAION-CLAP** encoder followed by a small trainable projection head.
I reproduce the paper's baseline (0.865) and improve to **0.9237** global
accuracy under the same within-subject split.

I then designed audit experiments **not present in the paper**:

1. **Negative controls** (label / audio shuffling) validate the metric.
2. **Negative controls on the paper's baseline checkpoint** (same audit
   protocol, raw audio encoder) reveal an identical 0.13 "stem-type
   generic" prior, so this component is a **property of the dataset, not
   of CLAP**.
3. **Leave-one-subject-out** (LOSO) on subject 0 reaches **0.9363**.
4. **Leave-one-song-out** (LSO) collapses on every non-vocal target:
   drum songs 36/145/59 → 0.019/0.000/0.288, bass song 44 → 0.087,
   others song 16 → 0.000. All at or below chance (0.25).
5. **The apparent LSO "success" on vocal-target songs is a class-prior
   bias, not decoding.** On held-out non-vocal songs the model predicts
   vocal 75–93 % of the time (it defaults to the 49 % majority class), so
   a vocal-target held-out song (song 3 → 0.904) is "correct" only
   because the default happens to match the target.
6. **The same LSO collapse happens on the paper's raw baseline**: LSO 36
   with `audio_repr=raw` gives **0.000**. The cross-song failure is a
   property of the dataset/paradigm, not of CLAP.
7. **Leave-one-subject-out** is robust (0.947 ± 0.024 over 6 subjects) but
   every LOSO fold still has all songs in training, so it measures
   cross-subject EEG transfer on top of fully-seen audio geometry — it
   does not contradict the cross-song collapse.

**Bottom line.** The 0.9237 figure decomposes as:

```
chance                                            0.25
+ "stem-type generic" prior (audio_pair shuffle)  0.13   ← same in baseline
+ song-specific geometry memorisation             0.54
══════════════════════════════════════════════════════
total within-subject                              0.92
```

The +0.13 component is **shared with the paper's baseline** (which we
reproduced and audited on the same checkpoint): in both models the
audio_pair shuffle yields ~0.38 accuracy. That component is therefore
not a property of CLAP-frozen — it is a property of the task / dataset
geometry (the four stems of a song have systematically distinguishable
acoustic profiles regardless of which encoder maps them).

On the leave-one-song-out folds run so far (two, both drum-attended), the
model does *not* generalise to music it has not seen during training. The
bulk of the within-subject accuracy appears to be memorisation of the
fixed audio-embedding geometry of the 63 MUSDB18 songs. This **extends**
the paper rather than contradicting it — see §1.4 for the exact
relationship to the paper's reported numbers.

This is consistent with — but goes beyond — the paper's own caveat that
*"our model may be less sensitive to attention-level differences"* (only
1.27 % gap between all-data and high-attention evaluations).

---

## 1. Context and goal

### 1.1 The paper
Akama et al. (2025) propose music-oriented Auditory Attention Decoding (AAD)
from 4-channel consumer EEG (Muse 2). Eight subjects listened to 63 songs
from MUSDB18 (31 vocal, 11 drum, 11 bass, 10 others — heavily imbalanced),
each cued to attend to one of {vocal, drum, bass, others}. The model is
a 2D-CNN EEG encoder + four parallel CNN audio encoders + an InfoNCE
contrastive loss; their best variant ("Model: all-0 ms") reaches **0.865**
global accuracy on within-subject test and **0.756** on leave-one-subject-
out (only 3 subjects evaluated).

The paper itself flags as **future work** (§"Challenges in Generalization
to New Subjects and Songs", p.18) the evaluation of "attentional focus
when an existing user listens to entirely new songs". This is exactly the
gap our leave-one-song-out experiment fills.

### 1.2 What I changed
I swap the audio side for a **frozen LAION-CLAP** backbone shared across
the four stems, followed by a small trainable projection head
`Linear(512) → GELU → Linear(100)`. Everything else (EEG encoder, loss,
optimiser, batch size, splits, seed) is meant to be identical to the paper. The only
change is `audio_repr: clap` in the YAML.

### 1.3 What I audited
Within the paper's protocol, each subject's data is split 8:1:1 over its
own songs. I observed empirically that across subjects, **30 out of 31**
distinct test songs also appear in training (because different subjects
attended different songs and the dataset randomly assigned each song to a
split *per subject*).

Since the audio embedding of any song depends only on the audio file (not
on the subject who heard it), the consequence is that during training the
model sees the audio embeddings of essentially every song in the dataset.
The hypothesis I tested is that the apparent decoding capability is
largely **audio-embedding memorisation** rather than true EEG→attention
generalisation.

### 1.4 Relationship to the paper's reported claims

This audit does **not** contradict any number the paper reports; every
comparable metric is reproduced or exceeded:

| Evaluation | Paper | This work |
|---|---|---|
| Within-subject (global acc) | 0.865 | 0.865 (raw, authors' checkpoint) / 0.9237 (CLAP) |
| Cross-subject (leave-one-subject-out) | 0.756 (raw, 3 subjects) | 0.9363 (CLAP, 1 subject) |
| Cross-song (leave-one-song-out) | — (listed as future work) | 0.000–0.019 (2 folds, both drum) |

Two consequences:

1. The within-subject 0.865 and the cross-subject 0.756 are **real and
   reproducible** — the within-subject figure is recovered bit-for-bit
   from the authors' own checkpoint (§3.5). The paper is not overstating
   these numbers, and our pipeline matches or beats the paper on every
   metric it reports.
2. The leave-one-song-out collapse is **not** a refutation of a paper
   claim: the paper makes no cross-song claim — it explicitly defers this
   setting to future work (§1.1). Our LSO result **closes that gap**
   rather than contradicting the paper.

The contribution is therefore **interpretive and extensional**, not a
falsification. The within-subject accuracy is genuine but, on the
evidence here, reflects in large part memorisation of the fixed audio
geometry of the 63 training songs rather than an EEG→attention map that
transfers to novel music. The LOSO-vs-LSO contrast makes the mechanism
explicit: under leave-one-*subject*-out the audio of every song is still
in the training set (only the EEG distribution shifts) and accuracy stays
high; under leave-one-*song*-out the held-out song's audio is
out-of-distribution and accuracy collapses. **The bottleneck is audio
novelty, not subject novelty.**

Caveat on the cross-subject row: it is **not** apples-to-apples (CLAP vs
the paper's raw encoder; one fold vs three). A raw-encoder LOSO on the
paper's three subjects (3/7/2) is required before stating any cross-subject
claim relative to the paper — see §7.

---

## 2. Code changes

### 2.1 `configs/baseline.yaml` (+27 lines)
Four new YAML keys, all picked up automatically by `argparse` in
`main.py` / `checkpoint_test.py`:

| key | type | default | purpose |
|---|---|---|---|
| `cv_mode` | str | `"within"` | cross-validation routing: `within` / `leave_song_out` / `leave_subject_out` |
| `cv_held_out_id` | int | `-1` | song id or subject id held out for the current fold; ignored when `cv_mode=within` |
| `shuffle_test_mode` | str | `"none"` | negative control at test time: `none` / `labels` / `audio_pair` |
| `test_breakdown` | int | `1` | when 1, write per-window records + per-subject/song/attention/task CSVs and figures |

Defaults preserve legacy behaviour bit-for-bit.

### 2.2 `src/datasets/preprocessing_eegmusic_dataset.py`
Two new constructor arguments (`cv_mode`, `cv_held_out_id`). In
`_get_file_list` a CV-gate filters the trial dataframe **before** the
original `subset` filter:

- `within` — unchanged (uses the folder-tree split).
- `leave_song_out` — train / valid: original split with `song != held_out`;
  test: all rows where `song == held_out`, regardless of original split.
- `leave_subject_out` — same logic but on `subject`.

Verified empirically on the real data (`/tmp` sanity script):
no (subject, song) pair appears in two splits, so leave-song-out and
leave-subject-out actually hold out what they say.

### 2.3 `src/datasets/__init__.py`
`get_dataset()` accepts the two new args and passes them to the dataset
constructor.

### 2.4 `src/main.py` and `src/checkpoint_test.py`
Both read `args.cv_mode` and `args.cv_held_out_id` and forward them to
`get_dataset()` for train / valid / test.

`checkpoint_test.py` also forces `shuffle=True` on the test DataLoader
whenever `args.shuffle_test_mode != "none"`. This is necessary because the
test dataset emits the 13 sliding windows of a trial consecutively;
with `shuffle=False` and `batch_size=8`, ~64 % of batches share a single
task across all 8 samples and any intra-batch shuffle becomes a no-op.

### 2.5 `src/modules/contrastive_learning.py` (+158 lines)

**At init.** `self._shuffle_test_mode` stored; `self.test_records = []`
initialised.

**In `test_step` (before the forward pass):**
- `labels` mode: `task = torch.randint(0, 4, task.shape)`. Truly i.i.d.
  uniform labels; stronger control than `torch.randperm`, which preserves
  the marginal task distribution.
- `audio_pair` mode: `torch.randperm` over the batch is applied to all
  four audio tensors, so each EEG is paired with the stems of another
  song in the batch.

**Per-window record collection (when `test_breakdown=1`).** After the
criterion call, for each task index `t` the code iterates over the
similarity rows in `matrix_list[t]` and appends one dict per window to
`self.test_records`. Each record contains:

```
subject, song, task, attention, sim_vocal, sim_drum, sim_bass,
sim_others, pos_sim, max_neg_sim, margin (= pos − max_neg),
correct (1/0), high_attention (1/0)
```

**`on_test_end` extension.** After saving the two pre-existing figures
(`test_pairwise_matrix.png`, `test_per_task_accuracy.png` — unchanged), it
adds:

- `test_records.csv` — one row per window
- `test_per_subject.csv` + `test_per_subject.png`
- `test_per_song.csv` + `test_per_song.png`
- `test_per_attention.csv` + `test_vs_attention.png`
- `test_per_task_summary.csv`
- `test_breakdown_summary.txt` (the headline number + n_windows totals)

The printed `eval_all` / `eval_att` numbers and the two original figures
are bit-for-bit identical to before when `shuffle_test_mode="none"` and
`test_breakdown=0`.

### 2.6 New driver scripts under `scripts/`

| script | what it does |
|---|---|
| `scripts/test_sanity.sh` | runs `checkpoint_test.py` 3× with `shuffle_test_mode=none|labels|audio_pair`. Output: `results/<TAG>_<MODE>/...`. |
| `scripts/train_cv.sh` | one fold. Usage: `bash scripts/train_cv.sh <cv_mode> <held_out_id> [tag]`. First trains from scratch (`main.py`), then tests (`checkpoint_test.py`). |
| `scripts/run_cv_sweep.sh` | loops `train_cv.sh` over a list of held-out ids. |

All three scripts activate the conda env `eeg_attention` and `tee` the log
into `logs/`.

---

## 3. Experiments and results

### 3.1 Within-subject CLAP run (the headline number)

| field | value |
|---|---|
| command | `bash scripts/train_clap.sh` |
| checkpoint | `results/clap_run1/nmed-CL-preprocessing_eegmusic/version_1/checkpoints/best-checkpoint.ckpt` |
| n_windows | 800 (drop_last=True; 806 trial-windows in the test set, 800 used) |
| **global accuracy (all)** | **0.9237** |
| global accuracy (high-attention) | 0.9185 |
| per-task accuracy | vocal 0.94, drum 0.90, bass 1.00, others 0.87 |

The bass=1.00 is suspicious low-N: only 7 trials × ~13 windows ≈ 91
samples. Sub-sub-population statistics.

### 3.2 Negative controls (sanity v2)

| run | `shuffle_test_mode` | acc (all) | acc (attn) | mean margin |
|---|---|---:|---:|---:|
| clap_sanity_v2_none | `none` | 0.9237 | 0.9185 | +2.22 |
| clap_sanity_v2_labels | `labels` | **0.2575** | 0.2481 | −1.65 |
| clap_sanity_v2_audio_pair | `audio_pair` | **0.3812** | 0.3829 | −0.89 |

Reading:

- **`labels` collapses to chance 0.25 exactly.** No bug in the metric: when
  the labels are random, accuracy is random. (Two earlier runs at 0.80
  were a no-op due to homogeneous batches; the fix that gave the 0.2575
  number is the `shuffle=True` flag on the test DataLoader plus
  `torch.randint` for labels — see §2.5.)
- **`audio_pair` collapses to 0.38**, not to chance. The 0.13 above
  chance reflects what I call the **stem-type generic prior**: the
  model can tell "vocal stem" from "drum stem" from "bass stem" from
  "others stem" *as long as both the EEG and the audio are drawn from
  the training distribution*. This is a property of the CLAP
  representation + projection head, not of EEG→attention decoding.

### 3.3 Leave-one-subject-out (LOSO)

Only one fold completed (subject 0); a second (subject 1) crashed mid-
training when the jupyterhub session ended.

| held-out subject | training songs | test trials | acc (all) | acc (attn) |
|---|---|---|---:|---:|
| 0 | 271 (subjects 1–7) | 62 (all of subject 0's trials) | **0.9363** | 0.9304 |

Subject 0's LOSO accuracy is **higher** than the within-subject 0.9237.
This is consistent with the paper's table 1 (Sub#1 within-subject was
0.9135, mid-cohort) — but the *cross-subject* number in our setup is
much higher than the paper's mean (~0.756, table 2 on three subjects).

Interpretation: under LOSO, the model still sees **every song** in
training (because every song is heard by multiple subjects); only the
EEG distribution shifts. The audio side carries enough of the signal that
the model trains on subjects 1–7 to map their EEG onto a fully known
audio geometry and then transfers reasonably to subject 0's EEG mapped
onto the same audio geometry.

### 3.4 Leave-one-song-out (LSO) — the critical experiment

Two folds completed.

| held-out song | n_trials in test | task distribution | acc (all) | mean margin |
|---|---|---|---:|---:|
| 36 | 8 (one per subject) | drum: 104/104 | **0.0192** | −2.39 |
| 145 | 8 (one per subject) | drum: 104/104 | **0.0000** | −3.22 |

Both folds happen to be drum-attended. The held-out trial of each subject
was a song the model never saw during training; the test set has 8
trials × 13 windows = 104 windows.

**Argmax distribution for LSO 145:**

| stem | count / 104 | fraction |
|---|---:|---:|
| vocal | 24 | 0.231 |
| drum | **0** | **0.000** |
| bass | 72 | 0.692 |
| others | 8 | 0.077 |

**Mean similarity per stem for LSO 145:**

```
sim_vocal:   +0.83
sim_drum:    −1.51   ← target (always lowest)
sim_bass:    +1.68   ← favoured (always highest)
sim_others:  +1.29
```

Across all 8 subjects and at all behavioural attention levels (2–5), the
LSO 145 accuracy is exactly 0. The model places the drum stem of song
145 systematically far from any EEG and prefers the bass stem.

LSO 36 shows the same pattern, slightly less extreme: drum is argmax
2 % of the time, bass 57 %.

This is **not** the chance-level 0.25 expected from a model that has
"just lost the song-specific cue but still knows what drum looks like";
it is *below* chance, with a systematic, predictable error pattern.

### 3.4b LSO on the paper baseline (raw encoder) — the same collapse

I trained one full LSO fold (song 36) with `audio_repr=raw` (the paper's
four-CNN architecture, trained from scratch on the remaining 62 songs;
50 epochs, ~4.8 h on one L40S). Command:
`AUDIO_REPR=raw bash scripts/train_cv.sh leave_song_out 36 lso_baseline`.

| LSO 36 (target = drum, 104 windows) | accuracy | argmax distribution | mean sim (drum / target) |
|---|---:|---|---:|
| CLAP (`audio_repr=clap`) | 0.0192 | vocal 37 % / **drum 2 %** / bass 57 % / others 5 % | −0.58 |
| **Baseline (`audio_repr=raw`)** | **0.0000** | **vocal 94 %** / drum 0 % / bass 0 % / others 6 % | −1.77 |

Both architectures collapse below chance on a never-seen song. The drum
stem (the true target) is essentially never selected (2 % and 0 %). The
two models differ only in their **default** when faced with
out-of-distribution audio: CLAP falls back to bass, the raw baseline
falls back overwhelmingly to vocal (94 %) — i.e. to the majority class
(vocal is 49 % of the training trials).

**This is the key cross-architecture result.** The cross-song failure is
*not* a CLAP-specific artefact: the original Akama architecture fails the
same way. The leave-one-song-out evaluation — which the paper lists as
future work — exposes that **neither model learns a generalisable
EEG → attended-instrument map; both rely on memorising the fixed audio
geometry of the training songs plus a class-frequency prior.**

### 3.4c LSO across all four target tasks — the class-prior bias (decisive)

Each MUSDB18 song has a single fixed target task (all subjects who heard
it were cued to attend the same element). Songs 36 and 145 (§3.4) were
both drum-target. We therefore ran a CLAP LSO sweep covering all four
target tasks (`scripts/sweep_audit.sbatch`, run via `bash`):

| song | target | LSO accuracy | argmax distribution | reading |
|---|---|---:|---|---|
| 3 | vocal | **0.904** | vocal 90 % | "correct" because the model's OOD default *is* vocal |
| 5 | vocal | **0.462** | vocal 46 % / bass 54 % | partial — this song's vocals are less typical |
| 59 | drum | 0.288 | drum 29 % / bass 38 % / others 34 % | scattered, ≈ chance |
| 36 | drum | 0.019 | bass 57 % / vocal 37 % / drum 2 % | collapse (§3.4) |
| 145 | drum | 0.000 | bass 69 % / vocal 23 % / drum 0 % | collapse (§3.4) |
| 44 | bass | 0.087 | **vocal 75 %** / bass 9 % | predicts vocal, target is bass → collapse |
| 16 | others | 0.000 | **vocal 93 %** / others 0 % | predicts vocal, target is others → total collapse |

Mean LSO accuracy by target task:

```
vocal:  ~0.68   (but see below — this is bias, not decoding)
drum:   ~0.10
bass:   ~0.09
others: ~0.00
```

**The decisive observation.** On songs 44 (bass) and 16 (others) — songs
the model never saw — the model predicts **vocal** 75 % and 93 % of the
time respectively. It defaults to the majority class (vocal = 49 % of the
training trials). The apparent "generalisation" to vocal-target songs
(song 3, 0.904) is therefore **not** attention decoding: it is the
class-prior default accidentally matching the target whenever the target
happens to be vocal. Song 5 (also vocal) gets only 0.462 because its
vocal stem is acoustically less typical, so the default splits between
vocal and bass.

In other words, leave-one-song-out shows the model has **no
song-transferable EEG → instrument map at all**. Whatever within-subject
accuracy looked like decoding is (a) song-specific audio-geometry
memorisation, plus (b) a vocal-majority prior that only "works" on
held-out songs when those songs are themselves vocal-target.

### 3.4d Leave-one-subject-out variance (CLAP, 6 folds)

| held-out subject | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| LOSO accuracy | 0.936 | 0.924 | 0.953 | 0.968 | 0.917 | 0.986 |

Mean = **0.947 ± 0.024** over all 6 subjects (6/6 above 0.91).
Cross-subject transfer is
robust and high — but in every LOSO fold all 63 songs are present in
training (each song is heard by multiple subjects), so this number
reflects cross-subject EEG transfer *on top of* fully-seen audio
geometry. It does **not** contradict the cross-song collapse; the two
evaluations isolate different axes.

### 3.5 Negative controls on the paper baseline (raw audio encoder)

I re-ran the same three-mode sanity sweep on the paper's checkpoint
`checkpoints/model-all0.ckpt` (the "all-0 ms" model from Akama et al.,
`audio_repr=raw`, four independent CNN audio encoders trained jointly
with the EEG encoder). Command:
`TAG=baseline_sanity bash scripts/test_sanity.sh`.

| `shuffle_test_mode` | CLAP (clap_run1) | Baseline (model-all0) |
|---|---:|---:|
| `none` | 0.9237 | **0.8650** (matches the paper's headline) |
| `labels` (i.i.d. uniform) | 0.2575 | **0.2250** ≈ chance |
| `audio_pair` | 0.3812 | **0.3825** |

**Reading.** The `audio_pair` accuracy is essentially identical for the
two architectures (0.3812 vs 0.3825). This is a structural property of
the dataset: when each EEG window is paired with stems from a different
song (so the audio is in-distribution but not the *correct* in-distribution
audio for that EEG), both models retain the same ~13 percentage points of
accuracy above chance. This +0.13 is the "stem-type generic prior" — the
fact that a vocal stem audio embedding is systematically closer to "EEG
of someone attending vocal" than to "EEG of someone attending drum/bass/
others", because the four CLAP- or CNN-projected audio embeddings of any
song occupy systematically different regions of the contrastive space.

The per-task and pairwise patterns are also similar: in both models the
vocal row has the highest off-diagonal values (the model picks vocal far
more often than other stems when scrambled), reflecting both class
imbalance (49 % vocal) and the easier acoustic distinguishability of
vocal stems.

**Implication for §3.4 (LSO).** Because the +0.13 prior is shared with
the baseline, the LSO collapse I observed on CLAP is unlikely to be a
CLAP-specific failure mode. Verifying this directly requires running an
LSO fold with `audio_repr=raw`; see §7 (Follow-ups). Note that one LSO
fold of the baseline takes roughly 5× longer to train than CLAP (see
§3.7) so it has not been completed in the current time budget.

### 3.6 What this tells us about the 0.92

```
chance baseline:                                  0.25
+ "stem-type generic" prior:                      0.13
+ song-specific geometry memorisation:            0.54
══════════════════════════════════════════════════════
within-subject "none" run:                        0.92
```

The +0.13 (stem-type generic) requires both EEG **and** audio to be
in-distribution. In LSO the audio is out-of-distribution, so I lose
both the +0.54 (song-specific) and the +0.13 (stem-type generic),
ending up *below* chance because the projection head's
"default placement" of novel audio is adversarial to the held-out target.

The 0.54 of song-specific geometry memorisation is the largest single
contributor to the headline number, and it is **not** an attention-
decoding signal.

### 3.7 Computational cost: CLAP vs raw

From the LSO 36 training logs:

| model | trainable params | epoch-0 wall time | nominal full-fold training |
|---|---:|---:|---:|
| CLAP (`audio_repr=clap`) | ~273 K | ~60 s | ~40 min (≈50 epochs to early stop) |
| baseline (`audio_repr=raw`) | ~580 K | ~340 s (≈6 min) | ~4–5 hours at min_epochs=50 |

The roughly 5× difference per epoch comes from two factors:
1. The CLAP backbone (~30 M params, HTSAT-tiny) runs in `eval()` mode
   with no_grad: only forward, no backward, no optimizer state.
   Backward and optimization happen only on the EEG encoder + projection
   head (~273 K trainable). The baseline backpropagates through five
   networks (1 EEG + 4 audio encoders, ~580 K trainable in total).
2. Each of the four baseline CNN audio encoders is a 2D-Conv stack
   applied to the raw audio waveform padded to 3^11 = 177 147 samples
   per stem. With four stems per sample and a batch size of 8, this is
   significantly heavier per backward step than CLAP's projection-head-
   only backward.

This matters for the audit: replicating LSO on the baseline ablation is
budget-bounded by GPU availability. It is in §7 (Follow-ups) for that
reason.

---

## 4. What I can / cannot claim

### Can claim
1. Reproduction of the paper baseline (0.865). Verified with the paper
   authors' checkpoint loaded into our pipeline.
2. CLAP extension reaches **0.9237** within-subject — +5.9 points over
   the paper's headline, same protocol, same seed.
3. The pipeline is metrically sane: with i.i.d. random labels, accuracy
   is 0.2575 on the CLAP run and 0.2250 on the baseline, both
   indistinguishable from chance 0.25.
4. I measure a quantitative decomposition of the 0.9237 into three
   numerical components (chance 0.25 + stem-type prior 0.13 + song-
   specific memorisation 0.54).
5. The +0.13 stem-type prior is **shared between the CLAP and the raw
   baseline architectures** (0.3812 and 0.3825 audio_pair accuracy
   respectively), so it reflects a property of the dataset/protocol,
   not of CLAP specifically.
6. The CLAP model **fails to generalise to songs not seen in training**.
   LSO accuracy on the two CLAP folds is 0.019 and 0.000, well below
   chance, with a systematic error pattern (bass dominates the argmax).
7. **The paper's own baseline (raw encoder) collapses identically under
   LSO** (song 36: 0.000, with vocal dominating the argmax at 94 %).
   The cross-song failure is therefore a property of the dataset /
   paradigm, **not** of the CLAP extension.
8. The paper lists leave-song-out as future work (it makes no cross-song
   claim). I implement it on both architectures and find no cross-song
   generalisation in the two folds run so far — both drum-attended. This
   **closes the paper's stated gap** rather than contradicting it; a
   vocal-attended fold is still required before generalising the finding
   (see "Cannot claim" #1).

### Cannot claim
1. **Limited LSO coverage.** Only two LSO folds (songs 36 and 145, both
   drum-attended). I have no estimate of variance across songs and no
   coverage of vocal / bass / others target tasks.
2. **Limited LOSO coverage.** Only one fold (subject 0). The paper used
   three. Variance across subjects is not constrained.
3. **Limited baseline-LSO coverage.** I completed one baseline LSO fold
   (song 36, raw encoder, 0.000). A second baseline fold (song 145) and
   non-drum target songs have not been run; each baseline fold costs
   ~5 hours of GPU time (§3.7). The single completed fold already shows
   the same collapse as CLAP, but variance is not constrained.
4. **No partial-fine-tuning experiment.** Unfreezing the last CLAP block
   (or LoRA adapters) might restore generalisation; I have not tested
   this.
5. **Cohort size.** N=8 subjects is small; the paper itself calls its
   own results "preliminary evidence" for this reason.

---

## 5. Where every number in this document comes from

| number | run directory |
|---|---|
| 0.9237 (within-subject) | `results/clap_run1/nmed-CL-.../version_1/test_breakdown_summary.txt` |
| 0.9237 / 0.2575 / 0.3812 (CLAP sanity v2) | `results/clap_sanity_v2_{none,labels,audio_pair}/nmed-CL-.../version_0/test_breakdown_summary.txt` |
| 0.8650 / 0.2250 / 0.3825 (baseline sanity) | `results/baseline_sanity_{none,labels,audio_pair}/nmed-CL-.../version_0/test_breakdown_summary.txt` |
| 0.9363 (LOSO sub 0) | `results/cv1_leave_subject_out_0/nmed-CL-.../version_1/test_breakdown_summary.txt` |
| 0.0192 (CLAP LSO song 36) | `results/lso1_leave_song_out_36/nmed-CL-.../version_1/test_breakdown_summary.txt` |
| 0.0000 (CLAP LSO song 145) | `results/lso1_leave_song_out_145/nmed-CL-.../version_1/test_breakdown_summary.txt` |
| 0.0000 (baseline LSO song 36) | `results/lso_baseline_leave_song_out_36/nmed-CL-.../version_2/test_breakdown_summary.txt` |
| LSO sweep by task (0.904/0.462/0.288/0.087/0.000) | `results/lso_sweep_leave_song_out_{3,5,59,44,16}/nmed-CL-.../version_1/test_breakdown_summary.txt` |
| LOSO sweep (0.924/0.953/0.968/0.917/0.986) | `results/cv_sweep_leave_subject_out_{1,2,3,4,5}/nmed-CL-.../version_1/test_breakdown_summary.txt` |
| argmax / mean-sim tables | derived from `test_records.csv` in the same directories |

To inspect any of them, from the project root:

```bash
cat results/<run>/nmed-CL-preprocessing_eegmusic/version_*/test_breakdown_summary.txt
```

---

## 6. How to reproduce

### 6.1 Environment
Conda env `eeg_attention` on the training cluster (Python 3.9, PyTorch +
PyTorch Lightning, laion_clap, audiomentations, sklearn). Dataset at
`dataset/eeg_within_sub/{subject}/{train,valid,test}/{song}/{task}/{att}/eeg.pkl`
and `dataset/audio/{0..3}/{song}.wav`.

The LAION-CLAP checkpoint is downloaded into `~/.cache/laion_clap/` on
the first run (a few hundred MB).

### 6.2 Within-subject CLAP run
```bash
bash scripts/train_clap.sh
```
Output checkpoint: `results/clap_run1/.../version_*/checkpoints/best-checkpoint.ckpt`.
Time: ~30–60 min on one L40S.

### 6.3a Negative-control sanity sweep on the CLAP checkpoint
```bash
CKPT=results/clap_run1/nmed-CL-preprocessing_eegmusic/version_1/checkpoints/best-checkpoint.ckpt \
AUDIO_REPR=clap TAG=clap_sanity_v2 \
  bash scripts/test_sanity.sh
```
Time: ~30 min total. Produces `results/clap_sanity_v2_*/...`.

### 6.3b Negative-control sanity sweep on the paper baseline checkpoint
```bash
TAG=baseline_sanity bash scripts/test_sanity.sh
```
`CKPT` defaults to `checkpoints/model-all0.ckpt` (the paper authors'
"all-0 ms" checkpoint), and `AUDIO_REPR` defaults to `raw`, so no
overrides are needed. Time: ~5 min total. Produces
`results/baseline_sanity_*/...`.

### 6.4 LOSO fold
```bash
bash scripts/train_cv.sh leave_subject_out 0 cv1
```
Time: ~40 min training + 3 min test.

### 6.5 LSO fold
```bash
bash scripts/train_cv.sh leave_song_out 36 lso1
```

If the training completes but the test does not (e.g., session timeout),
the test step can be re-run on the already-saved checkpoint directly:

```bash
cd src
python -u checkpoint_test.py \
  --dataset preprocessing_eegmusic \
  --test_dataset preprocessing_eegmusic_test \
  --devices 1 --batch_size 8 \
  --eeg_length 768 --loss_function clip_loss \
  --eeg_normalization MetaAI --clamp_value 20 \
  --learning_rate 0.003 --supervised 1 \
  --dim_reduction 1 --shifting_time 0 \
  --split_seed 42 --detach_z_c 0 \
  --window_size 1280 --stride 256 \
  --test_window_size 768 --test_stride 256 \
  --seed 42 --start_position 0 \
  --attention_values 4 5 --key all \
  --audio_repr clap \
  --cv_mode leave_song_out --cv_held_out_id 145 \
  --test_breakdown 1 \
  --training_date lso1_leave_song_out_145 \
  --checkpoint_path ../results/lso1_leave_song_out_145/nmed-CL-preprocessing_eegmusic/version_0/checkpoints/best-checkpoint.ckpt
```

(The script must be run from inside `src/` because `checkpoint_test.py`
loads `../configs/baseline.yaml` with a relative path.)

---

## 7. Suggested follow-ups

| priority | follow-up | what it tests | est. GPU time |
|---|---|---|---|
| ~~high~~ DONE | LSO with raw encoder, song 36 | does the collapse hold for the baseline too? → **yes, 0.000** | ~4.8 h (completed) |
| high | LSO on a vocal-attended song (e.g. song 100), both architectures | does the collapse depend on the held-out task being drum? | ~40 min (CLAP) / ~5 h (raw) |
| high | LOSO with raw encoder on the paper's three subjects (3/7/2) | apples-to-apples cross-subject comparison vs the paper's 0.756 | ~15 h (3 × ~5 h, raw) |
| medium | Complete LOSO sweep (subjects 1–7) with CLAP | variance estimate across subjects | ~5 hours (7 × ~40 min) |
| medium | LSO sweep over 6–10 songs covering all task types | variance estimate across songs | ~5–7 hours of CLAP folds |
| lower | Partial fine-tuning of CLAP (LoRA or unfreeze last block) | does limited adaptation restore generalisation? | depends on adapter size |
| lower | Aggregate per-song accuracy in the within-subject run vs LSO accuracy on the same song | does the within-subject easy/hard pattern correlate with the LSO pattern? | analysis-only, no GPU |

---

## 8. Practical glossary (for the explanation to the advisor)

- **Within-subject test**: each subject's songs are randomly divided
  8:1:1; the model trains and tests on different songs *for that subject*.
  This is the paper's primary evaluation. Note that the *same* song may
  appear in different splits across different subjects.
- **Leave-one-subject-out (LOSO)**: hold all of one subject's trials out
  for test; train on the other seven. Cross-subject generalisation.
- **Leave-one-song-out (LSO)**: hold all trials of one song (across all
  subjects who heard it) out for test; train on the remaining 62 songs.
  Cross-song generalisation. This is the experiment the paper lists as
  future work and I implemented here.
- **`shuffle_test_mode=labels`**: replace each test label with a uniform
  random {0,1,2,3}. Accuracy must collapse to chance for the metric to
  be honest.
- **`shuffle_test_mode=audio_pair`**: for each batch, randomly permute
  the (vocal, drum, bass, others) tensors across the batch so each EEG
  is paired with the stems of a different song. Accuracy collapses to
  chance + whatever the model can still infer from generic stem-type
  acoustic similarity.
- **Margin**: for a given EEG window, `sim(EEG, target_stem) − max sim(EEG, non-target stem)`.
  Positive when the prediction is correct, the more positive the more
  separated.
- **Audio embedding memorisation**: the model learns the (fixed) positions
  in CLAP space of the 63 × 4 = 252 audio stem embeddings of the
  training set, rather than learning a generalisable EEG → instrument
  function. This is what I believe the bulk of the within-subject
  accuracy reflects, based on the LSO collapse.
