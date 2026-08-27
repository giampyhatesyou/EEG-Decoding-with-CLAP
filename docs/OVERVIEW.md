# Overview — what each arm and each experiment is asking

A reader's map of the science. For *where the code lives*, see
[`CODE_TOUR.md`](CODE_TOUR.md); for *the rules every number obeys*, see
[`METHOD_RULES.md`](METHOD_RULES.md); for *the numbers themselves with their
thresholds*, see [`../scripts/replicate/README.md`](../scripts/replicate/README.md).

> **Reading order.** The sections below are ordered by how central the claim is,
> which is the right order for someone who already knows the field. Coming to it
> cold, read them in dependency order instead:
> **§0 → §5 → §3 → §4 → §2 → §1 → §6 → §7 → §8 → §9.**
> That way the concrete method (§5) and the two datasets (§3, §4) arrive before
> the measuring instrument (§2), and both arrive before the abstract distinction
> (§1) that explains why the results look the way they do.

---

## 0. The question

**Auditory attention decoding (AAD)**: a listener hears several sound sources at
once and attends to one of them. From the EEG alone, can we say which one?

For speech this is established. For **polyphonic music** it is much harder, and
this project is about why. Two datasets, because one of them cannot answer the
question at all — and showing that is half the result.

## 1. The distinction everything else rests on

**No model in this project optimises the decision it is judged on.**

| model family | what the training objective actually minimises/maximises | how the decision is taken |
|---|---|---|
| backward model (ridge / shrinkage) | reconstruction error of the attended source's representation | **afterwards**: argmax of the correlation between the reconstruction and each present source |
| multi-view CCA | canonical correlation between EEG views and stimulus views | afterwards: argmax of rho over present sources |
| contrastive (CLAP↔EEG) | InfoNCE — cosine similarity to the correct audio against negatives | afterwards: argmax of similarity over present sources |
| band-power LDA (Exp. 11/12) | class separation — and this one **does** optimise a decision, but of the **register**, not of attention | the LDA output itself |
| alpha laterality index | nothing is fitted: a fixed statistic | sign of the difference within a twin pair |

Everything above the last two rows optimises a **similarity**, and the decision is
a *read-out* performed after the fact. That gap is not a technicality: it is the
mechanism the thesis reports. A change that makes the EEG track the audio better
improves the similarity **for both candidate sources at once**, so it can leave
the decision exactly where it was. That is measured, not argued — see §6.

## 2. Four tasks, four nulls. Confusing them is the fastest way to be wrong

| task | question | n | null | costs held-out looks? |
|---|---|---|---|---|
| **own-vs-other** | given the EEG of a **solo**, was it this audio excerpt or someone else's? | 376 (188 mutual pairs x 2 directions) | **0.500 exact by symmetry** | **no** — it runs on held-out solo segments |
| **duo attention decision** | given the EEG of a **duo**, which of the two instruments was attended? | 154 stereo, 150 mono | 0.500 — *measured*, because every mixture appears with **both** instruments as target | **yes** |
| **paired forced choice** | within a **twin pair** (same subject, same mixture, both targets), which member is which? | 47 stereo + 42 mono | **0.500 exact by exchangeability** | yes |
| **match-mismatch** | does this EEG window go with this audio? | 536 | 0.5000, measured | no — solos only |

**own-vs-other measures tracking, not selection.** It bounds the attention
decision from above without predicting it: if the EEG does not follow even an
isolated sound, it cannot choose between two overlapping ones. The converse does
**not** hold, and §6 is the measurement that shows it. The bridge is one-way.

## 3. Chapter 1 — the Akama arm: a benchmark that can be won without reading attention

**Dataset**: 4-channel consumer EEG (Muse), listeners attending one of four fixed
instrument slots — vocals / drums / bass / other.

**What is optimised**: an InfoNCE contrastive loss between a 3 s EEG window and the
four audio stems. The EEG encoder is a 2D-CNN; the audio branch is the variable.

**The design is a grid, not a sequence** — two audio branches x three splits:

| split | contrastive on **raw audio** | contrastive with **CLAP** | what the split removes |
|---|---:|---:|---|
| within (same songs in train and test) | 0.875 MACRO / 0.865 GLOBAL | **0.946 / 0.935** (8 seeds) | nothing |
| leave-subject-out | 0.781 (3 folds) | **0.943** (6 folds) | the listener |
| leave-song-out | **0.142** (20 folds) | **0.268** (20 folds) | the song |

Chance is 0.25. The raw branch is four independent trainable CNNs, one per slot;
the CLAP branch is **one frozen LAION-CLAP tower** plus a single trainable
projection head (116,120 + 157,028 = **273,148** trainable parameters, ~158 M
frozen).

**What each change was optimising, and what it bought**: swapping raw audio for
CLAP is a better *audio representation*. It buys +0.071 within and +0.162
leave-subject-out — and moves leave-song-out from *below* chance to *at* chance,
which is not an improvement in kind. Read by columns: both branches survive
removing the listener and both collapse when the song is removed. Removing a
whole subject costs CLAP 0.003; removing a song costs it 0.678.

**The control that explains it**: the **audio alone, with no EEG at all**, reaches
**0.996** within-split. In this dataset each song has one target instrument, so
recognising the song is enough. The 0.865 must always be quoted next to the
0.996 and the 0.268 — the *order* of those three numbers is the chapter's result.

Two more controls close the alternatives: EEG alone scores 0.250 MACRO (it
answers "vocals" to everything — the majority prior), and permuting the labels
drops the same pipeline to 0.227, so the evaluation itself is honest.

> **The claim**: a model can top this benchmark without ever reading attention.
> That is why a second dataset was needed.

## 4. Chapter 2 — MAD-EEG: the dataset built so the shortcut cannot work

20-channel research EEG, 8 subjects, duos and trios of real instruments.
The property that matters: **the same mixture is presented with both of its
instruments as the target** (18 mixtures out of 18). So "which stem is attended"
is *not* a property of the stimulus, the decision is internal to one trial, and
stimulus identity cannot win it.

Everything here is pre-registered: threshold, unit of analysis and test are
written down before the code exists. The threshold is always the smallest integer
with one-sided exact binomial p < 0.05 — one rule, chosen once, never touched.

## 5. Stimulus reconstruction — the "safe family", explained

This is the classical AAD method and the project's anchor.

**A backward model.** Train a linear decoder that maps the EEG (band-passed 1–8 Hz,
downsampled to 64 Hz, with lags 0–250 ms) onto a *representation of the attended
source* — its log-mel spectrogram by default. Ridge regression, with lambda picked
on a split internal to the training material.

**What is optimised**: reconstruction error on training trials. Nothing else.

**How the decision is taken**: at test time, reconstruct once, then correlate the
reconstruction against *each source actually present in that trial* and take the
argmax. Chance = 1/n_present = 0.500 on a duo.

**Why the representation choice is the crucial one**: instruments in a duo can
share a rhythm — the same amplitude envelope — and differ spectrally. A broadband
envelope target discards exactly the information that separates them; a log-mel
target keeps it.

**The three anchors**, all against a threshold of 88/154 = 0.5714:

| | trained on | target | result |
|---|---|---|---|
| A1 | k-fold on the duos | mel-8 | **86/154 = 0.5584**, p 0.0853 — the project's highest attention number, and **not significant** |
| A2 | k-fold on the duos | envelope | 78/154 = 0.5065 |
| A3 | raw solos (the paper's protocol) | mel-8 | 74/154 = 0.4805 — below chance |

## 6. The genealogy: what each change was trying to optimise, and what happened

Each row is one change, behind one flag, with the old default untouched.

| # | change | hypothesis / what it optimises | outcome |
|---|---|---|---|
| **axes** | notch filter | remove line noise | `inner_val_r` 0.0582 → 0.0582, exactly zero |
| | **notch + ICA** | remove EOG/ECG artefacts | **0.0582 → 0.0673 (+16 %)** — the only axis that gains. Verified by its own control: frontal–EOG coupling 0.552 → 0.006 |
| | per-instrument decoders | the paper's Fig. 1 protocol | `inner_val_r` 0.1100, but a **metric artefact**: each source is reconstructed by its *own* decoder. Fair test on the same 70 segments: 36/70, p 0.29 |
| | shrinkage estimator | the paper's normalised reverse correlation | 0.0349 — worse |
| | 24 mel bands @ 256 Hz | the **published** hyperparameters | 0.0468, and duo accuracy 0.4870. The published settings make it *worse*, and the two halves worsen independently |
| **Exp. 6 [G]** | ridge → **multi-view CCA** | de Cheveigné et al.: CCA is more sensitive to small effects, and ours is small | 209/376 vs 208/376. Paired McNemar 66/131, **p = 0.5000** → **NO-GO**. Not "CCA loses": "CCA adds nothing" |
| **Exp. 7** | 8 configurations of band / bands / rate / estimator | is the 208/376 ceiling a property of *this* configuration? | max 213/376, none passes Bonferroni α = 0.05/8. The ceiling holds |
| **Exp. 8** | mel → **spectral flux** target | onsets, not energy, are what cortex tracks | **243/376 = 0.6463, p 0.0015** — the project's **only pre-registered positive** |
| **Exp. 9** | the same flux, on the **attention** decision | does the tracking gain transfer? | **76/154 and 77/154** — it does not. Bar was 90/154 |
| **Exp. 13** | six audio-only representations | which one makes the two stems of a duo *least* alike? | **GO for MFCC-13**: within 0.0533 vs a cross-song floor of 0.0256, on 32/36 mixtures |
| **Exp. 14** | MFCC-13 and mel-64, on own-vs-other | does dropping energy (MFCC discards coefficient 0) cost tracking? | both track: 214/376 (p 0.004224) and 212/376 (p 0.007623) |
| **Exp. 15** | MFCC-13 on the duo decision | it won separability *and* tracking — does it win the task? | D(mel) +0.1001 vs D(MFCC) −0.1119; **numerically worse** (65/154 vs 86/154). Inconclusive by power (MDD 0.2467), with the estimate on the opposite side |
| **Exp. 16B** | add a `band_power` view to the CCA | a richer EEG feature extraction | 202/376 — bar not crossed. And the *cause* is instructive: mean rho(own)/rho(other) nearly doubles while accuracy drops, because an argmax between two scores is scale-invariant |
| **Exp. 16A/17** | CLAP as the reconstruction target | the thesis premise: a learned audio representation should separate cello from flute | **the most entangled row measured**: within/floor 3.837 against a ceiling of 1.80, worse than 8 mel bands (3.131). Measured cause: CLAP answers both stems of a duo with a coherent common mode |
| **step C** | linear → **contrastive**, batch negatives | learn the front end instead of fixing it | **58/154 = 0.3766** — below chance, CI excludes 0.50 |
| **step D** | negatives restricted to the **same mixture** | align the objective with the task, remove the shortcut | 70/154 = 0.4545, and the pre-declared secondary went the **wrong way**: prior-following 0.7177 → **0.8145** |
| **Exp. 18/19** | deny identity: match-mismatch on **solos**, negatives ≥ 2 s away in the same recording | with no identity available, is there any learnable EEG↔audio signal? | S1 278/536 = 0.5187 at 10 epochs, **244/536 = 0.4552 at 120**. Over those 120 epochs the train loss falls 0.6962 → 0.3160 while held-out accuracy moves +0.0010 |

### The thread through the table

Exp. 8 and Exp. 9 together are the thesis. Flux **wins tracking by a wide,
pre-registered margin and transfers nothing**. Exp. 13–15 make it worse: MFCC-13
wins audio separability *and* tracking, and is numerically worse on the decision.

**The cause is measured in the stimuli, not inferred from the EEG**: within a duo
the two competing stems correlate **0.287 under flux against 0.177 under mel**,
and flux is the more similar of the two in **30 of 36** mixtures
(`docs/provenance/2026-08-11_exp9_flux_attention_mcnemar.txt`, audio-only block —
no EEG is read to establish this).

On the EEG side the gain accrues to *both* candidates. On the duo k-fold protocol,
mel gives r(attended) 0.0202 against r(best unattended) 0.0118, and flux gives
**0.0560 against 0.0513**: the correlations nearly triple while the
attended-minus-unattended differential goes **+0.0084 → +0.0047**, i.e. flat or
slightly down.

> Provenance note: those four correlations are printed in the runs' own
> `madeeg_summary.txt` under `runs/results/{canary_duo,exp9_P2_duo_flux}/`, which
> is gitignored — so from a clean clone they are degree-2 provenance (the value a
> rerun must produce) rather than degree 1. `exp09_flux_attention.sh` regenerates
> them. The accuracies and the stem correlations above are degree 1.

> **Ensemble music synchronises across sources exactly the component that EEG
> tracks best.** Every gain in tracking goes to both candidates and none to the
> decision. Choosing a front end by how well it is tracked — which is how the
> field chooses — is not justified on this task.

## 7. The external hypothesis: is attention spatial or spectral?

Two independent, dated proposals were tested in full rather than dismissed.

**Lateralised alpha (Exp. 6 [H])** — alpha power lateralisation is a tonic
correlate of attended *side*. Tested as a sign test on twin pairs, where every
subject, channel, impedance and session constant cancels algebraically. Result:
**23/44 = 0.5227** against a threshold of 28/44, with **power 0.371** against the
motivating effect. The mono control came out *higher* than the primary (24/42),
which is the pre-written falsifier's condition.

**Spectral register (Exp. 11/12)** — band power 1–40 Hz plus LDA, asking which of
the two registers is attended. Stereo **24/47** (threshold 30/47), mono 22/42,
pooled **46/89** (threshold 53/89), **power 0.88** against an effect of 0.65.
An injection sweep locates the detection floor: the design recovers a **+18.8 %**
modulation (38/47) and is blind to **+9.2 %** (26/47).

> A negative without a floor is not a result. These two have both: what is
> reported is *no register-linked effect of roughly +19 % or larger exists in
> these data*, which is a bound, not an absence.

**Why the two forms do not meet**: lateralised alpha is a **tonic** correlate,
while the safe family decides by **within-trial correlation**, which centres any
trial constant out. This is also why the CCA `position` view is provably inert.

## 8. The audits: what the published record actually rests on

- **Arm A — protocol parity.** The complete published package (per-instrument
  shrinkage decoders, 24 mel @ 256 Hz, F1 on whole trials), over three bands:
  F1 **0.5267 / 0.4800 / 0.5400** at n = 150, with attended correlations
  0.016–0.025 against a published median of 0.119. The foundational 79 % **does
  not reproduce from its stated methods**. Stated that way — not "false".
- **Arm D — the leakage audit.** An LDA on band power decodes **pseudo-random
  labels with zero attentional content** at **0.6209** under window-level CV
  (null 0.5136) and **0.4847** under trial-level CV (null 0.500 exact). The rule
  for reading this was written in the code before the run, and it fired: a
  published 92.6 % at 1 s windows is **accounted for without invoking attention**.
- **Prior-following — why step C landed below chance.** For each test trial, is
  the model's choice predicted by the majority target of that mixture *in its own
  training folds*? Step C: **89/124 = 0.7177** against a computed null of
  **0.4274** (not 0.5 — k-fold removes the test trial, so the count tips towards
  the other instrument). Aligning the loss made it **worse**: 0.8145. The
  arithmetic is exact — from C to D, +12 trials gained, *all twelve* in the cell
  where the prior was already right, *zero* where it was wrong.

## 9. What holds, in one page

- Everything that decodes on MAD-EEG is **stimulus** — identity or following.
  The attentional selection signal, wherever it was probed, sits **below every
  detection floor that was measured**.
- The linear family's honest ceiling on an *easier* task (own-vs-other) is
  **~0.55** against a null of 0.500 exact. Published attention accuracies on this
  dataset sit **at** that ceiling, not below it. The limit measured is that of the
  narrow-band linear front end — not of auditory attention, and not of a 20-channel
  montage.
- **Tracking does not predict selection.** One pre-registered positive in the whole
  project (Exp. 8), and it transfers nothing.
- Given any route to stimulus identity, learned models take it. Denied identity by
  construction, they fit and do not transfer: 39.4 minutes of solo EEG and 273,148
  parameters is not enough data. That is a statement about **scale**, not about the
  optimiser or the architecture.
- One signal survives, and it is small and exploratory: Exp. 19 S2, **113/188 =
  0.6011** against a repaired null of 0.5000. *Which instrument* decodes a little;
  *which instant* does not. It is below the pre-registered gate and opens nothing
  on its own.

### The data budget, which is the real currency

All **309 duos** were spent by 30 July 2026: every duo number after that date is
**exploratory by construction**, however clean its pre-written threshold. The
**trios** — 92 stereo + 93 mono — were never opened. They are one holdout, one
shot, with one threshold already written (38/90 against a null of 1/3).
