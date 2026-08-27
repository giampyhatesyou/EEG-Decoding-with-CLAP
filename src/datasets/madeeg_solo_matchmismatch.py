# CHANGED(baseline): NEW FILE — not in the Akama et al. upstream. Match-mismatch sampling
#   over the MAD-EEG *solos*, for Exp. 18 (pre-registration of 12 Aug 2026, stages S1/S2).
"""Match-mismatch pairs from the MAD-EEG solo trials: one EEG window, two audio candidates.

Why a second dataset class instead of an option on `MadeegContrastiveDataset`: that one
reads `madeeg_preprocessed.hdf5`, and the preprocessed release contains **no solo trial at
all** (154 duo + 92 trio; the note is in `madeeg_reconstruction.py`'s header). The solos
exist only in the raw release, as segments of the continuous record cut at the onsets
`madeeg_sequences_raw.yaml` declares, paired with the wavs in `stimuli/`. Reading them is
a different I/O path, not a different experiment -- everything downstream (EEG encoder,
CLAP head, `CLIP_Loss`, the training loop) is the code step C and step D already ran, and
`madeeg_contrastive_dataset.py` is not touched by this file, so no reported number moves.

What one item is:

    eeg    (20, eeg_length)          one window of one solo repetition, 256 Hz
    stems  (2, 1, audio_samples)     row 0 = the audio that was playing during that window
                                     row 1 = the MISMATCHED candidate (see below)
    target_idx                       always 0 -- but slot position carries no parameters:
                                     both rows go through the same audio encoder and are
                                     compared to the EEG by cosine, so "answer slot 0" is
                                     not a strategy the model can learn.

The two mismatch modes, which are the two stages of the contract:

  negative="temporal_offset" (S1)
      The wrong candidate is the SAME wav file -- same subject, same piece, same
      instrument, same repetition, same acoustics -- at a different instant, at least
      `min_offset_s` away. Recognising the recording is therefore useless by construction:
      both candidates are that recording. This is the property the Chapter 1 model did not
      have (it won by identifying the song).

  negative="cross_instrument" (S2)
      The wrong candidate is the solo of a DIFFERENT INSTRUMENT playing the SAME piece and
      the SAME theme, over the SAME span of time. Melody, tempo, onsets and instant are all
      held constant; only the instrument changes. That is the "which instrument" contrast
      the contract calls the missing ingredient.

Two numbers are declared HERE, in the code, before any run (method rules 3/5):

  SOLO_WINDOW = 512 samples = 2.0 s. Not a taste: it is the longest window for which every
      solo excerpt still yields a non-overlapping pair. The shortest excerpts are the four
      `pop_falldead` solos at 4.89 s; at a 3.0 s window (the step C/D value) they yield
      ZERO valid pairs and 112 of the 420 repetitions would silently vanish from training.

  MIN_OFFSET_S = SOLO_WINDOW / EEG_FS = 2.0 s, i.e. exactly the window length, so the two
      candidate spans share NO audio sample. Anything smaller and the "negative" literally
      contains part of the positive.

And one property that is engineered rather than assumed, because it is what makes the null
exactly 0.500 for S1: the index enumerates ORDERED pairs. If (a, b) is a valid
(positive start, negative start) pair inside a repetition, so is (b, a), and both are in
the index exactly once. Any rule that looks only at where the two candidates sit in the
excerpt therefore scores exactly 0.500 -- there is no positional prior to ride. For S2 the
same symmetry does NOT hold (the positive instrument is fixed by the recording, and the
subjects did not all hear the same solos), so the null there is measured, not assumed:
`pairwise_prior_null` below computes the best achievable instrument-identity-only rule.

Runnable checks (CPU, no training, writes nothing):

    python src/datasets/madeeg_solo_matchmismatch.py --madeeg_dir ~/madeeg --check_sampling
    python src/datasets/madeeg_solo_matchmismatch.py --madeeg_dir ~/madeeg \\
        --negative cross_instrument --check_sampling --check_null_by_split
"""
import os
import sys

import h5py
import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from madeeg_reconstruction import (AUDIO_FS, EEG_FS, _load_by_file,  # noqa: E402
                                   raw_eeg_bandpassed, solo_instrument)

# By FILE, not through the `datasets` package: `datasets/__init__.py` pulls in the Chapter 1
# Akama dataset and with it torchaudio, which the CPU box does not have. The same trick
# `madeeg_reconstruction.py` uses for the same reason.
MadeegContrastiveDataset = _load_by_file(
    "_madeeg_contrastive_dataset", "datasets",
    "madeeg_contrastive_dataset.py").MadeegContrastiveDataset

SOLO_WINDOW = 512                       # 2.0 s @ 256 Hz -- see module docstring
SOLO_STRIDE = 256                       # 1.0 s
MIN_OFFSET_S = SOLO_WINDOW / EEG_FS     # 2.0 s: the candidate spans do not overlap
# The raw release is NOT the authors' cleaned release: step C and step D read
# `madeeg_preprocessed.hdf5`, which the authors had already notch-filtered and ICA-cleaned
# (WASPAA 2019 Sec. 2). None of that exists on the raw path, and 50 Hz line noise would be
# most of what the EEG encoder sees. A single wide band-pass -- the same (1, 45) Hz the
# `--alpha_li` arm already uses as its "wide pre-filter only" -- removes drift and the line
# component without imposing the 1-8 Hz analysis band, which is a Chapter 2 choice that has
# no business being baked into a learned representation.
SOLO_BAND = (1.0, 45.0)
NEGATIVE_MODES = ("temporal_offset", "cross_instrument")


def solo_group(key):
    """`classique_morceau1_solo_Co_theme2_mono` -> (('classique','morceau1','theme2','mono'), 'Co').

    The group is everything except the instrument: same genre, same piece, same theme, same
    render. Two solo keys in one group are the same music played by different instruments,
    which is the only cross-instrument negative that isolates the instrument."""
    parts = key.split("_")
    i = parts.index("solo")
    return tuple(parts[:i] + parts[i + 2:]), parts[i + 1]


class MadeegSoloMatchMismatch(Dataset):
    """MAD-EEG solo repetitions as match-mismatch pairs. See the module docstring."""

    def __init__(self, madeeg_dir, negative, eeg_length=SOLO_WINDOW, stride=SOLO_STRIDE,
                 min_offset_s=MIN_OFFSET_S, band=SOLO_BAND, subjects=None,
                 eeg_normalization="MetaAI", clamp_value=20, verbose=True):
        # Asserted, not silently defaulted: a mistyped mode that fell back to the other one
        # would produce a stage-S2 write-up from a stage-S1 run (method rule 8).
        assert negative in NEGATIVE_MODES, f"unknown negative mode {negative!r}"
        self.negative = negative
        self.eeg_length = eeg_length
        self.stride = stride
        self.min_offset = int(round(min_offset_s * EEG_FS))
        self.min_offset_s = min_offset_s
        self.band = band
        self.eeg_normalization = eeg_normalization
        self.clamp_value = clamp_value

        raw_h5 = os.path.join(madeeg_dir, "madeeg_raw.hdf5")
        seq_yaml = os.path.join(madeeg_dir, "madeeg_sequences_raw.yaml")
        raw_yaml = os.path.join(madeeg_dir, "madeeg_raw.yaml")
        pre_yaml = os.path.join(madeeg_dir, "madeeg_preprocessed.yaml")
        self.stim_dir = os.path.join(madeeg_dir, "stimuli")
        for p in (raw_h5, seq_yaml, raw_yaml, pre_yaml, self.stim_dir):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found -- the solo arm needs the RAW release plus stimuli/ "
                    "(the preprocessed release has no solo trial)")
        sq = yaml.load(open(seq_yaml), Loader=yaml.FullLoader)
        raw_info = yaml.load(open(raw_yaml), Loader=yaml.FullLoader)
        meta = yaml.load(open(pre_yaml), Loader=yaml.FullLoader)
        s0 = next(iter(meta))
        pre_chs = list(meta[s0][next(iter(meta[s0]))]["eeg_info"]["ch_names"])

        subjects = sorted(subjects or sq)
        all_solo_keys = sorted({k for s in sq for k in sq[s] if "solo" in k})
        self.groups = {}
        for k in all_solo_keys:
            g, instr = solo_group(k)
            self.groups.setdefault(g, {})[instr] = k

        self._audio = {}                    # wav filename -> float32 mono waveform
        self.trials = []                    # (subject, solo key) -- the SPLIT unit
        self.reps = []                      # one entry per presented repetition
        self.pairs = []                     # one entry per (eeg window, candidate pair)
        dropped = {"eeg_past_end": 0, "no_wav": 0, "no_sibling": 0, "too_short": 0}

        with h5py.File(raw_h5, "r") as raw_f:
            for subj in subjects:
                raw_chs = list(raw_info[subj]["ch_names"])
                eeg_full = raw_eeg_bandpassed(raw_f, raw_chs, pre_chs, subj, band)
                n_samples = eeg_full.shape[1]
                for key in sorted(k for k in sq[subj] if "solo" in k):
                    rec = len(self.trials)
                    n_before = len(self.reps)
                    n_ech = [int(o) for o in sq[subj][key].get("n_ech", [])]
                    wavs = list(sq[subj][key].get("wav_files", []))
                    for i in range(min(len(n_ech), len(wavs))):
                        audio = self._wav(wavs[i], dropped)
                        if audio is None:
                            continue
                        seg = int(round(len(audio) / AUDIO_FS * EEG_FS))
                        a = n_ech[i]
                        if a + seg > n_samples:
                            dropped["eeg_past_end"] += 1
                            continue
                        if seg < self.eeg_length:
                            dropped["too_short"] += 1
                            continue
                        self.reps.append({
                            "rec": rec, "subject": subj, "key": key, "rep": i,
                            "wav": wavs[i], "instrument": solo_instrument(key),
                            "eeg": eeg_full[:, a:a + seg].astype(np.float32),
                        })
                    if len(self.reps) > n_before:
                        self.trials.append((subj, key))
                    else:
                        continue
                    for r in range(n_before, len(self.reps)):
                        self._index_repetition(r, dropped)
                del eeg_full

        self.index = [(p["rec"], j) for j, p in enumerate(self.pairs)]
        if not self.pairs:
            raise ValueError(f"no {negative} pairs could be built from {madeeg_dir}")
        if verbose:
            self.describe(dropped)

    # --- construction ----------------------------------------------------------
    def _wav(self, name, dropped=None):
        """One stimulus wav, mono float32, cached. 24 distinct solo keys x 4 = 96 files."""
        if name not in self._audio:
            path = os.path.join(self.stim_dir, name)
            if not os.path.exists(path):
                if dropped is not None:
                    dropped["no_wav"] += 1
                return None
            x, sr = sf.read(path, dtype="float32", always_2d=False)
            assert sr == AUDIO_FS, f"{name}: {sr} Hz, expected {AUDIO_FS}"
            self._audio[name] = x if x.ndim == 1 else x.mean(axis=1)
        return self._audio[name]

    def _index_repetition(self, r, dropped):
        """Every (positive window, negative candidate) pair this repetition contributes."""
        rep = self.reps[r]
        n_eeg = rep["eeg"].shape[1]
        starts = list(range(0, n_eeg - self.eeg_length + 1, self.stride))
        if self.negative == "temporal_offset":
            # ORDERED pairs, both directions: this is what pins the null at 0.500 exactly.
            for a in starts:
                for b in starts:
                    if abs(a - b) >= self.min_offset:
                        self.pairs.append({"rec": rep["rec"], "pos_rep": r, "pos_start": a,
                                           "neg_rep": r, "neg_start": b,
                                           "neg_wav": rep["wav"],
                                           "neg_instrument": rep["instrument"]})
            return
        group, instr = solo_group(rep["key"])
        siblings = [k for i, k in sorted(self.groups[group].items()) if i != instr]
        if not siblings:
            dropped["no_sibling"] += 1
            return
        for sib in siblings:
            sib_wav = f"{sib}_{rep['rep'] + 1}.wav"
            sib_audio = self._wav(sib_wav, dropped)
            if sib_audio is None:
                continue
            for a in starts:
                # Same instant on both sides: only the instrument differs. A span the
                # sibling render is too short to cover is dropped and counted, never
                # padded -- silence is a trivial negative.
                a1 = self._audio_bounds(a)[1]
                if a1 > len(sib_audio):
                    dropped["too_short"] += 1
                    continue
                self.pairs.append({"rec": rep["rec"], "pos_rep": r, "pos_start": a,
                                   "neg_rep": r, "neg_start": a, "neg_wav": sib_wav,
                                   "neg_instrument": solo_instrument(sib)})

    def _audio_bounds(self, start):
        """Audio bounds of an EEG window, computed in SECONDS -- 256 and 44100 share no
        useful common factor, and scaling the sample index instead would de-align the pair
        by a few thousand samples (the same reasoning as MadeegContrastiveDataset)."""
        a0 = int(round(start / EEG_FS * AUDIO_FS))
        return a0, a0 + int(round(self.eeg_length / EEG_FS * AUDIO_FS))

    # --- torch Dataset ---------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, n):
        p = self.pairs[n]
        rep = self.reps[p["pos_rep"]]
        eeg = rep["eeg"][:, p["pos_start"]:p["pos_start"] + self.eeg_length]
        # The SAME normalisation step C and step D used, called on the class that owns it
        # rather than copied, so the two arms cannot drift apart: per-channel RobustScaler
        # then clamp. It only reads `self.eeg_normalization` and `self.clamp_value`.
        eeg = MadeegContrastiveDataset._normalize(self, torch.from_numpy(eeg.copy()))

        p0, p1 = self._audio_bounds(p["pos_start"])
        n0, n1 = self._audio_bounds(p["neg_start"])
        pos = self._audio[rep["wav"]][p0:p1]
        neg = self._audio[p["neg_wav"]][n0:n1]
        stems = torch.from_numpy(np.stack([pos, neg])).unsqueeze(1)   # (2, 1, samples)

        return {
            "eeg": eeg,
            "stems": stems,
            "target_idx": 0,
            "n_present": 2,
            "subject": rep["subject"],
            "stim": rep["key"],
            "ensemble": "solo",
            "rep": rep["rep"],
            "pos_wav": rep["wav"],
            "neg_wav": p["neg_wav"],
            "pos_instrument": rep["instrument"],
            "neg_instrument": p["neg_instrument"],
            "pos_start_s": p["pos_start"] / EEG_FS,
            "neg_start_s": p["neg_start"] / EEG_FS,
        }

    # --- reporting -------------------------------------------------------------
    def describe(self, dropped=None):
        import collections
        eeg_s = sum(r["eeg"].shape[1] for r in self.reps) / EEG_FS
        uniq = {r["wav"] for r in self.reps}
        print(f"[solo-matchmismatch] negative={self.negative}  recordings={len(self.trials)}  "
              f"repetitions={len(self.reps)}  pairs={len(self.pairs)}")
        print(f"  window={self.eeg_length / EEG_FS:.1f}s stride={self.stride / EEG_FS:.1f}s "
              f"min_offset={self.min_offset_s:.1f}s band={self.band} Hz")
        print(f"  EEG behind it: {eeg_s / 60:.1f} min over {len(self.reps)} repetitions; "
              f"{len(uniq)} distinct stimulus wavs ({len({solo_group(r['key'])[0] for r in self.reps})} "
              f"piece/theme groups)")
        per_subj = collections.Counter(s for s, _ in self.trials)
        print(f"  recordings per subject: {dict(sorted(per_subj.items()))}")
        if dropped and any(dropped.values()):
            print(f"  dropped (counted, never padded): {dropped}")

    def _null_key(self, j, by_subject):
        """(group the rule may condition on, which side is the positive) for pair `j`."""
        p = self.pairs[j]
        rep = self.reps[p["pos_rep"]]
        if self.negative == "temporal_offset":
            a, b = p["pos_start"], p["neg_start"]
        else:
            a, b = rep["instrument"], p["neg_instrument"]
        group = (tuple(sorted((a, b))),)
        if by_subject:
            group = (rep["subject"],) + group
        return group, a

    def pairwise_prior_null(self, item_ids=None, by_subject=False):
        """The best accuracy reachable WITHOUT the EEG, by candidate identity alone.

        For each unordered candidate pair the best identity-only rule is "always answer the
        one that is more often the positive"; the decision is per pair, so no global
        ordering has to be searched and this bound is exact. For `temporal_offset` the
        identity is the start time, and the ordered-pair index makes every pair appear in
        both directions -> the bound comes out at exactly 0.500, which is the check that
        the construction did what it claims. For `cross_instrument` the identity is the
        instrument, and this is the honest null to print next to the accuracy: the subjects
        did not all hear the same solos, so it is NOT 0.500 a priori (method rule 4).

        CHANGED(baseline): Exp. 19 §2.1 adds `by_subject`. The subject is readable off the
        EEG for free -- different anatomy, different impedances -- so a rule of the form
        "for THIS subject, on this candidate pair, always answer i" needs no auditory
        decoding at all, and it is a strictly larger family than the one Exp. 18 measured.
        Measured on the Exp. 18 held-out split it reaches 0.8898, so it is not hypothetical;
        `balanced_indices` below is keyed on it for that reason, and both numbers get
        printed side by side rather than the smaller one alone.
        """
        import collections
        ids = range(len(self.pairs)) if item_ids is None else item_ids
        wins = collections.Counter()
        for j in ids:
            wins[self._null_key(j, by_subject)] += 1
        best = total = 0
        for group in {k[0] for k in wins}:
            counts = [wins[(group, side)] for side in group[-1]]
            best += max(counts)
            total += sum(counts)
        return best / total, total

    def balanced_indices(self, item_ids=None):
        """The largest subset of `item_ids` on which `pairwise_prior_null` is exactly 0.500.

        Exp. 19 §2.1, and it is a REPAIR OF THE NULL, not a hyperparameter: on the Exp. 18
        split the best EEG-free rule scored 0.6061 on the held-out pairs of S2, so a gate at
        0.70 was worth +0.094 instead of the +0.20 it was written to be. Every candidate
        pair here is kept in BOTH directions the same number of times, so answering by
        candidate identity -- or by subject and candidate identity -- scores exactly chance
        by construction, the same way S1's ordered-pair index already did.

        Two properties this deliberately has:

        * The key includes the SUBJECT (see `pairwise_prior_null`). Balancing on the
          instrument pair alone leaves the subject-conditional rule at 0.8884 on the Exp. 18
          held-out split -- i.e. "recognise whose EEG this is, then answer that subject's
          usual instrument", which is the same class of shortcut ("recognise the trial")
          that Chapter 1 won by and that the whole match-mismatch design exists to close.
        * It is deterministic and takes the surplus side evenly across the block rather than
          its first n items, so the survivors still span every recording rather than
          clustering on whichever one the index happens to list first.

        Applied to `temporal_offset` it is a provable no-op -- every (a, b) start pair is in
        the index together with (b, a), within the same repetition and so the same subject,
        so no group has a surplus side. Verified on the real index: 3346 -> 3346, 2810 ->
        2810, 536 -> 536, the returned list identical element by element. That is what keeps
        the S1 sampler of Exp. 18 bit-for-bit reproducible under this change.
        """
        import collections
        ids = sorted(range(len(self.pairs)) if item_ids is None else item_ids)
        sides = collections.defaultdict(list)
        for j in ids:
            group, positive = self._null_key(j, by_subject=True)
            sides[(group, positive)].append(j)
        keep = []
        for group in sorted({g for g, _ in sides}):
            # A missing side gives [] -> n = 0 -> the whole group is dropped, which is the
            # only honest option: there is no EEG for the direction that does not exist, so
            # it cannot be mirrored, only removed.
            blocks = [sides[(group, side)] for side in group[-1]]
            n = min(len(b) for b in blocks)
            for b in blocks:
                keep += [b[int(i * len(b) / n)] for i in range(n)]
        keep = sorted(keep)
        # The postcondition, checked rather than argued: if this does not come out at 0.500
        # the repair did not happen and every number downstream would be read against the
        # wrong chance level (method rule 4).
        if not keep:
            raise ValueError("balancing removed every pair -- no candidate pair of this "
                             "index exists in both directions")
        for by_subject in (False, True):
            got, _ = self.pairwise_prior_null(keep, by_subject=by_subject)
            assert abs(got - 0.5) < 1e-12, (
                f"balanced null is {got:.6f}, not 0.5 (by_subject={by_subject})")
        return keep


# --- checks -------------------------------------------------------------------------
def check_sampling(ds, n_content=200):
    """THE check the contract makes mandatory: where does the negative actually come from?

    Structural, over EVERY pair (method rule 7 -- count, do not generalise from one):
      1. `temporal_offset`: the negative wav is byte-for-byte the SAME FILE as the positive
         (same subject, same piece, same instrument, same repetition);
         `cross_instrument`: same piece and theme, DIFFERENT instrument, same instant.
      2. the offset is >= the declared MIN_OFFSET_S;
      3. the offset is never 0 -- the same instant is never a candidate against itself.
    Content, on a deterministic subsample: the audio the dataset actually hands back is the
    span it declares, and the two candidates are not the same waveform.
    """
    same_file = diff_instrument = 0
    min_off = float("inf")
    for p in ds.pairs:
        rep = ds.reps[p["pos_rep"]]
        off = abs(p["pos_start"] - p["neg_start"]) / EEG_FS
        if ds.negative == "temporal_offset":
            assert p["neg_wav"] == rep["wav"], (
                f"negative from another recording: {p['neg_wav']} vs {rep['wav']}")
            assert off >= ds.min_offset_s - 1e-9, f"offset {off:.3f}s < {ds.min_offset_s}s"
            assert p["pos_start"] != p["neg_start"], "negative at the SAME instant"
            same_file += 1
            min_off = min(min_off, off)
        else:
            g_pos, i_pos = solo_group(rep["key"])
            g_neg, i_neg = solo_group(p["neg_wav"].rsplit("_", 1)[0])
            assert g_pos == g_neg, f"negative from another piece/theme: {g_neg} vs {g_pos}"
            assert i_neg != i_pos, f"negative from the SAME instrument {i_pos}"
            assert p["pos_start"] == p["neg_start"], "cross-instrument must hold the instant"
            diff_instrument += 1
            min_off = 0.0

    step = max(1, len(ds) // n_content)
    checked = 0
    for j in range(0, len(ds), step):
        it = ds[j]
        p = ds.pairs[j]
        pos, neg = it["stems"][0, 0].numpy(), it["stems"][1, 0].numpy()
        n0, n1 = ds._audio_bounds(p["neg_start"])
        assert np.array_equal(neg, ds._audio[p["neg_wav"]][n0:n1]), \
            "the returned negative audio is not the span the sampler declares"
        assert not np.array_equal(pos, neg), "the two candidates are the same waveform"
        assert pos.shape == neg.shape == (int(round(ds.eeg_length / EEG_FS * AUDIO_FS)),)
        checked += 1

    null, n = ds.pairwise_prior_null()
    null_s, _ = ds.pairwise_prior_null(by_subject=True)
    print(f"[check_sampling] PASS  {len(ds.pairs)} pairs, mode {ds.negative}")
    if ds.negative == "temporal_offset":
        print(f"  negative from the SAME wav file: {same_file}/{len(ds.pairs)}   "
              f"smallest offset seen: {min_off:.3f}s (declared floor {ds.min_offset_s:.1f}s)   "
              f"offset 0 (same instant): 0/{len(ds.pairs)}")
    else:
        print(f"  negative from the same piece+theme and a DIFFERENT instrument: "
              f"{diff_instrument}/{len(ds.pairs)}   same instant on both sides: "
              f"{len(ds.pairs)}/{len(ds.pairs)}")
    print(f"  audio content verified against the declared span on {checked} items "
          f"(every {step}th)")
    print(f"  null WITHOUT the EEG (best candidate-identity-only rule): {null:.4f} on n={n}")
    print(f"  null WITHOUT the EEG, subject-conditional rule: {null_s:.4f} on n={n}")
    return {"pairs": len(ds.pairs), "checked": checked, "null": null, "null_subject": null_s,
            "min_offset_s": None if min_off == float("inf") else min_off}


def check_null_by_split(ds, valid_frac=0.2, seed=42, ceiling=0.52):
    """Exp. 19 gate 3: the null RE-MEASURED per split, raw and balanced, before any training.

    The number that matters is the HELD-OUT one -- that is the set the gate reads -- and it
    is not implied by the whole-set null: Exp. 18 measured 0.5063 over all of S2's pairs and
    0.6061 on the 21 held-out recordings of the very same index. So the split is rebuilt
    here exactly as the run rebuilds it, and both rule families are printed for each side.
    """
    from madeeg_contrastive import split_by_trial      # lazy: it imports torch and models

    train_ds, valid_ds, n_valid = split_by_trial(ds, valid_frac=valid_frac, seed=seed)
    print(f"\n[null-by-split] valid_frac={valid_frac} seed={seed}: "
          f"{n_valid} of {len(ds.trials)} recordings held out")
    print(f"{'split':>10}{'n':>8}{'null(pair)':>13}{'null(subj+pair)':>18}  balancing")
    out = {}
    for name, ids in (("whole", list(range(len(ds)))), ("train", list(train_ds.indices)),
                      ("held-out", list(valid_ds.indices))):
        for tag, sub in (("raw", ids), ("balanced", ds.balanced_indices(ids))):
            a, n = ds.pairwise_prior_null(sub)
            b, _ = ds.pairwise_prior_null(sub, by_subject=True)
            print(f"{name:>10}{n:>8}{a:>13.4f}{b:>18.4f}  {tag}"
                  + ("  <-- the gate reads this line" if name == "held-out"
                     and tag == "balanced" else ""))
            out[(name, tag)] = (a, b, n)
    held = out[("held-out", "balanced")]
    print(f"  contract §2.1 ceiling {ceiling}: balanced held-out null {held[0]:.4f} -> "
          + ("OK, the 0.70 gate means what it was written to mean"
             if held[0] <= ceiling else
             f"STILL ABOVE -- the gate becomes 'beat {held[0]:.4f}, exact one-sided "
             "binomial p <= 0.05'"))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MAD-EEG solo match-mismatch sampler -- checks only")
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--negative", default="temporal_offset", choices=list(NEGATIVE_MODES))
    ap.add_argument("--check_sampling", action="store_true")
    ap.add_argument("--check_null_by_split", action="store_true",
                    help="Exp. 19 gate 3: re-measure the null per split, raw and balanced, "
                         "before any training")
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    ds = MadeegSoloMatchMismatch(args.madeeg_dir, args.negative, subjects=args.subjects)
    if args.check_sampling:
        check_sampling(ds)
    if args.check_null_by_split:
        check_null_by_split(ds, valid_frac=args.valid_frac, seed=args.seed)
