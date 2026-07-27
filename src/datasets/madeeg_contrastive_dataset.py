# CHANGED(baseline): NEW FILE — not in the Akama et al. upstream. Feeds MAD-EEG to the
#   contrastive pipeline, the way madeeg_reconstruction.py feeds it to the ridge anchor.
"""MAD-EEG as a contrastive EEG<->audio dataset (step A: data only, no training).

Why this exists: MAD-EEG is the *positive* arm. The same mixture is attended with
different target instruments across trials (all 18 duo mixtures appear twice, once per
target), so "which instrument is attended" is not aliased with "which stimulus" — the
confound that sinks the Akama arm. See docs/MADEEG.md.

What one item is:

    eeg    (n_channels, eeg_length)          20 ch @ 256 Hz, one sliding window
    stems  (n_present, 1, audio_samples)     the isolated sources of THAT mixture,
                                             44.1 kHz mono, the same span of time
    target_idx                               which row of `stems` was attended
    n_present                                2 for a duo, 3 for a trio

Deliberately NOT four fixed slots. The Akama pipeline carries exactly four stems with
fixed instrument roles (vocal/drum/bass/others); MAD-EEG has two or three present out of
nine instruments. Mapping variable arity onto the four-slot loss is a methodological
change and belongs to a later step -- this file only makes the data available and honest
about its shape. In particular it never pads an absent stem with silence: silence is a
trivial negative to beat and would inflate accuracy without any attention decoding.

Smoke test (CPU, no training, writes nothing):

    python src/datasets/madeeg_contrastive_dataset.py --madeeg_dir ~/madeeg
"""
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

# read_f64 lives in the ridge script: MAD-EEG stores arrays in a float layout h5py cannot
# auto-map, and that helper is the fix. Same reader for both arms, by construction.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from madeeg_reconstruction import EEG_FS, read_f64  # noqa: E402

AUDIO_FS = 44100


class MadeegContrastiveDataset(Dataset):
    """One item = one sliding window of one trial, with that trial's isolated sources.

    Windows are cut from EEG and audio over the *same* span of wall-clock time; the two
    rates (256 Hz / 44100 Hz) are exact multiples of nothing in particular, so the audio
    bounds are computed from seconds rather than from a sample ratio.
    """

    def __init__(self, madeeg_dir, ensemble="duo", subjects=None,
                 eeg_length=768, stride=256,
                 eeg_normalization="MetaAI", clamp_value=20, preload=False,
                 synthetic_eeg=False, synth_snr=4.0, seed=42):
        self.h5_path = os.path.join(madeeg_dir, "madeeg_preprocessed.hdf5")
        meta_path = os.path.join(madeeg_dir, "madeeg_preprocessed.yaml")
        for p in (self.h5_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found -- run scripts/madeeg_setup.sh")
        self.meta = yaml.load(open(meta_path), Loader=yaml.FullLoader)

        self.eeg_length = eeg_length
        self.stride = stride
        self.eeg_normalization = eeg_normalization
        self.clamp_value = clamp_value
        self._h5 = None                     # opened lazily: HDF5 handles do not fork
        self._cache = {}                    # (subj, stim) -> whole-trial arrays; see _trial_arrays
        self._evict = True                  # flipped off by preload

        # Positive control. With synthetic_eeg the recorded EEG is replaced by a signal that
        # tracks the ATTENDED source (see _synth_eeg); everything downstream is untouched.
        self.synthetic_eeg = synthetic_eeg
        self.synth_snr = synth_snr
        self._synth_seed = seed
        self._synth_op = None               # one forward operator, shared by every trial

        keep = {"duo", "trio"} if ensemble == "both" else {ensemble}
        subjects = subjects or list(self.meta)
        self.trials = [(s, k) for s in subjects for k in self.meta[s]
                       if self.meta[s][k].get("ensemble") in keep]
        if not self.trials:
            raise ValueError(f"no {ensemble} trials for subjects {subjects}")

        # MAD-EEG trials are NOT all the same length: measured 19.7 / 20.0 / 24.0 / 28.0 s
        # across the 246 preprocessed trials. Indexing per trial rather than assuming a
        # uniform window count keeps the 16% of EEG that cropping everything to the
        # shortest trial would throw away.
        with h5py.File(self.h5_path, "r") as f:
            self.trial_samples = [f[s][k]["response"].shape[1] for s, k in self.trials]
        self.index = [(t, w)
                      for t, n in enumerate(self.trial_samples)
                      for w in range(max(1, (n - self.eeg_length) // self.stride + 1))]

        self.instruments = sorted({i for s, k in self.trials
                                   for i in self.meta[s][k]["instruments"]})

        # `preload` exists so the loader can SHUFFLE. Without it the two-trial cache forces
        # sequential access, and with batch_size 8 against ~26 windows per trial almost every
        # batch would be drawn from a single trial -- which quietly breaks InfoNCE, because
        # the "other samples" that serve as batch negatives would then be the SAME audio as
        # the positive. Held as float32 and trimmed to the sources actually present, the whole
        # duo set is ~1.6 GB, so it simply lives in RAM.
        if preload:
            for subj, stim in self.trials:
                self._load(subj, stim)
            self._evict = False

    # --- torch Dataset ---------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        trial_idx, window_idx = self.index[n]
        subj, stim = self.trials[trial_idx]
        m = self.meta[subj][stim]

        response, soli_full = self._trial_arrays(subj, stim)

        start = window_idx * self.stride
        eeg = response[:, start:start + self.eeg_length]
        eeg = self._normalize(torch.from_numpy(eeg.astype(np.float32)))

        # Same span of time on the audio side. Computed in seconds, not by scaling the
        # sample index: the two rates share no useful common factor and an off-by-a-few
        # thousand samples here would silently de-align EEG from the audio it is paired
        # with -- which is exactly the kind of bug that produces a plausible-looking
        # chance result nobody can explain.
        t0 = start / EEG_FS
        dur = self.eeg_length / EEG_FS
        a0 = int(round(t0 * AUDIO_FS))
        a1 = a0 + int(round(dur * AUDIO_FS))
        soli = soli_full[:, a0:a1]

        instruments = list(m["instruments"])
        n_present = len(instruments)
        stems = torch.from_numpy(soli).unsqueeze(1)   # (n_present, 1, samples); the empty
                                                      # third `soli` row of a duo is dropped
                                                      # at load time, not here

        return {
            "eeg": eeg,
            "stems": stems,
            "target_idx": instruments.index(m["target"]),
            "n_present": n_present,
            "instruments": instruments,
            "subject": subj,
            "stim": stim,
            "ensemble": m["ensemble"],
        }

    def _load(self, subj, stim):
        """Read one trial into the cache, as float32 and trimmed to the present sources.

        `read_f64` reads a whole HDF5 dataset -- it selects H5S_ALL, the price of the type
        conversion MAD-EEG's float layout needs -- so re-reading per window would mean
        ~175 GB of I/O per epoch. Storing float32 and dropping the empty third `soli` row
        of a duo halves it again: ~10 MB per trial, ~1.6 GB for all 154.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        grp = self._h5[subj][stim]
        m = self.meta[subj][stim]
        n_present = len(m["instruments"])
        response = read_f64(grp["response"]).astype(np.float32)
        soli = read_f64(grp["soli"])[:n_present].astype(np.float32)
        if self.synthetic_eeg:
            target_row = list(m["instruments"]).index(m["target"])
            response = self._synth_eeg(soli[target_row], response.shape,
                                       self.trials.index((subj, stim)))
        self._cache[(subj, stim)] = (response, soli)

    def _synth_eeg(self, src, shape, trial_no):
        """EEG that tracks the ATTENDED source: the positive control, as in the ridge arm.

        One fixed linear forward operator -- a per-channel weight and a causal lag of 0-100 ms
        -- applied to the attended source's envelope, plus per-trial noise. The operator is
        the same for every trial, so nothing about *which* source is attended is encoded in
        the operator itself; the only route from EEG to the answer is the audio content.

        It answers one question: if the EEG did follow the attended stem, would this loss,
        this decision rule and this k-fold recover it? A PASS says the wiring is sound and
        says nothing at all about real EEG -- an envelope at snr 4 is incomparably easier.
        """
        n_ch, n_eeg = shape
        # Envelope of the source, binned straight down to the EEG sample rate. Bin edges are
        # spread over the whole source, so envelope sample j covers the same span of time as
        # EEG sample j -- the same seconds-based alignment __getitem__ uses for the windows.
        edges = np.linspace(0, src.shape[0], n_eeg + 1).astype(np.int64)
        cumulative = np.concatenate([[0.0], np.cumsum(np.abs(src).astype(np.float64))])
        env = ((cumulative[edges[1:]] - cumulative[edges[:-1]])
               / np.maximum(edges[1:] - edges[:-1], 1))
        env = (env - env.mean()) / (env.std() + 1e-8)

        if self._synth_op is None:
            op = np.random.RandomState(self._synth_seed)
            self._synth_op = (op.standard_normal(n_ch), op.randint(0, 26, n_ch))
        weights, lags = self._synth_op
        noise = np.random.RandomState(self._synth_seed + 1 + trial_no)
        out = np.empty((n_ch, n_eeg), dtype=np.float32)
        for ch in range(n_ch):
            lag = int(lags[ch])
            shifted = np.roll(env, lag)
            shifted[:lag] = 0.0                       # causal: the EEG lags the stimulus
            out[ch] = weights[ch] * shifted + noise.standard_normal(n_eeg) / self.synth_snr
        return out

    def _trial_arrays(self, subj, stim):
        """Whole-trial arrays, cached. With preload=False only two trials are kept, which
        requires sequential access; see the note in __init__ on why training needs preload."""
        key = (subj, stim)
        if key not in self._cache:
            if self._evict and len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._load(subj, stim)
        return self._cache[key]

    def _normalize(self, eeg):
        """Per-channel RobustScaler + clamp -- the `MetaAI` scheme the Akama runs use."""
        if self.eeg_normalization != "MetaAI":
            return eeg
        for idx, ch in enumerate(eeg):
            scaled = RobustScaler().fit_transform(ch.reshape(-1, 1))
            eeg[idx] = torch.from_numpy(scaled.astype(np.float32)).view(-1)
        return torch.clamp(eeg, min=-self.clamp_value, max=self.clamp_value)


# --- smoke -------------------------------------------------------------------------
def smoke(madeeg_dir, ensemble="both", n_items=3):
    """Print what the adapter yields. No training, no claim, nothing written."""
    import collections

    ds = MadeegContrastiveDataset(madeeg_dir, ensemble=ensemble)
    total_s = sum(ds.trial_samples) / EEG_FS
    print(f"[madeeg-contrastive] ensemble={ensemble}  trials={len(ds.trials)}  "
          f"windows={len(ds)}  ({total_s / 60:.1f} min of EEG)")
    lens = collections.Counter(round(n / EEG_FS, 1) for n in ds.trial_samples)
    print(f"  trial lengths (s): {dict(sorted(lens.items()))}  "
          f"-> windows/trial {min(w for _, w in ds.index) + 1}..{max(w for _, w in ds.index) + 1}")
    print(f"  instruments ({len(ds.instruments)}): {' '.join(ds.instruments)}")

    per_subj = collections.Counter(s for s, _ in ds.trials)
    per_ens = collections.Counter(ds.meta[s][k]["ensemble"] for s, k in ds.trials)
    per_target = collections.Counter(ds.meta[s][k]["target"] for s, k in ds.trials)
    print(f"  trials per subject: {dict(sorted(per_subj.items()))}")
    print(f"  trials per ensemble: {dict(per_ens)}")
    print(f"  attended instrument: {dict(per_target.most_common())}")

    # The premise of this whole arm: the same mixture attended with different targets. A
    # mixture presented with only ONE target is aliased -- for those trials stimulus
    # identity predicts the answer, which is precisely the Akama confound in miniature.
    mixtures, mix_ens = collections.defaultdict(set), {}
    for s, k in ds.trials:
        mix = k.rsplit("_", 1)[0]
        mixtures[mix].add(ds.meta[s][k]["target"])
        mix_ens[mix] = ds.meta[s][k]["ensemble"]
    by_ens = collections.defaultdict(collections.Counter)
    for mix, targets in mixtures.items():
        by_ens[mix_ens[mix]][len(targets)] += 1
    for e in sorted(by_ens):
        print(f"  {e} mixtures: {sum(by_ens[e].values())}, targets per mixture "
              f"{dict(sorted(by_ens[e].items()))}")
    aliased = sorted(m for m, t in mixtures.items() if len(t) == 1)
    n_trials_aliased = sum(1 for s, k in ds.trials if k.rsplit("_", 1)[0] in aliased)
    if aliased:
        print(f"  ALIASED -- one target only, so stimulus identity predicts it "
              f"({n_trials_aliased} trials): {', '.join(aliased)}")
    else:
        print("  every mixture appears with >1 attended target -> attention is identifiable")

    print()
    for i in (0, len(ds) // 2, len(ds) - 1)[:n_items]:
        it = ds[i]
        print(f"  item {i}: eeg={tuple(it['eeg'].shape)} "
              f"[{it['eeg'].min():+.2f},{it['eeg'].max():+.2f}]  "
              f"stems={tuple(it['stems'].shape)}  n_present={it['n_present']}  "
              f"target={it['instruments'][it['target_idx']]} of {it['instruments']}  "
              f"subj={it['subject']} {it['ensemble']}")
    return ds


def _check_soli_row_order(ds):
    """Row i of `soli` must be instrument i of `instruments` -- checked on the audio itself.

    Within one piece and theme the three duo mixtures are mixed from the same three rendered
    solo tracks (CoFl, CoOb, FlOb out of Co, Fl, Ob). So the track two of those mixtures
    share has to sit in the row `instruments` names, in BOTH of them -- a specific
    permutation, not merely "some rows are equal".

    Worth its own check because a transposition here is invisible everywhere else: training
    and evaluation index `stems` the same way, so a swapped pair would keep scoring exactly
    as it does now while the model was in fact being trained on the UNATTENDED source.
    """
    import itertools

    representative = {}
    for subj, stim in ds.trials:
        representative.setdefault(stim.rsplit("_", 1)[0], (subj, stim))
    groups = {}
    for mix in representative:
        piece, rest = mix.split("_duo_")
        groups.setdefault((piece, rest.split("_")[1]), []).append(mix)

    def corr(u, v):
        du, dv = u - u.mean(), v - v.mean()
        den = float(np.sqrt((du * du).sum() * (dv * dv).sum()))
        return float((du * dv).sum() / den) if den > 1e-12 else 0.0

    checked = 0
    for mixes in groups.values():
        loaded = {m: (list(ds.meta[representative[m][0]][representative[m][1]]["instruments"]),
                      ds._trial_arrays(*representative[m])[1]) for m in mixes}
        for m1, m2 in itertools.combinations(sorted(mixes), 2):
            instr1, soli1 = loaded[m1]
            instr2, soli2 = loaded[m2]
            n = min(soli1.shape[1], soli2.shape[1])
            for inst in sorted(set(instr1) & set(instr2)):
                r1, r2 = instr1.index(inst), instr2.index(inst)
                r = corr(soli1[r1, :n], soli2[r2, :n])
                assert r > 0.99, (
                    f"{inst}: {m1} row {r1} vs {m2} row {r2} r={r:.4f} -- the rows of `soli` "
                    "do not follow `instruments`, so target_idx points at the wrong stem")
                checked += 1
    return checked


def _self_check(madeeg_dir):
    """One runnable check of the part that can silently be wrong: EEG/audio alignment."""
    ds = MadeegContrastiveDataset(madeeg_dir, ensemble="duo")
    it0, it1 = ds[0], ds[1]
    eeg_dur = it0["eeg"].shape[1] / EEG_FS
    audio_dur = it0["stems"].shape[2] / AUDIO_FS
    assert abs(eeg_dur - audio_dur) < 1e-3, f"span mismatch: eeg {eeg_dur}s vs audio {audio_dur}s"
    # Consecutive windows must advance by `stride` on BOTH sides, or the pairing drifts.
    assert not torch.equal(it0["eeg"], it1["eeg"]), "consecutive windows are identical"
    assert not torch.equal(it0["stems"], it1["stems"]), "consecutive audio spans are identical"
    assert it0["stems"].shape[0] == it0["n_present"] == 2, "duo must yield exactly 2 stems"
    assert ds[0]["target_idx"] < ds[0]["n_present"], "target index out of range"
    # EEG and audio must also cover the same span at the TRIAL level, or every window past
    # the first is offset by a growing amount.
    resp, soli = ds._trial_arrays(*ds.trials[0])
    assert abs(resp.shape[1] / EEG_FS - soli.shape[1] / AUDIO_FS) < 0.05, (
        f"trial durations differ: eeg {resp.shape[1] / EEG_FS:.3f}s vs "
        f"audio {soli.shape[1] / AUDIO_FS:.3f}s")
    n_rows = _check_soli_row_order(ds)
    print(f"[self-check] PASS  window = {eeg_dur:.3f} s on both sides, "
          f"stride advances both, duo yields 2 stems, "
          f"{n_rows} shared-instrument `soli` rows in the order `instruments` declares")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MAD-EEG contrastive adapter -- smoke only")
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--ensemble", default="both", choices=["duo", "trio", "both"])
    args = ap.parse_args()
    smoke(args.madeeg_dir, args.ensemble)
    _self_check(args.madeeg_dir)
