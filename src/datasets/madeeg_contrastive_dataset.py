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
                 eeg_normalization="MetaAI", clamp_value=20):
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
        n_present = len(instruments)                # `soli` always has 3 rows; row 3 is
        stems = soli[:n_present]                    # empty for a duo, so slice it off
        stems = torch.from_numpy(stems.astype(np.float32)).unsqueeze(1)  # (n, 1, samples)

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

    def _trial_arrays(self, subj, stim):
        """Whole-trial `response` and `soli`, cached across the windows that share a trial.

        `read_f64` reads a whole HDF5 dataset -- it selects H5S_ALL, which is the price of
        the type conversion MAD-EEG's float layout needs. `soli` is (3, 1.2 M) float64 =
        ~30 MB, so re-reading it per window would mean ~175 GB of I/O per epoch. Indexing
        is trial-major, so a two-entry cache turns that back into one read per trial.
        CEILING: with a shuffling DataLoader consecutive items are no longer in the same
        trial and this degrades to one read per item. If that shows up as a bottleneck,
        the fix is a hyperslab read (h5py selection instead of H5S_ALL) so only the
        window's samples cross the wire -- not a bigger cache.
        """
        key = (subj, stim)
        if key not in self._cache:
            if self._h5 is None:
                self._h5 = h5py.File(self.h5_path, "r")
            grp = self._h5[subj][stim]
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (read_f64(grp["response"]), read_f64(grp["soli"]))
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
    print(f"[self-check] PASS  window = {eeg_dur:.3f} s on both sides, "
          f"stride advances both, duo yields 2 stems")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MAD-EEG contrastive adapter -- smoke only")
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--ensemble", default="both", choices=["duo", "trio", "both"])
    args = ap.parse_args()
    smoke(args.madeeg_dir, args.ensemble)
    _self_check(args.madeeg_dir)
