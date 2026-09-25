"""Band-power change produced by each injected amplitude of the detection floor (Ch. 9, tab:floor).

WHAT IT MEASURES: the ratio of 8-13 Hz power on the nine injection channels after and
  before `inject(eeg, ch, a)` of src/madeeg_spectral_attention.py, over all duo trials
  of the preprocessed release (stereo). The injection is applied to EVERY trial, so no
  label is read: it opens no split and spends no look.
COST: ~1 min CPU. GPU: no.
EXPECTED (printed in tab:floor, until now with no file behind them):
  a=0.20 x1.393   a=0.10 x1.188   a=0.05 x1.092

Usage: /opt/miniconda3/bin/python scripts/replicate/floor_power_ratio.py --madeeg_dir ~/madeeg
"""
import argparse
import os
import sys

import h5py
import numpy as np
import yaml
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from madeeg_spectral_attention import EEG_FS, FREQS, INJECT_BAND, INJECT_CHANNELS, N_CYCLES, inject  # noqa: E402
from mne.time_frequency import tfr_array_morlet  # noqa: E402

AMPS = (0.20, 0.10, 0.05, 0.02)
BAND_FREQS = (FREQS >= INJECT_BAND[0]) & (FREQS <= INJECT_BAND[1])


def welch_band(x):
    f, p = welch(x, fs=EEG_FS, nperseg=512, axis=-1)
    return p[:, (f >= INJECT_BAND[0]) & (f <= INJECT_BAND[1])].sum(axis=1)


def morlet_band(x):
    # the pipeline's own power: Morlet, averaged over the trial, at the FREQS inside 8-13 Hz
    p = tfr_array_morlet(x[np.newaxis], sfreq=EEG_FS, freqs=FREQS[BAND_FREQS],
                         n_cycles=N_CYCLES[BAND_FREQS], output="power", decim=8,
                         zero_mean=True, verbose=False)[0]
    return p.mean(axis=2).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    md = os.path.expanduser(ap.parse_args().madeeg_dir)
    f = h5py.File(os.path.join(md, "madeeg_preprocessed.hdf5"), "r")
    meta = yaml.safe_load(open(os.path.join(md, "madeeg_preprocessed.yaml")))

    before = {"welch": [], "morlet": []}
    after = {a: {"welch": [], "morlet": []} for a in AMPS}
    n = 0
    for subj in sorted(f.keys()):
        for stim in sorted(f[subj].keys()):
            m = meta[subj][stim]
            if m.get("ensemble") != "duo":
                continue
            eeg = np.asarray(f[subj][stim]["response"], dtype=np.float64)
            ch = list(m["eeg_info"]["ch_names"])
            idx = [ch.index(c) for c in INJECT_CHANNELS if c in ch]
            assert len(idx) == len(INJECT_CHANNELS), (subj, stim)
            before["welch"].append(welch_band(eeg[idx]))
            before["morlet"].append(morlet_band(eeg[idx]))
            for a in AMPS:
                x = inject(eeg, ch, a)[idx]
                after[a]["welch"].append(welch_band(x))
                after[a]["morlet"].append(morlet_band(x))
            n += 1
    print(f"duo trials (preprocessed, stereo): {n}; channels: {','.join(INJECT_CHANNELS)}; "
          f"band {INJECT_BAND[0]:.0f}-{INJECT_BAND[1]:.0f} Hz; no label read")
    print("ratio = 8-13 Hz power after / before injection, pooled over trials and channels "
          "(median of per-trial-channel ratios in brackets)")
    for a in AMPS:
        cells = []
        for k in ("welch", "morlet"):
            b, x = np.concatenate(before[k]), np.concatenate(after[a][k])
            cells.append(f"{k} x{x.sum() / b.sum():.3f} [{np.median(x / b):.3f}]")
        print(f"  a={a:.2f}  ideal (1+a)^2 = {(1 + a) ** 2:.4f}   " + "   ".join(cells))


if __name__ == "__main__":
    main()
