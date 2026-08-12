"""EXP. 11 -- the attended spectral REGISTER from the EEG frequency decomposition.

Vault contract (written BEFORE this file existed, and not touched after):
  "Cap. 2 -- Exp. 11 (ESPLORATIVO): il registro spettrale attenzionato, criterio
   pre-registrato (11 ago 2026)".

THE HYPOTHESIS (external, dated, not born from our data). The supervisor's mail of
~8/8/2026: "The instruments would trigger responses in vastly different frequency
ranges ... one would expect to find this decodable ... when using the frequency
decomposition as the features for the decoder (see Ingmar DeVries' paper from the lab
in 2021)". Feature recipe = de Vries, Marinato & Baldauf (2021) J Neurosci 41(41),
frozen in the Exp. 5 contract of 9/8 and never executed there (the SIDE decision was
dropped on 10/8). This file changes ONLY the dependent variable: REGISTER, not SIDE.

THE DESIGN, and why it is immune to the confound this project has already demonstrated:
  * MAD-EEG presents the SAME duo mixture twice, once per attended instrument:
        classique_morceau1_duo_CoFl_theme1_stereo_Co   (attend Co)
        classique_morceau1_duo_CoFl_theme1_stereo_Fl   (attend Fl)
    47 such pairs exist (8 subjects). Unit of analysis = the PAIR: a forced choice
    between its two trials. Acoustics, subject, session and spatial render are
    IDENTICAL across the two, so they cancel; the null is 0.500 EXACT by symmetry,
    not assumed (Comandamenti #4).
  * Label that generalises across pairs: y=1 iff the attended instrument is the one
    with the HIGHER spectral centroid, measured from the stems (`soli`) -- audio only,
    fixed a priori, 0 ambiguous orderings out of 18 duo mixtures.
  * CV = leave-one-subject-out: the test pair's subject is entirely out of training.
    Trial fingerprint (armD: LDA at 0.62 on zero-attention labels under window-CV) is
    unavailable by construction, and so is the per-subject group prior.

DECLARED BEFORE THE RUN, in this file, and not editable afterwards:
  * PRIMARY threshold  30/47 = 0.6383 (exact one-sided binomial p = 0.039470;
    29 -> 0.071933, so there was no freedom of choice). Power 0.35 @0.60, 0.63 @0.65,
    0.86 @0.70.
  * POSITIVE CONTROL threshold >= 0.90 (>= 43/47), crossed BEFORE the real number is
    computed -- this script REFUSES to compute the real number if it fails
    (Comandamenti #3). Dose-response reported at half amplitude.
  * NEGATIVE CONTROL (training labels permuted, 20 seeds) must land in [0.40, 0.60];
    outside that band the claimed null is wrong and the primary is not reported.
  * SECONDARY, pre-declared with direction: dose-response on register distance, split
    at the median declared in the contract (1188.3 Hz) -> 22 "far" vs 25 "near",
    predicted far > near. Descriptive; no formal test of the difference (n too small).
  * SECONDARY, descriptive: the 13 pairs where the louder source is the LOWER one --
    the gain confound is imbalanced 27/13 and cannot be removed, only probed.

EXPLORATORY BY CONSTRUCTION: the duos are spent material (ledger of looks). No trio is
touched by this file. No sign is ever flipped (Comandamenti #6): if the register decodes
below chance, that is explained, not inverted.

Run (CPU, ~10'):
  /opt/miniconda3/bin/python src/madeeg_spectral_attention.py --madeeg_dir ~/madeeg
"""
import os
import sys
import argparse
from math import comb

import numpy as np
import h5py
import yaml
import librosa
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.time_frequency import tfr_array_morlet

EEG_FS = 256.0
FREQS = np.logspace(np.log10(1.0), np.log10(40.0), 25)   # de Vries: 1-40 Hz, 25 log steps
N_CYCLES = np.linspace(3.0, 12.0, 25)                    # de Vries: 3-12 cycles
DECIM = 8                                                # power is a smooth envelope

PRIMARY_THRESHOLD = 30          # /47, exact binomial p = 0.0395
CONTROL_THRESHOLD = 0.90        # positive control, declared here before the run
NULL_BAND = (0.40, 0.60)        # negative control must land inside
DELTA_MEDIAN_HZ = 1188.3        # declared in the contract from stimulus audio alone
INJECT_CHANNELS = ["P3", "P1", "Pz", "P2", "P4", "POz", "O1", "Oz", "O2"]
PRE_CH_ORDER = ["F3", "F1", "Fz", "F2", "F4", "C3", "C1", "Cz", "C2", "C4",
                "CPz", "P3", "P1", "Pz", "P2", "P4", "POz", "O1", "Oz", "O2"]
THRESHOLDS = {"mono": 27, "stereo": 30, "pooled": 53}   # pre-written 9/8, reverified 11/8
INJECT_BAND = (8.0, 13.0)
N_NULL_SEEDS = 20

CAVEAT = [
    "!! EXPLORATORY BY CONSTRUCTION: the duos are spent material (ledger of looks).",
    "!! Unit = the PAIR (same mixture, two attended instruments); null = 0.500 EXACT by",
    "!! symmetry, not assumed. Criterion, thresholds and secondaries were written in the",
    "!! vault contract BEFORE this file existed and are not editable after the fact.",
]


def exact_p(k, n):
    """One-sided exact binomial p against p0 = 0.5."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def stem_centroids(f, meta):
    """(mixture key) -> {instrument: median spectral centroid}. Audio only, no EEG."""
    out = {}
    for subj in sorted(f.keys()):
        for stim in sorted(f[subj].keys()):
            m = meta[subj][stim]
            if m.get("ensemble") != "duo":
                continue
            key = (m["genre"], m["song"], m["theme"], tuple(m["instruments"]), m["spatial"])
            if key in out:
                continue
            soli = np.asarray(f[subj][stim]["soli"], dtype=np.float64)
            sr = int(m["wav_info"]["sfreq"])
            out[key] = {ins: float(np.median(librosa.feature.spectral_centroid(
                y=soli[i], sr=sr, n_fft=4096, hop_length=2048)[0]))
                for i, ins in enumerate(m["instruments"])}
    return out


def mixture_key(m):
    return (m["genre"], m["song"], m["theme"], tuple(m["instruments"]), m["spatial"])


def build_index(f, meta, cents):
    """Every duo trial with its register label, plus the list of complete pairs."""
    trials, groups = [], {}
    for subj in sorted(f.keys()):
        for stim in sorted(f[subj].keys()):
            m = meta[subj][stim]
            if m.get("ensemble") != "duo":
                continue
            c = cents[mixture_key(m)]
            hi = max(c, key=c.get)
            rec = dict(subj=subj, stim=stim, target=m["target"], hi=hi,
                       y=int(m["target"] == hi), dcent=abs(c[list(c)[0]] - c[list(c)[1]]),
                       gains={ins: float(g) for ins, g in
                              zip(m["instruments"], m["wav_info"]["gains"])})
            trials.append(rec)
            groups.setdefault((subj,) + mixture_key(m), {})[m["target"]] = len(trials) - 1
    pairs = []
    for key, d in groups.items():
        if len(d) != 2:
            continue
        i_hi = next(i for _, i in d.items() if trials[i]["y"] == 1)
        i_lo = next(i for _, i in d.items() if trials[i]["y"] == 0)
        gains = trials[i_hi]["gains"]                       # {instrument: gain}
        hi_ins = trials[i_hi]["hi"]
        lo_ins = next(ins for ins in gains if ins != hi_ins)
        pairs.append(dict(subj=key[0], i_hi=i_hi, i_lo=i_lo, dcent=trials[i_hi]["dcent"],
                          gain_hi=gains[hi_ins], gain_lo=gains[lo_ins]))
    return trials, pairs


def parse_key(k):
    """`classique_morceau1_duo_CoFl_theme1_mono_Co` (or `..._stereo_lcr_Co`) ->
    genre, song, instruments, theme, spatial, target. Instrument codes are 2 chars.
    Validated against madeeg_preprocessed.yaml on the stereo keys, where truth exists."""
    p = k.split("_")
    i = p.index("duo")
    instr = (p[i + 1][:2], p[i + 1][2:])
    spatial = "mono" if "_mono_" in k else "stereo"
    return dict(genre=p[0], song=p[1], instruments=instr, theme=p[i + 2],
                spatial=spatial, target=p[-1])


def build_raw_trials(raw_f, raw_info, sq, stimuli_dir, cents_nospatial, subjects, want, diag):
    """EXP. 12 -- duo trials rebuilt from the CONTINUOUS raw release.

    The mono duos exist only here. Segmentation convention inherited from
    madeeg_reconstruction.build_solo_trials: each repetition starts at sample n_ech[i] and
    lasts as long as its wav. Alignment canary (contract Exp. 12 par.2): on stereo trials,
    which exist in BOTH releases, the correlation against the authors' preprocessed trial
    peaks sharply at offset 0 and collapses at +-0.12 s.

    Cleaning = notch 50 Hz + ICA on the EOG/ECG references, reusing the repo's
    clean_continuous (with its documented channel-typing trap). Both renders go through
    THIS pipeline, so they are comparable to each other; comparison with the preprocessed
    release (Exp. 11) is indicative, not paired.
    """
    import librosa
    from madeeg_reconstruction import clean_continuous
    trials = []
    for subj in subjects:
        chs = [c["ch_name"] for c in raw_info[subj]["chs"]]
        X = clean_continuous(np.asarray(raw_f[subj], dtype=np.float64), chs,
                             raw_info[subj]["chs"], "notch_ica", subj, 42, diag)
        idx = [chs.index(c) for c in PRE_CH_ORDER]
        X = X[idx]
        for k in sorted(sq[subj]):
            if "duo" not in k.split("_"):
                continue
            m = parse_key(k)
            if m["spatial"] not in want:
                continue
            key = (m["genre"], m["song"], m["theme"], tuple(sorted(m["instruments"])))
            if key not in cents_nospatial:
                continue
            segs, ok = [], True
            for i, w in enumerate(sq[subj][k]["wav_files"]):
                path = os.path.join(stimuli_dir, w)
                if not os.path.exists(path):
                    ok = False
                    break
                audio, sr = librosa.load(path, sr=None, mono=True)
                a = int(sq[subj][k]["n_ech"][i])
                b = a + int(round(len(audio) / sr * EEG_FS))
                if b > X.shape[1]:
                    ok = False
                    break
                segs.append(X[:, a:b])
            if not ok or not segs:
                continue
            c = cents_nospatial[key]
            hi = max(c, key=c.get)
            trials.append(dict(subj=subj, stim=k, target=m["target"], hi=hi,
                               y=int(m["target"] == hi),
                               dcent=abs(c[list(c)[0]] - c[list(c)[1]]),
                               spatial=m["spatial"], eeg=np.concatenate(segs, axis=1)))
    return trials


def pairs_from(trials):
    """Complete pairs: same subject, same mixture, same render, both targets present."""
    groups = {}
    for i, t in enumerate(trials):
        m = parse_key(t["stim"])
        groups.setdefault((t["subj"], m["genre"], m["song"], m["theme"],
                           tuple(sorted(m["instruments"])), m["spatial"]), {})[t["target"]] = i
    out = []
    for key, d in groups.items():
        if len(d) != 2:
            continue
        hi = [i for i in d.values() if trials[i]["y"] == 1]
        lo = [i for i in d.values() if trials[i]["y"] == 0]
        if len(hi) != 1 or len(lo) != 1:
            continue
        out.append(dict(subj=key[0], spatial=key[5], i_hi=hi[0], i_lo=lo[0],
                        dcent=trials[hi[0]]["dcent"], gain_hi=0.0, gain_lo=0.0))
    return out


def inject(eeg, ch_names, amp):
    """Add `amp` x the 8-13 Hz component on posterior channels: the positive control."""
    if amp <= 0:
        return eeg
    b, a = butter(4, [INJECT_BAND[0] / (EEG_FS / 2), INJECT_BAND[1] / (EEG_FS / 2)], btype="band")
    out = eeg.copy()
    idx = [ch_names.index(c) for c in INJECT_CHANNELS if c in ch_names]
    out[idx] = out[idx] + amp * filtfilt(b, a, eeg[idx], axis=1)
    return out


def trial_features(eeg):
    """(n_ch, T) -> (n_ch * n_freqs,) log Morlet power, averaged over the whole trial."""
    p = tfr_array_morlet(eeg[np.newaxis], sfreq=EEG_FS, freqs=FREQS, n_cycles=N_CYCLES,
                         output="power", decim=DECIM, zero_mean=True, verbose=False)[0]
    return np.log(p.mean(axis=2) + 1e-20).ravel()


def compute_features(f, meta, trials, amp):
    """Feature matrix over all duo trials. amp > 0 injects the label-tied control signal."""
    X = np.empty((len(trials), 20 * len(FREQS)))
    for i, t in enumerate(trials):
        if "eeg" in t:                                  # EXP. 12: rebuilt from the raw release
            eeg, ch = t["eeg"], PRE_CH_ORDER
        else:                                           # EXP. 11 path, unchanged
            eeg = np.asarray(f[t["subj"]][t["stim"]]["response"], dtype=np.float64)
            ch = list(meta[t["subj"]][t["stim"]]["eeg_info"]["ch_names"])
        X[i] = trial_features(inject(eeg, ch, amp if t["y"] == 1 else 0.0))
        if (i + 1) % 25 == 0:
            print(f"    ... {i + 1}/{len(trials)} trial", flush=True)
    return X


def zscore_within_subject(X, trials):
    """Per (channel, frequency), inside each subject. Unsupervised and pair-symmetric:
    it cannot create the contrast being measured, only make subjects comparable."""
    Z = X.copy()
    for subj in sorted({t["subj"] for t in trials}):
        rows = [i for i, t in enumerate(trials) if t["subj"] == subj]
        mu, sd = Z[rows].mean(axis=0), Z[rows].std(axis=0)
        Z[rows] = (Z[rows] - mu) / (sd + 1e-12)
    return Z


def loso_decisions(Z, trials, pairs, y_override=None):
    """Leave-one-subject-out. One forced choice per pair. Returns a bool per pair."""
    y = np.array([t["y"] for t in trials]) if y_override is None else y_override
    subj_of = np.array([t["subj"] for t in trials])
    hits = []
    for p in pairs:
        tr = subj_of != p["subj"]
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(Z[tr], y[tr])
        s = clf.decision_function(Z[[p["i_hi"], p["i_lo"]]])
        hits.append(bool(s[0] > s[1]))          # ties count as wrong, declared
    return np.array(hits)


def write(out_dir, name, lines):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    txt = "\n".join(CAVEAT + lines)
    open(os.path.join(d, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n[{name.split('_')[0]}] -> {d}\n")
    return txt


def report(hits, label):
    k, n = int(hits.sum()), len(hits)
    return f"{label}: {k}/{n} = {k / n:.4f}  (null 0.500 exact by symmetry, p = {exact_p(k, n):.4f})"


def concentration_map(md):
    """Per-trial concentration 1-5 from behavioural_data.xlsx, where `stimuli order` and
    `concentration levels` are 78-entry lists stored as strings, one row per subject."""
    import ast
    import pandas as pd
    df = pd.read_excel(os.path.join(md, "behavioural_data.xlsx"))
    out = {}
    for _, r in df.iterrows():
        order = ast.literal_eval(r["stimuli order"])[0]
        lev = ast.literal_eval(r["concentration levels"])[0]
        assert len(order) == len(lev), "order/levels misaligned -- STOP"
        out[f"{int(r['subject ID']):04d}"] = dict(zip(order, lev))
    return out


def run_exp12(args, out, md):
    """EXP. 12 -- the same test on the MONO duos, fresh material for this variable.
    Contract: 'Cap. 2 -- Exp. 12 ... criterio pre-registrato (11 ago 2026)'."""
    fp = h5py.File(os.path.join(md, "madeeg_preprocessed.hdf5"), "r")
    meta = yaml.safe_load(open(os.path.join(md, "madeeg_preprocessed.yaml")))
    print("[exp12] centroids from the stereo stems (stems are shared by both renders)...")
    cents = stem_centroids(fp, meta)
    cents_ns = {(g, so, th, tuple(sorted(i))): c for (g, so, th, i, _), c in cents.items()}

    n_bad = sum(1 for s in meta for k in meta[s]
                if meta[s][k].get("ensemble") == "duo"
                and (parse_key(k)["target"] != meta[s][k]["target"]
                     or tuple(sorted(parse_key(k)["instruments"])) != tuple(sorted(meta[s][k]["instruments"]))))
    print(f"[exp12] key-parser canary against the preprocessed metadata: {n_bad} mismatches")
    assert n_bad == 0, "the key parser disagrees with the metadata -- STOP"
    fp.close()

    raw_info = yaml.safe_load(open(os.path.join(md, "madeeg_raw.yaml")))
    sq = yaml.safe_load(open(os.path.join(md, "madeeg_sequences_raw.yaml")))
    fr = h5py.File(os.path.join(md, "madeeg_raw.hdf5"), "r")
    diag = {}
    print("[exp12] rebuilding duo trials from the raw release (notch + ICA)...")
    trials = build_raw_trials(fr, raw_info, sq, os.path.join(md, "stimuli"), cents_ns,
                              sorted(fr.keys()), {"mono", "stereo"}, diag)
    fr.close()
    pairs = pairs_from(trials)
    mono = [p for p in pairs if p["spatial"] == "mono"]
    ster = [p for p in pairs if p["spatial"] == "stereo"]
    print(f"[exp12] {len(trials)} duo trials rebuilt · pairs: mono {len(mono)}, "
          f"stereo {len(ster)}, pooled {len(pairs)}")
    for s in sorted(diag):
        print(f"    ICA {s}: {diag[s]}")
    assert len(mono) == 42 and len(ster) == 47, (
        f"contract says 42 mono + 47 stereo, found {len(mono)} + {len(ster)} -- STOP")

    idx_of = {id(p): i for i, p in enumerate(pairs)}
    sub = lambda h, ps: h[[idx_of[id(p)] for p in ps]]

    print(f"\n[exp12] POSITIVE CONTROL on the PRIMARY set (mono), threshold "
          f"{CONTROL_THRESHOLD:.2f}")
    Zc = zscore_within_subject(compute_features(None, None, trials, 1.0), trials)
    hc = sub(loso_decisions(Zc, trials, pairs), mono)
    passed = hc.mean() >= CONTROL_THRESHOLD
    write(out, "exp12_control_a10", [
        "!! POSITIVE CONTROL, NOT A RESULT: label-tied 8-13 Hz power added to the real EEG.",
        report(hc, "positive control a=1.0 (mono)"),
        f"declared threshold >= {CONTROL_THRESHOLD:.2f} -> [{'PASSED' if passed else 'FAILED'}]",
    ] + [f"ICA {s}: {diag[s]}" for s in sorted(diag)])
    if not passed:
        print("[exp12] positive control FAILED -> the real number is NOT computed.")
        sys.exit(1)

    print("\n[exp12] sensitivity floor on the primary set (controls only)")
    floor = []
    for amp in (0.20, 0.10, 0.05):
        Za = zscore_within_subject(compute_features(None, None, trials, amp), trials)
        h = sub(loso_decisions(Za, trials, pairs), mono)
        floor.append((amp, int(h.sum())))
        print(f"    a={amp:.2f}: {int(h.sum())}/{len(mono)}", flush=True)

    print("\n[exp12] PRIMARY -- real EEG, no injection")
    Z = zscore_within_subject(compute_features(None, None, trials, 0.0), trials)
    hits = loso_decisions(Z, trials, pairs)
    h_mono, h_ster = sub(hits, mono), sub(hits, ster)

    rng = np.random.RandomState(args.seed)
    y_true = np.array([t["y"] for t in trials])
    nulls = []
    for _ in range(N_NULL_SEEDS):
        y_perm = y_true.copy()
        for s in sorted({t["subj"] for t in trials}):
            rows = [i for i, t in enumerate(trials) if t["subj"] == s]
            y_perm[rows] = y_true[rng.permutation(rows)]
        nulls.append(sub(loso_decisions(Z, trials, pairs, y_override=y_perm), mono).mean())
    null_mean = float(np.mean(nulls))

    med = float(np.median([p["dcent"] for p in mono]))
    far = np.array([p["dcent"] > med for p in mono])
    conc = concentration_map(md)
    hi5 = np.array([conc[p["subj"]].get(trials[p["i_hi"]]["stim"]) == 5
                    and conc[p["subj"]].get(trials[p["i_lo"]]["stim"]) == 5 for p in mono])

    k = int(h_mono.sum())
    lines = [
        "source = RAW release, notch + ICA, segmentation inherited from build_solo_trials;",
        "alignment canary in the contract: correlation vs the preprocessed trials peaks at",
        "offset 0 and collapses at +-0.12 s. Both renders go through THIS pipeline, so they",
        "are comparable to each other; comparison with Exp. 11 is indicative, not paired.",
        "",
        report(h_mono, "PRIMARY   mono (fresh material for this variable)"),
        f"declared threshold {THRESHOLDS['mono']}/{len(mono)} = "
        f"{THRESHOLDS['mono'] / len(mono):.4f} -> "
        f"[{'PASSED' if k >= THRESHOLDS['mono'] else 'NOT PASSED'}]",
        "",
        "SECONDARY (declared, does not promote anything to confirmatory):",
        "  " + report(h_ster, "stereo from raw (pipeline comparability vs Exp. 11's 24/47)")
        + f"  threshold {THRESHOLDS['stereo']}/{len(ster)}",
        "  " + report(hits, "pooled 89") + f"  threshold {THRESHOLDS['pooled']}/{len(pairs)}",
        "",
        f"SECONDARY dose-response on register (direction predicted far > near), median "
        f"{med:.1f} Hz:",
        f"  far  (n={int(far.sum())}): " + report(h_mono[far], "acc"),
        f"  near (n={int((~far).sum())}): " + report(h_mono[~far], "acc"),
        "",
        f"DESCRIPTIVE, no threshold: pairs with BOTH trials at concentration 5 "
        f"(n={int(hi5.sum())}): " + report(h_mono[hi5], "acc") if hi5.sum() else "",
        "",
        f"NEGATIVE CONTROL (training labels permuted, {N_NULL_SEEDS} seeds): {null_mean:.4f}"
        f"  band [{NULL_BAND[0]}, {NULL_BAND[1]}] -> "
        f"[{'OK' if NULL_BAND[0] <= null_mean <= NULL_BAND[1] else 'NULL IS WRONG'}]",
        "SENSITIVITY FLOOR (controls only): "
        + " · ".join(f"a={a:.2f}: {kk}/{len(mono)}" for a, kk in floor),
        f"positive control a=1.0: {int(hc.sum())}/{len(mono)} (PASSED before the above)",
    ]
    write(out, "exp12_primary", lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--log_dir", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control_sweep", action="store_true",
                    help="Controls only: sweep the injected amplitude down to find the "
                         "sensitivity FLOOR of this design. Declared necessary after the "
                         "a=1.0 and a=0.5 controls both saturated at 47/47 -- by the "
                         "contract's own rule a control that does not degrade does not "
                         "bound anything. Injects synthetic signal only; it spends no new "
                         "look on the attention decision and does not touch the primary.")
    ap.add_argument("--source", choices=["preprocessed", "raw"], default="preprocessed",
                    help="preprocessed = EXP. 11 (stereo duos, authors' cleaning; DEFAULT, "
                         "untouched). raw = EXP. 12: duo trials rebuilt from the continuous "
                         "raw release, which is the only place the MONO duos exist.")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.log_dir or os.path.join(root, "runs", "results")
    md = os.path.expanduser(args.madeeg_dir)
    if args.source == "raw":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        return run_exp12(args, out, md)

    f = h5py.File(os.path.join(md, "madeeg_preprocessed.hdf5"), "r")
    meta = yaml.safe_load(open(os.path.join(md, "madeeg_preprocessed.yaml")))

    print("[exp11] spectral centroids from the stems (audio only, no EEG)...")
    cents = stem_centroids(f, meta)
    trials, pairs = build_index(f, meta, cents)
    n_pairs = len(pairs)
    print(f"[exp11] {len(trials)} duo trials, {n_pairs} complete pairs, "
          f"{sum(t['y'] for t in trials)} trials labelled high-register")
    assert n_pairs == 47, f"contract says 47 pairs, found {n_pairs} -- STOP"

    if args.control_sweep:
        rows = []
        for amp in (0.40, 0.20, 0.10, 0.05, 0.02, 0.01):
            Za = zscore_within_subject(compute_features(f, meta, trials, amp), trials)
            h = loso_decisions(Za, trials, pairs)
            rows.append((amp, int(h.sum())))
            print(f"  a={amp:.2f}  {int(h.sum())}/{n_pairs} = {h.mean():.4f}  "
                  f"p={exact_p(int(h.sum()), n_pairs):.4f}", flush=True)
        floor = next((a for a, k in rows if k < PRIMARY_THRESHOLD), None)
        write(out, "exp11_control_sweep", [
            "!! CONTROLS ONLY, NOT A RESULT. Injected synthetic 8-13 Hz power modulation at",
            "!! decreasing amplitude, to bound what effect size this design could have seen.",
            "!! Run BECAUSE the a=1.0 and a=0.5 controls both saturated at 47/47: by the",
            "!! contract's own rule, a control that does not degrade bounds nothing.",
        ] + [f"  a={a:.2f}: {k}/{n_pairs} = {k / n_pairs:.4f}  p={exact_p(k, n_pairs):.4f}"
             for a, k in rows] + [
            f"sensitivity floor: the first amplitude that falls BELOW the primary threshold "
            f"({PRIMARY_THRESHOLD}/{n_pairs}) is a = {floor}" if floor else
            f"NO FLOOR FOUND down to a = {rows[-1][0]}: the design still passes the primary "
            f"threshold at the smallest amplitude tested.",
        ])
        f.close()
        return

    # ---- POSITIVE CONTROL FIRST. The real number is not computed until it passes. ----
    print(f"\n[exp11] POSITIVE CONTROL a=1.0 (threshold {CONTROL_THRESHOLD:.2f} = "
          f"{int(np.ceil(CONTROL_THRESHOLD * n_pairs))}/{n_pairs}) -- injecting into REAL EEG")
    Zc = zscore_within_subject(compute_features(f, meta, trials, 1.0), trials)
    hc = loso_decisions(Zc, trials, pairs)
    passed = hc.mean() >= CONTROL_THRESHOLD
    write(out, "exp11_control_a10", [
        "!! POSITIVE CONTROL, NOT A RESULT: a synthetic 8-13 Hz power modulation tied to",
        "!! the true label was ADDED to the real EEG on posterior channels. It says the",
        "!! wiring recovers a signal that is present by construction. It says NOTHING",
        "!! about the real data.",
        f"channels={','.join(INJECT_CHANNELS)}  band={INJECT_BAND[0]}-{INJECT_BAND[1]} Hz  amplitude=1.0",
        report(hc, "positive control a=1.0"),
        f"declared threshold >= {CONTROL_THRESHOLD:.2f} -> "
        f"[{'PASSED' if passed else 'FAILED'}]",
    ])
    if not passed:
        print("[exp11] positive control FAILED -> the real number is NOT computed "
              "(Comandamenti #3). Fix the wiring.")
        f.close()
        sys.exit(1)

    print("\n[exp11] POSITIVE CONTROL dose-response a=0.5")
    Zh = zscore_within_subject(compute_features(f, meta, trials, 0.5), trials)
    hh = loso_decisions(Zh, trials, pairs)
    write(out, "exp11_control_a05", [
        "!! POSITIVE CONTROL at HALF amplitude, NOT A RESULT. Declared before the run: a",
        "!! control that passes at full amplitude but does not degrade at half amplitude",
        "!! is not measuring what it claims to measure.",
        report(hh, "positive control a=0.5"),
    ])

    # ---- the real number ----
    print("\n[exp11] PRIMARY -- real EEG, no injection")
    Z = zscore_within_subject(compute_features(f, meta, trials, 0.0), trials)
    hits = loso_decisions(Z, trials, pairs)
    k = int(hits.sum())

    rng = np.random.RandomState(args.seed)
    y_true = np.array([t["y"] for t in trials])
    null_accs = []
    for _ in range(N_NULL_SEEDS):
        y_perm = y_true.copy()
        for subj in sorted({t["subj"] for t in trials}):       # permute inside subject
            rows = [i for i, t in enumerate(trials) if t["subj"] == subj]
            y_perm[rows] = y_true[rng.permutation(rows)]
        null_accs.append(loso_decisions(Z, trials, pairs, y_override=y_perm).mean())
    null_mean = float(np.mean(null_accs))
    null_ok = NULL_BAND[0] <= null_mean <= NULL_BAND[1]

    far = np.array([p["dcent"] > DELTA_MEDIAN_HZ for p in pairs])
    rev = np.array([p["gain_hi"] < p["gain_lo"] for p in pairs])
    lines = [
        f"features = log Morlet power, {len(FREQS)} log-spaced freqs 1-40 Hz, 3-12 cycles,",
        f"           20 channels -> {20 * len(FREQS)} features/trial; z-scored within subject",
        f"classifier = shrinkage LDA (lsqr, Ledoit-Wolf)   CV = leave-one-subject-out",
        "",
        report(hits, "PRIMARY  attended register"),
        f"declared threshold {PRIMARY_THRESHOLD}/{n_pairs} = {PRIMARY_THRESHOLD / n_pairs:.4f} -> "
        f"[{'PASSED' if k >= PRIMARY_THRESHOLD else 'NOT PASSED'}]",
        f"declared power: 0.35 @0.60 · 0.63 @0.65 · 0.86 @0.70",
        "",
        f"NEGATIVE CONTROL (training labels permuted, {N_NULL_SEEDS} seeds): "
        f"{null_mean:.4f}  declared band [{NULL_BAND[0]}, {NULL_BAND[1]}] -> "
        f"[{'OK' if null_ok else 'NULL IS WRONG -- primary not reportable'}]",
        "",
        "SECONDARY (pre-declared, direction predicted far > near; descriptive, no formal",
        "test of the difference -- n too small):",
        f"  far  (delta > {DELTA_MEDIAN_HZ} Hz, n={int(far.sum())}): " + report(hits[far], "acc"),
        f"  near (delta <= {DELTA_MEDIAN_HZ} Hz, n={int((~far).sum())}): " + report(hits[~far], "acc"),
        "",
        f"SECONDARY (descriptive, underpowered): pairs where the LOUDER source is the LOWER",
        f"  register, n={int(rev.sum())}: " + report(hits[rev], "acc"),
        "",
        f"positive control a=1.0: {int(hc.sum())}/{n_pairs} (PASSED before this was computed)",
        f"positive control a=0.5: {int(hh.sum())}/{n_pairs}",
    ]
    write(out, "exp11_primary", lines)

    import pandas as pd
    pd.DataFrame([dict(subj=p["subj"], dcent=p["dcent"], gain_hi=p["gain_hi"],
                       gain_lo=p["gain_lo"], hit=bool(h), hit_ctrl=bool(c))
                  for p, h, c in zip(pairs, hits, hc)]).to_csv(
        os.path.join(out, "exp11_primary", "pairs.csv"), index=False)
    f.close()


if __name__ == "__main__":
    main()
