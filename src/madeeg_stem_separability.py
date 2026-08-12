"""EXP. 13 -- the audio-only separability GATE for arm C.

Vault contract, written BEFORE this file existed and not touched after:
  "Cap. 2 -- Exp. 13: gate di separabilita' solo-audio per il braccio C, criterio
   pre-registrato (12 ago 2026)".

THE QUESTION. Exp. 9 measured that the bottleneck of the duo decision lives in the
STIMULI: under the pipeline's own front end the two competing stems of a duo correlate
0.2874 (flux) and 0.1772 (mel-8) after band-pass 1-8 Hz + z-score, and the flux -- the
better tracker -- is the MORE similar of the two on 30/36 duos. Arm C promises to LEARN a
selective representation. This gate asks whether a FIXED one already exists: is there a
representation, computable from the stems and compatible with the pipeline protocol
(64 Hz, 1-8 Hz band-pass, z-score), under which the two stems of a duo become nearly as
uncorrelated as stems of DIFFERENT songs?

Necessary, not sufficient: a GO does not say the EEG tracks that representation; a NO-GO
says the separability ceiling is a property of ensemble music under the whole family
tested, and arm C loses its fixed-target premise.

ZERO LOOKS SPENT. Audio only. This file opens `madeeg_preprocessed.hdf5` and reads ONLY
the `soli` datasets and the metadata; it never touches ['response'], never reads a trial,
never reads a label. No EEG number can come out of it.

DECLARED HERE, BEFORE THE RUN (contract SS2/SS3/SS5, verbatim):
  * CANARY, crossed FIRST: re-running the Exp. 9 audio-only diagnostic must reproduce
    mel-8 mean 0.1772 / median 0.1442 and flux mean 0.2874 / median 0.2417, with flux >
    mel on 30/36, tolerance +-0.0010. If it does not, this script EXITS and no candidate
    is computed (Comandamenti #3). The canary goes through madeeg_reconstruction's own
    source_repr(), so what it validates is the shared feature path, not a copy of it.
  * NULL, computed from the data alone (Comandamenti #4): for EVERY representation, the
    cross-song floor -- all stem pairs from different `morceau`, same processing. Printed
    (mean, sd, n pairs) next to every statistic. Deterministic, no seed.
  * PRIMARY CRITERION, both required:
      1. |corr| within-duo strictly lower than mel-8 on >= 26/36 duos
         (exact binomial P(X>=26 | 36, 0.5) = 0.0057 <= 0.05/6, Bonferroni on K=6);
      2. gap <= 0.05, gap = mean |corr| within-duo - mean |corr| cross-song floor OF THE
         SAME representation.
    GO = at least one candidate passes both. NO-GO = none does.

Run (CPU, ~2'):
  /opt/miniconda3/bin/python src/madeeg_stem_separability.py --madeeg_dir ~/madeeg
"""
import os
import sys
import argparse
from math import comb

import numpy as np
import h5py
import yaml
from scipy.fft import dct
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from madeeg_reconstruction import source_repr, pearson, bandpass   # noqa: E402  the Exp. 9 path

TARGET_FS = 64.0
BAND = (1.0, 8.0)
COMPRESSION = 0.3            # unused by the mel/flux branches; passed for signature parity

# ---- canary, contract SS2. Not editable after the fact. ----
CANARY = {"mel-8": (0.1772, 0.1442), "flux": (0.2874, 0.2417)}
CANARY_COUNT = 30            # flux more similar than mel-8, out of 36
CANARY_TOL = 0.0010

# ---- primary criterion, contract SS5. Not editable after the fact. ----
CRIT1_MIN = 26               # /36, exact binomial p = 0.005657 <= 0.05/6
CRIT2_GAP = 0.05

# ---- the K = 6 candidates, contract SS4. Declared before the run. ----
CANDIDATES = [
    ("C1 mel-24",      ("mel", 24)),
    ("C2 mel-64",      ("mel", 64)),
    ("C3 MFCC-13",     ("mfcc", 13)),
    ("C4 flux-8band",  ("flux_mel", 8)),
    ("C5 flux-24band", ("flux_mel", 24)),
    ("C6 contrast-6",  ("contrast", 6)),
]
REFERENCES = [("mel-8", ("mel", 8)), ("flux", ("flux", 1))]

CAVEAT = [
    "!! AUDIO ONLY, ZERO LOOKS SPENT: no EEG was read. This file opens the preprocessed",
    "!! HDF5 for its `soli` datasets and metadata only -- never ['response'], never a",
    "!! trial, never a label. Nothing here is a statement about EEG.",
    "!! GATE, not a result: a GO is a NECESSARY condition for arm C's fixed-target route,",
    "!! not evidence that the EEG tracks the winning representation. A NO-GO says the",
    "!! ceiling is a property of ensemble music under the whole family tested.",
    "!! Criterion, thresholds and the K=6 candidate list were written in the vault",
    "!! contract of 12/8/2026 BEFORE this file existed, and are not editable after.",
    "!! The 36 units are 18 distinct duo mixtures x 2 attended targets. The two stim ids",
    "!! of a mixture carry the SAME audio, so the paired counts are exactly 2x the count",
    "!! over 18 mixtures and the binomial on 36 treats duplicated units as independent.",
    "!! Applied as pre-registered; the effective independent n is 18, stated as a limit.",
]


def exact_p(k, n):
    """One-sided exact binomial p against p0 = 0.5."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _post(rep, out_len):
    """The tail of source_repr(): resample to the EEG grid, then band-pass each band to
    1-8 Hz and z-score it. Only C3/C6 need it -- everything else goes through
    source_repr() itself. Equivalence with source_repr's own tail is ASSERTED on real
    audio in check_post_matches_source_repr() before any number is produced, so this
    cannot silently drift from the Exp. 9 protocol."""
    if rep.shape[1] != out_len:
        rep = resample(rep, out_len, axis=1)
    out = np.empty((rep.shape[0], out_len))
    for b in range(rep.shape[0]):
        r = bandpass(rep[b], TARGET_FS, BAND[0], BAND[1])
        out[b] = (r - r.mean()) / (r.std() + 1e-8)
    return out


def _stft_geometry(sr):
    """The pipeline's STFT geometry: hop = fs_audio / fs_target, n_fft the next power of
    two above 2*hop. Same numbers source_repr uses, so C3/C6 are frame-aligned with the
    mel branch rather than merely similar to it."""
    hop = max(1, int(round(sr / TARGET_FS)))
    return hop, int(2 ** np.ceil(np.log2(2 * hop)))


def _raw_mfcc(x, sr, n_coef):
    """DCT-II (ortho) of the log-mel-64, coefficients 1..n_coef -- the 0th, i.e. overall
    energy, is dropped as the contract specifies. Same log(S + 1e-6) as the mel branch."""
    import librosa
    hop, n_fft = _stft_geometry(sr)
    S = librosa.feature.melspectrogram(y=x, sr=int(sr), n_fft=n_fft, hop_length=hop, n_mels=64)
    return dct(np.log(S + 1e-6), axis=0, type=2, norm="ortho")[1:n_coef + 1]


def _raw_contrast(x, sr, n_bands):
    """Octave-band spectral contrast (peak - valley). librosa returns n_bands + 1 rows:
    the extra row is the sub-band below fmin. Reported as such, not silently trimmed."""
    import librosa
    hop, n_fft = _stft_geometry(sr)
    S = np.abs(librosa.stft(y=x, n_fft=n_fft, hop_length=hop))
    return librosa.feature.spectral_contrast(S=S, sr=int(sr), n_bands=n_bands)


def represent(x, sr, spec):
    """(stem @ sr) -> (n_bands, out_len), band-matched to the EEG grid. out_len comes from
    the AUDIO duration, never from an EEG trial."""
    kind, param = spec
    out_len = int(round(x.shape[0] / sr * TARGET_FS))
    if kind in ("mel", "flux", "flux_mel"):
        return source_repr(x, sr, out_len, TARGET_FS, BAND, COMPRESSION, param, kind)
    if kind == "mfcc":
        return _post(_raw_mfcc(x, sr, param), out_len)
    if kind == "contrast":
        return _post(_raw_contrast(x, sr, param), out_len)
    raise ValueError(f"unknown representation kind {kind!r}")


def corr(A, B):
    """The Exp. 9 statistic: mean over bands of the per-band Pearson r, signed. Pairs of
    different length (cross-song only -- within-duo stems are same-length by
    construction) are truncated to the shorter one."""
    T = min(A.shape[1], B.shape[1])
    return float(np.mean([pearson(A[b, :T], B[b, :T]) for b in range(A.shape[0])]))


def load_stems(md):
    """(18 unique duo mixtures, 24 distinct stems). Reads `soli` and metadata only.

    A stem is keyed by (genre, song, theme, spatial, instrument): the same instrument of
    the same theme is reused across mixtures, and those occurrences are asserted to be
    byte-identical rather than assumed to be."""
    meta = yaml.safe_load(open(os.path.join(md, "madeeg_preprocessed.yaml")))
    f = h5py.File(os.path.join(md, "madeeg_preprocessed.hdf5"), "r")
    first = {}
    for s in sorted(f.keys()):
        for k in sorted(f[s].keys()):
            if meta[s][k].get("ensemble") == "duo":
                first.setdefault(k, s)          # the same stim id recurs across subjects
    units, stems = [], {}
    for k, s in sorted(first.items()):
        m = meta[s][k]
        sr = int(m["wav_info"]["sfreq"])
        keys = tuple((m["genre"], m["song"], m["theme"], m["spatial"], ins)
                     for ins in m["instruments"])
        soli = np.asarray(f[s][k]["soli"], dtype=np.float64)
        for i, sk in enumerate(keys):
            if sk in stems:
                assert np.array_equal(stems[sk][0], soli[i]), (
                    f"the same stem {sk} differs between mixtures -- STOP")
            else:
                stems[sk] = (soli[i], sr)
        units.append((k, keys))                 # 36 stim ids = 18 mixtures x 2 targets
    f.close()
    return units, stems


def check_post_matches_source_repr(stems):
    """_post() must be the tail of source_repr(), not an approximation of it. Checked on
    real audio: the mel-8 raw representation pushed through _post has to equal
    source_repr(kind='mel', n_mels=8) to floating point."""
    import librosa
    x, sr = next(iter(stems.values()))
    hop, n_fft = _stft_geometry(sr)
    out_len = int(round(x.shape[0] / sr * TARGET_FS))
    S = librosa.feature.melspectrogram(y=x, sr=int(sr), n_fft=n_fft, hop_length=hop, n_mels=8)
    mine = _post(np.log(S + 1e-6), out_len)
    theirs = source_repr(x, sr, out_len, TARGET_FS, BAND, COMPRESSION, 8, "mel")
    assert np.allclose(mine, theirs, atol=1e-12), "_post drifted from source_repr -- STOP"
    return float(np.max(np.abs(mine - theirs)))


def stats(units, stems, spec):
    """Within-duo statistics + the cross-song floor for ONE representation."""
    reps = {sk: represent(x, sr, spec) for sk, (x, sr) in stems.items()}
    within = np.array([corr(reps[a], reps[b]) for _, (a, b) in units])
    keys = sorted(reps)
    floor = np.array([corr(reps[keys[i]], reps[keys[j]])
                      for i in range(len(keys)) for j in range(i + 1, len(keys))
                      if keys[i][:2] != keys[j][:2]])          # different (genre, song)
    return within, floor, next(iter(reps.values())).shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--out", default="", help="provenance file (default: "
                                              "docs/provenance/2026-08-12_exp13_stem_separability.txt)")
    args = ap.parse_args()
    md = os.path.expanduser(args.madeeg_dir)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(root, "docs", "provenance",
                                        "2026-08-12_exp13_stem_separability.txt")

    L = list(CAVEAT)

    def say(line=""):
        L.append(line)
        print(line, flush=True)

    units, stems = load_stems(md)
    say()
    say(f"units: {len(units)} unique duo stim ids from {len({tuple(v) for _, v in units})} "
        f"distinct mixtures · {len(stems)} distinct stems")
    say(f"protocol: representation at {TARGET_FS:g} Hz -> band-pass {BAND[0]}-{BAND[1]} Hz "
        f"-> z-score per band -> mean over bands of the per-band Pearson r (signed),")
    say("          identical to the Exp. 9 audio-only diagnostic; the criterion then takes |corr|.")
    say(f"_post/source_repr equivalence on real audio: max abs diff "
        f"{check_post_matches_source_repr(stems):.3e} (must be 0 to fp)")

    # ---- CANARY FIRST. No candidate is computed until it passes. ----
    say()
    say(f"=== CANARY (contract SS2, tolerance +-{CANARY_TOL:.4f}) -- crossed BEFORE anything else ===")
    ref = {}
    for name, spec in REFERENCES:
        w, fl, nb = stats(units, stems, spec)
        ref[name] = (w, fl, nb)
        em, ed = CANARY[name]
        ok = abs(w.mean() - em) <= CANARY_TOL and abs(np.median(w) - ed) <= CANARY_TOL
        say(f"  {name:6s} mean {w.mean():.4f} (expected {em:.4f})  "
            f"median {np.median(w):.4f} (expected {ed:.4f})  -> [{'PASSED' if ok else 'FAILED'}]")
        if not ok:
            say("  CANARY FAILED -> no candidate is computed (Comandamenti #3).")
            open(out_path, "w").write("\n".join(L) + "\n")
            sys.exit(1)
    n_more = int((ref["flux"][0] > ref["mel-8"][0]).sum())
    ok = n_more == CANARY_COUNT
    say(f"  flux more similar than mel-8 on {n_more}/{len(units)} (expected {CANARY_COUNT}) "
        f"mean diff {np.mean(ref['flux'][0] - ref['mel-8'][0]):+.4f}  -> [{'PASSED' if ok else 'FAILED'}]")
    if not ok:
        say("  CANARY FAILED -> no candidate is computed (Comandamenti #3).")
        open(out_path, "w").write("\n".join(L) + "\n")
        sys.exit(1)
    say("  CANARY PASSED: the shared feature path reproduces Exp. 9. It licenses the wiring")
    say("  and nothing else.")

    mel8 = np.abs(ref["mel-8"][0])
    mel8_floor = np.abs(ref["mel-8"][1])
    say()
    say(f"=== CROSS-SONG FLOOR (the null, computed from the data -- Comandamenti #4) ===")
    say(f"  all stem pairs from different (genre, song): n = {len(mel8_floor)} pairs, "
        f"deterministic, no seed. Unequal-length pairs truncated to the shorter stem.")

    rows = []
    for name, spec in REFERENCES + CANDIDATES:
        w, fl, nb = ref[name] if name in ref else stats(units, stems, spec)
        aw, af = np.abs(w), np.abs(fl)
        k = int((aw < mel8).sum())
        gap = aw.mean() - af.mean()
        c1 = k >= CRIT1_MIN
        c2 = gap <= CRIT2_GAP
        rows.append(dict(name=name, nb=nb, mean=aw.mean(), med=float(np.median(aw)), signed=w.mean(),
                         fmean=af.mean(), fsd=af.std(ddof=1), fn=len(af), k=k, gap=gap,
                         c1=c1, c2=c2, passed=bool(c1 and c2) and name.startswith("C")))

    say()
    say("=== TABLE -- every statistic with its own null next to it ===")
    say(f"  criterion 1: |corr| within-duo strictly below mel-8 on >= {CRIT1_MIN}/{len(units)}"
        f"  (null {len(units)//2}/{len(units)} = 0.500 exact by symmetry, "
        f"P(X>={CRIT1_MIN}|36,0.5) = {exact_p(CRIT1_MIN, len(units)):.6f})")
    say(f"  criterion 2: gap = mean|corr| within-duo - mean|corr| cross-song floor <= {CRIT2_GAP:.2f}")
    say()
    say(f"  {'representation':15s} {'rows':>4s} {'mean':>7s} {'median':>7s} {'signed':>7s} | "
        f"{'floor':>7s} {'sd':>6s} {'n':>4s} | {'gap':>7s} {'w/floor':>8s} | "
        f"{'vs mel-8':>9s} {'p':>8s} | c1  c2  gate")
    for r in rows:
        tag = "ref" if r["name"] in ("mel-8", "flux") else ("PASS" if r["passed"] else "no")
        say(f"  {r['name']:15s} {r['nb']:4d} {r['mean']:7.4f} {r['med']:7.4f} {r['signed']:+7.4f} | "
            f"{r['fmean']:7.4f} {r['fsd']:6.4f} {r['fn']:4d} | {r['gap']:+7.4f} "
            f"{r['mean'] / r['fmean']:8.3f} | "
            f"{r['k']:5d}/{len(units)} {exact_p(r['k'], len(units)):8.6f} | "
            f"{'Y' if r['c1'] else 'n'}   {'Y' if r['c2'] else 'n'}   {tag}")
    say("  (mel-8 vs itself is 0/36 by definition; flux is the second Exp. 9 reference,")
    say("   shown for continuity -- neither is a candidate.)")
    say()
    say("  READ THE GATE WITH THIS: `w/floor` is the within-duo mean divided by this")
    say("  representation's own floor -- arithmetic on the two columns to its left, NOT a")
    say("  pre-registered criterion and NOT used for the verdict. It is printed because")
    say("  NEITHER pre-registered criterion is invariant to a uniform contraction of the")
    say("  correlation scale: shrinking within-duo AND floor by the same factor lowers the")
    say("  gap toward 0 and wins the paired count against mel-8, without any shared")
    say("  component having been removed. `w/floor` is the part of the comparison that")
    say("  survives such a rescaling. The verdict below applies the criterion AS WRITTEN.")

    cand = [r for r in rows if r["name"].startswith("C")]
    winners = [r["name"] for r in cand if r["passed"]]
    say()
    say("=== descriptive ranking of the 6 gaps (contract SS6: the only secondary allowed) ===")
    for i, r in enumerate(sorted(cand, key=lambda r: r["gap"]), 1):
        say(f"  {i}. {r['name']:15s} gap {r['gap']:+.4f}   "
            f"(within {r['mean']:.4f} vs floor {r['fmean']:.4f})")
    say()
    say(f"=== VERDICT: {'GO' if winners else 'NO-GO'} ===")
    say(f"  candidates passing BOTH criteria: {', '.join(winners) if winners else 'none'}")
    if winners:
        say("  Pre-declared reading (contract SS6): arm C has a named target, and the")
        say("  SparrKULee download + baseline reproduction + MAD-EEG adaptation with the")
        say("  winning representation as candidate target are justified. Any EEG use stays")
        say("  behind a NEW pre-registration.")
        say("  What this GO does NOT say: nothing about the EEG (none was read), and nothing")
        say("  about WHY the winner wins -- see the `w/floor` note above, which is the first")
        say("  thing to settle before spending 18.4 GiB and GPU time on it.")
    if not winners:
        say("  No fixed representation in the declared family makes the two competing stems of")
        say("  a duo look like stems of different songs. The fixed-target route of arm C loses")
        say("  its premise; the remaining premise is the weak one (learned features find what no")
        say("  fixed feature shows), and spending GPU on it is A.'s decision, with this number.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(L) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
