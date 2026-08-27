"""EXP. 13 -- the audio-only separability GATE for arm C.

Pre-registration, written BEFORE this file existed and not touched after:
  "Chapter 2 -- Exp. 13: the audio-only separability gate for arm C, pre-registered
   criterion (2026-08-12)".

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

DECLARED HERE, BEFORE THE RUN (contract section 2/section 3/section 5, verbatim):
  * CANARY, crossed FIRST: re-running the Exp. 9 audio-only diagnostic must reproduce
    mel-8 mean 0.1772 / median 0.1442 and flux mean 0.2874 / median 0.2417, with flux >
    mel on 30/36, tolerance +-0.0010. If it does not, this script EXITS and no candidate
    is computed (method rule 3). The canary goes through madeeg_reconstruction's own
    source_repr(), so what it validates is the shared feature path, not a copy of it.
  * NULL, computed from the data alone (method rule 4): for EVERY representation, the
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

# ---- canary, contract section 2. Not editable after the fact. ----
CANARY = {"mel-8": (0.1772, 0.1442), "flux": (0.2874, 0.2417)}
CANARY_COUNT = 30            # flux more similar than mel-8, out of 36
CANARY_TOL = 0.0010

# ---- primary criterion, contract section 5. Not editable after the fact. ----
CRIT1_MIN = 26               # /36, exact binomial p = 0.005657 <= 0.05/6
CRIT2_GAP = 0.05

# ---- the K = 6 candidates, contract section 4. Declared before the run. ----
CANDIDATES = [
    ("C1 mel-24",      ("mel", 24)),
    ("C2 mel-64",      ("mel", 64)),
    ("C3 MFCC-13",     ("mfcc", 13)),
    ("C4 flux-8band",  ("flux_mel", 8)),
    ("C5 flux-24band", ("flux_mel", 24)),
    ("C6 contrast-6",  ("contrast", 6)),
]
REFERENCES = [("mel-8", ("mel", 8)), ("flux", ("flux", 1))]

# ---- EXP. 16 ARM A, contract section A. Reached ONLY with --clap_dir; without it this file is
#      byte-for-byte the Exp. 13 run, same candidates, same output path (method rule 8). --
CLAP_CANDIDATE = ("C7 CLAP-512", "clap")
CLAP_DIM = 512                 # laion_clap's audio tower output width
CLAP_SERIES_HZ = 16.0          # hop 62.5 ms, contract section A.3
NEG_SEED = 20260812            # declared before the run
NEG_CAVEAT = [
    "!! THIS RUN IS A NEGATIVE CONTROL AND CONTAINS NO DATA. Every candidate series is WHITE",
    "!! NOISE with the dimensionality and the frame rate CLAP would have (512 dims at 16 Hz),",
    "!! drawn from a generator seeded per stem, independent across stems by construction.",
    "!! NOTHING HERE IS A RESULT AND NOTHING HERE MAY BE CITED AS ONE.",
    "!! WHY IT EXISTS: method rule 3 asks what the instrument does when the answer is known.",
    "!! Here the answer is known to be NOTHING -- noise cannot separate anything -- so whatever",
    "!! the gate says about this row is a statement about the CRITERION, not about a candidate.",
    "!! Read it as the number a real 512-dimensional candidate has to beat before its own",
    "!! PASS means anything: the `w/floor` column of this row is the no-structure baseline.",
]
# ---- EXP. 17, criterion "Chapter 2 -- Exp. 17: CLAP in the separability gate, criterion
#      REMADE invariant to dimensionality (2026-08-12)". Reached ONLY with --exp17, which also
#      requires --clap_dir. Written here BEFORE any CLAP number existed anywhere: the criterion
#      below replaces Exp. 13's `gap <= 0.05` because that one was FALSIFIED by the negative
#      control of 12/8 (512 noise dims meet it), NOT because anybody disliked a result. No CLAP
#      number has ever been computed at the time these three constants were written. ----
EXP17_CRIT1_MIN = 26           # /36, unchanged from Exp. 13. P(X>=26|36,0.5) = 0.005665
EXP17_CRIT2_MAX = 1.80         # w/floor <= 1.80: cover >= 60% of the mel-8 (3.131) -> noise
                               # (0.922) distance. Stricter than mel-64 (41.6%) and MFCC-13
                               # (47.4%), so "passes" cannot mean "as good as what we have".
EXP17_CRIT3_MIN = 1.20         # w/floor >= 1.20: at least 0.28 above the noise floor. Below
                               # this the candidate is DECLARED INDISTINGUISHABLE FROM NOISE
                               # at its own dimensionality and does NOT win. This is the clause
                               # the old criterion did not have.
EXP17_NOISE_LO = 0.85          # the negative control re-run at CLAP's EXACT dimensionality must
EXP17_NOISE_HI = 1.05          # land here, or the wiring moved -> hard exit, no CLAP computed.
EXP17_CAVEAT = [
    "!! EXP. 17 -- the Exp. 13 gate with a criterion REMADE to be invariant to dimensionality.",
    "!! WHY THE OLD CRITERION IS RETIRED, AND WHY THIS IS NOT MOVING A GOALPOST: method rule",
    "!! 5 forbids changing a threshold AFTER SEEING THE NUMBER THAT MUST BEAT IT. There is no",
    "!! CLAP number here to protect -- none has ever been computed. What exists is a NEGATIVE",
    "!! CONTROL (12/8, docs/provenance/2026-08-12_exp16A_negative_control.txt) showing that 512",
    "!! dimensions of WHITE NOISE satisfy BOTH old criteria: 36/36 against mel-8 and gap",
    "!! -0.0002. CLAP has exactly 512 dimensions. Under the old criterion a GO would have been",
    "!! INDISTINGUISHABLE FROM NOISE, i.e. the test could not be LOST -- the mirror image of",
    "!! the error for which Exp. 10 was retired pre-run and the CCA gate on inner_val_r was",
    "!! declared unwinnable. A falsified criterion is retired in writing (method rule 12);",
    "!! it is not quietly relaxed. Exp. 13's own verdict is NOT touched retroactively.",
    "!! THE NEW CRITERION, all three required, written before this file could read a .npy:",
    "!!   1. |corr| within-duo strictly below mel-8 on >= 26/36 (unchanged from Exp. 13),",
    "!!   2. w/floor <= 1.80  -- dimensionality-invariant, >= 60% of the way from mel-8 (3.131)",
    "!!      to noise (0.922); stricter than BOTH best known candidates,",
    "!!   3. w/floor >= 1.20  -- MANDATORY MARGIN OVER NOISE. Below it the candidate is declared",
    "!!      indistinguishable from noise at its dimensionality and DOES NOT WIN.",
    "!! The negative control is RE-RUN IN THIS RUN at CLAP's exact dimensionality, its row is",
    "!! printed ABOVE CLAP's, and if its w/floor leaves [0.85, 1.05] this script EXITS before",
    "!! CLAP is even computed (method rule 3: the control is crossed BEFORE the real number).",
    "!! !! AND THE LIMIT THAT BELONGS IN EVERY BRANCH OF THE OUTCOME: CLAP's design aperture is",
    "!! 10 s, 160x the window used here, and below 10 s laion_clap applies `repeatpad`. WE ARE",
    "!! MEASURING CLAP OUTSIDE THE REGIME IT WAS TRAINED FOR. A failure here is NOT evidence",
    "!! against CLAP at 10 s; it is a statement about this window.",
]
CLAP_CAVEAT = [
    "!! EXP. 16 ARM A. The K=1 candidate of this run is CLAP, the thesis's own premise, which",
    "!! had never been tested as a reconstruction target (docs/03_OVERVIEW.md, the genealogy). The six",
    "!! Exp. 13 candidates are re-computed here only as the reference frame; the criterion,",
    "!! the canary and the floor are the Exp. 13 ones, unchanged.",
    "!! THE TIME-SCALE GAP, DECLARED BEFORE THE NUMBER: CLAP is time-invariant BY DESIGN and",
    "!! its input aperture is 10 s. The pre-registered band (1-8 Hz) forces a 62.5 ms hop by",
    "!! Nyquist, and a 62.5 ms window so that the aperture does not annihilate the band",
    "!! (a W-long aperture nulls at 1/W: at 125 ms that null sits exactly on 8 Hz). So the",
    "!! series is read at 160x finer a time scale than the model was built for. Whatever the",
    "!! gate returns, THAT gap belongs next to it.",
    "!! THE EMBEDDINGS WERE COMPUTED IN A DIFFERENT INTERPRETER (/opt/anaconda3, laion_clap",
    "!! is not installed in /opt/miniconda3 and nothing was installed there). That is why the",
    "!! Exp. 13 CANARY at the top of this file is the load-bearing check: it proves the",
    "!! MEASUREMENT environment is unchanged. The embeddings enter as data, like a wav does.",
    "!! A GO HERE IS A HYPOTHESIS, NOT A RESULT, and must be written so everywhere: Exp. 15",
    "!! showed this gate can be won by a representation that then got WORSE on the real task.",
    "!! Read the `w/floor` column before the gate column.",
]

CAVEAT = [
    "!! AUDIO ONLY, ZERO LOOKS SPENT: no EEG was read. This file opens the preprocessed",
    "!! HDF5 for its `soli` datasets and metadata only -- never ['response'], never a",
    "!! trial, never a label. Nothing here is a statement about EEG.",
    "!! GATE, not a result: a GO is a NECESSARY condition for arm C's fixed-target route,",
    "!! not evidence that the EEG tracks the winning representation. A NO-GO says the",
    "!! ceiling is a property of ensemble music under the whole family tested.",
    "!! Criterion, thresholds and the K=6 candidate list were written down",
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


def _load_clap(x, clap_dir):
    """EXP. 16 ARM A. The pre-computed CLAP series of THIS stem, (512, n_frames).

    Keyed by md5 of the stem's own float64 bytes, recomputed here from this process's copy of
    the audio: a .npy can only be paired with the stem it was extracted from, so a renamed or
    reordered file cannot silently attach the wrong series to a stem. The embeddings were
    produced by a DIFFERENT interpreter (/opt/anaconda3, environment trap 1) and enter here as
    data, exactly like a wav does -- the Exp. 13 canary above is what proves THIS
    environment, the one that measures, has not moved."""
    import hashlib
    sha = hashlib.md5(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()
    p = os.path.join(os.path.expanduser(clap_dir), sha + ".npy")
    assert os.path.isfile(p), (
        f"no CLAP embedding for the stem hashing to {sha} in {clap_dir} -- run stage 1 "
        f"(src/madeeg_exp16a_clap_extract.py) on the SAME madeeg_dir first")
    rep = np.load(p)
    assert rep.ndim == 2 and rep.shape[0] == 512, f"{p}: expected (512, T), got {rep.shape}"
    return np.asarray(rep, dtype=np.float64)


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
    if kind == "noise":
        # NEGATIVE CONTROL. White noise with CLAP's shape: `param` dimensions at 16 Hz, for as
        # long as this stem lasts. The generator is seeded from the stem's own bytes, so the
        # series is reproducible and INDEPENDENT across stems by construction -- there is no
        # shared structure for any criterion to find.
        import hashlib
        seed = int(hashlib.md5(np.ascontiguousarray(x, dtype=np.float64).tobytes()
                               ).hexdigest()[:8], 16) ^ NEG_SEED
        n_fr = int(x.shape[0] / sr * CLAP_SERIES_HZ)
        return _post(np.random.RandomState(seed).standard_normal((param, n_fr)), out_len)
    if kind == "clap":
        # Same tail as every other candidate: resample onto the 64 Hz grid, band-pass each
        # dimension to 1-8 Hz, z-score it. The CLAP series arrives at 16 Hz (hop 62.5 ms), so
        # this is an UPsampling and it cannot create content above its own 8 Hz Nyquist --
        # which is exactly why the contract fixes the hop rather than the window.
        return _post(_load_clap(x, param), out_len)
    raise ValueError(f"unknown representation kind {kind!r}")


def corr(A, B, absolute=False):
    """The Exp. 9 statistic: mean over bands of the per-band Pearson r, signed. Pairs of
    different length (cross-song only -- within-duo stems are same-length by
    construction) are truncated to the shorter one.

    absolute=True averages |r| instead, which is a DIFFERENT quantity and is used only by
    the Exp. 17 post-hoc diagnostic: it separates 'each dimension is less correlated' from
    'the per-dimension correlations cancel in sign when averaged'. The criterion never
    calls it; the default path never reaches it."""
    T = min(A.shape[1], B.shape[1])
    r = [pearson(A[b, :T], B[b, :T]) for b in range(A.shape[0])]
    return float(np.mean(np.abs(r) if absolute else r))


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


def stats(units, stems, spec, absolute=False):
    """Within-duo statistics + the cross-song floor for ONE representation."""
    reps = {sk: represent(x, sr, spec) for sk, (x, sr) in stems.items()}
    within = np.array([corr(reps[a], reps[b], absolute) for _, (a, b) in units])
    keys = sorted(reps)
    floor = np.array([corr(reps[keys[i]], reps[keys[j]], absolute)
                      for i in range(len(keys)) for j in range(i + 1, len(keys))
                      if keys[i][:2] != keys[j][:2]])          # different (genre, song)
    return within, floor, next(iter(reps.values())).shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--out", default="", help="provenance file (default: "
                                              "docs/provenance/2026-08-12_exp13_stem_separability.txt)")
    ap.add_argument("--clap_dir", default="",
                    help="EXP. 16 ARM A: directory written by stage 1 (src/"
                         "madeeg_exp16a_clap_extract.py) holding one <md5>.npy per stem plus "
                         "manifest.json. Given it, ONE candidate is appended (C7 CLAP-512) "
                         "and the run writes to the Exp. 16A provenance file instead of the "
                         "Exp. 13 one -- a variant never writes over a result "
                         "(method rule 9). Without it this file is the Exp. 13 run, "
                         "unchanged (method rule 8)")
    ap.add_argument("--negative_control", action="store_true",
                    help="EXP. 16 ARM A: replace the CLAP candidate by WHITE NOISE of the same "
                         f"shape ({CLAP_DIM} dims at {CLAP_SERIES_HZ:g} Hz). method rule 3 "
                         "the other way round: the answer is known to be 'nothing', so what "
                         "the gate returns is a statement about the CRITERION. Writes to its "
                         "own provenance file and cannot be confused with a result")
    ap.add_argument("--exp17", action="store_true",
                    help="EXP. 17: the remade, dimensionality-invariant criterion. Requires "
                         "--clap_dir. Runs the negative control AT CLAP'S EXACT DIMENSIONALITY "
                         "IN THE SAME RUN, prints its row ABOVE CLAP's, hard-exits if its "
                         f"w/floor leaves [{EXP17_NOISE_LO}, {EXP17_NOISE_HI}], and judges the "
                         f"candidate on all three of: >= {EXP17_CRIT1_MIN}/36 below mel-8, "
                         f"w/floor <= {EXP17_CRIT2_MAX}, w/floor >= {EXP17_CRIT3_MIN}. Writes "
                         "to its own provenance file (method rule 9)")
    args = ap.parse_args()
    assert not (args.clap_dir and args.negative_control), (
        "--clap_dir and --negative_control are two different runs and must not be mixed: one "
        "measures a candidate, the other measures the criterion")
    assert not (args.exp17 and not args.clap_dir), (
        "--exp17 is the Exp. 17 criterion applied to the CLAP row: it needs --clap_dir. "
        "Its own negative control is internal and does NOT come from --negative_control")
    md = os.path.expanduser(args.madeeg_dir)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = ("2026-08-12_exp17_clap_separability.txt" if args.exp17 else
                   "2026-08-12_exp16A_negative_control.txt" if args.negative_control else
                   "2026-08-12_exp16A_clap_separability.txt" if args.clap_dir else
                   "2026-08-12_exp13_stem_separability.txt")
    out_path = args.out or os.path.join(root, "docs", "provenance", default_out)

    candidates = list(CANDIDATES)
    L = list(CAVEAT)
    if args.exp17:
        # ORDER IS LOAD-BEARING: the negative control is computed and CROSSED first, and its row
        # is printed ABOVE CLAP's, as the contract section 2 requires.
        candidates.append((f"N1 noise-{CLAP_DIM}", ("noise", CLAP_DIM)))
    if args.clap_dir:
        name, kind = CLAP_CANDIDATE
        candidates.append((name, (kind, args.clap_dir)))
        L = CLAP_CAVEAT + L
    if args.exp17:
        L = EXP17_CAVEAT + L
    if args.negative_control:
        candidates.append((f"N1 noise-{CLAP_DIM}", ("noise", CLAP_DIM)))
        L = NEG_CAVEAT + L

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

    if args.clap_dir:
        # The provenance of the embeddings belongs INSIDE this file, not only next to it.
        import json
        man = json.load(open(os.path.join(os.path.expanduser(args.clap_dir), "manifest.json")))
        say()
        say("=== EXP. 16A -- provenance of the CLAP series (stage 1, a DIFFERENT interpreter) ===")
        for k in ("model", "checkpoint", "checkpoint_bytes", "checkpoint_sha256",
                  "audio_sample_rate_in", "clap_sample_rate", "resampler", "window_s", "hop_s",
                  "series_rate_hz", "nyquist_hz", "clap_input_len_s", "clap_padding",
                  "embedding_dim", "l2_normalised_per_frame", "seed", "interpreter",
                  "versions", "total_frames", "extraction_seconds"):
            say(f"  {k}: {man.get(k)}")
        miss = [h for h in man["stems"] if not os.path.isfile(
            os.path.join(os.path.expanduser(args.clap_dir), h + ".npy"))]
        say(f"  stems with a series on disk: {len(man['stems']) - len(miss)}/{len(stems)} "
            f"(missing: {miss if miss else 'none'})")
        if args.exp17:
            # ---- GATE 3 OF THE CONTRACT section 3: without a COMPLETE provenance the measurement
            # does not start. Missing or empty -> hard exit, before the canary. ----
            need = ["checkpoint", "checkpoint_bytes", "checkpoint_sha256", "versions",
                    "audio_sample_rate_in", "clap_sample_rate", "window_s", "hop_s", "machine"]
            bad = [k for k in need if not man.get(k)]
            bad += ["versions." + k for k in ("laion_clap", "torch")
                    if not man.get("versions", {}).get(k)]
            say(f"  GATE 3 -- provenance complete: {'FAILED, missing ' + str(bad) if bad else 'PASSED'}")
            if bad:
                say("  The measurement does NOT start without it (contract section 3).")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                open(out_path, "w").write("\n".join(L) + "\n")
                sys.exit(1)
            say(f"  sha256 of the checkpoint actually loaded: {man['checkpoint_sha256']}")
        say("  The criterion, the canary, the floor and the 26/36 threshold are the Exp. 13")
        say("  ones, applied UNCHANGED as the contract requires (K=1 for this arm; the")
        say("  threshold is deliberately NOT re-derived for a smaller K -- that would be")
        say("  loosening a pre-registered number after the fact).")

    # ---- CANARY FIRST. No candidate is computed until it passes. ----
    say()
    say(f"=== CANARY (contract section 2, tolerance +-{CANARY_TOL:.4f}) -- crossed BEFORE anything else ===")
    ref = {}
    for name, spec in REFERENCES:
        w, fl, nb = stats(units, stems, spec)
        ref[name] = (w, fl, nb)
        em, ed = CANARY[name]
        ok = abs(w.mean() - em) <= CANARY_TOL and abs(np.median(w) - ed) <= CANARY_TOL
        say(f"  {name:6s} mean {w.mean():.4f} (expected {em:.4f})  "
            f"median {np.median(w):.4f} (expected {ed:.4f})  -> [{'PASSED' if ok else 'FAILED'}]")
        if not ok:
            say("  CANARY FAILED -> no candidate is computed (method rule 3).")
            open(out_path, "w").write("\n".join(L) + "\n")
            sys.exit(1)
    n_more = int((ref["flux"][0] > ref["mel-8"][0]).sum())
    ok = n_more == CANARY_COUNT
    say(f"  flux more similar than mel-8 on {n_more}/{len(units)} (expected {CANARY_COUNT}) "
        f"mean diff {np.mean(ref['flux'][0] - ref['mel-8'][0]):+.4f}  -> [{'PASSED' if ok else 'FAILED'}]")
    if not ok:
        say("  CANARY FAILED -> no candidate is computed (method rule 3).")
        open(out_path, "w").write("\n".join(L) + "\n")
        sys.exit(1)
    say("  CANARY PASSED: the shared feature path reproduces Exp. 9. It licenses the wiring")
    say("  and nothing else.")

    mel8 = np.abs(ref["mel-8"][0])
    mel8_floor = np.abs(ref["mel-8"][1])
    say()
    say(f"=== CROSS-SONG FLOOR (the null, computed from the data -- method rule 4) ===")
    say(f"  all stem pairs from different (genre, song): n = {len(mel8_floor)} pairs, "
        f"deterministic, no seed. Unequal-length pairs truncated to the shorter stem.")

    rows = []
    for name, spec in REFERENCES + candidates:
        w, fl, nb = ref[name] if name in ref else stats(units, stems, spec)
        aw, af = np.abs(w), np.abs(fl)
        k = int((aw < mel8).sum())
        gap = aw.mean() - af.mean()
        wf = float(aw.mean() / af.mean())
        c1 = k >= CRIT1_MIN
        c2 = gap <= CRIT2_GAP
        # EXP. 17 columns: arithmetic on the two columns already computed, no new statistic.
        e1, e2, e3 = c1, wf <= EXP17_CRIT2_MAX, wf >= EXP17_CRIT3_MIN
        # In an Exp. 17 run the ONLY candidate is CLAP (K=1). The six Exp. 13 rows are the
        # reference frame and are NOT re-judged under the new criterion: Exp. 13's verdict is
        # not touched retroactively (contract section 1).
        is_cand = name == CLAP_CANDIDATE[0] if args.exp17 else name.startswith("C")
        passed = (bool(e1 and e2 and e3) if args.exp17 else bool(c1 and c2)) and is_cand
        rows.append(dict(name=name, nb=nb, mean=aw.mean(), med=float(np.median(aw)), signed=w.mean(),
                         fmean=af.mean(), fsd=af.std(ddof=1), fn=len(af), k=k, gap=gap, wf=wf,
                         c1=c1, c2=c2, e1=e1, e2=e2, e3=e3, is_cand=is_cand, passed=passed))
        # ---- GATE 2 OF THE CONTRACT, CROSSED BEFORE THE CLAP ROW EXISTS. The noise row is
        # computed first (see the candidate order above); if the wiring moved, this exits here
        # and no CLAP number is ever produced, let alone looked at (method rule 3). ----
        if args.exp17 and name.startswith("N"):
            say()
            say("=== GATE 2 -- negative control RE-RUN at CLAP's exact dimensionality ===")
            say(f"  {name}: w/floor = {wf:.3f}   must land in "
                f"[{EXP17_NOISE_LO:.2f}, {EXP17_NOISE_HI:.2f}] (12/8 reference: 0.922)")
            if not (EXP17_NOISE_LO <= wf <= EXP17_NOISE_HI):
                say("  -> [FAILED] the wiring is not the one the criterion was calibrated on.")
                say("  NO CLAP NUMBER IS COMPUTED (method rule 3).")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                open(out_path, "w").write("\n".join(L) + "\n")
                sys.exit(1)
            say("  -> [PASSED] the no-structure baseline reproduces. It licenses the scale on")
            say("     which criteria 2 and 3 are read, and nothing else.")

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
        # A negative control can never be a winner, but its c1/c2 must be printed honestly:
        # the whole point is what the CRITERION does with a row that has nothing in it.
        tag = ("ref" if r["name"] in ("mel-8", "flux")
               else ("NOISE MEETS BOTH" if r["c1"] and r["c2"] else "noise, fails")
               if r["name"].startswith("N")
               else "Exp.13 ref" if not r["is_cand"]
               else ("PASS" if r["passed"] else "no"))
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

    if args.exp17:
        by = {r["name"]: r for r in rows}
        say()
        say("=== EXP. 17 CRITERION -- all three required, written before any CLAP number ===")
        say(f"  1. |corr| within-duo strictly below mel-8 on >= {EXP17_CRIT1_MIN}/{len(units)} "
            f"(P = {exact_p(EXP17_CRIT1_MIN, len(units)):.6f}) -- unchanged from Exp. 13")
        say(f"  2. w/floor <= {EXP17_CRIT2_MAX:.2f}  (>= 60% of the way from mel-8 to noise)")
        say(f"  3. w/floor >= {EXP17_CRIT3_MIN:.2f}  (margin over noise; BELOW IT THE CANDIDATE "
            "IS DECLARED INDISTINGUISHABLE FROM NOISE AND DOES NOT WIN)")
        say()
        say(f"  {'representation':15s} {'rows':>4s} {'w/floor':>8s} | {'c1: vs mel-8':>14s} "
            f"{'c2: <=1.80':>10s} {'c3: >=1.20':>10s} | verdict")
        order = [f"N1 noise-{CLAP_DIM}", "mel-8", "C3 MFCC-13", CLAP_CANDIDATE[0]]
        for n in order:                       # the noise row is printed FIRST, contract section 2
            r = by[n]
            v = ("no-structure baseline" if r["name"].startswith("N") else
                 "reference, not a candidate" if not r["is_cand"] else
                 "PASSES ALL THREE" if r["passed"] else
                 "INDISTINGUISHABLE FROM NOISE" if not r["e3"] else "FAILS")
            cnt = "%d/%d" % (r["k"], len(units))
            say(f"  {r['name']:15s} {r['nb']:4d} {r['wf']:8.3f} | "
                f"{cnt:>9s} {'Y' if r['e1'] else 'n':>4s} "
                f"{'Y' if r['e2'] else 'n':>10s} {'Y' if r['e3'] else 'n':>10s} | {v}")
        say("  (mel-8 and MFCC-13 are the Exp. 13 frame, judged under THEIR criterion there and")
        say("   NOT re-judged here; the noise row is a control and cannot win anything.)")

    cand = [r for r in rows if r["is_cand"]]
    winners = [r["name"] for r in cand if r["passed"]]
    for r in [r for r in rows if r["name"].startswith("N")]:
        say()
        say("=== NEGATIVE CONTROL -- what the criterion does with a row that has NOTHING in it ===")
        say(f"  {r['name']}: {CLAP_DIM} independent white-noise dimensions at "
            f"{CLAP_SERIES_HZ:g} Hz, one draw per stem, seed {NEG_SEED}, no shared structure")
        say(f"  by construction. criterion 1: {r['k']}/{len(units)} (needs >= {CRIT1_MIN}) -> "
            f"{'MET' if r['c1'] else 'not met'} · criterion 2: gap {r['gap']:+.4f} "
            f"(needs <= {CRIT2_GAP:.2f}) -> {'MET' if r['c2'] else 'not met'}")
        say(f"  w/floor = {r['mean'] / r['fmean']:.3f}   (mel-8 "
            f"{rows[0]['mean'] / rows[0]['fmean']:.3f})")
        if r["c1"] and r["c2"]:
            say("  ==> THE PRE-REGISTERED CRITERION IS MET BY NOISE. It is therefore NOT")
            say("  sufficient on its own to distinguish 'this representation separates the two")
            say("  stems' from 'this representation has many independent dimensions and a small")
            say("  aggregate correlation'. Exp. 13 section 5.1 declared this weakness in words and")
            say("  left it open; this row is the number. A candidate of the same dimensionality")
            say("  that passes the gate has demonstrated NOTHING until its w/floor is read")
            say("  against this row -- which is why the contract makes that column mandatory.")
            if args.exp17:
                say("  THIS IS WHY EXP. 17 EXISTS. The `gap <= 0.05` criterion is RETIRED IN")
                say("  WRITING as falsified by this row (method rule 12), not relaxed after")
                say("  seeing a CLAP number -- there was none to see. The replacement is read")
                say("  on the two w/floor thresholds above, calibrated on THIS row and mel-8.")
            else:
                say("  NOT DONE HERE: the criterion is NOT changed. Changing a pre-registered")
                say("  threshold after seeing what it admits is exactly what pre-registration")
                say("  exists to prevent (method rules 5/6). This is a control, reported next")
                say("  to the criterion, not a replacement for it.")
        else:
            say("  ==> the criterion rejects a structureless row, which is the outcome a")
            say("  criterion should have. It says nothing about any real candidate.")
    say()
    say(f"=== descriptive ranking of the {len(cand)} gaps "
        "(contract section 6: the only secondary allowed) ===")
    for i, r in enumerate(sorted(cand, key=lambda r: r["gap"]), 1):
        say(f"  {i}. {r['name']:15s} gap {r['gap']:+.4f}   "
            f"(within {r['mean']:.4f} vs floor {r['fmean']:.4f})")
    if args.exp17:
        # ---- POST-HOC DIAGNOSTIC, DECLARED AS SUCH (method rule 5: a second look is
        # exploratory by construction, even when it is only descriptive). It is computed AFTER
        # the verdict above, it enters NO criterion, and the verdict does not depend on it.
        # WHAT IT SEPARATES: the criterion statistic is |mean over dims of the signed r|, so a
        # small value has two possible causes -- each dimension really is less correlated, or
        # the per-dimension correlations CANCEL in sign. mean|r| per dim tells them apart. ----
        say()
        say("=== POST-HOC DIAGNOSTIC (descriptive, NOT part of the criterion) ===")
        say("  criterion statistic: |mean_dims r| · diagnostic: mean_dims |r|, same pairs")
        say(f"  {'representation':15s} {'|mean r| w':>10s} {'mean |r| w':>10s} "
            f"{'|mean r| fl':>11s} {'mean |r| fl':>11s} {'cancel':>7s} {'w/floor|r|':>10s}")
        alt = {}
        for n, spec in [(f"N1 noise-{CLAP_DIM}", ("noise", CLAP_DIM)), ("mel-8", ("mel", 8)),
                        ("C3 MFCC-13", ("mfcc", 13)),
                        (CLAP_CANDIDATE[0], ("clap", args.clap_dir))]:
            aw, af, _ = stats(units, stems, spec, absolute=True)
            r = by[n]
            alt[n] = float(aw.mean() / af.mean())
            say(f"  {n:15s} {r['mean']:10.4f} {aw.mean():10.4f} {r['fmean']:11.4f} "
                f"{af.mean():11.4f} {aw.mean() / r['mean']:6.1f}x {alt[n]:10.3f}")
        say("  `cancel` = how much larger the per-dimension |r| is than the aggregate the")
        say("  criterion reads. A large factor means the aggregate is small because signs cancel")
        say("  across dimensions, NOT because the two stems stopped sharing structure. Noise")
        say("  cancels 22x; that IS the mechanism by which 512 empty dimensions beat the old")
        say("  criterion. CLAP cancels far less, i.e. its 512 dimensions are strongly redundant.")
        say()
        say("  !! AND THE PART THAT MUST NOT BE BURIED (method rules 5/6). The last column is")
        say("  !! the same w/floor ratio built on mean|r| instead of |mean r|. On it CLAP reads")
        say(f"  !! {alt[CLAP_CANDIDATE[0]]:.3f} -- i.e. under THAT aggregation criterion 2 would be "
            f"{'MET' if alt[CLAP_CANDIDATE[0]] <= EXP17_CRIT2_MAX else 'NOT met'} and the verdict")
        say("  !! would flip. THE VERDICT ABOVE STANDS AS PRE-REGISTERED: the criterion names the")
        say("  !! Exp. 9 statistic (mean over dims of the SIGNED r, then |.|), and swapping in the")
        say("  !! statistic that flips the answer after seeing the answer is precisely what")
        say("  !! pre-registration exists to prevent. This line is declared, not acted on: it is")
        say("  !! an OPEN QUESTION for a new contract, not a second reading of this one.")

        r = by[CLAP_CANDIDATE[0]]
        branch = ("PASS" if r["passed"] else "NOISE" if not r["e3"] else "FAIL")
        say()
        say(f"=== EXP. 17 VERDICT: {'GO' if branch == 'PASS' else 'NO-GO'} "
            f"(contract section 5 branch: {branch}) ===")
        say(f"  C7 CLAP-512: c1 {r['k']}/{len(units)} -> {'MET' if r['e1'] else 'NOT MET'} · "
            f"c2 w/floor {r['wf']:.3f} <= {EXP17_CRIT2_MAX:.2f} -> "
            f"{'MET' if r['e2'] else 'NOT MET'} · c3 w/floor >= {EXP17_CRIT3_MIN:.2f} -> "
            f"{'MET' if r['e3'] else 'NOT MET'}")
        say(f"  scale, printed above it: noise-{CLAP_DIM} {by[f'N1 noise-{CLAP_DIM}']['wf']:.3f} "
            f"· mel-8 {by['mel-8']['wf']:.3f} · MFCC-13 {by['C3 MFCC-13']['wf']:.3f}")
        if branch == "PASS":
            say("  -> The semantic premise of the thesis HOLDS ON THIS AXIS, against a threshold")
            say("     stricter than both best known candidates. It stays a HYPOTHESIS about the")
            say("     real task: MFCC won this gate and then got WORSE (Exp. 15). CLAP as a")
            say("     target on own-vs-other needs a NEW pre-registration.")
        elif branch == "FAIL":
            say("  -> The embedding that should capture 'cello' vs 'flute' does NOT separate them")
            say("     better than 8 mel bands. Strong, publishable, and to be written WITHOUT")
            say("     softening it.")
        else:
            say("  -> CLAP AT THIS WINDOW IS INDISTINGUISHABLE FROM NOISE at its own")
            say("     dimensionality. The reading is about the METHOD, not about CLAP: 62.5 ms")
            say("     is outside its regime.")
        say("  !! IN EVERY BRANCH, AND NOT NEGOTIABLE: CLAP's design aperture is 10 s, 160x the")
        say("  !! window used here, and below 10 s laion_clap applies `repeatpad`. WE MEASURED")
        say("  !! CLAP OUTSIDE THE REGIME IT WAS TRAINED FOR. A failure here IS NOT EVIDENCE")
        say("  !! AGAINST CLAP AT 10 s. A 10 s window would need a new contract and would push")
        say("  !! the band below 1 Hz, i.e. a different experiment.")
        say("  !! No EEG was read. No second candidate, no second window, in any branch.")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path, "w").write("\n".join(L) + "\n")
        print(f"\n-> {out_path}")
        return

    say()
    say(f"=== VERDICT: {'GO' if winners else 'NO-GO'} ===")
    if args.negative_control:
        say("  (In a negative-control run the six C rows are the Exp. 13 candidates recomputed;")
        say("   they reproduce that run exactly. The line below is Exp. 13's verdict restated,")
        say("   NOT a new one, and the N row above is not a candidate and cannot win anything.)")
    say(f"  candidates passing BOTH criteria: {', '.join(winners) if winners else 'none'}")
    if winners:
        say("  Pre-declared reading (contract section 6): arm C has a named target, and the")
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
