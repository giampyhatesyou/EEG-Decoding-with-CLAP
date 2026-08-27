"""EXP. 16 ARM A, STAGE 1 -- CLAP embeddings of the duo stems, as a TIME SERIES.

Pre-registration, written BEFORE this file existed and not touched after:
  "Chapter 2 -- Exp. 16: the thesis premise put to the test -- CLAP as a target and rich
   EEG features, pre-registered criterion (2026-08-12)", section A.

WHY TWO STAGES, AND WHY THIS ONE IS NOT THE MEASUREMENT.
`laion_clap` is NOT installed in /opt/miniconda3 (the canonical interpreter of every MAD-EEG
number) and IS installed in /opt/anaconda3 with torch 2.11. environment trap 1: Anaconda base
produces different floats at 2e-5 and makes the md5 canaries fail. So this file ONLY writes
embeddings to disk, under /opt/anaconda3, and takes NO measurement; the gate itself runs in
/opt/miniconda3 from those .npy, where the Exp. 13 canary must reproduce exactly. Nothing is
installed in /opt/miniconda3.

  Stage 1 (this file, /opt/anaconda3/bin/python) -> <out_dir>/<sha>.npy + manifest.json
  Stage 2 (/opt/miniconda3/bin/python)           -> src/madeeg_stem_separability.py --clap_dir

ZERO LOOKS SPENT. Audio only: it reuses madeeg_stem_separability.load_stems(), which opens
the preprocessed HDF5 for its `soli` datasets and metadata and never touches ['response'],
never reads a trial, never reads a label.

THE RECIPE, FIXED HERE BEFORE ANY NUMBER EXISTS (contract section A.3):
  * hop = 62.5 ms  ->  series rate 16 Hz  ->  Nyquist 8 Hz, exactly the top of the
    pre-registered 1-8 Hz band. The contract calls this non-negotiable and forbids
    compensating a slower hop by resampling. MEASURED cost at this hop: ~62 ms of CPU per
    window (HTSAT-tiny, batch 32, this machine) x 8882 windows over the 555 s of duo stem
    audio = ~9 minutes, i.e. inside the 90-minute budget. The declared fallback recipe
    (hop 125 ms, band 1-4 Hz, mel-8 re-run in the same band) is therefore NOT used.
  * window = 62.5 ms = hop, consecutive and non-overlapping. Three independent reasons, all
    written before any number:
      (a) the contract asks for the SHORTEST window the model accepts without artificial
          padding. MEASURED in laion_clap.training.data.get_audio_features: anything shorter
          than 10 s is `repeatpad`, i.e. the waveform is REPEATED n = floor(480000/len)
          times and only the remainder is zero-filled. 3000 samples divides 480000 exactly
          (160 repeats), so this window is padded with NO zeros at all -- the model sees a
          seamless loop, never silence.
      (b) a sliding aperture of length W attenuates the series like |sinc(f W)|: at W = 125
          ms the first null falls exactly on 8 Hz and the top of the pre-registered band is
          annihilated by the aperture itself. At W = 62.5 ms the first null is at 16 Hz and
          the whole 1-8 Hz band survives (0.64 at 8 Hz). Any longer window measures the
          smoothing, not CLAP.
      (c) W = hop makes the frames disjoint, so no correlation between neighbouring frames
          is manufactured by overlap.
  * !! AND THE LIMIT THAT COMES WITH IT, DECLARED NOW AND NOT AFTER THE NUMBER: CLAP's
    design aperture is 10 s and the model is time-invariant BY DESIGN (contract section A.3,
    Registro section 2.5). 62.5 ms is 160x shorter. This is not a defect of the implementation, it
    is the pre-registered band and CLAP's own time scale being 160x apart. Whatever this
    gate returns, THAT is the finding, and it belongs next to the number.

Run (CPU, ~10', interpreter is NOT optional):
  /opt/anaconda3/bin/python src/madeeg_exp16a_clap_extract.py --madeeg_dir ~/madeeg \\
      --out_dir runs/clap_exp16a

PREREQUISITE, and it is NOT satisfied automatically: laion_clap's default checkpoint
(630k-audioset-best.pt, 1 863 587 645 bytes) is NOT in this machine's cache, and
CLAP_Module.load_ckpt() would DOWNLOAD it from
https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt into
/opt/anaconda3/lib/python3.12/site-packages/laion_clap/. Downloading 1.74 GiB is A.'s
decision, not an agent's. Pass --ckpt <path> once the file exists; this script REFUSES to
trigger the download by itself.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import platform

import numpy as np


def _dist_version(name):
    """Installed distribution version, for packages that expose no __version__."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as e:                       # noqa: BLE001 -- reported, never guessed
        return f"unknown ({type(e).__name__})"

SR_CLAP = 48000
WIN_S = 0.0625                 # 62.5 ms -- see the recipe above. Not editable after the fact.
HOP_S = 0.0625                 # 62.5 ms -> 16 Hz series -> Nyquist 8 Hz
BATCH = 32
SEED = 20260812                # declared before the run; the path that consumes it is not
                               # reached (rand_trunc only fires for audio LONGER than 10 s)


def stem_sha(x):
    """The key of a stem: md5 of its float64 bytes, exactly as the HDF5 stores them.

    Deliberately not a human-readable name. Stage 2 recomputes this hash from ITS OWN copy of
    the audio, so a .npy can only ever be matched to the stem it was computed from -- a
    filename typo or a reordered stem list cannot silently pair the wrong series."""
    return hashlib.md5(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def frames(x48, win, hop):
    """(T,) -> (n_frames, win), consecutive windows. The tail shorter than one window is
    dropped: padding it would put silence in the model's mouth. _post() in stage 2 resamples
    the series onto the 64 Hz grid derived from the AUDIO duration anyway."""
    n = 1 + (len(x48) - win) // hop if len(x48) >= win else 0
    return np.stack([x48[i * hop:i * hop + win] for i in range(n)]) if n else np.zeros((0, win))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="", help="path to a LOCAL laion_clap checkpoint. "
                                               "Required: this script never downloads.")
    ap.add_argument("--amodel", default="HTSAT-tiny",
                    help="must match the checkpoint; HTSAT-tiny is what src/models/"
                         "clap_encoder.py uses in this repo")
    args = ap.parse_args()

    if not (args.ckpt and os.path.isfile(os.path.expanduser(args.ckpt))):
        sys.exit(
            "--ckpt must point to an existing laion_clap checkpoint file.\n"
            "laion_clap's own load_ckpt() would DOWNLOAD 630k-audioset-best.pt "
            "(1 863 587 645 bytes) from huggingface.co/lukewys/laion_clap.\n"
            "Downloading it is A.'s decision, not this script's -- STOP.")
    ckpt = os.path.expanduser(args.ckpt)

    import torch
    import scipy
    from scipy.signal import resample_poly
    import laion_clap

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from madeeg_stem_separability import load_stems      # the SAME 24 stems as Exp. 13

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    md = os.path.expanduser(args.madeeg_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    units, stems = load_stems(md)
    print(f"{len(units)} duo stim ids · {len(stems)} distinct stems", flush=True)

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel=args.amodel)
    model.load_ckpt(ckpt=ckpt)
    model.eval()

    win = int(round(WIN_S * SR_CLAP))
    hop = int(round(HOP_S * SR_CLAP))
    assert 480000 % win == 0, f"window {win} does not divide the 10 s CLAP input -> zero padding"

    man = dict(
        experiment="Exp. 16 arm A stage 1 (extraction only -- no measurement here)",
        contract="Chapter 2 -- Exp. 16 ... pre-registered criterion (2026-08-12), section A",
        model="laion_clap.CLAP_Module(enable_fusion=False, amodel=%r)" % args.amodel,
        checkpoint=os.path.abspath(ckpt),
        checkpoint_bytes=os.path.getsize(ckpt),
        checkpoint_sha256=hashlib.sha256(open(ckpt, "rb").read()).hexdigest(),
        audio_sample_rate_in=int(next(iter(stems.values()))[1]),
        clap_sample_rate=SR_CLAP,
        resampler="scipy.signal.resample_poly(x, 160, 147)  # 44100 -> 48000, polyphase",
        window_s=WIN_S, hop_s=HOP_S, series_rate_hz=1.0 / HOP_S, nyquist_hz=0.5 / HOP_S,
        clap_input_len_s=10.0,
        clap_padding="repeatpad: the %d-sample window is repeated %d times to fill 480000; "
                     "no zeros are added" % (win, 480000 // win),
        embedding_dim=512,
        l2_normalised_per_frame=None,          # MEASURED below, not assumed
        batch=BATCH, seed=SEED,
        interpreter=sys.executable,
        machine=platform.node(), platform=platform.platform(),
        device="cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      numpy=np.__version__, scipy=scipy.__version__,
                      # laion_clap ships no __version__ attribute; the DISTRIBUTION version is
                      # the only honest answer and the provenance gate asks for it by name.
                      laion_clap=_dist_version("laion_clap")),
        caveats=[
            "EXTRACTION ONLY. No statistic is computed here and no number in this directory "
            "is a result. The gate runs in /opt/miniconda3 (environment trap 1).",
            "CLAP's design aperture is 10 s and the model is time-invariant by design; the "
            "pre-registered band forces a 62.5 ms window. The two time scales are 160x "
            "apart, and that gap must be reported next to whatever the gate returns.",
            "ZERO LOOKS SPENT: audio only, no EEG opened, no label read.",
        ],
        stems={},
    )

    t0 = time.time()
    norms = []
    for i, (sk, (x, sr)) in enumerate(sorted(stems.items()), 1):
        sha = stem_sha(x)
        x48 = resample_poly(np.asarray(x, dtype=np.float64), 160, 147)
        F = frames(x48, win, hop)
        out = np.empty((len(F), 512), dtype=np.float32)
        for b in range(0, len(F), BATCH):
            chunk = torch.from_numpy(np.ascontiguousarray(F[b:b + BATCH], dtype=np.float32))
            with torch.no_grad():
                out[b:b + BATCH] = model.get_audio_embedding_from_data(
                    x=chunk, use_tensor=True).cpu().numpy()
        norms.append(float(np.linalg.norm(out, axis=1).mean()))
        np.save(os.path.join(out_dir, sha + ".npy"), out.T)          # (512, n_frames)
        man["stems"][sha] = dict(key=list(sk), n_samples=int(len(x)), sr=int(sr),
                                 duration_s=len(x) / sr, n_frames=int(len(F)))
        print(f"  [{i:2d}/{len(stems)}] {sha[:8]} {sk[4]:<12s} {len(x)/sr:6.2f}s "
              f"-> {len(F):5d} frames  ({time.time() - t0:.0f}s elapsed)", flush=True)

    man["l2_normalised_per_frame"] = dict(
        mean_frame_norm=float(np.mean(norms)),
        note="1.0 means laion_clap L2-normalises every frame, so the series carries "
             "DIRECTION only and the per-band z-score of the protocol is applied to a "
             "already-unit-norm vector. Measured, not assumed.")
    man["extraction_seconds"] = round(time.time() - t0, 1)
    man["total_frames"] = int(sum(v["n_frames"] for v in man["stems"].values()))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n-> {out_dir}/manifest.json  ({man['total_frames']} frames, "
          f"{man['extraction_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
