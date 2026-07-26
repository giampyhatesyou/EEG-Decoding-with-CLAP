# CHANGED(baseline): NEW FILE — not in the Akama et al. upstream. Contrastive CLAP<->EEG
#   training on MAD-EEG (step B: wiring only -- does the loss go down?).
"""CLAP <-> EEG contrastive training on MAD-EEG duos.

A separate entrypoint, deliberately, exactly as `madeeg_reconstruction.py` is a separate
entrypoint for the ridge anchor. The Akama pipeline (`main.py` + `contrastive_learning.py`)
is hard-wired to four instrument slots at every level -- dataset tuple, LightningModule
`batch[:5]`, model forward, test records. A MAD-EEG duo has two sources. Bending that
pipeline to variable arity would be a large edit to the file that produces every Chapter 1
number, for no Chapter 1 benefit; keeping the two arms apart costs one small training loop
and risks nothing.

Reused unchanged: `CLAPEncoder` (frozen LAION-CLAP + trainable head), `SampleCNN2DEEG`
(channel-agnostic, so 20-channel EEG needs no change), `CLIP_Loss` (now slot-parametric),
`MadeegContrastiveDataset`.

What this step claims: **only that the model trains** -- the loss decreases in-distribution
and the trainable head moves. It reports no accuracy. The within-trial decision against the
competing source, and the comparison against chance (0.50 for duos) and against the linear
anchor, come after.

    python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --smoke     # CPU, ~2 min
    python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --epochs 10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.madeeg_contrastive_dataset import MadeegContrastiveDataset  # noqa: E402
from models import CLAPEncoder, SampleCNN2DEEG  # noqa: E402
from modules.clip_loss import CLIP_Loss  # noqa: E402


def split_by_trial(dataset, valid_frac=0.2, seed=42):
    """Hold out whole TRIALS, never windows.

    Windows of one trial overlap and share their audio, so splitting at window level would
    put near-duplicates of a training item in validation and make the loss look better than
    it is. Splitting by trial is the weakest split that is not self-deceiving; it is still
    *in-distribution* (same subjects, same mixtures) which is all this step needs -- the
    honest per-subject k-fold belongs to the evaluation step.
    """
    g = torch.Generator().manual_seed(seed)
    n_trials = len(dataset.trials)
    order = torch.randperm(n_trials, generator=g).tolist()
    n_valid = max(1, int(round(n_trials * valid_frac)))
    valid_trials = set(order[:n_valid])
    train_idx = [i for i, (t, _) in enumerate(dataset.index) if t not in valid_trials]
    valid_idx = [i for i, (t, _) in enumerate(dataset.index) if t in valid_trials]
    return Subset(dataset, train_idx), Subset(dataset, valid_idx), n_valid


def stratified_folds(dataset, n_folds, seed):
    """k folds over TRIALS, balanced within each subject.

    Splitting by trial and not by window is not a detail: windows of one trial overlap and
    share their audio, so a window-level split would score the model on near-duplicates of
    what it trained on. Stratifying by subject keeps every fold representative of all eight.
    Every trial lands in exactly one test fold, so the folds together yield one prediction
    per trial -- the same 154 independent decisions the ridge anchor reports.
    """
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]
    by_subject = {}
    for t, (subj, _) in enumerate(dataset.trials):
        by_subject.setdefault(subj, []).append(t)
    for subj in sorted(by_subject):
        trials = np.array(by_subject[subj])
        rng.shuffle(trials)
        for i, t in enumerate(trials):
            folds[i % n_folds].append(int(t))
    return [sorted(f) for f in folds]


@torch.no_grad()
def decide_per_trial(dataset, test_trials, encoder_eeg, encoder_audio, device, batch_size):
    """One decision per trial: is the attended source the closer one, on average?

    Per window we take the margin sim(EEG, attended) - sim(EEG, competitor); the trial is
    correct when the mean margin over its windows is positive. Averaging the margin rather
    than voting per window keeps a trial where every window leans slightly the right way
    from being decided by a couple of noisy ones -- and it is the trial, not the window,
    that is the independent unit.
    """
    encoder_eeg.eval()
    encoder_audio.eval()
    idx = [i for i, (t, _) in enumerate(dataset.index) if t in set(test_trials)]
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False)
    cos = torch.nn.CosineSimilarity(dim=1)

    margins = {}
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stems = batch["stems"].to(device)
        target = batch["target_idx"]
        z_eeg = encoder_eeg(eeg)
        sims = torch.stack([cos(z_eeg, encoder_audio(stems[:, s]))
                            for s in range(stems.shape[1])], dim=1)      # (B, n_present)
        for i in range(eeg.size(0)):
            attended = sims[i, target[i]]
            competitor = torch.cat([sims[i, :target[i]], sims[i, target[i] + 1:]]).max()
            margins.setdefault(batch["stim"][i], []).append(float(attended - competitor))
    return margins


def build_model(device, clap_hidden=256, clap_pretrained=""):
    """EEG encoder + a single shared CLAP head, both emitting 100-d embeddings.

    One shared audio encoder, as `main.py` does for audio_repr=clap: the stems are the same
    kind of object whichever slot they land in, and sharing keeps the trainable parameter
    count at one projection head. `SampleCNN2DEEG` ignores its `out_dim` argument -- its
    projector is fixed at 100 -- which is why the two sides meet without any adapter.
    """
    encoder_eeg = SampleCNN2DEEG(out_dim=100, kernal_size=3).to(device)
    encoder_audio = CLAPEncoder(out_dim=100, hidden_dim=clap_hidden,
                                pretrained=clap_pretrained).to(device)
    return encoder_eeg, encoder_audio


def run_epoch(loader, encoder_eeg, encoder_audio, criterion, optimizer, device, max_batches=None):
    train = optimizer is not None
    encoder_eeg.train(train)
    encoder_audio.train(train)
    total, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        eeg = batch["eeg"].to(device)
        stems = batch["stems"].to(device)              # (B, n_present, 1, samples)
        task = batch["target_idx"].to(device)
        with torch.set_grad_enabled(train):
            z_eeg = encoder_eeg(eeg)
            z_stems = [encoder_audio(stems[:, s]) for s in range(stems.shape[1])]
            # Slots 3 and 4 do not exist for a duo and are passed as None -- never as
            # silence, which would be a trivial negative and would flatter the result.
            z_stems += [None] * (4 - len(z_stems))
            # attention_score is an Akama construct (a self-reported 1-5 rating); MAD-EEG
            # has none. A constant outside `attention_values` makes the high-attention
            # branch inactive rather than pretending every trial was highly attended.
            att = torch.zeros(eeg.size(0), dtype=torch.long, device=device)
            out = criterion(z_eeg, *z_stems, task, att, [4, 5])
            loss = out["all"]["loss"]
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss) * eeg.size(0)
        seen += eeg.size(0)
    return total / max(seen, 1)


def run_kfold(ds, args, device):
    """Step C: one prediction per trial, k folds, pooled model. Reports against 0.50.

    The criterion was fixed in the vault BEFORE this function produced a number:
    >= 88/154 = 0.5714, one-sided binomial p < 0.05. Anything below is reported as not
    distinguishable from chance -- including anything that merely beats the linear anchor,
    whose own 0.5584 is itself not significant (p = 0.085).
    """
    from scipy import stats

    folds = stratified_folds(ds, args.kfold, args.seed)
    print(f"[kfold] {args.kfold} folds over {len(ds.trials)} trials, stratified by subject: "
          f"{[len(f) for f in folds]}")

    records = []
    for f, test_trials in enumerate(folds):
        torch.manual_seed(args.seed + f)          # fresh model per fold, deterministic
        train_idx = [i for i, (t, _) in enumerate(ds.index) if t not in set(test_trials)]
        loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size,
                            shuffle=True, drop_last=True, num_workers=args.workers)
        encoder_eeg, encoder_audio = build_model(device)
        trainable = [p for p in list(encoder_eeg.parameters()) + list(encoder_audio.parameters())
                     if p.requires_grad]
        criterion = CLIP_Loss(args.batch_size, args.temperature, world_size=1)
        optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

        t0 = time.time()
        for ep in range(args.epochs):
            tr = run_epoch(loader, encoder_eeg, encoder_audio, criterion, optimizer,
                           device, args.max_batches or None)
        margins = decide_per_trial(ds, test_trials, encoder_eeg, encoder_audio,
                                   device, args.batch_size)
        for t in test_trials:
            subj, stim = ds.trials[t]
            m = float(np.mean(margins[stim]))
            records.append({"fold": f, "subject": subj, "stim": stim,
                            "ensemble": ds.meta[subj][stim]["ensemble"],
                            "n_present": len(ds.meta[subj][stim]["instruments"]),
                            "mean_margin": m, "correct": int(m > 0)})
        acc = np.mean([r["correct"] for r in records if r["fold"] == f])
        print(f"  fold {f}: train_loss={tr:.4f}  test trials={len(test_trials)}  "
              f"acc={acc:.3f}  ({time.time() - t0:.0f}s)")

    n = len(records)
    k = sum(r["correct"] for r in records)
    test = stats.binomtest(k, n, 0.5, alternative="greater")
    lo, hi = stats.binomtest(k, n, 0.5).proportion_ci(0.95)
    threshold = next(x for x in range(n + 1)
                     if stats.binomtest(x, n, 0.5, alternative="greater").pvalue < 0.05)

    print(f"\n{'=' * 62}")
    if args.max_batches:
        print(f"!! --max_batches={args.max_batches}: UNDERTRAINED, this accuracy is NOT a result\n")
    print(f"[step C] {k}/{n} = {k / n:.4f}   chance 0.50")
    print(f"  one-sided binomial p = {test.pvalue:.4f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  pre-registered threshold: {threshold}/{n} = {threshold / n:.4f}")
    verdict = ("ABOVE CHANCE" if k >= threshold else
               "NOT DISTINGUISHABLE FROM CHANCE")
    print(f"  --> {verdict}")
    print(f"  (linear anchor, same 154 trials: 0.5584, p=0.085 -- itself not significant,")
    print(f"   so beating it is not the criterion)")

    print("\n  per subject (descriptive only -- one subject needs 0.737 to be significant):")
    for subj in sorted({r["subject"] for r in records}):
        rs = [r for r in records if r["subject"] == subj]
        print(f"    {subj}: {sum(r['correct'] for r in rs)}/{len(rs)} = "
              f"{np.mean([r['correct'] for r in rs]):.3f}")

    out_dir = os.path.join(_runs_dir(), args.training_date)
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "madeeg_contrastive_records.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(out_dir, "madeeg_contrastive_summary.txt"), "w") as fh:
        fh.write(f"== MAD-EEG contrastive CLAP<->EEG, step C ==\n"
                 f"ensemble={args.ensemble} folds={args.kfold} epochs={args.epochs} "
                 f"lr={args.learning_rate} batch={args.batch_size} seed={args.seed}\n"
                 f"n_trials={n} chance=0.500\n"
                 f"accuracy: {k}/{n} = {k / n:.4f}\n"
                 f"one-sided binomial p={test.pvalue:.4f}  95% CI=[{lo:.3f}, {hi:.3f}]\n"
                 f"pre-registered threshold {threshold}/{n}={threshold / n:.4f} -> {verdict}\n")
    print(f"\n  written to {out_dir}")
    return records


def _runs_dir():
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_paths", os.path.join(repo, "src", "utils", "paths.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_log_dir()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--madeeg_dir", required=True)
    ap.add_argument("--ensemble", default="duo", choices=["duo", "trio"],
                    help="one arity per run; mixing duos and trios needs variable arity")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=0.003)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--eeg_length", type=int, default=768)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs of 5 batches on CPU: proves it wires up, claims nothing")
    ap.add_argument("--kfold", type=int, default=0,
                    help="step C: k folds over trials, one prediction per trial, "
                         "reported against chance 0.50 with the pre-registered threshold")
    ap.add_argument("--training_date", default="madeeg_clap_kfold",
                    help="subdirectory under the run-output dir")
    ap.add_argument("--max_batches", type=int, default=0,
                    help="cap batches per epoch -- for checking the pipeline end to end "
                         "cheaply. A capped run is undertrained: its accuracy is not a result")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preload=True for the k-fold run: it is what makes shuffling affordable, and without
    # shuffling almost every batch comes from one trial -- so the InfoNCE batch negatives
    # would be the same audio as the positive. The smoke keeps the light path.
    ds = MadeegContrastiveDataset(args.madeeg_dir, ensemble=args.ensemble,
                                  eeg_length=args.eeg_length, stride=args.stride,
                                  preload=bool(args.kfold))
    if args.kfold:
        print(f"[madeeg-contrastive] device={device} ensemble={args.ensemble} "
              f"trials={len(ds.trials)} windows={len(ds)}")
        run_kfold(ds, args, device)
        return
    train_ds, valid_ds, n_valid = split_by_trial(ds, seed=args.seed)
    print(f"[madeeg-contrastive] device={device} ensemble={args.ensemble} "
          f"trials={len(ds.trials)} ({n_valid} held out for validation)")
    print(f"  windows: train={len(train_ds)} valid={len(valid_ds)}")

    # shuffle=False: the dataset caches two trials at a time, and shuffling would turn one
    # HDF5 read per trial into one per window (see MadeegContrastiveDataset._trial_arrays).
    # Consecutive windows of a trial are correlated, so this is a real trade-off, not a
    # free choice -- revisit with a hyperslab read before drawing conclusions from a run.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=args.workers)

    encoder_eeg, encoder_audio = build_model(device)
    trainable = [p for p in list(encoder_eeg.parameters()) + list(encoder_audio.parameters())
                 if p.requires_grad]
    frozen = sum(p.numel() for p in encoder_audio.parameters() if not p.requires_grad)
    print(f"  trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(frozen CLAP backbone: {frozen:,})")

    criterion = CLIP_Loss(args.batch_size, args.temperature, world_size=1)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    epochs, max_batches = (2, 5) if args.smoke else (args.epochs, None)
    head_before = encoder_audio.proj[0].weight.detach().clone()

    print(f"\n{'epoch':>6}{'train':>10}{'valid':>10}{'sec':>8}")
    history = []
    for ep in range(epochs):
        t0 = time.time()
        tr = run_epoch(train_loader, encoder_eeg, encoder_audio, criterion, optimizer,
                       device, max_batches)
        va = run_epoch(valid_loader, encoder_eeg, encoder_audio, criterion, None,
                       device, max_batches)
        history.append((tr, va))
        print(f"{ep:>6}{tr:>10.4f}{va:>10.4f}{time.time() - t0:>8.1f}")

    head_moved = float((encoder_audio.proj[0].weight - head_before).abs().max())
    print(f"\n[check] CLAP projection head moved by {head_moved:.2e} (must be > 0)")
    print(f"[check] train loss {history[0][0]:.4f} -> {history[-1][0]:.4f}")
    assert head_moved > 0, "the trainable head did not move -- nothing was learned"
    if args.smoke:
        assert all(torch.isfinite(torch.tensor(h)).all() for h in history), "non-finite loss"
        print("[smoke] PASS -- wiring is sound. This says nothing about accuracy.")


if __name__ == "__main__":
    main()
