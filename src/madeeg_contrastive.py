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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MadeegContrastiveDataset(args.madeeg_dir, ensemble=args.ensemble,
                                  eeg_length=args.eeg_length, stride=args.stride)
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
