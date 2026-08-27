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
    python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --kfold 5            # step C
    python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --self_test          # control
    python src/madeeg_contrastive.py --madeeg_dir ~/madeeg --kfold 5 \
        --loss within_mixture                                                   # step D

The post-mortem of a finished k-fold run lives in `madeeg_diagnose.py`, which needs neither
torch nor the HDF5 so that any number can be re-checked offline.
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
from madeeg_diagnose import trial_decision  # noqa: E402
from models import SampleCNN2DEEG  # noqa: E402
from modules.clip_loss import CLIP_Loss  # noqa: E402

# CHANGED(baseline): Exp. 18. The two match-mismatch stages are two more values of the
# EXISTING --loss flag, not a second program. What they change is where the wrong candidate
# comes from -- which is a property of the SAMPLER, not of the arithmetic: with two
# candidates and the positive in slot 0 the objective is the step-D one, unchanged
# (`CLIP_Loss` is not edited by this experiment, and its Chapter 1 canaries are untouched
# by construction). See src/datasets/madeeg_solo_matchmismatch.py.
MATCH_MISMATCH = ("temporal_offset", "cross_instrument")

# CHANGED(baseline): Exp. 19. Three numbers declared HERE, in the code, before the run
# (method rules 3/5). They come from the pre-registration of 13 Aug 2026 and none of them
# is tunable from the command line, precisely so that a run cannot quietly move one.
MM_GATE = 0.70              # §1 and §2.2 -- the SAME bar as Exp. 18. The budget changed;
#                             the criterion did not, and lowering it after seeing a number
#                             would be the move pre-registration exists to prevent.
NULL_CEILING = 0.52         # §2.1 -- above this the sampler's null is "dirty" and the gate
#                             stops meaning what it was written to mean, so it switches to
#                             "beat the MEASURED null with an exact one-sided binomial".
CONVERGED_TRAIN_LOSS = 0.50  # §1 -- the boundary between the contract's first two readings.
#                             Not a taste: 0.50 is the loss the contract's own arithmetic
#                             uses to size the 120-epoch budget (Exp. 18's measured slope of
#                             0.00166/epoch from 0.6962 reaches it in ~110 epochs). A run
#                             that early-stops still ABOVE it never fitted its training data
#                             -- "the optimiser finished and there was nothing to find". One
#                             that early-stops BELOW it did fit and failed to generalise --
#                             "273k parameters memorised 2700 pairs". Same stop, opposite
#                             cause, and the number that separates them is written first.


# The one sentence that has to survive the terminal (method rule 9), so it is written
# into the summary FILE above the numbers and not only printed.
EPOCH_RULE = (
    "!! THE REPORTED NUMBER IS THE ONE AT THE STOPPING EPOCH, and the stopping epoch is\n"
    "!! defined by the TRAIN loss alone. It is NOT the best held-out accuracy seen along\n"
    "!! the way, and the best held-out accuracy is not a result of this run at any epoch:\n"
    "!! picking the epoch by looking at the held-out set is selecting on the test set.\n"
    "!! The full curve below is printed for readability only -- every row of it except\n"
    "!! the last is a diagnostic, never a number to quote.")


def branch_of_contract(args, stopped_early, final_loss, acc, passed):
    """Which of the four readings the Exp. 19 contract §1 declared in advance has fired.

    Written as a table before this code existed, and the boundary between the
    first two is `CONVERGED_TRAIN_LOSS`, declared above for the reason given there.
    """
    if passed:
        return (f"held-out {acc:.4f} clears the gate: Exp. 18 was UNDER-TRAINED. "
                "The own-vs-other gauge follows, under a NEW pre-registration.")
    if not stopped_early:
        return (f"NOT CONCLUSIVE -- the train loss was still improving by more than "
                f"{args.early_stop_min_delta} when the {args.epochs}-epoch cap was reached "
                f"(last {final_loss:.4f}). The budget was not enough. No third run without "
                "a new contract.")
    if final_loss < CONVERGED_TRAIN_LOSS:
        return (f"PURE OVERFITTING -- the train loss converged well below "
                f"{CONVERGED_TRAIN_LOSS} ({final_loss:.4f}) and the held-out set stayed at "
                f"{acc:.4f}. The limit is the amount of data, now demonstrated instead of "
                "inferred.")
    return (f"THE QUESTION IS CLOSED: it was not the epochs. The optimiser stopped "
            f"improving at a train loss of {final_loss:.4f}, still at or above "
            f"{CONVERGED_TRAIN_LOSS} (chance = log 2 = 0.6931), and the held-out set is "
            f"{acc:.4f}. Exp. 18's negative becomes definitive and stronger, because it now "
            "carries the proof that the optimiser had finished.")


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
    """Per trial, the mean similarity of EVERY present stem over that trial's windows.

    The decision itself is `trial_decision`: the argmax of those means, correct when it is
    the attended stem. Averaging over windows rather than voting per window keeps a trial
    where every window leans slightly the right way from being decided by a couple of noisy
    ones -- and it is the trial, not the window, that is the independent unit.

    CHANGED(baseline): this used to accumulate the per-window margin
    sim(attended) - max sim(competitor) directly. On a duo the two are the same number,
    because the mean of a difference is the difference of the means -- verified against
    step C's archived records by `madeeg_diagnose.py --check_rule`, which must return
    58/154. Stated as per-stem means it also holds when three stems are present, where
    "average then take the best competitor" and "take the best competitor then average"
    part company.
    """
    encoder_eeg.eval()
    encoder_audio.eval()
    idx = [i for i, (t, _) in enumerate(dataset.index) if t in set(test_trials)]
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False)
    cos = torch.nn.CosineSimilarity(dim=1)

    # Keyed by (subject, stim), NOT by stim. A stimulus id is not unique: 36 distinct ids
    # cover the 154 duo trials, because several subjects heard the same excerpt with the
    # same target. Keying by stim alone merged those subjects' windows into one list and
    # handed every trial that shared an id the identical margin -- which silently destroys
    # the per-trial independence the whole binomial test rests on, and averages one
    # subject's EEG with another's. Found by checking the records for duplicate margins.
    per_window = {}
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stems = batch["stems"].to(device)
        z_eeg = encoder_eeg(eeg)
        sims = torch.stack([cos(z_eeg, encoder_audio(stems[:, s]))
                            for s in range(stems.shape[1])], dim=1)      # (B, n_present)
        for i in range(eeg.size(0)):
            key = (batch["subject"][i], batch["stim"][i])
            per_window.setdefault(key, []).append([float(x) for x in sims[i]])

    # One list per test trial, no merging. This assertion is the check that would have
    # caught the bug above at once instead of after a full GPU run.
    assert len(per_window) == len(test_trials), (
        f"{len(per_window)} similarity groups for {len(test_trials)} test trials -- "
        "decisions are being merged, so they are not independent")
    return {k: [sum(col) / len(col) for col in zip(*v)] for k, v in per_window.items()}


class _FrozenRandomAudio(torch.nn.Module):
    """CHANGED(baseline): CPU stand-in for the FROZEN CLAP tower. Wiring checks ONLY.

    The LAION-CLAP checkpoint is 1.74 GiB and is not on the CPU box (fetching it is A.'s
    decision -- Exp. 16A), and that box has no torchaudio either, so `CLAPEncoder` cannot be
    constructed there at all. What the optimisation-sanity check of Exp. 18 §4.3 has to
    exercise is sampler -> loss -> optimizer -> decision rule; CLAP contributes no gradient
    to any of it, because it is frozen. So the tower is replaced by a FIXED (seeded, never
    updated) projection of a coarse envelope of the waveform to the same 512 dims, and the
    trainable head is byte-for-byte the same module CLAPEncoder carries.

    Anything this produces is a statement about the wiring and never about CLAP's features
    or about the data -- which is why `--clap_stub` writes that sentence into the summary
    FILE and not only onto the terminal.
    """

    CLAP_EMBED_DIM = 512

    def __init__(self, out_dim=100, hidden_dim=256, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("mix", torch.randn(self.CLAP_EMBED_DIM, self.CLAP_EMBED_DIM,
                                                generator=g) / self.CLAP_EMBED_DIM ** 0.5)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.CLAP_EMBED_DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        env = torch.nn.functional.adaptive_avg_pool1d(x.abs(), self.CLAP_EMBED_DIM).squeeze(1)
        env = (env - env.mean(dim=1, keepdim=True)) / (env.std(dim=1, keepdim=True) + 1e-8)
        return self.proj(torch.tanh(env @ self.mix))


def build_model(device, clap_hidden=256, clap_pretrained="", clap_stub=False):
    """EEG encoder + a single shared CLAP head, both emitting 100-d embeddings.

    One shared audio encoder, as `main.py` does for audio_repr=clap: the stems are the same
    kind of object whichever slot they land in, and sharing keeps the trainable parameter
    count at one projection head. `SampleCNN2DEEG` ignores its `out_dim` argument -- its
    projector is fixed at 100 -- which is why the two sides meet without any adapter.
    """
    encoder_eeg = SampleCNN2DEEG(out_dim=100, kernal_size=3).to(device)
    if clap_stub:
        encoder_audio = _FrozenRandomAudio(out_dim=100, hidden_dim=clap_hidden).to(device)
    else:
        # Imported here, not at module load: the import pulls torchaudio, and every
        # EEG-side check on a CPU box would otherwise fail for a reason unrelated to it.
        from models import CLAPEncoder
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
    """One prediction per trial, k folds, pooled model. Reports against 0.50.

    The criterion was fixed in writing BEFORE this function produced a number:
    >= 88/154 = 0.5714, one-sided binomial p < 0.05. Anything below is reported as not
    distinguishable from chance -- including anything that merely beats the linear anchor,
    whose own 0.5584 is itself not significant (p = 0.085).

    `--loss within_mixture` makes this step D. Everything else is frozen at step C's
    values -- epochs, lr, batch, temperature, seed, folds, decision unit -- so any
    difference is attributable to the one factor that changed. Step D is a SECOND LOOK at
    the same trials, chosen after step C's number was known: the same 0.5714 is printed,
    but as a descriptor, and the summary says so in the file itself.
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
        encoder_eeg, encoder_audio = build_model(device, clap_pretrained=args.clap_pretrained)
        trainable = [p for p in list(encoder_eeg.parameters()) + list(encoder_audio.parameters())
                     if p.requires_grad]
        criterion = CLIP_Loss(args.batch_size, args.temperature, world_size=1,
                              negatives=args.loss)
        optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

        t0 = time.time()
        losses = []
        for ep in range(args.epochs):
            tr = run_epoch(loader, encoder_eeg, encoder_audio, criterion, optimizer,
                           device, args.max_batches or None)
            losses.append(tr)
        mean_sims = decide_per_trial(ds, test_trials, encoder_eeg, encoder_audio,
                                     device, args.batch_size)
        for t in test_trials:
            subj, stim = ds.trials[t]
            meta = ds.meta[subj][stim]
            instruments = list(meta["instruments"])
            m = trial_decision(mean_sims[(subj, stim)], instruments.index(meta["target"]))
            records.append({"fold": f, "subject": subj, "stim": stim,
                            "ensemble": meta["ensemble"],
                            "n_present": len(instruments),
                            "mean_margin": m, "correct": int(m > 0)})
        acc = np.mean([r["correct"] for r in records if r["fold"] == f])
        # The first epoch's loss is printed too, because its SCALE is a wiring check: with
        # within-mixture negatives a duo has 2 logits, so an untrained model sits at
        # log 2 = 0.693. Batch negatives put it near 3. A first epoch far from the expected
        # value means the mode is not the one that was asked for.
        print(f"  fold {f}: train_loss={losses[0]:.4f}->{tr:.4f}  "
              f"test trials={len(test_trials)}  acc={acc:.3f}  ({time.time() - t0:.0f}s)")

    n = len(records)
    k = sum(r["correct"] for r in records)
    test = stats.binomtest(k, n, 0.5, alternative="greater")
    lo, hi = stats.binomtest(k, n, 0.5).proportion_ci(0.95)
    threshold = next(x for x in range(n + 1)
                     if stats.binomtest(x, n, 0.5, alternative="greater").pvalue < 0.05)

    step = "D" if args.loss == "within_mixture" else "C"
    print(f"\n{'=' * 62}")
    if args.max_batches:
        print(f"!! --max_batches={args.max_batches}: UNDERTRAINED, this accuracy is NOT a result\n")
    if getattr(args, "self_test", False):
        print("!! --self_test: the EEG is SYNTHETIC. This accuracy is a control, NOT a result\n")
    if step == "D":
        print("!! step D is a SECOND LOOK at the same 154 trials, decided after seeing step\n"
              "!! C's number. It is EXPLORATORY BY CONSTRUCTION -- no threshold reached here\n"
              "!! makes it confirmatory, whatever it says. Step C's 58/154 verdict stands.\n")
    print(f"[step {step}] {k}/{n} = {k / n:.4f}   chance 0.50   (loss negatives: {args.loss})")
    print(f"  one-sided binomial p = {test.pvalue:.4f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  pre-registered threshold: {threshold}/{n} = {threshold / n:.4f}")
    verdict = ("ABOVE CHANCE" if k >= threshold else
               "NOT DISTINGUISHABLE FROM CHANCE")
    print(f"  --> {verdict}" + ("   (descriptive: step D cannot be confirmatory)"
                                if step == "D" else ""))
    print(f"  (linear anchor, same 154 trials: 0.5584, p=0.085 -- itself not significant,")
    print(f"   so beating it is not the criterion)")

    if getattr(args, "self_test", False):
        # Threshold declared here, in the code, before the control was ever run, and not
        # guessed: applying THIS decision rule (mean margin over a trial's windows) directly
        # to the envelopes decides 154/154 of the duo trials correctly, so the signal is
        # fully separable in principle and only the wiring is on trial.
        passed = k / n >= 0.90
        print(f"\n[self-test] {'PASS' if passed else 'FAIL'} -- synthetic EEG tracking the "
              f"attended source, {k}/{n} = {k / n:.4f} against a threshold of 0.90")
        print("  PASS: this loss, this decision rule and this k-fold do recover a")
        print("        source-tracking signal. It says nothing about real EEG.")
        print("  FAIL: the defect is in the wiring, not in the data.")

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
    # The banner has to live in the FILE, not only on stdout: the summary outlives the
    # terminal, and a file that reads "step C ... ABOVE CHANCE" with no mention of the
    # synthetic EEG is a number waiting to be quoted as a result by mistake.
    caveat = ""
    if getattr(args, "self_test", False):
        caveat = (f"!! POSITIVE CONTROL -- the EEG is SYNTHETIC (a fixed linear operator on\n"
                  f"!! the attended source's envelope, snr={args.self_test_snr}). This accuracy is a\n"
                  f"!! control on the wiring, NOT a result, and NOT a statement about real EEG.\n"
                  f"!! The 'ABOVE CHANCE' verdict below is the control's, not step C's.\n\n")
    if args.max_batches:
        caveat += (f"!! --max_batches={args.max_batches}: UNDERTRAINED by construction.\n"
                   f"!! This accuracy is not a result.\n\n")
    if step == "D":
        caveat += ("!! STEP D -- within-mixture negatives. A SECOND LOOK at the same 154\n"
                   "!! trials, decided after step C's number was known, therefore\n"
                   "!! EXPLORATORY BY CONSTRUCTION. The threshold below is a descriptor,\n"
                   "!! not a confirmatory verdict; step C's 58/154 = 0.3766 stands.\n\n")
    with open(os.path.join(out_dir, "madeeg_contrastive_summary.txt"), "w") as fh:
        fh.write(caveat)
        fh.write(f"== MAD-EEG contrastive CLAP<->EEG, step {step} ==\n"
                 f"ensemble={args.ensemble} folds={args.kfold} epochs={args.epochs} "
                 f"lr={args.learning_rate} batch={args.batch_size} seed={args.seed} "
                 f"loss={args.loss}\n"
                 f"n_trials={n} chance=0.500\n"
                 f"accuracy: {k}/{n} = {k / n:.4f}\n"
                 f"one-sided binomial p={test.pvalue:.4f}  95% CI=[{lo:.3f}, {hi:.3f}]\n"
                 f"pre-registered threshold {threshold}/{n}={threshold / n:.4f} -> {verdict}\n")
    print(f"\n  written to {out_dir}")
    return records


@torch.no_grad()
def match_mismatch_records(subset, encoder_eeg, encoder_audio, device, batch_size):
    """One decision per pair: is the matched candidate the more similar of the two?

    The rule is the step-D margin with the competing stem replaced by the mismatched
    candidate -- sim(matched) - sim(mismatched) -- so training and evaluation optimise the
    same function, exactly as they do for the duo (see `compute_task_loss`). Slot 0 is
    always the matched one, which is not a shortcut: both rows go through the SAME audio
    encoder and are compared by cosine, so slot position carries no parameter.
    """
    encoder_eeg.eval()
    encoder_audio.eval()
    cos = torch.nn.CosineSimilarity(dim=1)
    records = []
    for batch in DataLoader(subset, batch_size=batch_size, shuffle=False):
        z_eeg = encoder_eeg(batch["eeg"].to(device))
        stems = batch["stems"].to(device)
        margin = cos(z_eeg, encoder_audio(stems[:, 0])) - cos(z_eeg, encoder_audio(stems[:, 1]))
        for i in range(margin.size(0)):
            records.append({"subject": batch["subject"][i], "stim": batch["stim"][i],
                            "rep": int(batch["rep"][i]),
                            "pos_instrument": batch["pos_instrument"][i],
                            "neg_instrument": batch["neg_instrument"][i],
                            "pos_start_s": float(batch["pos_start_s"][i]),
                            "neg_start_s": float(batch["neg_start_s"][i]),
                            "margin": float(margin[i]), "correct": int(float(margin[i]) > 0)})
    return records


def run_matchmismatch(ds, args, device):
    """Exp. 18, stages S1/S2: train on solo match-mismatch pairs, score held-out recordings.

    The gate was written down BEFORE this function existed (Exp. 18 §4.2): held-out
    accuracy >= 0.70 against a chance of 0.500. Below it the encoder learned nothing and the
    contract stops -- no own-vs-other gauge, no claim, no second look.

    Two properties this function will not let itself get wrong:

    * The held-out unit is the RECORDING -- one (subject, solo key), with all four of its
      presented repetitions travelling together. The four repetitions are the SAME six
      seconds of audio played four times, so splitting between them would put a near-copy
      of a training item in the test set. `split_by_trial` already holds out whole trials
      and is reused unchanged; the solo dataset just declares its recordings as `trials`.
    * The null is PRINTED, never assumed (method rule 4). For `temporal_offset` the index
      carries every (start, start) pair in both directions, so the best rule that ignores
      the EEG scores exactly 0.500 -- and the number is computed rather than asserted. For
      `cross_instrument` the subjects did not all hear the same solos, so it is NOT 0.500
      and the measured value is what the accuracy has to be read against.

    CHANGED(baseline): Exp. 19 (pre-registration, 13 Aug 2026) adds two things and only two,
    both behind flags that default to off so an Exp. 18 command still reproduces Exp. 18:

    * `--early_stop_patience` / `--early_stop_min_delta` (§1): a 120-epoch cap with early
      stopping on the TRAIN loss. This changes the compute budget, NOT the criterion --
      the bar is still 0.70. The reported number is the one at the stopping epoch.
    * `--balance_pairs` (§2.1): the repair of S2's null, which Exp. 18 measured at 0.6061 on
      the held-out pairs, i.e. a "gate" a model that never looks at the EEG could nearly
      walk through. Nothing about learning rate, architecture or temperature moves: one
      variable at a time, and here the variable is the budget plus that repair.
    """
    from scipy import stats

    train_ds, valid_ds, n_valid = split_by_trial(ds, valid_frac=args.valid_frac, seed=args.seed)
    # CHANGED(baseline): Exp. 19 §2.1 -- repair the null BEFORE training, per split.
    # Per split and not once over the whole index, because the two are different numbers:
    # Exp. 18 measured 0.5063 over all of S2 and 0.6061 on the held-out side of the very
    # same index, and it is the held-out one the gate reads. Off by default (method rule 8), and a provable no-op on S1 anyway -- see `balanced_indices`.
    if args.balance_pairs:
        tr_ids, va_ids = ds.balanced_indices(train_ds.indices), ds.balanced_indices(valid_ds.indices)
        print(f"[balance] train {len(train_ds)} -> {len(tr_ids)} pairs, "
              f"held-out {len(valid_ds)} -> {len(va_ids)} pairs "
              f"({len({ds.pairs[j]['rec'] for j in va_ids})} of {n_valid} held-out "
              "recordings still represented)")
        train_ds, valid_ds = Subset(ds, tr_ids), Subset(ds, va_ids)
    if args.overfit:
        # Sanity of the optimisation (Exp. 18 §4.3), NOT a result: memorise a tiny subset.
        # Spread across the index so the subset spans several recordings -- memorising one
        # recording's windows would be a weaker check than the contract asks for.
        step = max(1, len(ds) // args.overfit)
        ids = list(range(0, len(ds), step))[:args.overfit]
        train_ds = valid_ds = Subset(ds, ids)
        print(f"[overfit] {len(ids)} items drawn every {step}th from {len(ds)}: "
              f"a WIRING check, never a result")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers)
    torch.manual_seed(args.seed)
    encoder_eeg, encoder_audio = build_model(device, clap_pretrained=args.clap_pretrained,
                                             clap_stub=args.clap_stub)
    trainable = [p for p in list(encoder_eeg.parameters()) + list(encoder_audio.parameters())
                 if p.requires_grad]
    # Two candidates, positive in slot 0: that IS the within-mixture objective. The negative
    # construction moved into the sampler; the arithmetic did not move at all.
    criterion = CLIP_Loss(args.batch_size, args.temperature, world_size=1,
                          negatives="within_mixture")
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    print(f"[{args.loss}] train pairs={len(train_ds)} held-out pairs={len(valid_ds)} "
          f"({n_valid} of {len(ds.trials)} recordings held out)  "
          f"trainable params: {sum(p.numel() for p in trainable):,}")

    # Gate 3 of the Exp. 19 contract: the null is re-measured and printed BEFORE a single
    # gradient step, so it cannot be quietly read off after the accuracy is known.
    null, n_null = ds.pairwise_prior_null(valid_ds.indices)
    null_subj, _ = ds.pairwise_prior_null(valid_ds.indices, by_subject=True)
    print(f"[null] best EEG-free rule on these {n_null} held-out pairs: {null:.4f} "
          f"(candidate identity)   {null_subj:.4f} (subject + candidate identity)"
          + ("" if null <= NULL_CEILING else
             f"   !! above the {NULL_CEILING} ceiling: the gate becomes 'beat {null:.4f}'"))

    print(f"\n{'epoch':>6}{'train_loss':>12}{'held_out_acc':>14}{'sec':>8}")
    curve, best_loss, stale, stopped_early = [], float("inf"), 0, False
    for ep in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(loader, encoder_eeg, encoder_audio, criterion, optimizer,
                       device, args.max_batches or None)
        records = match_mismatch_records(valid_ds, encoder_eeg, encoder_audio,
                                         device, args.batch_size)
        acc = sum(r["correct"] for r in records) / len(records)
        curve.append((ep, tr, acc))
        print(f"{ep:>6}{tr:>12.4f}{acc:>14.4f}{time.time() - t0:>8.1f}")
        # CHANGED(baseline): Exp. 19 §1 -- early stop on the TRAIN loss, never on the
        # held-out accuracy. Stopping on held-out would be choosing the reported epoch by
        # looking at the test set, which is the same class of error as flipping the sign;
        # the train loss is a property of the optimiser alone and knows nothing about the
        # number being gated. Off by default, so an Exp. 18 command still runs Exp. 18.
        if args.early_stop_patience:
            if best_loss - tr > args.early_stop_min_delta:
                best_loss, stale = tr, 0
            else:
                stale += 1
                if stale >= args.early_stop_patience:
                    stopped_early = True
                    print(f"[early stop] no train-loss improvement > "
                          f"{args.early_stop_min_delta} for {args.early_stop_patience} "
                          f"epochs; stopping at epoch {ep} of a {args.epochs}-epoch cap")
                    break

    n = len(records)
    k = sum(r["correct"] for r in records)
    test = stats.binomtest(k, n, null, alternative="greater")

    # The honest independent unit. Pairs inside one recording reuse the same EEG windows,
    # so the binomial over pairs is anti-conservative and is printed as a descriptor only.
    by_rec = {}
    for r in records:
        by_rec.setdefault((r["subject"], r["stim"]), []).append(r["correct"])
    rec_acc = [sum(v) / len(v) for v in by_rec.values()]

    print(f"\n{'=' * 62}")
    if args.clap_stub:
        print("!! --clap_stub: the frozen CLAP tower is a RANDOM PROJECTION. This number is\n"
              "!! a wiring check, NOT a result, and says nothing about CLAP or about EEG.\n")
    if args.overfit:
        print(f"!! --overfit {args.overfit}: trained and scored on the SAME items. This is the\n"
              "!! optimisation-sanity check of Exp. 18 §4.3; ~1.0 means the wiring can learn,\n"
              "!! and nothing else. It is NOT held-out accuracy.\n")
    if args.max_batches:
        print(f"!! --max_batches={args.max_batches}: UNDERTRAINED, this accuracy is NOT a result\n")
    if args.early_stop_patience:
        print(EPOCH_RULE + "\n")
    print(f"[{args.loss}] {k}/{n} = {k / n:.4f}   "
          f"null {null:.4f} (best rule that ignores the EEG, measured on these {n_null} pairs)"
          f"   {null_subj:.4f} (same, allowed to condition on the subject)")
    print(f"  one-sided binomial vs the measured null {null:.4f}: p = {test.pvalue:.4g}  "
          f"(descriptive: pairs inside a recording share EEG windows)")
    print(f"  per held-out recording: {len(rec_acc)} recordings, mean {np.mean(rec_acc):.4f}, "
          f"{sum(a > 0.5 for a in rec_acc)}/{len(rec_acc)} above 0.5")
    print(f"  reported at epoch {curve[-1][0]} of {len(curve)} run "
          f"(cap {args.epochs}); train loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f}"
          + ("  [early stop fired]" if stopped_early else ""))

    # The gate. Which of the two forms applies is decided by the MEASURED null, exactly as
    # contract §2.1 writes it -- never by which one the accuracy would pass.
    if null <= NULL_CEILING:
        gate_text = (f"held-out accuracy >= {MM_GATE:.2f}, against a measured null of "
                     f"{null:.4f} <= {NULL_CEILING}")
        passed = k / n >= MM_GATE
    else:
        gate_text = (f"null {null:.4f} is ABOVE the {NULL_CEILING} ceiling, so the gate is "
                     f"'beat it, exact one-sided binomial p <= 0.05' (p = {test.pvalue:.4g})"
                     f"; the absolute {MM_GATE:.2f} bar is reported alongside, not instead")
        passed = test.pvalue <= 0.05
    # A control never writes over a result (method rule 9): an overfit run is scored on
    # its own training items, so it has no held-out accuracy and the gate does not apply to
    # it. Printing "PASSED" there would leave a line waiting to be quoted by mistake.
    verdict = ("not applicable -- this run has no held-out set" if args.overfit else
               "PASSED" if passed else "NOT PASSED -- the contract stops here")
    print(f"  pre-registered gate: {gate_text}\n  -->  {verdict}")

    # CHANGED(baseline): Exp. 19 §1 -- say WHICH of the four declared readings fired, so the
    # log cannot be read as any of the other three later on.
    reading = branch_of_contract(args, stopped_early, curve[-1][1], k / n, passed)
    if args.early_stop_patience and not args.overfit:
        print(f"\n  [Exp. 19 §1] {reading}")

    out_dir = os.path.join(_runs_dir(), args.training_date)
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "madeeg_matchmismatch_records.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    caveat = ""
    if args.clap_stub:
        caveat += ("!! --clap_stub -- the FROZEN CLAP tower was replaced by a fixed random\n"
                   "!! projection because the checkpoint is not on this machine. Every number\n"
                   "!! below is a check on the WIRING, not a result, and not a statement\n"
                   "!! about CLAP features or about EEG.\n\n")
    if args.overfit:
        caveat += (f"!! --overfit {args.overfit} -- trained and scored on the SAME {args.overfit}\n"
                   "!! items. This is the optimisation-sanity check (Exp. 18 §4.3). ~1.0 means\n"
                   "!! the wiring can learn. It is NOT held-out accuracy and NOT a result.\n\n")
    if args.max_batches:
        caveat += (f"!! --max_batches={args.max_batches}: UNDERTRAINED by construction.\n"
                   f"!! This accuracy is not a result.\n\n")
    # CHANGED(baseline): Exp. 19. Two caveats that have to be INSIDE the file, above the
    # numbers, because the file outlives the terminal (method rule 9).
    if args.early_stop_patience:
        caveat += EPOCH_RULE + "\n\n"
    if args.loss == "cross_instrument":
        caveat += ("!! STAGE S2 IS EXPLORATORY BY CONSTRUCTION. Exp. 18 §3 subordinated it\n"
                   "!! to S1; the decision to run it anyway was taken AFTER S1's number was\n"
                   "!! known (Exp. 19 §0). Nothing in this file can support a confirmatory\n"
                   "!! claim, whatever it says. A success here would need a fresh\n"
                   "!! pre-registered replication on material not yet looked at.\n\n")
    with open(os.path.join(out_dir, "madeeg_matchmismatch_summary.txt"), "w") as fh:
        fh.write(caveat)
        fh.write(f"== MAD-EEG solo match-mismatch, negatives={args.loss} ==\n"
                 f"epoch_cap={args.epochs} lr={args.learning_rate} batch={args.batch_size} "
                 f"seed={args.seed} temperature={args.temperature} valid_frac={args.valid_frac}\n"
                 f"early_stop: patience={args.early_stop_patience} "
                 f"min_delta={args.early_stop_min_delta} on the TRAIN loss "
                 f"-> {'fired' if stopped_early else 'did not fire'}\n"
                 f"balance_pairs={args.balance_pairs} "
                 f"(Exp. 19 §2.1 null repair; a no-op on temporal_offset)\n"
                 f"window={ds.eeg_length / 256:.1f}s stride={ds.stride / 256:.1f}s "
                 f"min_offset={ds.min_offset_s:.1f}s band={ds.band} Hz\n"
                 f"recordings={len(ds.trials)} repetitions={len(ds.reps)} pairs={len(ds)} "
                 f"held_out_recordings={n_valid}\n"
                 f"\nSTOPPING EPOCH {curve[-1][0]} of {len(curve)} run, cap {args.epochs}\n"
                 f"accuracy AT THE STOPPING EPOCH: {k}/{n} = {k / n:.4f}\n"
                 f"null (best EEG-free rule, measured on the held-out pairs): {null:.4f} "
                 f"by candidate identity, {null_subj:.4f} conditioning on the subject too\n"
                 f"one-sided binomial vs the measured null {null:.4f}: p={test.pvalue:.4g} "
                 f"(descriptive: pairs share EEG windows)\n"
                 f"per-recording mean over {len(rec_acc)} held-out recordings: "
                 f"{np.mean(rec_acc):.4f}\n"
                 f"train loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f}\n"
                 f"pre-registered gate: {gate_text}\n  -> {verdict}\n")
        if args.early_stop_patience and not args.overfit:
            fh.write(f"Exp. 19 §1 reading: {reading}\n")
        # The full curve, in the file: it is what tells a later reader whether the loss had
        # flattened or was still falling, and it is the evidence behind the reading above.
        fh.write("\nfull curve (diagnostic only -- the result is the LAST row):\n"
                 " epoch  train_loss  held_out_acc\n")
        for e, t, a in curve:
            fh.write(f"{e:>6}{t:>12.4f}{a:>14.4f}\n")
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
    ap.add_argument("--self_test", action="store_true",
                    help="POSITIVE CONTROL: swap every trial's EEG for a synthetic signal "
                         "that tracks ITS attended source, then run the k-fold unchanged. "
                         "Its accuracy is a control, never a result")
    ap.add_argument("--self_test_snr", type=float, default=4.0,
                    help="--self_test only: signal/noise std ratio of the synthetic EEG")
    ap.add_argument("--loss", default="batch",
                    choices=["batch", "within_mixture", *MATCH_MISMATCH],
                    help="where the negatives come from. 'batch' is step C and stays the "
                         "default so no reported number moves. 'within_mixture' is step D: "
                         "the only negatives are the competing stems of the SAME trial. "
                         "'temporal_offset' and 'cross_instrument' are Exp. 18 on the SOLOS "
                         "(match-mismatch): same recording at a declared offset, and a "
                         "different instrument playing the same piece")
    # CHANGED(baseline): Exp. 18 -- the three flags the two new --loss values need.
    ap.add_argument("--valid_frac", type=float, default=0.2,
                    help="Exp. 18 only: fraction of solo RECORDINGS held out (all four "
                         "repetitions of a recording travel together)")
    ap.add_argument("--overfit", type=int, default=0,
                    help="Exp. 18 §4.3 sanity: train AND score on this many items. ~1.0 "
                         "means the wiring can learn; it is never a result")
    ap.add_argument("--clap_pretrained", default="",
                    help="path to a local LAION-CLAP .pt. Empty (the default) is what steps "
                         "C and D used -- laion_clap's own bundled checkpoint -- so passing "
                         "nothing changes nothing; on a cluster, pointing at the file that "
                         "is already on disk saves a 1.74 GiB download at run time")
    # CHANGED(baseline): Exp. 19 -- the epoch budget and the null repair. Both default to
    # OFF, so every Exp. 18 command still runs Exp. 18 (method rule 8).
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="Exp. 19 §1: stop after this many epochs without a TRAIN-loss "
                         "improvement greater than --early_stop_min_delta. 0 (the default) "
                         "means no early stopping, i.e. exactly the Exp. 18 behaviour. The "
                         "reported number is always the one at the stopping epoch, never "
                         "the best held-out accuracy seen along the way")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.002,
                    help="Exp. 19 §1: the smallest TRAIN-loss drop that counts as progress")
    ap.add_argument("--balance_pairs", action="store_true",
                    help="Exp. 19 §2.1: keep every candidate pair in both directions the "
                         "same number of times, per subject, so the best rule that ignores "
                         "the EEG scores exactly 0.500. A repair of the null, not a "
                         "hyperparameter; provably a no-op on --loss temporal_offset")
    ap.add_argument("--clap_stub", action="store_true",
                    help="replace the FROZEN CLAP tower with a fixed random projection. For "
                         "CPU wiring checks where the 1.74 GiB checkpoint is absent. Every "
                         "number it produces is marked as a check inside the summary file")
    args = ap.parse_args()

    # method rule 8: a value that is not one of the four must BREAK. argparse already
    # rejects a typo; this second gate is here because the failure mode that cost us a
    # write-up was a mode silently behaving like the previous one, and it is cheap.
    assert args.loss in ("batch", "within_mixture", *MATCH_MISMATCH), \
        f"unknown --loss {args.loss!r}"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A control must never overwrite the records of the real run -- and neither must a
    # different objective. Step D writes beside step C, not over it.
    if args.self_test:
        args.kfold = args.kfold or 5
    if args.training_date == "madeeg_clap_kfold":
        if args.self_test:
            args.training_date = "madeeg_clap_selftest"
        if args.loss == "within_mixture":
            args.training_date += "_within"
        if args.loss in MATCH_MISMATCH:
            # A control never writes over a result, and neither does a different budget
            # (method rule 9): the Exp. 19 run lands beside the Exp. 18 one, not on it.
            exp = "exp19" if (args.early_stop_patience or args.balance_pairs) else "exp18"
            args.training_date = f"madeeg_{exp}_{args.loss}"
            if args.overfit:
                args.training_date += "_overfit"
            if args.clap_stub:
                args.training_date += "_clapstub"

    # CHANGED(baseline): Exp. 18. A different dataset (the solos live only in the RAW
    # release) feeding the SAME encoders, the SAME loss and the SAME training loop. Nothing
    # above this branch is reachable with these two --loss values, and nothing below it is
    # reachable with the other two -- so `--loss batch` is bit-for-bit what it was.
    if args.loss in MATCH_MISMATCH:
        assert not args.kfold and not args.self_test, (
            "--kfold and --self_test are the duo k-fold of steps C/D; Exp. 18 trains on the "
            "SOLOS and holds out recordings. Drop them.")
        assert args.ensemble == "duo", (
            "--ensemble has no meaning on the solos and the contract forbids opening duos "
            "or trios here; leave it at its default")
        from datasets.madeeg_solo_matchmismatch import MadeegSoloMatchMismatch
        ds = MadeegSoloMatchMismatch(args.madeeg_dir, negative=args.loss)
        print(f"[madeeg-contrastive] device={device} Exp.18 negatives={args.loss}")
        run_matchmismatch(ds, args, device)
        return

    # preload=True for the k-fold run: it is what makes shuffling affordable, and without
    # shuffling almost every batch comes from one trial -- so the InfoNCE batch negatives
    # would be the same audio as the positive. The smoke keeps the light path.
    ds = MadeegContrastiveDataset(args.madeeg_dir, ensemble=args.ensemble,
                                  eeg_length=args.eeg_length, stride=args.stride,
                                  preload=bool(args.kfold), synthetic_eeg=args.self_test,
                                  synth_snr=args.self_test_snr, seed=args.seed)
    if args.kfold:
        print(f"[madeeg-contrastive] device={device} ensemble={args.ensemble} "
              f"trials={len(ds.trials)} windows={len(ds)}"
              + (f" SYNTHETIC EEG (snr={args.self_test_snr})" if args.self_test else ""))
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

    encoder_eeg, encoder_audio = build_model(device, clap_pretrained=args.clap_pretrained)
    trainable = [p for p in list(encoder_eeg.parameters()) + list(encoder_audio.parameters())
                 if p.requires_grad]
    frozen = sum(p.numel() for p in encoder_audio.parameters() if not p.requires_grad)
    print(f"  trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(frozen CLAP backbone: {frozen:,})")

    criterion = CLIP_Loss(args.batch_size, args.temperature, world_size=1,
                          negatives=args.loss)
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
