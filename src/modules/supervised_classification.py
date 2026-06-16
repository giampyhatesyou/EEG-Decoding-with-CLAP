# CHANGED(baseline): NEW FILE — not present in the Akama et al. upstream. Diagnostic extension.
"""Unimodal supervised classification baselines (supervisor-proposed diagnostics).

These are the diagnostic experiments #1 and #2, deliberately
*outside* the minimal-change contrastive baseline. They live in their own
``LightningModule`` so the contrastive loss (``clip_loss.py::compute_task_loss``) and the
contrastive evaluation metric (``EEGContrastiveLearning.compute_evaluation_matrix``) are
never touched. Selection happens in ``main.py`` / ``checkpoint_test.py`` via the
``objective`` flag:

  ``objective="classify_eeg"``   -> EEG encoder -> Linear(100->4) -> CrossEntropy(task)   [#2]
  ``objective="classify_audio"`` -> 4 audio-stem encoders -> concat(4x100) -> Linear(400->4)
                                    -> CrossEntropy(task)                                  [#1]

Scientific reading (see docs/METHODOLOGY.md):

* **#2 EEG-only** forces the EEG encoder to build label-discriminative features on its own.
  High argmax accuracy => attention is decodable from EEG alone; chance-level (~25%) =>
  the contrastive model's apparent success leaned on the audio side / song structure.

* **#1 audio-only is a NEGATIVE CONTROL.** The classifier sees the song's four stems with
  *no* cue about what the subject attended; the stems are fixed per song while the label
  varies per trial, so the experiment *should* sit at chance (~25%). Above-chance accuracy
  is **not** performance — it quantifies song->label leakage in the split.

The metric here (4-class argmax accuracy, chance = 25%) is a **different quantity** from the
contrastive pairwise metric (also ~25% chance but a different definition). The two are
reported separately and must not be conflated.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from utils import get_logger


class SupervisedClassification(LightningModule):
    """Supervised 4-class classifier over a single modality.

    Parameters
    ----------
    preprocess_dataset, key:
        kept for signature parity with ``EEGContrastiveLearning``; unused by the loss.
    args:
        the parsed config Namespace, saved as hyperparameters so the run's
        ``hparams.yaml`` records ``objective`` / ``audio_repr`` / ``cv_*`` (read back by
        ``run.py``'s checkpoint-matching guard).
    modality:
        ``"eeg"``   -> use ``encoder_eeg`` only.
        ``"audio"`` -> use the four audio-stem encoders.
    encoder_*:
        the *same* encoder instances ``main.py`` builds for the contrastive path, so a
        ``classify_audio`` checkpoint stays compatible with the chosen ``audio_repr``.
    """

    debug_logger = get_logger("dataloader_debug")
    NUM_CLASSES = 4
    EMBED_DIM = 100  # SampleCNN2DEEG.projector2 / CLAPEncoder out_dim
    TASK_NAMES = ["Vocal", "Drum", "Bass", "Others"]

    def __init__(self, preprocess_dataset, args, modality,
                 encoder_eeg=None, encoder_vocal=None, encoder_drum=None,
                 encoder_bass=None, encoder_others=None, key="all"):
        super().__init__()
        self.save_hyperparameters(args)

        if modality not in ("eeg", "audio"):
            raise ValueError(f"modality must be 'eeg' or 'audio', got {modality!r}")
        self.modality = modality
        self.key = key
        self.attention_values = args.attention_values
        self.preprocess_dataset = preprocess_dataset

        if modality == "eeg":
            if encoder_eeg is None:
                raise ValueError("classify_eeg requires encoder_eeg")
            self.encoder_eeg = encoder_eeg
            in_dim = self.EMBED_DIM
        else:
            for name, enc in (("vocal", encoder_vocal), ("drum", encoder_drum),
                              ("bass", encoder_bass), ("others", encoder_others)):
                if enc is None:
                    raise ValueError(f"classify_audio requires encoder_{name}")
            self.encoder_vocal = encoder_vocal
            self.encoder_drum = encoder_drum
            self.encoder_bass = encoder_bass
            self.encoder_others = encoder_others
            in_dim = 4 * self.EMBED_DIM

        # Classification head: the only NEW trainable weights vs the baseline encoders.
        self.classifier = nn.Linear(in_dim, self.NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()

        # Per-window test records (mirrors the contrastive module's pattern so the
        # run.py breakdown parser keeps working). Validation accuracy counters.
        self.test_records = []
        self._val_correct = 0
        self._val_total = 0

    # --- forward -------------------------------------------------------------
    def _embed(self, batch):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        if self.modality == "eeg":
            return self.encoder_eeg(eeg)
        # Fixed stem order (vocal, drum, bass, others). The label is NOT recoverable
        # from this ordering; only song->label memorisation can lift accuracy off
        # chance, which is exactly what this control measures.
        return torch.cat((
            self.encoder_vocal(m_v),
            self.encoder_drum(m_d),
            self.encoder_bass(m_b),
            self.encoder_others(m_o),
        ), dim=1)

    def forward(self, batch):
        return self.classifier(self._embed(batch))

    # --- train ---------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        task = batch[5]
        logits = self.forward(batch)
        loss = self.criterion(logits, task)
        # Same key the baseline logs, so EarlyStopping/ModelCheckpoint config in
        # main.py needs no change (it monitors Valid/loss; Loss/train mirrors it).
        self.log("Loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # --- validation ----------------------------------------------------------
    def on_validation_epoch_start(self):
        self._val_correct = 0
        self._val_total = 0

    def validation_step(self, batch, batch_idx):
        task = batch[5]
        logits = self.forward(batch)
        loss = self.criterion(logits, task)
        preds = logits.argmax(dim=1)
        self._val_correct += (preds == task).sum().item()
        self._val_total += task.size(0)
        # Monitored by EarlyStopping/ModelCheckpoint (monitor="Valid/loss" in main.py).
        self.log("Valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        acc = self._val_correct / self._val_total if self._val_total else float("nan")
        self.log("Accuracy/valid_all", acc, prog_bar=True, logger=True)
        self.debug_logger.info(
            f"--- Valid Epoch {self.current_epoch}: acc={acc:.4f} "
            f"n={self._val_total} (chance={1.0/self.NUM_CLASSES:.2f}) ---"
        )

    # --- test ----------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        subject = batch[7]
        song = batch[8]

        logits = self.forward(batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        attention_values = set(self.attention_values)
        task_l = task.detach().cpu().tolist()
        pred_l = preds.detach().cpu().tolist()
        att_l = attention_score.detach().cpu().tolist()
        sub_l = subject.detach().cpu().tolist()
        song_l = song.detach().cpu().tolist()
        probs_l = probs.detach().cpu().tolist()
        for i in range(len(task_l)):
            t = int(task_l[i])
            p = int(pred_l[i])
            att = int(att_l[i])
            self.test_records.append({
                "subject": int(sub_l[i]),
                "song": int(song_l[i]),
                "task": t,
                "pred": p,
                "attention": att,
                "correct": int(p == t),
                "prob_vocal": float(probs_l[i][0]),
                "prob_drum": float(probs_l[i][1]),
                "prob_bass": float(probs_l[i][2]),
                "prob_others": float(probs_l[i][3]),
                "high_attention": int(att in attention_values),
            })

    def on_test_end(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(self.test_records) == 0:
            print("[supervised] no test records collected")
            return

        df = pd.DataFrame(self.test_records)
        df_att = df[df["high_attention"] == 1]

        global_acc = df["correct"].mean()
        global_acc_att = df_att["correct"].mean() if len(df_att) > 0 else float("nan")
        chance = 1.0 / self.NUM_CLASSES

        # Per-class accuracy (recall over the true label).
        per_task = (df.groupby("task")["correct"].mean()
                      .reindex(range(self.NUM_CLASSES)))

        # ---- Console output ----
        print("")
        print("===== SUPERVISED TEST RESULTS =====")
        print(f"objective: classify_{self.modality}  |  chance = {chance:.2f}")
        print(f"Global accuracy (all data):        {global_acc:.4f}")
        print(f"Global accuracy (high attention):  {global_acc_att:.4f}")
        print(f"Per-class accuracy (all):          {[round(float(x), 4) for x in per_task.tolist()]}")
        if self.modality == "audio":
            print("NOTE: classify_audio is a negative control — expected ~chance; "
                  "above-chance accuracy quantifies song->label leakage, not performance.")
        print("===================================")

        save_dir = self.logger.log_dir if self.logger is not None else "test_results"
        os.makedirs(save_dir, exist_ok=True)

        # ---- Per-window records + aggregates (same filenames/keys as the
        #      contrastive breakdown so run.py's parser is reused unchanged). ----
        df.to_csv(os.path.join(save_dir, "test_records.csv"), index=False)

        def _agg(grp):
            return pd.Series({
                "n_windows": len(grp),
                "accuracy": grp["correct"].mean(),
            })

        for col, fname in [
            ("subject", "test_per_subject.csv"),
            ("song", "test_per_song.csv"),
            ("attention", "test_per_attention.csv"),
            ("task", "test_per_task_summary.csv"),
        ]:
            df.groupby(col).apply(_agg).reset_index().to_csv(
                os.path.join(save_dir, fname), index=False)

        summary_lines = [
            f"== Supervised test (objective=classify_{self.modality}) ==",
            f"chance: {chance:.4f}",
            f"n_windows total: {len(df)}",
            f"n_windows high-attention: {len(df_att)}",
            f"global accuracy (records, all): {global_acc:.4f}",
            f"global accuracy (records, attn): {global_acc_att:.4f}",
        ]
        with open(os.path.join(save_dir, "test_breakdown_summary.txt"), "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print("\n".join(summary_lines))

        # ---- Figure 1: confusion matrix (rows = true, cols = predicted) ----
        cm = np.zeros((self.NUM_CLASSES, self.NUM_CLASSES), dtype=float)
        for t, p in zip(df["task"], df["pred"]):
            cm[int(t), int(p)] += 1
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(self.NUM_CLASSES)); ax.set_yticks(range(self.NUM_CLASSES))
        ax.set_xticklabels(self.TASK_NAMES); ax.set_yticklabels(self.TASK_NAMES)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class (attended)")
        ax.set_title(f"classify_{self.modality}  |  global acc {global_acc:.3f} "
                     f"(chance {chance:.2f})")
        for i in range(self.NUM_CLASSES):
            for j in range(self.NUM_CLASSES):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "test_confusion_matrix.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 2: per-class accuracy bar (with chance line) ----
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(self.NUM_CLASSES)
        ax.bar(x, [float(v) if not np.isnan(v) else 0.0 for v in per_task.tolist()],
               color="steelblue", label=f"all data (global={global_acc:.3f})")
        ax.axhline(y=chance, color="gray", linestyle="--", alpha=0.6,
                   label=f"chance ({chance:.0%})")
        ax.set_xticks(x); ax.set_xticklabels(self.TASK_NAMES)
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
        ax.set_title(f"Per-class accuracy (classify_{self.modality})")
        ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "test_per_task_accuracy.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Figures saved to: {save_dir}")
        return super().on_test_end()

    # --- optimizer -----------------------------------------------------------
    def configure_optimizers(self):
        # Module.parameters() already de-duplicates a shared encoder referenced by
        # several attributes (the clap case), so a plain filter on requires_grad
        # gives exactly the trainable set: the head, plus the used encoder(s) when
        # they are trainable (frozen CLAP contributes only its projection head).
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, self.hparams.learning_rate)
        return {"optimizer": optimizer}
