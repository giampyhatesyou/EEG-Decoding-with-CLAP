# CHANGED(baseline): this LightningModule is substantially modified from the Akama upstream. PRESERVED
#                    unchanged (verified by diff): forward(), compute_evaluation_matrix() [the metric],
#                    configure_optimizers(). Loss arithmetic lives in clip_loss.py and is untouched.
#                    Everything marked `CHANGED(baseline)` below is observability/audit/lifecycle work.
import torch
from pytorch_lightning import LightningModule
# CHANGED(baseline): removed top-level `from simclr.modules import NT_Xent` (now lazy-imported in
#                    configure_criterion, only on the non-CLIP fallback path)
from . import CLIP_Loss  # CHANGED(baseline): was `from modules import CLIP_Loss`
from itertools import chain
from utils import get_logger
# CHANGED(baseline): removed dead `import torch.nn.functional as F` (no F. use in this module)
import pandas as pd
import os
import numpy as np

class EEGContrastiveLearning(LightningModule):
    debug_logger = get_logger("dataloader_debug")

    def __init__(self, preprocess_dataset, args, encoder_raw_e, encoder_vocal, encoder_drum, encoder_bass, encoder_others, key):
        super().__init__()

        self.save_hyperparameters(args)

        self.encoder_raw_e = encoder_raw_e
        self.encoder_vocal = encoder_vocal
        self.encoder_drum = encoder_drum
        self.encoder_bass = encoder_bass
        self.encoder_others = encoder_others
        self.criterion = self.configure_criterion()
        self.attention_values = args.attention_values
        self.audio_sample_rate=args.audio_sample_rate
        self.eeg_sample_rate=args.eeg_sample_rate

        self.key=key

        self.preprocess_dataset = preprocess_dataset
        # Reset per-epoch in on_validation_epoch_start; initialised here for the test path.
        self.matrix_list_all = []
        self.matrix_list_attention = []

        # CHANGED(baseline): removed 11 dead __init__ fields inherited from upstream (last_epoch_*
        #                    embeddings/labels, train_log_df, valid_log_df, validation_end_values,
        #                    test_result, label_accuracy_count, subject_accuracy_count, test_data_length):
        #                    written but never read anywhere in the repo.
        # CHANGED(baseline): added the two fields below (test_records, _shuffle_test_mode); upstream __init__
        #                    ended at subject_accuracy_count.
        # Per-window test records populated in test_step / consumed in on_test_end.
        # Each row: subject, song, task, attention, sim per stem, pos, max_neg,
        # margin, correct, group ("all"/"attention"). Kept as a flat list of
        # dicts so we can dump it straight to CSV without further bookkeeping.
        self.test_records = []
        self._shuffle_test_mode = getattr(args, "shuffle_test_mode", "none")

    def forward(self, eeg, m_v, m_d, m_b, m_o):
        z_eeg = self.encoder_raw_e(eeg)
        z_v = self.encoder_vocal(m_v)
        z_d = self.encoder_drum(m_d)
        z_b = self.encoder_bass(m_b)
        z_o = self.encoder_others(m_o)

        return z_eeg, z_v, z_d, z_b, z_o

    # CHANGED(baseline): new method — not in upstream. Resets per-epoch train observability counters.
    def on_train_epoch_start(self):
        self.train_epoch_stats = {
            "total_samples": 0,
            "high_attention_samples": 0,
            "no_attention_batches": 0,
            "skipped_tasks_batches": {0: 0, 1: 0, 2: 0, 3: 0},
            "task_counts": {0: 0, 1: 0, 2: 0, 3: 0}
        }

    def training_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        subject = batch[7]
        song = batch[8]
    
        z_eeg, z_v, z_d, z_b, z_o = self.forward(eeg, m_v, m_d, m_b, m_o)
        if self.hparams.detach_z_c:
            z_v = z_v.detach()
            z_d = z_d.detach()
            z_b = z_b.detach()
            z_o = z_o.detach()

        similarity_dict = self.criterion(
            z_eeg, z_v, z_d, z_b, z_o, task, attention_score, self.attention_values)

        # CHANGED(baseline): start — read the new `stats` dict and aggregate per-epoch observability counters
        stats = similarity_dict.get("stats")
        if stats and hasattr(self, 'train_epoch_stats'):
            self.train_epoch_stats["total_samples"] += stats["all"]["total_samples"]
            self.train_epoch_stats["high_attention_samples"] += stats["attention"]["high_attention_count"]
            if stats["attention"]["high_attention_count"] == 0:
                self.train_epoch_stats["no_attention_batches"] += 1
            
            for i in range(4):
                self.train_epoch_stats["task_counts"][i] += stats["all"]["task_counts"][i]
                if stats["all"]["skipped_tasks"][i]:
                    self.train_epoch_stats["skipped_tasks_batches"][i] += 1
            
            debug_batch_logging = getattr(self.hparams, 'debug_batch_logging', False)
            if debug_batch_logging:
                skipped_tasks = [i for i, skipped in stats['all']['skipped_tasks'].items() if skipped]
                self.debug_logger.debug(
                    f"Batch {batch_idx}: total={stats['all']['total_samples']}, high_attn={stats['attention']['high_attention_count']}, skipped={skipped_tasks}"
                )

        if self.key == 'all':
            loss = similarity_dict["all"]["loss"]
        elif self.key == 'high_attention':
            if similarity_dict["attention"]["loss"] is None:
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            else:
                loss = similarity_dict["attention"]["loss"]
        else:
            raise ValueError('Please input training data category (all or high_attention)')
        # CHANGED(baseline): end — per-epoch stats aggregation

        self.log("Loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # CHANGED(baseline): upstream logged only via debug_logger, never self.log

        # CHANGED(baseline): removed last-epoch collection into last_epoch_train_embeddings (dead field, see __init__)
        return loss

    # CHANGED(baseline): new method — not in upstream. Prints the per-epoch train observability summary.
    def on_train_epoch_end(self):
        if hasattr(self, 'train_epoch_stats'):
            summary = (
                f"--- Train Epoch {self.current_epoch} Summary ---\n"
                f"  Total samples: {self.train_epoch_stats['total_samples']}\n"
                f"  High attention samples: {self.train_epoch_stats['high_attention_samples']}\n"
                f"  Batches w/o high attention: {self.train_epoch_stats['no_attention_batches']}\n"
                f"  Task distribution: {list(self.train_epoch_stats['task_counts'].values())}\n"
                f"  Batches skipping tasks: {list(self.train_epoch_stats['skipped_tasks_batches'].values())}\n"
                f"---------------------------------------"
            )
            self.debug_logger.info(summary)

    # CHANGED(baseline): new method — replaces upstream `on_validation_start` (which only did self.eval()).
    #                    Resets the validation matrices each epoch; upstream accumulated them in __init__
    #                    and never reset them, so validation accuracy was not per-epoch.
    def on_validation_epoch_start(self):
        self.val_epoch_stats = {
            "total_samples": 0,
            "high_attention_samples": 0,
            "task_counts": {0: 0, 1: 0, 2: 0, 3: 0}
        }
        self.matrix_list_all = []
        self.matrix_list_attention = []

    def validation_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        
        z_eeg, z_v, z_d, z_b, z_o = self.forward(eeg, m_v, m_d, m_b, m_o)
        if self.hparams.detach_z_c:
            z_v = z_v.detach()
            z_d = z_d.detach()
            z_b = z_b.detach()
            z_o = z_o.detach()
            
        similarity_dict = self.criterion(
            z_eeg, z_v, z_d, z_b, z_o, task, attention_score, self.attention_values)

        # CHANGED(baseline): start — per-epoch val stats. Upstream validation_step instead did a lot of
        #                    debug_logger.info()/_get_tensor_value() bookkeeping (positive/filtered task
        #                    averages) that was computed and discarded; that has been removed here.
        stats = similarity_dict.get("stats")
        if stats and hasattr(self, 'val_epoch_stats'):
            self.val_epoch_stats["total_samples"] += stats["all"]["total_samples"]
            self.val_epoch_stats["high_attention_samples"] += stats["attention"]["high_attention_count"]
            for i in range(4):
                self.val_epoch_stats["task_counts"][i] += stats["all"]["task_counts"][i]

        if self.key == 'all':
            loss = similarity_dict["all"]["loss"]
        elif self.key == 'high_attention':
            if similarity_dict["attention"]["loss"] is None:
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            else:
                loss = similarity_dict["attention"]["loss"]
        else:
            raise ValueError('Please input training data category (all or high_attention)')

        self.matrix_list_all.append(similarity_dict["all"]["matrix_list"])
        self.matrix_list_attention.append(similarity_dict["attention"]["matrix_list"])
        # CHANGED(baseline): end — per-epoch val stats / removed discarded bookkeeping

        self.log("Valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  # CHANGED(baseline): upstream never logged Valid/loss, so EarlyStopping/ModelCheckpoint on it never fired
        return loss

    # CHANGED(baseline): new method — not in upstream. Logs per-epoch valid accuracy (all + high-attention)
    #                    via the preserved compute_evaluation_matrix, and prints the val summary.
    def on_validation_epoch_end(self):
        eval_all = self.compute_evaluation_matrix(self.matrix_list_all)
        eval_att = self.compute_evaluation_matrix(self.matrix_list_attention)
        
        # eval_all[6] is the global ratio (accuracy)
        self.log("Accuracy/valid_all", eval_all[6], prog_bar=True, logger=True)
        self.log("Accuracy/valid_attention", eval_att[6], prog_bar=True, logger=True)

        if hasattr(self, 'val_epoch_stats'):
            summary = (
                f"--- Valid Epoch {self.current_epoch} Summary ---\n"
                f"  Total samples: {self.val_epoch_stats['total_samples']}\n"
                f"  High attention samples: {self.val_epoch_stats['high_attention_samples']}\n"
                f"  Task distribution: {list(self.val_epoch_stats['task_counts'].values())}\n"
                f"  Global Accuracy (All): {eval_all[6]:.4f}\n"
                f"  Global Accuracy (Att): {eval_att[6]:.4f}\n"
                f"----------------------------------------"
            )
            self.debug_logger.info(summary)


    # NOT CHANGED(baseline): this method is identical to upstream (verified by diff). It is
    #                        the evaluation metric (global accuracy = P[sim(EEG,target) > max sim(EEG,other)],
    #                        per-task and pairwise). Do NOT modify without an explicit methodological decision.
    def compute_evaluation_matrix(self, all_lists):
        if not all_lists or all(len(task) == 0 for group in all_lists for task in group):
            print("all_lists is empty")
            nan_matrix = torch.full((4, 4), float('nan'))
            nan_column = torch.full((4, 1), float('nan'))
            return [nan_matrix, nan_matrix, nan_matrix, nan_column, nan_column, nan_column,
                    float('nan'), float('nan'), float('nan')]

        first_nonempty = next((lists for lists in all_lists if any(len(t) > 0 for t in lists)), None)
        num_tasks = len(first_nonempty) if first_nonempty is not None else 4

        ratio_num_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        ratio_den_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        
        pos_sum_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        pos_cnt_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        neg_sum_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
        neg_cnt_44 = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)

        ratio_num_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        ratio_den_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        
        pos_sum_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        pos_cnt_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        neg_sum_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)
        neg_cnt_41 = torch.zeros((num_tasks, 1), dtype=torch.float32)

        total_number = 0
        larger_number = 0

        global_pos_sum = 0
        global_pos_cnt = 0
        global_neg_sum = 0
        global_neg_cnt = 0
        # mcnemar_results = []
        for lists in all_lists:
            for task_idx, task_list in enumerate(lists):
                if len(task_list) == 0:
                    continue

                task_tensor = torch.as_tensor(task_list, dtype=torch.float32)  
                pos_values = task_tensor[:, task_idx]  

                for neg_idx in range(num_tasks):
                    if task_idx == neg_idx:
                        continue
                    neg_values = task_tensor[:, neg_idx]

                    larger_count = (pos_values > neg_values).sum().item()
                    total_count = len(pos_values)
                    ratio_num_44[task_idx, neg_idx] += larger_count
                    ratio_den_44[task_idx, neg_idx] += total_count

                    differences = pos_values - neg_values
                    pos_sel = differences[differences > 0]
                    neg_sel = differences[differences < 0]
                    if len(pos_sel) > 0:
                        pos_sum_44[task_idx, neg_idx] += pos_sel.sum().item()
                        pos_cnt_44[task_idx, neg_idx] += len(pos_sel)
                    if len(neg_sel) > 0:
                        neg_sum_44[task_idx, neg_idx] += neg_sel.sum().item()
                        neg_cnt_44[task_idx, neg_idx] += len(neg_sel)

                max_values, _ = task_tensor[:, [i for i in range(num_tasks) if i != task_idx]].max(dim=1) 
                larger = (pos_values > max_values).sum().item()
                total = len(pos_values)
                total_number += total
                larger_number += larger

                ratio_num_41[task_idx, 0] += larger
                ratio_den_41[task_idx, 0] += total

                diffs = pos_values - max_values
                pos41 = diffs[diffs > 0]
                neg41 = diffs[diffs < 0]
                if len(pos41) > 0:
                    pos_sum_41[task_idx, 0] += pos41.sum().item()
                    pos_cnt_41[task_idx, 0] += len(pos41)
                    global_pos_sum += pos41.sum().item()
                    global_pos_cnt += len(pos41)
                if len(neg41) > 0:
                    neg_sum_41[task_idx, 0] += neg41.sum().item()
                    neg_cnt_41[task_idx, 0] += len(neg41)
                    global_neg_sum += neg41.sum().item()
                    global_neg_cnt += len(neg41)
    
            # ########### Mcnemar #############
            # flags = (pos_values > max_values).int()
            # mcnemar_results.append(flags)
            # flattened = []
            # for t in mcnemar_results:
            #     flattened.extend(t.cpu().numpy().tolist())
            # df = pd.DataFrame(flattened)
            # df.to_excel("Mcnemar_results.xlsx", index=False, header=False)
            # ######################################################

        ratio_matrix = ratio_num_44/ratio_den_44  
        pos_diff     = pos_sum_44/pos_cnt_44    
        neg_diff     = neg_sum_44/neg_cnt_44    

        ratio_4      = ratio_num_41/ratio_den_41
        pos_diff_4   = pos_sum_41/pos_cnt_41
        neg_diff_4   = neg_sum_41/neg_cnt_41

        ratio = (larger_number / total_number) if total_number > 0 else float('nan')
        positive_difference = (global_pos_sum / global_pos_cnt) if global_pos_cnt > 0 else float('nan')
        negative_difference = (global_neg_sum / global_neg_cnt) if global_neg_cnt > 0 else float('nan')

        return [ratio_matrix, pos_diff, neg_diff,
                ratio_4, pos_diff_4, neg_diff_4,
                ratio, positive_difference, negative_difference]


    def test_step(self, batch, batch_idx):
        eeg, m_v, m_d, m_b, m_o = batch[:5]
        task = batch[5]
        attention_score = batch[6]
        subject = batch[7]
        song = batch[8]

        # CHANGED(baseline): start — negative-control shuffles (audit). Not in upstream test_step.
        # --- Negative-control shuffles (audit only; no effect when "none"). ---
        # Applied BEFORE the forward pass so the criterion sees the perturbed
        # pairing/label. We rely on pl.seed_everything() upstream for repro.
        #
        # IMPORTANT: these only behave as proper negative controls if the
        # test DataLoader is shuffled. The Akama test set lays out the 13
        # sliding windows of one trial consecutively, so with shuffle=False
        # ~64% of batches share a single task across all 8 samples and an
        # intra-batch permutation is a no-op. checkpoint_test.py forces
        # shuffle=True whenever shuffle_test_mode != "none".
        if self._shuffle_test_mode == "labels":
            # Uniform i.i.d. labels — stronger than randperm because randperm
            # preserves the marginal task distribution. With i.i.d. labels and
            # an honest evaluator, accuracy MUST collapse to chance (1/4).
            task = torch.randint(
                0, 4, task.shape, device=task.device, dtype=task.dtype
            )
        elif self._shuffle_test_mode == "audio_pair":
            perm = torch.randperm(eeg.size(0), device=eeg.device)
            m_v = m_v[perm]
            m_d = m_d[perm]
            m_b = m_b[perm]
            m_o = m_o[perm]
        elif self._shuffle_test_mode != "none":
            raise ValueError(
                f"Unknown shuffle_test_mode={self._shuffle_test_mode!r}; "
                "expected one of {'none','labels','audio_pair'}."
            )
        # CHANGED(baseline): end — negative-control shuffles

        z_eeg, z_v, z_d, z_b, z_o = self.forward(eeg, m_v, m_d, m_b, m_o)
        if self.hparams.detach_z_c:
            z_v = z_v.detach()
            z_d = z_d.detach()
            z_b = z_b.detach()
            z_o = z_o.detach()
        similarity_dict = self.criterion(
            z_eeg, z_v, z_d, z_b, z_o, task, attention_score,self.attention_values)

        matrix_all = similarity_dict["all"]["matrix_list"]
        matrix_attention = similarity_dict["attention"]["matrix_list"]

        self.matrix_list_all.append(matrix_all)
        self.matrix_list_attention.append(matrix_attention)

        # CHANGED(baseline): start — per-window record collection (audit breakdowns). Not in upstream.
        #                    Upstream test_step instead had discarded _get_tensor_value()/positive-average
        #                    bookkeeping here; that has been removed.
        # --- Per-window record collection (used by on_test_end for breakdowns).
        # matrix_all[k] is the ordered list of per-slot similarity rows for samples in
        # this batch whose task == k, in the same order as `task == k` boolean indexing of
        # (subject, song, attention_score). We zip them back together so each window keeps
        # its metadata.
        # CHANGED(baseline): the slot count comes from matrix_all rather than being fixed
        # at four. Akama mixtures always hold four stems; a MAD-EEG duo holds two, and the
        # decision is then between two sources, not four -- so chance is 1/n_slots and the
        # unused sim_* columns stay empty instead of raising IndexError.
        if getattr(self.hparams, "test_breakdown", 0):
            attention_values = set(self.attention_values)
            n_slots = len(matrix_all)
            for task_idx in range(n_slots):
                task_mask = (task == task_idx)
                if task_mask.sum() == 0:
                    continue
                sub_t = subject[task_mask].detach().cpu().tolist()
                song_t = song[task_mask].detach().cpu().tolist()
                att_t = attention_score[task_mask].detach().cpu().tolist()
                rows = matrix_all[task_idx]
                for i, row in enumerate(rows):
                    sims = [float(v) for v in row]
                    pos = sims[task_idx]
                    neg = [s for j, s in enumerate(sims) if j != task_idx]
                    max_neg = max(neg)
                    padded = sims + [None] * (4 - len(sims))
                    self.test_records.append({
                        "subject": int(sub_t[i]),
                        "song": int(song_t[i]),
                        "task": int(task_idx),
                        "attention": int(att_t[i]),
                        "sim_vocal": padded[0],
                        "sim_drum": padded[1],
                        "sim_bass": padded[2],
                        "sim_others": padded[3],
                        "n_slots": len(sims),
                        "pos_sim": pos,
                        "max_neg_sim": max_neg,
                        "margin": pos - max_neg,
                        "correct": int(pos > max_neg),
                        "high_attention": int(int(att_t[i]) in attention_values),
                    })
        # CHANGED(baseline): end — per-window record collection
        # CHANGED(baseline): removed two compute_evaluation_matrix calls here whose results were
        #                    discarded (the real evaluation runs once in on_test_end). They re-ran
        #                    the full O(windows) aggregation on every test batch for nothing.

    # CHANGED(baseline): upstream on_test_end was effectively empty (commented bootstrap + super() call).
    #                    The whole body below — console metrics, figures, breakdown CSVs — is new. It uses
    #                    the preserved compute_evaluation_matrix and never feeds back into training.
    def on_test_end(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        eval_all = self.compute_evaluation_matrix(self.matrix_list_all)
        eval_att = self.compute_evaluation_matrix(self.matrix_list_attention)

        # ---- Console output ----
        print("")
        print("========== TEST RESULTS ==========")
        print(f"Global Accuracy (All data):        {eval_all[6]:.4f}")
        print(f"Global Accuracy (High attention):  {eval_att[6]:.4f}")
        print(f"Per-task acc (All):    {eval_all[3].squeeze().tolist()}")
        print(f"Per-task acc (Att):    {eval_att[3].squeeze().tolist()}")
        print("Pairwise matrix 4x4 (All):"); print(eval_all[0])
        print("Pairwise matrix 4x4 (Att):"); print(eval_att[0])
        print("==================================")

        # ---- Save figures ----
        save_dir = self.logger.log_dir if self.logger is not None else "test_results"
        os.makedirs(save_dir, exist_ok=True)
        task_names = ["Vocal", "Drum", "Bass", "Others"]

        def _to_np(t):
            import torch as _torch
            if isinstance(t, _torch.Tensor):
                return t.detach().cpu().numpy().astype(float)
            return np.asarray(t, dtype=float)

        # --- Figure 1: Pairwise discrimination heatmaps ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, ev, title in [(axes[0], eval_all, "All data"), (axes[1], eval_att, "High attention")]:
            mat = _to_np(ev[0])
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(task_names); ax.set_yticklabels(task_names)
            ax.set_xlabel("Negative class"); ax.set_ylabel("Positive class (target)")
            ax.set_title(f"{title}  |  Global acc: {float(ev[6]):.3f}")
            for i in range(4):
                for j in range(4):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color="white" if v > 0.5 else "black", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("Pairwise discrimination (P[sim(EEG, target) > sim(EEG, other)])", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "test_pairwise_matrix.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # --- Figure 2: Per-task accuracy bar chart ---
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(4); width = 0.35
        acc_all = _to_np(eval_all[3]).squeeze()
        acc_att = _to_np(eval_att[3]).squeeze()
        ax.bar(x - width/2, acc_all, width, label=f"All data (global={float(eval_all[6]):.3f})", color="steelblue")
        ax.bar(x + width/2, acc_att, width, label=f"High attention (global={float(eval_att[6]):.3f})", color="coral")
        ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.6, label="Chance (25%)")
        ax.set_xticks(x); ax.set_xticklabels(task_names)
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Per-task accuracy: correct stream identification")
        ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "test_per_task_accuracy.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # --- Breakdown CSVs + plots (only when test_breakdown is on). ---
        # All of this is observability-only: it never feeds back into training,
        # never alters the printed `eval_all` / `eval_att` numbers above. It
        # simply re-aggregates the per-window records we collected in test_step.
        if getattr(self.hparams, "test_breakdown", 0) and len(self.test_records) > 0:
            df = pd.DataFrame(self.test_records)
            shuffle_tag = getattr(self, "_shuffle_test_mode", "none")
            df.to_csv(os.path.join(save_dir, "test_records.csv"), index=False)

            def _agg(grp):
                return pd.Series({
                    "n_windows": len(grp),
                    "accuracy": grp["correct"].mean(),
                    "margin_mean": grp["margin"].mean(),
                    "margin_std": grp["margin"].std(ddof=0),
                })

            # Per-subject / per-song / per-attention aggregates.
            for key, fname in [
                ("subject", "test_per_subject.csv"),
                ("song", "test_per_song.csv"),
                ("attention", "test_per_attention.csv"),
                ("task", "test_per_task_summary.csv"),
            ]:
                agg = df.groupby(key).apply(_agg).reset_index()
                agg.to_csv(os.path.join(save_dir, fname), index=False)

            # High-attention subset summary mirrors the "Att" branch already
            # printed above, but computed from the same per-window records so
            # it stays internally consistent with the breakdown CSVs.
            df_att = df[df["high_attention"] == 1]
            if len(df_att) > 0:
                summary_lines = [
                    f"== Test breakdown (shuffle_test_mode={shuffle_tag}) ==",
                    f"n_windows total: {len(df)}",
                    f"n_windows high-attention: {len(df_att)}",
                    f"global accuracy (records, all): {df['correct'].mean():.4f}",
                    f"global accuracy (records, attn): {df_att['correct'].mean():.4f}",
                    f"global margin mean (all): {df['margin'].mean():+.4f}",
                    f"global margin mean (attn): {df_att['margin'].mean():+.4f}",
                ]
                with open(os.path.join(save_dir, "test_breakdown_summary.txt"), "w") as f:
                    f.write("\n".join(summary_lines) + "\n")
                print("\n".join(summary_lines))

            # --- Figure: per-subject accuracy + margin distribution ---
            sub_agg = df.groupby("subject").apply(_agg).reset_index()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            ax1.bar(sub_agg["subject"].astype(str), sub_agg["accuracy"], color="steelblue")
            ax1.axhline(0.25, color="gray", ls="--", alpha=0.6, label="chance")
            ax1.set_xlabel("subject"); ax1.set_ylabel("accuracy")
            ax1.set_ylim(0, 1.05)
            ax1.set_title("Per-subject accuracy"); ax1.legend()
            ax2.bar(sub_agg["subject"].astype(str), sub_agg["margin_mean"],
                    yerr=sub_agg["margin_std"], color="coral", capsize=3)
            ax2.axhline(0.0, color="gray", ls="--", alpha=0.6)
            ax2.set_xlabel("subject"); ax2.set_ylabel("mean margin (pos - max_neg)")
            ax2.set_title("Per-subject margin (mean ± std)")
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, "test_per_subject.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Figure: per-song accuracy (sorted) ---
            song_agg = df.groupby("song").apply(_agg).reset_index().sort_values("accuracy")
            fig, ax = plt.subplots(figsize=(min(0.25 * len(song_agg) + 4, 22), 5))
            ax.bar(song_agg["song"].astype(str), song_agg["accuracy"], color="seagreen")
            ax.axhline(0.25, color="gray", ls="--", alpha=0.6, label="chance")
            ax.set_xlabel("song id (sorted by accuracy)"); ax.set_ylabel("accuracy")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Per-song accuracy (n_songs={len(song_agg)})")
            ax.tick_params(axis='x', rotation=90, labelsize=8)
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, "test_per_song.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Figure: accuracy & margin vs attention score ---
            att_agg = df.groupby("attention").apply(_agg).reset_index().sort_values("attention")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            ax1.bar(att_agg["attention"].astype(int).astype(str), att_agg["accuracy"], color="steelblue")
            ax1.axhline(0.25, color="gray", ls="--", alpha=0.6, label="chance")
            for x, n in zip(att_agg["attention"].astype(int).astype(str), att_agg["n_windows"]):
                ax1.text(x, 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=8, color="white")
            ax1.set_xlabel("behavioural attention score"); ax1.set_ylabel("accuracy")
            ax1.set_ylim(0, 1.05)
            ax1.set_title("Accuracy vs attention score"); ax1.legend()

            ax2.bar(att_agg["attention"].astype(int).astype(str), att_agg["margin_mean"],
                    yerr=att_agg["margin_std"], color="coral", capsize=3)
            ax2.axhline(0.0, color="gray", ls="--", alpha=0.6)
            ax2.set_xlabel("behavioural attention score"); ax2.set_ylabel("mean margin")
            ax2.set_title("Margin vs attention score (mean ± std)")
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, "test_vs_attention.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Figures saved to: {save_dir}")
        return super().on_test_end()


    def configure_criterion(self):
        # CHANGED(baseline): `devices` instead of upstream's deprecated `gpus`
        #                    (upstream: `if self.hparams.accelerator == "dp" and self.hparams.gpus: ... / self.hparams.gpus`)
        if self.hparams.accelerator == "dp" and hasattr(self.hparams, 'devices') and self.hparams.devices:
            batch_size = int(self.hparams.batch_size / self.hparams.devices)
        else:
            batch_size = self.hparams.batch_size

        if self.hparams.loss_function == "clip_loss":
            print("use CLIP_loss as criterion")
            criterion = CLIP_Loss(
                batch_size, self.hparams.temperature, world_size=1)
        else:
            from simclr.modules import NT_Xent  # CHANGED(baseline): lazy import (was top-of-module)
            print('use NT_Xent as criterion')
            criterion = NT_Xent(
                batch_size, self.hparams.temperature, world_size=1)
        return criterion

    # NOT CHANGED(baseline): identical to upstream (verified by diff). Adam over all 5 encoders' params.
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.encoder_raw_e.parameters(), self.encoder_vocal.parameters(
        ), self.encoder_drum.parameters(), self.encoder_bass.parameters(), self.encoder_others.parameters()), self.hparams.learning_rate)
        return {"optimizer": optimizer}

    # CHANGED(baseline): removed 9 upstream methods that had no call site in this repo:
    #   on_validation_start, val_dataloader, _shared_step, Kfold_log, save_checkpoint,
    #   load_checkpoint, _get_tensor_value, _get_eeg, _get_audio.
