# CHANGED(baseline): pyrefly linter directives added below (tooling only, no runtime effect)
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.distributed as dist
import numpy as np

class CLIP_Loss(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(CLIP_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, filtered_size):
        N = filtered_size
        mask = torch.ones((N, N), dtype=bool)
        for i in range(filtered_size):
            mask[i, i] = 0
        return mask

    def isin(self, elements, test_elements):
        if not isinstance(elements, torch.Tensor):
            elements = torch.tensor(elements)
        if not isinstance(test_elements, torch.Tensor):
            test_elements = torch.tensor(test_elements)

        if test_elements.numel() == 0:
            return torch.zeros_like(elements, dtype=torch.bool)

        return (elements[..., None] == test_elements).any(-1)

    def forward(self, eeg, m_v, m_d, m_b, m_o, task, attention_score, attention_values):
        """CHANGED(baseline): a trailing slot may be None, for datasets whose mixtures hold
        fewer than four sources (MAD-EEG duos hold two). Absent slots are DROPPED, never
        filled with silence: a silent stem is a trivial negative to beat, and padding with
        it would raise accuracy without any attention being decoded. The Akama path passes
        four tensors and is unaffected."""
        if self.world_size > 1:
            raise NotImplementedError()
        stems = [m for m in (m_v, m_d, m_b, m_o) if m is not None]
        n_slots = len(stems)
        losses, positive_list, negative_av, matrix_list = self.compute_task_loss(
            eeg, stems, task)

        attention_mask = self.isin(attention_score, torch.tensor(attention_values, device=attention_score.device))
        if attention_mask.sum() > 0:
            filtered_eeg = eeg[attention_mask]
            filtered_task = task[attention_mask]
         

            filtered_stems = [m[attention_mask] for m in stems]
            filtered_losses, filtered_positive_list, filtered_negative_av, filtered_matrix = self.compute_task_loss(
                filtered_eeg, filtered_stems, filtered_task
            )
        else:
            # CHANGED(baseline): removed `print(f"No samples with attention_score {attention_values}")`
            filtered_losses, filtered_positive_list, filtered_negative_av, filtered_matrix = None, [None] * n_slots, None, [[] for _ in range(n_slots)]

        # CHANGED(baseline): start — structured observability stats (per-task counts, skipped-task
        #                    flags, high-attention count). Pure bookkeeping: does NOT touch the loss.
        task_counts_all = {i: (task == i).sum().item() for i in range(n_slots)}
        stats = {
            "all": {
                "task_counts": task_counts_all,
                "skipped_tasks": {i: c == 0 for i, c in task_counts_all.items()},
                "total_samples": eeg.size(0)
            },
            "attention": {
                "high_attention_count": attention_mask.sum().item(),
                "attention_values": attention_values,
            }
        }
        
        if attention_mask.sum() > 0:
            task_counts_att = {i: (filtered_task == i).sum().item() for i in range(n_slots)}
            stats["attention"]["task_counts"] = task_counts_att
            stats["attention"]["skipped_tasks"] = {i: c == 0 for i, c in task_counts_att.items()}
        else:
            stats["attention"]["task_counts"] = {i: 0 for i in range(n_slots)}
            stats["attention"]["skipped_tasks"] = {i: True for i in range(n_slots)}
        # CHANGED(baseline): end — stats

        return {
            "all": {
                "loss": losses,
                "positive_list": positive_list,
                "negative_av": negative_av,
                "matrix_list": matrix_list
            },
            "attention": {
                "loss": filtered_losses,
                "positive_list": filtered_positive_list,
                "negative_av": filtered_negative_av,
                "matrix_list": filtered_matrix
            },
            "stats": stats  # CHANGED(baseline): new key carrying the observability stats above
        }

    def compute_task_loss(self, eeg, stems, task):
        """InfoNCE per attended slot. `stems[s]` is the audio embedding of slot s, (B, D).

        CHANGED(baseline): this was four copy-pasted branches, one per instrument slot,
        hard-wiring the number of slots at four. It is now one loop over `len(stems)`
        slots, which is what lets MAD-EEG in: there a mixture holds two or three sources,
        not four. The arithmetic is unchanged for four slots -- same similarity matrix,
        same positive, same negative set, same cross-entropy. The negatives are gathered
        in a different ORDER than the old code gathered them, which cannot change the
        value: the label is always 0 (the positive comes first) and log-sum-exp over the
        remaining logits is invariant to their permutation.

        For attended slot t and batch size B, sim is (B, S*B), read as S blocks of B
        columns. The positive is the diagonal of block t: this EEG window against the
        source it was attending. The negatives are
          * the off-diagonal of block t -- the same instrument in OTHER trials, and
          * every column of every other block -- which contains, on its diagonal, the
            competing sources of the SAME mixture. That within-mixture contrast is the
            part that matters scientifically: it cannot be won by recognising the
            stimulus, because the competing source is in the same stimulus.
        """
        n_slots = len(stems)
        losses, positive_list, negative_list = [], [], []
        matrix_list = [[] for _ in range(n_slots)]

        for t in range(n_slots):
            task_mask = (task == t)
            if task_mask.sum() == 0:
                positive_list.append(None)
                continue

            eeg_t = eeg[task_mask]
            filtered_size = eeg_t.size(0)
            z_audio = torch.cat([s[task_mask] for s in stems], dim=0)

            sim = self.similarity_f(eeg_t.unsqueeze(1), z_audio.unsqueeze(0)) / self.temperature
            blocks = [sim[:, s * filtered_size:(s + 1) * filtered_size] for s in range(n_slots)]
            diagonals = [torch.diagonal(b) for b in blocks]

            positive_samples = diagonals[t].reshape(filtered_size, 1)
            for i in range(filtered_size):
                matrix_list[t].append([d[i].item() for d in diagonals])

            mask = self.mask_correlated_samples(filtered_size)
            parts = [blocks[t][mask].reshape(filtered_size, filtered_size - 1)]
            parts += [blocks[s] for s in range(n_slots) if s != t]
            negative_samples = torch.cat(parts, dim=1).reshape(filtered_size, -1)

            logits = torch.cat((positive_samples, negative_samples), dim=1)
            labels = torch.zeros(filtered_size).to(positive_samples.device).long()
            losses.append(self.criterion(logits, labels))
            positive_list.append(positive_samples.mean())
            negative_list.append(negative_samples.mean())

        loss = sum(losses) / len(losses)
        negative_av = sum(negative_list) / len(negative_list)

        return loss, positive_list, negative_av, matrix_list


# CHANGED(baseline): new -- self-check for the slot loop. Run: python src/modules/clip_loss.py
# It pins the arithmetic against a value computed by hand, so a future edit to the loop
# cannot quietly change what the loss means.
if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    D, TEMP = 8, 0.5

    # One sample, two slots, attending slot 0. Everything is hand-computable:
    #   logits = [cos(e,a)/T, cos(e,b)/T], label 0
    #   loss   = -log softmax(logits)[0] = log(1 + exp((cos(e,b) - cos(e,a)) / T))
    e, a, b = torch.randn(1, D), torch.randn(1, D), torch.randn(1, D)
    cos = torch.nn.CosineSimilarity(dim=1)
    expected = math.log(1 + math.exp((float(cos(e, b)) - float(cos(e, a))) / TEMP))
    got = float(CLIP_Loss(1, TEMP, 1)(e, a, b, None, None,
                                      torch.tensor([0]), torch.tensor([4]), [4, 5])["all"]["loss"])
    assert abs(got - expected) < 1e-5, f"duo InfoNCE: got {got}, hand-computed {expected}"

    # Four slots: the positive must be logits[0], so a perfectly-aligned positive drives
    # the loss towards zero while an anti-aligned one does not.
    eeg = torch.randn(4, D)
    stems = [torch.randn(4, D) for _ in range(4)]
    stems[0] = eeg.clone()                       # slot 0 == the EEG it is paired with
    task = torch.zeros(4).long()
    out = CLIP_Loss(4, TEMP, 1)(eeg, *stems, task, torch.full((4,), 4), [4, 5])
    aligned = float(out["all"]["loss"])
    stems[0] = -eeg.clone()
    anti = float(CLIP_Loss(4, TEMP, 1)(eeg, *stems, task, torch.full((4,), 4), [4, 5])["all"]["loss"])
    assert aligned < anti, f"aligned positive should cost less: {aligned} vs {anti}"
    assert len(out["all"]["matrix_list"]) == 4, "four stems must report four slots"
    assert all(len(row) == 4 for row in out["all"]["matrix_list"][0]), "row per slot"

    print(f"[clip_loss self-check] PASS  duo InfoNCE matches hand computation "
          f"({got:.6f}); aligned {aligned:.4f} < anti-aligned {anti:.4f}")
