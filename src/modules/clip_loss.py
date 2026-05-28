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
        if self.world_size > 1:
            raise NotImplementedError()
        losses, positive_list, negative_av,matrix_list= self.compute_task_loss(
            eeg, m_v, m_d, m_b, m_o, task)

        attention_mask = self.isin(attention_score, torch.tensor(attention_values, device=attention_score.device))
        if attention_mask.sum() > 0:
            filtered_eeg = eeg[attention_mask]
            filtered_m_v = m_v[attention_mask]
            filtered_m_d = m_d[attention_mask]
            filtered_m_b = m_b[attention_mask]
            filtered_m_o = m_o[attention_mask]
            filtered_task = task[attention_mask]
         

            filtered_losses, filtered_positive_list, filtered_negative_av, filtered_matrix = self.compute_task_loss(
                filtered_eeg, filtered_m_v, filtered_m_d, filtered_m_b, filtered_m_o, filtered_task
            )
        else:
            filtered_losses, filtered_positive_list, filtered_negative_av, filtered_matrix = None, [None,None,None,None], None, [[] for _ in range(4)]

        task_counts_all = {i: (task == i).sum().item() for i in range(4)}
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
            task_counts_att = {i: (filtered_task == i).sum().item() for i in range(4)}
            stats["attention"]["task_counts"] = task_counts_att
            stats["attention"]["skipped_tasks"] = {i: c == 0 for i, c in task_counts_att.items()}
        else:
            stats["attention"]["task_counts"] = {i: 0 for i in range(4)}
            stats["attention"]["skipped_tasks"] = {i: True for i in range(4)}

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
            "stats": stats
        }

    def compute_task_loss(self, eeg, m_v, m_d, m_b, m_o, task):
        task_mask0 = (task == 0)
        task_mask1 = (task == 1)
        task_mask2 = (task == 2)
        task_mask3 = (task == 3)
        
        losses = []
        positive_list = []
        negative_list = []
        matrix_list = [[] for _ in range(4)]

        if task_mask0.sum() > 0:
            eeg_0 = eeg[task_mask0]
            m_v_0 = m_v[task_mask0]
            m_d_0 = m_d[task_mask0]
            m_b_0 = m_b[task_mask0]
            m_o_0 = m_o[task_mask0]
          

            filtered_size0 = eeg_0.size(0)
            N0 = 2*filtered_size0
            mask0 = self.mask_correlated_samples(filtered_size0)

            z_audio_0 = torch.cat((m_v_0, m_d_0, m_b_0, m_o_0), dim=0)
            z_eeg_0 = eeg_0

            sim_e_m_0 = self.similarity_f(z_eeg_0.unsqueeze(
                1), z_audio_0.unsqueeze(0)) / self.temperature

            positive_e_m_0 = torch.diagonal(sim_e_m_0[:, :filtered_size0])
            positive_samples_0 = positive_e_m_0.reshape(filtered_size0, 1)

            v_d_e_m_0 = torch.diagonal(sim_e_m_0[:, filtered_size0:2*filtered_size0])
            v_b_e_m_0 = torch.diagonal(sim_e_m_0[:, 2*filtered_size0:3*filtered_size0])
            v_o_e_m_0 = torch.diagonal(sim_e_m_0[:, 3*filtered_size0:])

            pos_0 = positive_e_m_0
            v_d_0 = v_d_e_m_0 
            v_b_0 = v_b_e_m_0
            v_o_0 = v_o_e_m_0 

            for i in range(len(pos_0)):  
                matrix_list[0].append([pos_0[i].item(), v_d_0[i].item(), v_b_0[i].item(), v_o_0[i].item()])

            negative_e_m_01 = sim_e_m_0[:,
                                        :filtered_size0][mask0].reshape(filtered_size0, filtered_size0-1)
            negative_e_m_02 = sim_e_m_0[:, filtered_size0:]
            negative_e_m_0 = torch.cat(
                (negative_e_m_01, negative_e_m_02), dim=1)
            
            negative_samples_0 = negative_e_m_0.reshape(filtered_size0, -1)
         

            logits_0 = torch.cat(
                (positive_samples_0, negative_samples_0), dim=1)
            labels_0 = torch.zeros(int(N0/2)).to(positive_samples_0.device).long()
            loss_0 = self.criterion(logits_0, labels_0)
            losses.append(loss_0)
            positive_list.append(positive_samples_0.mean())
            negative_list.append(negative_samples_0.mean())

        else:
            positive_list.append(None)

        if task_mask1.sum() > 0:
            eeg_1 = eeg[task_mask1]
            m_v_1 = m_v[task_mask1]
            m_d_1 = m_d[task_mask1]
            m_b_1 = m_b[task_mask1]
            m_o_1 = m_o[task_mask1]
 
            filtered_size1 = eeg_1.size(0)
            N1 = 2*filtered_size1
            mask1 = self.mask_correlated_samples(filtered_size1)

            z_audio_1 = torch.cat((m_v_1, m_d_1, m_b_1, m_o_1), dim=0)
            z_eeg_1 = eeg_1

            sim_e_m_1 = self.similarity_f(z_eeg_1.unsqueeze(
                1), z_audio_1.unsqueeze(0)) / self.temperature

            positive_e_m_1 = torch.diagonal(
                sim_e_m_1[:, filtered_size1:2*filtered_size1])
        
            positive_samples_1 = positive_e_m_1.reshape(filtered_size1, 1)
            d_v_e_m_1 = torch.diagonal(sim_e_m_1[:, :filtered_size1])
            d_b_e_m_1 = torch.diagonal(sim_e_m_1[:, 2*filtered_size1:3*filtered_size1])
            d_o_e_m_1 = torch.diagonal(sim_e_m_1[:, 3*filtered_size1:])

            d_v_1 = d_v_e_m_1 
            pos_1 = positive_e_m_1 
            d_b_1 = d_b_e_m_1 
            d_o_1 = d_o_e_m_1 

            for i in range(len(pos_1)):  
                matrix_list[1].append([d_v_1[i].item(), pos_1[i].item(), d_b_1[i].item(), d_o_1[i].item()]) 

            negative_e_m_11 = sim_e_m_1[:, filtered_size1:2 *
                                        filtered_size1][mask1].reshape(filtered_size1, filtered_size1-1)
            negative_e_m_12_part1 = sim_e_m_1[:, :filtered_size1]
            negative_e_m_12_part2 = sim_e_m_1[:, 2*filtered_size1:]
            negative_e_m_12 = torch.cat(
                (negative_e_m_12_part1, negative_e_m_12_part2), dim=1)
            negative_e_m_1 = torch.cat(
                (negative_e_m_11, negative_e_m_12), dim=1)

            negative_samples_1 = negative_e_m_1.reshape(filtered_size1, -1)

            logits_1 = torch.cat(
                (positive_samples_1, negative_samples_1), dim=1)
            labels_1 = torch.zeros(int(N1/2)).to(positive_samples_1.device).long()
            loss_1 = self.criterion(logits_1, labels_1)
            losses.append(loss_1)
            positive_list.append(positive_samples_1.mean())
            negative_list.append(negative_samples_1.mean())
            
        else:
            positive_list.append(None)

        if task_mask2.sum() > 0:
            eeg_2 = eeg[task_mask2]
            m_v_2 = m_v[task_mask2]
            m_d_2 = m_d[task_mask2]
            m_b_2 = m_b[task_mask2]
            m_o_2 = m_o[task_mask2]
          


            filtered_size2 = eeg_2.size(0)
            N2 = 2*filtered_size2
            mask2 = self.mask_correlated_samples(filtered_size2)

            z_audio_2 = torch.cat((m_v_2, m_d_2, m_b_2, m_o_2), dim=0)
            z_eeg_2 = eeg_2

            sim_e_m_2 = self.similarity_f(z_eeg_2.unsqueeze(
                1), z_audio_2.unsqueeze(0)) / self.temperature
          
            positive_e_m_2 = torch.diagonal(
                sim_e_m_2[:, 2*filtered_size2:3*filtered_size2])
          
            positive_samples_2 = positive_e_m_2.reshape(filtered_size2, 1)
            b_v_e_m_2 = torch.diagonal(sim_e_m_2[:, :filtered_size2])
            b_d_e_m_2 = torch.diagonal(sim_e_m_2[:, filtered_size2:2*filtered_size2])
            b_o_e_m_2 = torch.diagonal(sim_e_m_2[:, 3*filtered_size2:])
            
            b_v_2 = b_v_e_m_2 
            b_d_2 = b_d_e_m_2 
            pos_2 = positive_e_m_2 
            b_o_2 = b_o_e_m_2

            for i in range(len(pos_2)):  
                matrix_list[2].append([b_v_2[i].item(), b_d_2[i].item(), pos_2[i].item(), b_o_2[i].item()])
            negative_e_m_21 = sim_e_m_2[:, 2*filtered_size2:3 *
                                        filtered_size2][mask2].reshape(filtered_size2, filtered_size2-1)
            negative_e_m_22_part1 = sim_e_m_2[:, :2*filtered_size2]
            negative_e_m_22_part2 = sim_e_m_2[:, 3*filtered_size2:]
            negative_e_m_22 = torch.cat(
                (negative_e_m_22_part1, negative_e_m_22_part2), dim=1)
            negative_e_m_2 = torch.cat(
                (negative_e_m_21, negative_e_m_22), dim=1)

            negative_samples_2 = negative_e_m_2.reshape(filtered_size2, -1)

            logits_2 = torch.cat(
                (positive_samples_2, negative_samples_2), dim=1)
            labels_2 = torch.zeros(int(N2/2)).to(positive_samples_2.device).long()
            loss_2 = self.criterion(logits_2, labels_2)
            losses.append(loss_2)
            positive_list.append(positive_samples_2.mean())
            negative_list.append(negative_samples_2.mean())
        
        else:
            positive_list.append(None)

        if task_mask3.sum() > 0:
            eeg_3 = eeg[task_mask3]
            m_v_3 = m_v[task_mask3]
            m_d_3 = m_d[task_mask3]
            m_b_3 = m_b[task_mask3]
            m_o_3 = m_o[task_mask3]
 

            filtered_size3 = eeg_3.size(0)
            N3 = 2*filtered_size3
            mask3 = self.mask_correlated_samples(filtered_size3)

            z_audio_3 = torch.cat((m_v_3, m_d_3, m_b_3, m_o_3), dim=0)
            z_eeg_3 = eeg_3

            sim_e_m_3 = self.similarity_f(z_eeg_3.unsqueeze(
                1), z_audio_3.unsqueeze(0)) / self.temperature
         
            positive_e_m_3 = torch.diagonal(sim_e_m_3[:, 3*filtered_size3:])
        
            positive_samples_3 = positive_e_m_3.reshape(filtered_size3, 1)

            o_v_e_m_3 = torch.diagonal(sim_e_m_3[:, :filtered_size3])
            o_d_e_m_3 = torch.diagonal(sim_e_m_3[:, filtered_size3:2*filtered_size3])
            o_b_e_m_3 = torch.diagonal(sim_e_m_3[:, 2*filtered_size3:3*filtered_size3])
            
            o_v_3 = o_v_e_m_3
            o_d_3 = o_d_e_m_3 
            o_b_3 = o_b_e_m_3
            pos_3 = positive_e_m_3

            for i in range(len(pos_3)):  
                matrix_list[3].append([o_v_3[i].item(), o_d_3[i].item(), o_b_3[i].item(), pos_3[i].item()])
            negative_e_m_31 = sim_e_m_3[:, 3 *
                                        filtered_size3:][mask3].reshape(filtered_size3, filtered_size3-1)
            negative_e_m_32 = sim_e_m_3[:, :3*filtered_size3]
            negative_e_m_3 = torch.cat(
                (negative_e_m_31, negative_e_m_32), dim=1)
            
            negative_samples_3 = negative_e_m_3.reshape(filtered_size3, -1)

            logits_3 = torch.cat(
                (positive_samples_3, negative_samples_3), dim=1)
            labels_3 = torch.zeros(int(N3/2)).to(positive_samples_3.device).long()
            loss_3 = self.criterion(logits_3, labels_3)
            losses.append(loss_3)
            positive_list.append(positive_samples_3.mean())
            negative_list.append(negative_samples_3.mean())
      
        else:
            positive_list.append(None)

        loss = sum(losses) / len(losses)
        negative_av = sum(negative_list) / len(negative_list)
        
        return loss, positive_list, negative_av, matrix_list
