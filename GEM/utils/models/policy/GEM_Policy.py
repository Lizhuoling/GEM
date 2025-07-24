import pdb
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from GEM.utils.detr.models.GEM import get_GEM_model

class GEM_policy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = get_GEM_model(cfg)
        self.model = model.cuda()

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.status_cls_loss_weight = 1

    def __call__(self, repr, past_action, action, past_action_is_pad, action_is_pad, status, task_instruction_list,  video_frame_id, dataset_type, **kwargs):
        if action is not None: # training or validation time
            means, variances, mixture_weights, status_pred = self.model(repr, past_action, action, past_action_is_pad, action_is_pad, status, task_instruction_list, video_frame_id, dataset_type)
            loss_dict = dict()
            
            diff = action[None, :, :, None] - means # Left shape: (num_dec, B, num_query, num_mixture, state_dim)
            precisions = 1.0 / (variances + 1e-8) # Left shape: (num_dec, B, num_query, num_mixture, state_dim)
            exp_term = -0.5 * (diff**2) * precisions    # Left shape: (num_dec, B, num_query, num_mixture, state_dim)
            norm_term = 0.5 * (torch.log(precisions + 1e-8) - math.log(2 * torch.pi)).sum(dim=-1)   # Left shape: (num_dec, B, num_query, num_mixture)
            log_probs = exp_term.sum(dim=-1) + norm_term  # Left shape: (num_dec, B, num_query, num_mixture)
            weighted_log_probs = log_probs + torch.log(mixture_weights + 1e-8)  # Left shape: (num_dec, B, num_query, num_mixture)
            log_likelihood = torch.logsumexp(weighted_log_probs, dim=-1)    # Left shape: (num_dec, B, num_query）
            likehood_loss = -log_likelihood.sum(-1) # Left shape: (num_dec, B)
            valid_count = torch.clip((~action_is_pad)[None].sum(dim = -1), min = 1)  # Left shape: (num_dec,)
            avg_likehood_loss = (likehood_loss / valid_count).mean(dim = -1)    # Left shape: (num_dec,)
            total_loss = avg_likehood_loss.sum()
            
            if self.cfg['POLICY']['STATUS_PREDICT']:
                for dec_cnt in range(status_pred.shape[0]):
                    status_pred_loss = self.status_cls_loss_weight * self.CrossEntropyLoss(status_pred[dec_cnt, :, :], status.long())
                    loss_dict[f'status_pred_{dec_cnt}'] = status_pred_loss.item()
                    total_loss = total_loss + status_pred_loss
            
            for dec_id in range(avg_likehood_loss.shape[0]):
                loss_dict[f'dec_{dec_id}'] = avg_likehood_loss[dec_id].item()
            loss_dict['total_loss'] = total_loss.item()
            return total_loss, loss_dict
        else: # inference time
            means, variances, mixture_weights, status_pred = self.model(repr, past_action, action, past_action_is_pad, action_is_pad, status, task_instruction_list, video_frame_id, dataset_type)
            _, k_indices = torch.max(mixture_weights, dim=-1)  # k_indices shape: (B, num_query)
            B, num_query, num_mixture, action_dim = means.shape
            indices = k_indices[:, :, None, None].expand(-1, -1, 1, action_dim)   # Left shape: (B, num_query, 1, action_dim)
            a_hat = torch.gather(means, 2, indices).squeeze(2)  # means shape: (B, num_query, num_mixture, state_dim), a_hat shape: (B, num_query, state_dim)
            return a_hat, status_pred