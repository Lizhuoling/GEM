# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import pdb
import copy
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .backbone import build_backbone
from .transformer import build_transformer, TransformerDecoderLayer

import numpy as np

class GEM(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, state_dim, chunk_size, camera_names, cfg):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.cfg = cfg
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.num_mixture = cfg['POLICY']['MIXTURE_GAUSSIAN_NUM']
        self.output_dim = self.num_mixture * (2 * state_dim + 1)
        self.camera_names = camera_names
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        if self.cfg['POLICY']['STATUS_PREDICT']:
            query_num = 1 + chunk_size
        else:
            query_num = chunk_size
        
        dataset_dir = cfg['DATA']['DATASET_DIR']
        if type(dataset_dir) == str:
            self.skill_num = 1
            self.skill_types = [None,]
        elif type(dataset_dir) == list:
            self.skill_types = list(set([ele[0] for ele in dataset_dir]))
            self.skill_num = len(self.skill_types)
        self.skill_embed = nn.Embedding(self.skill_num, hidden_dim)
        
        self.input_proj = nn.Linear(self.backbone.num_features, hidden_dim)
        self.pc_flag_embed = nn.Embedding(1, hidden_dim) 
        self.query_embed = nn.Embedding(query_num, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.output_dim) # Decode transformer output as action.
        self.past_action_mlp = nn.Linear(state_dim, hidden_dim)  # Past action information encoding

        if self.cfg['POLICY']['STATUS_PREDICT']:
            self.status_head = nn.Linear(hidden_dim, self.cfg['POLICY']['STATUS_NUM'])
            
            
        if self.cfg['POLICY']['MEMORY_SIZE'] > 0:
            self.memory_flag_embed = nn.Embedding(1, hidden_dim)
            if self.cfg['POLICY']['MEMORY_MODE'] == 'RNN':
                self.memory_encoder = RNNMemoryEncoder(hidden_dim, cfg)
            elif self.cfg['POLICY']['MEMORY_MODE'] == 'LSTM':
                self.memory_encoder = LSTMMemoryEncoder(hidden_dim, cfg)

    def forward(self, repr, past_action, action, past_action_is_pad, action_is_pad, status, task_instruction_list, video_frame_id, dataset_type):
        """
        repr: (batch, n_point, feat_len)
        past_action: (batch, past_action_len, action_dim)
        action: (batch, chunk_size, action_dim)
        past_action_is_pad: (batch, past_action_len)\
        action_is_pad: (batch, chunk_size)
        status: (batch,)
        task_instruction_list: A list with the length of batch, each element is a string.
        """
        is_training = action is not None # train or val
        bs = past_action.shape[0]
        
        if self.cfg['TRAIN']['LR_BACKBONE'] > 0:
            feature, repr_is_pad = self.backbone(repr)
        else:
            with torch.no_grad():
                feature, repr_is_pad = self.backbone(repr)
                feature, repr_is_pad = feature.detach(), repr_is_pad.detach()
        src = self.input_proj(feature).permute(1, 0, 2)  # Left shape: (L, B, C)
        pc_flag_embed = self.pc_flag_embed.weight   # Left shape: (1, C)
        pos = pc_flag_embed[None].expand(src.shape[0], bs, -1)  # Left shape: (L, B, C)
        mask = repr_is_pad.clone()   # Left shape: (B, L)
        
        if self.cfg['POLICY']['MEMORY_SIZE'] > 0:
            memory_tokens = self.memory_encoder(obs = src.detach(), obs_mask = mask, obs_pos = pos, video_frame_id = video_frame_id)
            src = torch.cat((src, memory_tokens), dim = 0)  # Left shape: (L + memory_size, B, C)
            pos = torch.cat((pos, self.memory_flag_embed.weight[None].expand(memory_tokens.shape[0], bs, -1)), dim = 0)  # Left shape: (L + memory_size, B, C)
            mask = torch.cat((mask, torch.zeros((bs, self.cfg['POLICY']['MEMORY_SIZE']), dtype = torch.bool).cuda()), dim = 1) # Left shape: (B, L + memory_size)

        # past action
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)   # Left shape: (num_query, B, C)
        past_action_src = self.past_action_mlp(past_action).permute(1, 0, 2)   # (past_action_len, B, C)
        past_action_src = past_action_src[-1:]  # Left shape: (1, B, C)
        query_emb = query_emb + past_action_src # Left shape: (num_query, B, C)
        
        # Skill type
        all_skill_embed = self.skill_embed.weight[None]   # Left shape: (1, skill_num, C)
        if self.skill_num == 1:
            skill_embed = all_skill_embed.expand(-1, bs, -1)  # Left shape: (1, 1, C)
        else:
            skill_idx = [self.skill_types.index(ele) for ele in dataset_type]
            skill_embed = all_skill_embed[:, skill_idx] # Left shape: (1, B, C)
        query_emb = query_emb + skill_embed
    
        hs = self.transformer(src, mask, query_emb, pos) # Left shape: (num_dec, B, num_query, C)
        if self.cfg['POLICY']['STATUS_PREDICT']:
            status_hs = hs[:, :, 0] # Left shape: (num_dec, B, C)
            hs = hs[:, :, 1:]
            status_pred = self.status_head(status_hs)  # left shape: (num_dec, B, num_status)
            if not is_training: status_pred = status_pred[-1].argmax(dim = -1)  # Left shape: (B,)
        else:
            status_pred = None

        output = self.action_head(hs)    # left shape: (num_dec, B, num_query, output_dim)
        means, vars_log, pi_logits = torch.split(output, [self.num_mixture * self.state_dim, self.num_mixture * self.state_dim, self.num_mixture], dim=-1)  # means shape: (num_dec, B, num_query, num_mixture * state_dim)
        means = means.view(means.shape[0], means.shape[1], means.shape[2], self.num_mixture, self.state_dim)  # means shape: (num_dec, B, num_query, num_mixture, state_dim)
        variances = torch.exp(vars_log).view(vars_log.shape[0], vars_log.shape[1], vars_log.shape[2], self.num_mixture, self.state_dim) # Left shape: (num_dec, B, num_query, num_mixture, state_dim)
        mixture_weights = F.softmax(pi_logits, dim=-1)  # mixture_weights shape: (num_dec, B, num_query, num_mixture)
        
        if not is_training:
            means = means[-1]   # Left shape: (B, num_query, num_mixture, state_dim)
            variances = variances[-1]   # Left shape: (B, num_query, num_mixture, state_dim)
            mixture_weights = mixture_weights[-1]  # mixture_weights shape: (B, num_query, num_mixture)
        
        return means, variances, mixture_weights, status_pred
    
def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk

def build_memory_decoder(cfg):
    d_model = cfg['POLICY']['HIDDEN_DIM']
    dropout = cfg['POLICY']['DROPOUT']
    nhead = cfg['POLICY']['NHEADS']
    dim_feedforward = cfg['POLICY']['DIM_FEEDFORWARD']
    activation = "relu"
    normalize_before = False
    return TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)

class RNNMemoryEncoder(nn.Module):
    def __init__(self, hidden_dim, cfg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.memory_decoder = build_memory_decoder(cfg) 
        self.last_video_frame_id = None
        self.sample_interval = self.cfg['EVAL']['EXEC_CHUNK']
        
        self.memory_states = None
        
    def forward(self, obs, obs_mask, obs_pos, video_frame_id):
        '''
        Input:
            obs shape: (L, B, C)
            obs_mask shape: (B, L)
            obs_pos shape: (L, B, C)
            video_frame_id shape: (B,)
        Output:
            self.memory_states shape: (memory_size, B, C)
        '''
        bs = obs.shape[1]
        if self.memory_states is None:
            self.memory_states = torch.zeros((self.cfg['POLICY']['MEMORY_SIZE'], bs, self.hidden_dim), dtype = torch.float32).cuda()   # Left shape: (memory_size, B, C)
        if self.last_video_frame_id is None:
            is_new_segment = torch.ones((bs,), dtype = torch.bool).cuda()
        else:
            is_new_segment = (video_frame_id != self.last_video_frame_id + self.sample_interval)
        self.last_video_frame_id = video_frame_id.clone()
        self.memory_states[:, is_new_segment] = torch.zeros((self.cfg['POLICY']['MEMORY_SIZE'], is_new_segment.sum(), self.hidden_dim), dtype = torch.float32).cuda()
        self.memory_states = self.memory_decoder(tgt = self.memory_states.detach(), memory = obs, memory_key_padding_mask = obs_mask, pos=obs_pos,)  # Left shape: (memory_size, B, C)
        return self.memory_states

class LSTMMemoryEncoder(nn.Module):
    def __init__(self, hidden_dim, cfg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.memory_decoder = build_memory_decoder(cfg) 
        self.last_video_frame_id = None
        self.sample_interval = self.cfg['EVAL']['EXEC_CHUNK']
        self.memory_size = self.cfg['POLICY']['MEMORY_SIZE']
        
        self.cell_states = None
        self.hidden_states = None
        self.forget_gate_map = nn.Linear(self.hidden_dim, 1)
        self.input_gate_map = nn.Linear(self.hidden_dim, 1)
        self.output_gate_map = nn.Linear(self.hidden_dim, 1)
        self.cell_candidate_map = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, obs, obs_mask, obs_pos, video_frame_id):
        bs = obs.shape[1]
        if self.cell_states is None:
            self.cell_states = torch.zeros((self.memory_size, bs, self.hidden_dim), dtype = torch.float32).cuda()   # Left shape: (memory_size, B, C)
        if self.hidden_states is None:
            self.hidden_states = torch.zeros((self.memory_size, bs, self.hidden_dim), dtype = torch.float32).cuda()   # Left shape: (memory_size, B, C)
            
        if self.last_video_frame_id is None:
            is_new_segment = torch.ones((bs,), dtype = torch.bool).cuda()
        else:
            is_new_segment = (video_frame_id != self.last_video_frame_id + self.sample_interval)
        self.last_video_frame_id = video_frame_id.clone()
        self.cell_states[:, is_new_segment] = torch.zeros((self.memory_size, is_new_segment.sum(), self.hidden_dim), dtype = torch.float32).cuda()
        self.hidden_states[:, is_new_segment] = torch.zeros((self.memory_size, is_new_segment.sum(), self.hidden_dim), dtype = torch.float32).cuda()
        
        features = self.memory_decoder(tgt = self.hidden_states.detach(), memory = obs, memory_key_padding_mask = obs_mask, pos=obs_pos,)  # Left shape: (memory_size, B, C)
        forget_gate = self.forget_gate_map(features).sigmoid()  # Left shape: (memory_size, B, 1)
        input_gate = self.input_gate_map(features).sigmoid()    # Left shape: (memory_size, B, 1)
        output_gate = self.output_gate_map(features).sigmoid()  # Left shape: (memory_size, B, 1)
        
        cell_candidate = self.cell_candidate_map(features)     # Left shape: (memory_size, B, C)
        self.cell_states = forget_gate * self.cell_states.detach() + input_gate * cell_candidate # Left shape: (memory_size, B, C)
        self.hidden_states = output_gate * torch.tanh(self.cell_states) # Left shape: (memory_size, B, C)
        return self.hidden_states
        

def get_GEM_model(cfg):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = GEM(
        backbone,
        transformer,
        state_dim=cfg['POLICY']['STATE_DIM'],
        chunk_size=cfg['POLICY']['CHUNK_SIZE'],
        camera_names=cfg['DATA']['CAMERA_NAMES'],
        cfg = cfg,
    )

    return model
