import numpy as np
import torch
from torch import nn
import open3d as o3d
import einops
import pdb
import sonata
from sonata.model import PointTransformerV3

class CustomSonata(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        level_feat_ch = [512, 384, 192]
        self.use_level_num = 3
        self.num_features = 0
        for i in range(self.use_level_num):
            self.num_features += level_feat_ch[i]
            
        # Load sonata
        ckpt = torch.load(self.cfg['POLICY']['BACKBONE_PATH'])
        ckpt['config']['enc_patch_size'] = (64, 64, 64, 64, 64)
        ckpt['config']['enable_flash'] = False
        self.sonata = PointTransformerV3(**ckpt["config"])
        self.sonata.load_state_dict(ckpt["state_dict"])
        # Data preproceess
        self.transform = sonata.transform.GEM_preprocess(cfg)
        
    def forward(self, repr_list):
        bs = len(repr_list)
        scene_coord_idxs, depth_idxs, cls_idxs, normal_idxs = self.prepare_collect_idxs()
        assert (scene_coord_idxs is not None) and (cls_idxs is not None)
        
        batch_coord = []
        batch_depth = []
        batch_cls = []
        batch_normal = []
        batch_idxs = []
        for batch_idx, repr in enumerate(repr_list):           
            if scene_coord_idxs is not None:
                sample_coord = repr[:, scene_coord_idxs]
            else:
                sample_coord = np.zeros((repr.shape[0], 3), dtype = np.float32)
            batch_coord.append(sample_coord)
            
            if depth_idxs is not None:
                sample_depth = repr[:, depth_idxs]
            else:
                sample_depth = np.zeros((repr.shape[0], 1), dtype = np.float32)
            batch_depth.append(sample_depth)
            
            if cls_idxs is not None:
                sample_cls = repr[:, cls_idxs]
            else:
                sample_cls = np.zeros((repr.shape[0], 1), dtype = np.float32)
            batch_cls.append(sample_cls)

            if normal_idxs is not None:
                sample_normal = repr[:, normal_idxs]
            else:
                sample_normal = np.zeros((repr.shape[0], 3), dtype = np.float32)
            batch_normal.append(sample_normal)
            
            sample_batch_idxs = np.ones((repr.shape[0],), dtype = np.int32) * batch_idx
            batch_idxs.append(sample_batch_idxs)
        
        batch_coord = np.concatenate(batch_coord, axis = 0) # Left shape: (n, 3)
        batch_depth = np.concatenate(batch_depth, axis = 0) # Left shape: (n, 1)
        batch_cls = np.concatenate(batch_cls, axis = 0) # Left shape: (n, 1)
        batch_color = np.concatenate((batch_depth, batch_depth, batch_cls), axis = 1).astype(np.float32) # Left shape: (n, 3)
        batch_normal = np.concatenate(batch_normal, axis = 0)   # Left shape: (n, 3)
        batch_idxs = np.concatenate(batch_idxs, axis = 0)   # Left shape: (n)
        
        data = dict(coord=batch_coord, color=batch_color, depth = batch_depth, normal=batch_normal, segment=batch_cls, batch=batch_idxs, batch_size = bs)
        data = self.transform(data)

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        
        point = self.sonata(data)
        for _ in range(self.use_level_num - 1):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
            
        feature = point.feat    # (new_n, c)
        feature_batch_idxs = point.batch    # (new_n,)
        batch_repr_list = []
        batch_repr_is_pad_list = []
        batch_max_point_num = 0
        for i in range(bs):
            batch_feat = feature[feature_batch_idxs == i]
            batch_repr_list.append(batch_feat)
            if batch_feat.shape[0] > batch_max_point_num:
                batch_max_point_num = batch_feat.shape[0]
        for i in range(bs):
            pad_feat = torch.zeros((batch_max_point_num - batch_repr_list[i].shape[0], batch_repr_list[i].shape[1]), dtype = torch.float32).cuda()
            repr_is_pad = torch.cat((torch.zeros((batch_repr_list[i].shape[0],), dtype = torch.bool), torch.ones((pad_feat.shape[0],), dtype = torch.bool)), dim = 0).cuda()
            batch_repr_is_pad_list.append(repr_is_pad)
            batch_repr_list[i] = torch.cat((batch_repr_list[i], pad_feat), dim = 0)
            
        batch_repr = torch.stack(batch_repr_list, dim = 0)    # (bs, max_n, c)
        batch_repr_is_pad = torch.stack(batch_repr_is_pad_list, dim = 0)    # (bs, max_n)
        
        return batch_repr, batch_repr_is_pad 
        
    def prepare_collect_idxs(self,):
        repr_idxs = []
        for repr_range in self.cfg['DATA']['INPUT_REPR_KEY']:
            repr_idxs += list(range(repr_range[0], repr_range[1]))
        repr_idxs = set(repr_idxs)
        scene_coord_idxs = [0, 1, 2] if all(item in repr_idxs for item in [0, 1, 2]) else None
        depth_idxs = [3,] if all(item in repr_idxs for item in [3,]) else None
        cls_idxs = [4,] if all(item in repr_idxs for item in [4,]) else None
        normal_idxs = [8, 9, 10] if all(item in repr_idxs for item in [8, 9, 10]) else None
        return scene_coord_idxs, depth_idxs, cls_idxs, normal_idxs

def get_custom_sonata(cfg):
    return CustomSonata(cfg)