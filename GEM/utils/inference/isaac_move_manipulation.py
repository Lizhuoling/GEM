import pdb
import sys
import math
import scipy
import cv2
import os
import copy
import time
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from collections import deque
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from scipy.spatial.transform import Rotation

from GEM.utils.models.model_cfg.utils import read_model_cfg
from GEM.configs.utils import load_yaml_with_base
from GEM.utils.models.perception_models import HSVColorInstSegment
from GEM.utils.utils import quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply, quaternion_angle_diff, get_quaternion_A2B, quaternion_rotate, euler_zyx_to_quaternion, normal_from_cross_product
from GEM.utils.model_zoo import load_policy_and_stats

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class PointMoveManipulationTestEnviManager:
    def __init__(self, cfg, policy, stats):
        self.cfg = cfg
        self.policy = policy
        self.model_cfg = read_model_cfg(cfg['MODEL_CFG_PATH'])
        self.model_cfg['stats'] = stats
        
    def inference(self,):
        rewards = np.zeros((self.cfg['EVAL']['TEST_ENVI_NUM'],), dtype = np.float32)
        envi_manager = PointMoveManipulationManager(cfg = self.cfg, model_cfg = self.model_cfg,)

        with torch.no_grad():
            for env_id in range(self.cfg['EVAL']['TEST_ENVI_BATCH_NUM']):
                print("Start inference on environment {}...".format(env_id))
                while envi_manager.step <= self.cfg['EVAL']['INFERENCE_MAX_STEPS']:
                    envi_manager.run_one_step(policy = self.policy)
                    cv2.waitKey(1)
                rewards[env_id :  env_id + 1] = envi_manager.get_reward() # reward is a numpy array with the shape of (1,)
                envi_manager.reset()
        
        average_reward = np.mean(rewards)
        success_rate = np.sum(rewards) / (self.cfg['EVAL']['TEST_ENVI_NUM'] * self.model_cfg['envi_cfg']['num_obj_per_envi'])
        reward_info = dict(
            success_rate = success_rate,
            average_reward = average_reward
        )
        print(f'\average_reward: {average_reward} success_rate: {success_rate}\n')
        return reward_info

class PointMoveManipulationManager:
    def __init__(self, cfg, model_cfg):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.cam_visualization = self.model_cfg['envi_cfg']['cam_visualization']
        
        if self.model_cfg['envi_cfg']['envi_name'] == 'DynamicSimulator':
            from GEM.data_envi.dynamic_simulator import DynamicSimulator
            self.envi = DynamicSimulator(num_envs = 1, num_obj_per_envi = self.model_cfg['envi_cfg']['num_obj_per_envi'], belt_speed = self.model_cfg['envi_cfg']['belt_speed'], obj_init_range = self.model_cfg['envi_cfg']['obj_init_range'], \
                move_mode = self.model_cfg['envi_cfg']['move_mode'], obj_mode = self.model_cfg['envi_cfg']['obj_mode'], robot_name = self.model_cfg['envi_cfg']['robot_name'], seed = None, vis_marker = False)
        elif self.model_cfg['envi_cfg']['envi_name'] == 'MoveInsert':
            from GEM.data_envi.move_insert import MoveInsert
            self.envi = MoveInsert(num_envs = 1, num_obj_per_envi = self.model_cfg['envi_cfg']['num_obj_per_envi'], belt_speed = self.model_cfg['envi_cfg']['belt_speed'], obj_init_range = self.model_cfg['envi_cfg']['obj_init_range'], \
                obj_mode = self.model_cfg['envi_cfg']['obj_mode'], container_mode = self.model_cfg['envi_cfg']['container_mode'], robot_name = self.model_cfg['envi_cfg']['robot_name'], seed = None, vis_marker = False)
        self.envi_empty_run(iters = 1)
        self.step = 0
        
        self.percetion_manager = EnvironmentPerceptionManager(cfg, model_cfg, self.envi)
        self.control_manager = RobotControlManager(cfg, model_cfg, self.envi)
        
    def run_one_step(self, policy = None):
        self.envi.update_simulator_before_ctrl()
        vision_obs_dict = self.get_vision_observations()
        if vision_obs_dict is None: return None
        cur_time = self.envi.get_time()
        
        start_perception_time = time.time() 
        obs_dict = self.percetion_manager.process_obs(vision_obs_dict = vision_obs_dict, cur_skill = self.control_manager.cur_skill,  cur_time = cur_time)
        
        start_control_time = time.time()
        self.control_manager.execute_action(obs_dict, policy = policy)
        #print(f"Perception time: {start_control_time - start_perception_time}, Control time: {time.time() - start_control_time}, Total time: {time.time() - start_perception_time}")
        
        if self.cam_visualization and ('handcam_tracktarget_mask' in obs_dict.keys()):
            self.visualize_obs(vision_obs_dict, obs_dict)
        
        self.envi.update_simulator_after_ctrl()
        self.step += 1   
        
    def visualize_obs(self, vision_obs_dict, obs_dict):
        cmap = plt.get_cmap('Spectral_r')
        top_rgb_image = vision_obs_dict['top_rgb'][0].cpu().numpy().astype(np.uint8)
        hand_rgb_image = vision_obs_dict['hand_rgb'][0].cpu().numpy().astype(np.uint8)
        top_bgr_image = cv2.cvtColor(top_rgb_image, cv2.COLOR_RGBA2BGR)
        hand_bgr_image = cv2.cvtColor(hand_rgb_image, cv2.COLOR_RGBA2BGR)

        topcam_tracktarget_mask = obs_dict['topcam_target_mask'].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy().astype(np.uint8) * 255
        handcam_tracktarget_mask = obs_dict['handcam_tracktarget_mask'].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy().astype(np.uint8) * 255
        handcam_grasptarget_mask = obs_dict['handcam_grasptarget_mask'].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy().astype(np.uint8) * 255
        
        #hand_bgr_image = cv2.circle(hand_bgr_image, (160, 147), 5, (0, 0, 255), -1)
        '''hand_depth = vision_obs_dict['hand_depth'][0].cpu().numpy()
        norm_hand_depth = (hand_depth - np.min(hand_depth)) / min(max(np.max(hand_depth) - np.min(hand_depth), 1e-6), 1e3)
        hand_depth_vis = np.ascontiguousarray(cmap(norm_hand_depth)[:, :, :3])
        hand_depth_vis = (hand_depth_vis * 255).astype(np.uint8)'''
        
        '''grasp_pos3d = self.envi.get_grasp_pos3d()   # Left shape: (1, 3)
        handcam_grasp_target_uv = self.envi.project_target_world3d_to_hand_camuv(grasp_pos3d)[0]  # Left shape: (2,)
        cv2.circle(hand_bgr_image, (int(handcam_grasp_target_uv[0]), int(handcam_grasp_target_uv[1])), 5, (0, 0, 255), -1)'''
        
        vis_frame = np.concatenate((top_bgr_image, hand_bgr_image, topcam_tracktarget_mask, handcam_tracktarget_mask), axis = 1)
        cv2.imshow('vis', vis_frame)
        
    def envi_empty_run(self, iters = 1):
        '''
        Description:
            Run the simulation without control. This is used to initialize the environment.
        '''
        for _ in range(iters):
            self.envi.update_simulator_before_ctrl()
            self.envi.update_simulator_after_ctrl()
            
    def get_envs_seed(self,):
        return self.envi.random_seed
            
    def get_vision_observations(self,):
        vision_obs_list = self.envi.get_vision_observations()
        if vision_obs_list is None: return None
        vision_obs_dict = self.dict_to_cudatensor(vision_obs_list)
        return vision_obs_dict
        
    def dict_to_cudatensor(self, vis_obs):
        '''
        Description:
            Convert all the numpy arrays in vision observation dict as torch tensors on GPUs.
        '''
        keys = vis_obs[0].keys()
        obs_dict = {}
        for key in keys:
            obs_dict[key] = np.stack([ele[key] for ele in vis_obs], axis = 0)
        return {k: torch.Tensor(v).cuda() for k, v in obs_dict.items()}
    
    def get_reward(self,):
        return self.envi.get_reward()
    
    def clean_up(self,):
        self.envi.clean_up()
    
    def reset(self,):
        self.clean_up()
        self.__init__(cfg = self.cfg, model_cfg = self.model_cfg)
    
class EnvironmentPerceptionManager:
    def __init__(self, cfg, model_cfg, envi):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.inst_seg_model = self.get_instance_seg_model()
        self.obj_pos_estimator = Obj3DEstimator(cfg, model_cfg)
        self.step = 0
        self.hand_cam_proper_range = ((0.1 * self.envi.img_width, 0.1 * self.envi.img_height), (0.9 * self.envi.img_width, 0.9 * self.envi.img_height))
            
    def get_instance_seg_model(self,):
        if self.model_cfg['observation_cfg']['perception_model_name'] == 'HSVColorInstSegment':
            return HSVColorInstSegment()
        else:
            raise NotImplementedError
        
    def process_obs(self, vision_obs_dict, cur_skill, cur_time):
        obs_info = copy.deepcopy(vision_obs_dict)
        
        '''track_obj_pos3d = self.obj_pos_estimator.predict_pos(cur_time)
        track_obj_vel3d = self.obj_pos_estimator.predict_vel(cur_time)'''
        track_obj_pos3d, track_obj_vel3d = self.obj_pos_estimator.simple_pred_pos_and_vel(cur_time)
        if track_obj_vel3d is None:
            track_obj_vel3d = torch.zeros((1, 3), dtype = torch.float32).cuda()
        #track_obj_pos3d = None
        #track_obj_vel3d = torch.Tensor([0, self.model_cfg['envi_cfg']['belt_speed'], 0])[None].cuda()
        
        # Use instance segmentation model to get more precise environment observation.
        select_obj_idx = None
        if track_obj_pos3d == None or self.step % self.model_cfg['observation_cfg']['instance_seg_per_steps'] == 0:
            topcam_world_coors = self.envi.get_top_cam_world_coors(vision_obs_dict['top_depth'])[0]   # Left shape: (img_height, img_width, 3)
            inst_seg_result = self.inst_seg_model(vision_obs_dict,) # A list. list dim: num_obj, 2 (color name, mask)
            topcam_inst_seg = inst_seg_result['top_inst_seg']
            handcam_inst_seg = inst_seg_result['hand_inst_seg']
            if len(topcam_inst_seg) != 0:
                topcam_inst_seg_cls = [inst_cls for inst_cls, _ in topcam_inst_seg]
                topcam_inst_seg_mask = torch.stack([seg_result for _, seg_result in topcam_inst_seg], dim = 0) # Left shape: (num_obj, img_h, img_w)
            else:
                topcam_inst_seg_cls = []
                topcam_inst_seg_mask = torch.zeros((0, self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
            topcam_inst_seg_mask = (1 - F.max_pool2d(1 - topcam_inst_seg_mask.unsqueeze(1).float(), kernel_size=5, stride=1, padding=2)).bool().squeeze(1)  # Erosion operation. Left shape: (num_obj, img_h, img_w)
            topcam_mask_world_coors = topcam_inst_seg_mask.unsqueeze(-1) * topcam_world_coors[None]    # Left shape: (num_obj, img_h, img_w, 3)
            obj_center_coor3ds = topcam_mask_world_coors.sum(dim = (1, 2)) / topcam_inst_seg_mask.sum(dim = (1, 2)).unsqueeze(-1).clamp(min = 1)    # Left shape: (num_obj, 3)
            # Use the top center of an object to represent this object.
            obj_target_coor3ds = obj_center_coor3ds.clone()
            topcam_mask_world_coors[topcam_mask_world_coors == 0] = -1e3    # Left shape: (num_obj, img_h, img_w, 3)
            obj_target_coor3ds[:, 2] = topcam_mask_world_coors[..., 2].amax(dim = (1, 2))
            # Select the object to operate using the top camera information.
            track_obj_pos3d, select_obj_idx = self.model_cfg['topcam_schedule_on_objects_func'](mean_obj_coor3ds = obj_target_coor3ds, obj_cls = topcam_inst_seg_cls, operation_range = self.model_cfg['envi_cfg']['robot_operation_range'], cur_skill = cur_skill)   # Decide the target object.
            # Get the segmentation mask and point representation of the track target.
            if select_obj_idx != None:
                topcam_target_mask = topcam_inst_seg_mask[select_obj_idx] # Left shape: (img_h, img_w)
                track_obj_center3d = obj_center_coor3ds[select_obj_idx] # Left shape: (3,)
                handcam_track_target_uv = self.envi.project_target_world3d_to_hand_camuv(track_obj_center3d[None])[0]  # Left shape: (2,)
                if (handcam_track_target_uv[0] < self.hand_cam_proper_range[0][0]) or (handcam_track_target_uv[1] < self.hand_cam_proper_range[0][1]) or (handcam_track_target_uv[0] > self.hand_cam_proper_range[1][0]) \
                    or (handcam_track_target_uv[1] > self.hand_cam_proper_range[1][1]): handcam_track_target_uv = None    # Check if the target object is in the proper range of the hand camera.
                handcam_tracktarget_repr, handcam_tracktarget_mask = self.get_handcam_target_repr(target_handcam_uv = handcam_track_target_uv, target_cls = self.model_cfg['get_track_target_cls_func'](cur_skill), handcam_inst_seg = handcam_inst_seg, 
                                                                        hand_rgb = vision_obs_dict['hand_rgb'],  hand_depth = vision_obs_dict['hand_depth'], target_type = 'track_obj')
            else:   # No suitable track target is found
                topcam_target_mask = torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
                handcam_tracktarget_mask = torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
                handcam_tracktarget_repr = torch.zeros((0, 11), dtype = torch.float32).cuda()
                self.obj_pos_estimator.reset_deque()
            # Get the segmentation mask and point representation of the target at the gripper pixel.
            grasp_pos3d = self.envi.get_grasp_pos3d()   # Left shape: (1, 3)
            handcam_grasp_target_uv = self.envi.project_target_world3d_to_hand_camuv(grasp_pos3d)[0]  # Left shape: (2,)
            handcam_grasptarget_repr, handcam_grasptarget_mask = self.get_handcam_target_repr(target_handcam_uv = handcam_grasp_target_uv, target_cls = None, handcam_inst_seg = handcam_inst_seg, 
                                                                    hand_rgb = vision_obs_dict['hand_rgb'], hand_depth = vision_obs_dict['hand_depth'], target_type = 'grasp_obj')
            
            # Update the observation information.
            obs_info.update(dict(
                topcam_inst_seg_results = topcam_inst_seg,
                handcam_inst_seg_results = handcam_inst_seg,
                topcam_target_mask = topcam_target_mask,
                handcam_tracktarget_mask = handcam_tracktarget_mask,
                handcam_tracktarget_repr = handcam_tracktarget_repr,
                handcam_grasptarget_mask = handcam_grasptarget_mask,
                handcam_grasptarget_repr = handcam_grasptarget_repr,
            ))
        
        obs_info.update(dict(
            track_obj_pos3d = track_obj_pos3d,
            track_obj_vel3d = track_obj_vel3d,
        ))
        if select_obj_idx != None:
            self.obj_pos_estimator.append_value(track_obj_pos3d, cur_time)
        self.step += 1
        
        return obs_info
        
    def get_handcam_target_repr(self, target_handcam_uv, target_cls, handcam_inst_seg, hand_rgb, hand_depth, target_type):
        '''
        Description:
            Given the target handcam uv coordinate, segment the target image region, derive the 3D points, and then get the new 3D center coordinates in the environment world coordinate system.
        Input:
            target_handcam_uv shape: (2,)
            hand_rgb shape: (1, img_h, img_w, 3)
            hand_depth shape: (1, img_h, img_w)
        '''
        target_seg_result = self.get_target_inst_seg(target_handcam_uv, handcam_inst_seg, target_cls = target_cls, target_type = target_type)   # Left format: (cls_name, seg_mask)
        hand_coors = self.envi.get_hand_cam_cam_coors(hand_depth) # Left shape: (1, img_h, img_w, 3)
        hand_normals = normal_from_cross_product(hand_coors)    # Left shape: (1, img_h, img_w, 3)
        cls_mask = torch.zeros((hand_coors.shape[1], hand_coors.shape[2], 1), dtype = torch.float32).cuda() # background cls id is 0
        if target_seg_result[0] != None: 
            if target_type == 'track_obj':
                cls_mask[target_seg_result[1]] = 1 # The object to track motion
            elif target_type == 'grasp_obj':
                cls_mask[target_seg_result[1]] = 2  # The object graspped in hand
            else:
                raise NotImplementedError
                
        envs_target_cls = cls_mask.unsqueeze(0) # Left shape: (1, img_h, img_w, 1)
        # repr definition: 0-2: coors, 3: depth, 4: cls, 5-7: rgb, 8-11: normals
        handcam_repr = torch.cat((hand_coors, hand_depth.unsqueeze(-1), envs_target_cls, hand_rgb, hand_normals), dim = -1)   # Left shape: (1, img_h, img_w, 11)
        # Get the point representation
        seg_clsname, seg_mask = target_seg_result
        if seg_clsname != None:
            dilation_seg_mask = seg_mask[None, :, :, None].float()  # Left shape: (1, img_h, img_w, 1)
            dilation_seg_mask = F.max_pool2d(dilation_seg_mask, kernel_size=5, stride=1, padding=2)[0, :, :, 0].bool()  # Left shape: (img_h, img_w). Dilation operation.
            target_repr = handcam_repr[0][dilation_seg_mask]
        else:
            target_repr = torch.zeros((0, 11), dtype = torch.float32).cuda()  
        return target_repr, seg_mask    # target_repr shape: (n, 11), seg_mask shape: (img_h, img_w)
    
    def get_target_inst_seg(self, point, inst_seg, target_cls, target_type):
        '''
        Input:
            point: (2,)
            inst_seg: A list with each element as an instance segmentation instance. An instance format is (cls_name, mask), where mask shape is (h, w).
        '''
        if point == None:
            return [None, torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()]
        
        if target_cls is not None:
            inst_seg = [(cls_name, seg_result) for cls_name, seg_result in inst_seg if cls_name == target_cls]
        
        instance_seg_mask = torch.zeros((self.envi.img_height, self.envi.img_width), dtype = torch.bool).cuda()
        instance_cls_name = None
        
        if target_type == 'track_obj':
            if len(inst_seg) == 1:
                instance_cls_name, instance_seg_mask = inst_seg[0]
            elif len(inst_seg) > 1:
                mask_min_area = 1e6
                for cls_name, seg_result in inst_seg:
                    if self.point_inside_mask(point, seg_result):  # The point is within this mask
                        if seg_result.sum() < mask_min_area:
                            mask_min_area = seg_result.sum()
                            instance_seg_mask = seg_result
                            instance_cls_name = cls_name
        elif target_type == 'grasp_obj':
            mask_min_area = 1e6
            for cls_name, seg_result in inst_seg:
                if self.point_inside_mask(point, seg_result):  # The point is within this mask
                    if seg_result.sum() < mask_min_area:
                        mask_min_area = seg_result.sum()
                        instance_seg_mask = seg_result
                        instance_cls_name = cls_name
        
        return [instance_cls_name, instance_seg_mask]
        
    def point_inside_mask(self, point, mask):
        if point.isnan().any(): return False
        point = point.cpu().numpy()
        mask = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [max(contours, key=lambda ele: ele.shape[0])]
        contours = [cv2.convexHull(c, returnPoints=True) for c in contours]
        assert len(contours) == 1, 'The mask should only have one contour'
        contour = contours[0]
        result = cv2.pointPolygonTest(contour, point, False)
        if result > 0:  # The point is inside the contour
            return True
        else:
            return False

class RobotControlManager:
    def __init__(self, cfg, model_cfg, envi):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi  
        self.cur_skill = globals()[self.model_cfg['execution_cfg']['InitSkill']](cfg, model_cfg, envi, {'call_flag': 1})
        self.cur_skill.checkupdate = self.model_cfg[self.model_cfg['execution_cfg']['InitSkill'] + '_checkupdate'].__get__(self.cur_skill, self.cur_skill.__class__)
        
        self.robot_controller = KinematicsController(cfg, model_cfg, envi)
        
    def execute_action(self, obs_dict, policy = None):
        ctrl_signal = self.cur_skill(obs_dict, policy)
        action = self.robot_controller(ctrl_signal, obs_dict)   # Left shape: (1, 9)
        self.envi.execute_action_ik(action)
        # Check whether the skill needs to be updated.
        new_skill_name, last_skill_info_dict = self.cur_skill.checkupdate(obs_dict, ctrl_signal)
        if new_skill_name != None:
            self.cur_skill = globals()[new_skill_name](self.cfg, self.model_cfg, self.envi, last_skill_info_dict)
            self.cur_skill.checkupdate = self.model_cfg[new_skill_name + '_checkupdate'].__get__(self.cur_skill, self.cur_skill.__class__)
    
class TargetTrackSkill:
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi  
        self.last_skill_info_dict = last_skill_info_dict
        
        if 'target_track_handcam_dist' not in last_skill_info_dict.keys():
            self.handcam_track_dist = self.model_cfg['execution_cfg']['target_track_handcam_dist']
        else:
            self.handcam_track_dist = last_skill_info_dict['target_track_handcam_dist']
        
        if 'norm_gripper_ctrl' not in last_skill_info_dict.keys():
            self.norm_gripper_ctrl = torch.Tensor([self.model_cfg['execution_cfg']['target_track_gripper_ctrl'], self.model_cfg['execution_cfg']['target_track_gripper_ctrl']]).cuda()[None, :]  # (1, 2)
        else:
            self.norm_gripper_ctrl = torch.Tensor([last_skill_info_dict['norm_gripper_ctrl'], last_skill_info_dict['norm_gripper_ctrl']]).cuda()[None, :]  # (1, 2)
        
        if 'target_track_max_pos_noise' in last_skill_info_dict.keys():
            self.pos_noise = 2 * (torch.rand((3,), dtype=torch.float32, device='cuda') - 0.5) * last_skill_info_dict['target_track_max_pos_noise']
        else:
            self.pos_noise = torch.zeros((3,), dtype=torch.float32, device='cuda')
        
        self.handcam_track_ori = self.dire_vector_to_quat(v_target = self.model_cfg['execution_cfg']['target_track_handcam_ori'])
        
        self.step_cnt = 0
        self.hand_target_y = []
        self.hand_pos_y = []
        
    def __call__(self, obs_dict, policy = None):
        track_obj_pos3d = obs_dict['track_obj_pos3d'][None]   # Left shape: (1, 3)
        cams_ori = torch.Tensor(self.handcam_track_ori)[None].cuda() # Left shape: (1, 4)
        handcam_track_dist = torch.Tensor([self.handcam_track_dist,]).cuda() # Left shape: (1,)
        handcam_target_pos3d = self.compute_handcam_target_pos3d(target_pos = track_obj_pos3d, cam_ori = cams_ori, cam_target_dist = handcam_track_dist)    # Left shape: (1, 3)
        handcam_target_pos3d = handcam_target_pos3d + self.pos_noise[None]
        
        handcam2handend_offset, handcam2handend_ori = self.envi.compute_handcam2handend_transform() # handcam2handend_offset shape: (3,), handcam2handend_ori shape: (4,)
        handends_target_pos = handcam_target_pos3d + handcam2handend_offset[None] # Left shape: (1, 3)
        handends_target_ori = quaternion_rotate(cams_ori, handcam2handend_ori[None]) # Left shape: (1, 4)
        handends_target_ori = quaternion_multiply(torch.Tensor([-0.7071, 0, -0.7071, 0]).cuda()[None], handends_target_ori)
        norm_gripper_ctrl = self.norm_gripper_ctrl.clone()  # (1, 2)
        stable_speed = obs_dict['track_obj_vel3d']  # Left shape: (1, 3)
        
        ctrl_signal = dict(end_pos = handends_target_pos, end_ori = handends_target_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed)
        
        '''if self.step_cnt < 120:
            self.hand_target_y.append(handends_target_pos[0, 1].item())
            self.hand_pos_y.append(self.envi.get_robotend_pos()[0, 1].item())
        else:
            font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
            times_new_roman_font = fm.FontProperties(fname=font_path, size=24)
            #delta = self.hand_target_y[-1] - self.hand_pos_y[-1]
            #self.hand_pos_y = [ele + delta for ele in self.hand_pos_y]
            plt.figure(figsize=(12, 6))
            x = list(range(len(self.hand_target_y)))
            plt.scatter(x, self.hand_target_y, color='#1f77b4', label='Control Target', zorder=2)
            plt.scatter(x, self.hand_pos_y, color='#d62728', label='Real State', zorder=2)
            plt.xticks(fontproperties=times_new_roman_font)
            plt.yticks(fontproperties=times_new_roman_font)
            plt.xlabel('TimeStamp', fontsize=24, fontproperties=times_new_roman_font)
            plt.ylabel('Y-axis Position', fontsize=24, fontproperties=times_new_roman_font)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(prop= times_new_roman_font)
            plt.tight_layout()
            plt.savefig('velocity30', bbox_inches='tight', pad_inches=0, dpi=300)
            pdb.set_trace()
        self.step_cnt += 1'''
        
        return ctrl_signal
    
    def checkupdate(self, **kwargs):
        pass
        
    def dire_vector_to_quat(self, v_target, v_initial = [1, 0, 0]):
        '''
        Description:
            Compute the quaternion that transforms the direction represented by default_dire to dire_vector.
        Input:
            v_target: (3,) np.ndarray, the target direction vector.
            v_initial: (3,) np.ndarray, the initial direction vector.
        '''
        
        v_target = v_target / np.linalg.norm(v_target)
        axis = np.cross(v_initial, v_target)
        axis = axis / np.linalg.norm(axis)
        theta = math.acos(np.dot(v_initial, v_target))
        q_w = math.cos(theta / 2)
        q_x = axis[0] * math.sin(theta / 2)
        q_y = axis[1] * math.sin(theta / 2)
        q_z = axis[2] * math.sin(theta / 2)
        q = np.array([q_x, q_y, q_z, q_w])
        return q
    
    def compute_handcam_target_pos3d(self, target_pos, cam_ori, cam_target_dist):
        """
        Description:
            calculate the moving target positions of cameras.
        Input:
            target_pos: The 3D positions of targets. shape: (n, 3)
            cam_ori: The target orientation quaternion of the cameras. shape: (n, 4)
            cam_target_dist: The distance between cameras and targets. shape: (n, )
        """
        rotation_matrices = quaternion_to_rotation_matrix(cam_ori)
        # Compute the direction vector from camera to target (unit vector along the z-axis in camera frame)
        direction_vector = torch.tensor([1.0, 0.0, 0.0], device=target_pos.device).repeat(target_pos.shape[0], 1)
        # Transform the direction vector to the world frame
        direction_vector_world = torch.bmm(rotation_matrices, direction_vector.unsqueeze(2)).squeeze(2)
        # Normalize the direction vector to ensure it is a unit vector
        direction_vector_world = direction_vector_world / direction_vector_world.norm(dim=1, keepdim=True)
        # Compute the camera positions by moving backwards from the target positions along the direction vector
        camera_world_position = target_pos - direction_vector_world * cam_target_dist.unsqueeze(1)
        camera_world_position += torch.Tensor(self.model_cfg['execution_cfg']['target_track_handcam_offset']).cuda()[None]
        
        return camera_world_position
    
class TargetPickSkill:
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi 
        self.last_skill_info_dict = last_skill_info_dict
        
        self.default_gripper_ctrl = 1.0
        self.init_action = torch.Tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, self.default_gripper_ctrl], dtype = np.float32))[None].cuda() # Shape: (1, 8). Format: (delta_x, delta_y, delta_z, quat_x, quat_y, quat_z, quat_w, gripper_ctrl)
        self.action_mean, self.action_std = torch.Tensor(self.model_cfg['stats']['action_mean'])[None].cuda(), torch.Tensor(self.model_cfg['stats']['action_std'])[None].cuda() # Both shapes: (1, 8)
        self.norm_init_action = (self.init_action - self.action_mean) / self.action_std   # Left shape: (1, 8)
        self.norm_past_action = self.norm_init_action.clone()    # Left shape: (1, 8)
        
        self.action_exec_cnt = 0
        self.action_pred = None
        self.frame_id = 0
        
    def __call__(self, obs_dict, policy = None):
        ctrl_signal = {}
        # There is policy action prediction has not executed completely.
        if self.action_pred is not None and self.action_exec_cnt < self.action_pred.shape[0]:
            action_to_execute = self.action_pred[self.action_exec_cnt][None].clone()   # Left shape: (1, 8)
            self.action_exec_cnt += 1
            self.frame_id += 1
        # The perception model does not work in this step, so keep the last action.
        elif 'handcam_tracktarget_repr' not in obs_dict.keys():
            action_to_execute = self.norm_past_action * self.action_std + self.action_mean  # Left shape: (1, 8)
        else:
            batch_reprs, batch_past_action, batch_past_action_is_pad, video_frame_id = self.prepare_policy_input(obs_dict['handcam_tracktarget_repr'])
            norm_action_pred, status_pred = policy(repr = batch_reprs, past_action = batch_past_action, action = None, past_action_is_pad = batch_past_action_is_pad, \
                                action_is_pad = None, status = None, task_instruction_list = None, video_frame_id = video_frame_id, dataset_type = ['pick',])  # norm_action_pred shape: (1, chunk_size, 8), status_pred shape: (1,)
            action_pred = norm_action_pred[0] * self.action_std + self.action_mean
            self.action_pred = action_pred[:self.cfg['EVAL']['EXEC_CHUNK']] # Left shape: (exec_chunk, 8)
            self.action_exec_cnt = 0      
            action_to_execute = self.action_pred[self.action_exec_cnt][None].clone()   # Left shape: (1, 8)
            self.action_exec_cnt += 1  
            self.frame_id += 1  
            ctrl_signal['status_pred'] = status_pred
            
        self.norm_past_action = (action_to_execute - self.action_mean) / self.action_std    # Update the last executed action.
        translation_offset = action_to_execute[:, :3]   # Left shape: (1, 3)
        orientation_offset = action_to_execute[:, 3:7]  # Left shape: (1, 4)
        norm_gripper_ctrl = action_to_execute[:, 7:8].expand(-1, 2) # Left shape: (1, 2)
        stable_speed = torch.zeros((1, 3), dtype = torch.float32).cuda()  # Left shape: (1, 3)
        ctrl_signal.update(dict(translation_offset = translation_offset, orientation_offset = orientation_offset, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed))
        return ctrl_signal
        
    def prepare_policy_input(self, repr):
        repr_idxs = []
        for repr_range in self.cfg['DATA']['INPUT_REPR_KEY']:
            repr_idxs += list(range(repr_range[0], repr_range[1]))      
        repr = repr.cpu().numpy()
        repr = repr[:, repr_idxs]   # Left shape: (num_point, state_dim)
        batch_reprs = [repr]
        batch_past_action = self.norm_past_action[None] # Left shape: (1, 1, 8) corresponding to (num_envs, past_action_len, state_dim)
        batch_past_action_is_pad = torch.zeros((1, 1), dtype = torch.bool).cuda()  # Left shape: (1, 1) corresponding to (num_envs, past_action_len)
        video_frame_id = torch.Tensor([self.frame_id,]).long().cuda()   # Left shape: (1, )
        return batch_reprs, batch_past_action, batch_past_action_is_pad, video_frame_id
        
    def checkupdate(self, **kwargs):
        pass
    
class TargetTrackPickSkill(TargetTrackSkill, TargetPickSkill):
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        TargetTrackSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        TargetPickSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
    def __call__(self, obs_dict, policy = None):
        # First obtain the control signal from kinematics based target tracking.
        track_ctrl_signal = TargetTrackSkill.__call__(self, obs_dict, policy)
        # Then obtain the control signal from learning based target picking.
        pick_ctrl_signal = TargetPickSkill.__call__(self, obs_dict, policy)
        # Add the two control signals.
        end_pos = track_ctrl_signal['end_pos'] + pick_ctrl_signal['translation_offset']
        end_ori = quaternion_multiply(pick_ctrl_signal['orientation_offset'], track_ctrl_signal['end_ori'])
        norm_gripper_ctrl = pick_ctrl_signal['norm_gripper_ctrl']
        stable_speed = track_ctrl_signal['stable_speed']
        ctrl_signal = dict(end_pos = end_pos, end_ori = end_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed)
        if 'status_pred' in pick_ctrl_signal.keys():
            ctrl_signal['status_pred'] = pick_ctrl_signal['status_pred']
        return ctrl_signal
        
    def checkupdate(self, **kwargs):
        pass
    
class TargetPutSkill():
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
        self.default_gripper_ctrl = 0.0
        self.init_action = torch.Tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, self.default_gripper_ctrl], dtype = np.float32))[None].cuda() # Shape: (1, 8). Format: (delta_x, delta_y, delta_z, quat_x, quat_y, quat_z, quat_w, gripper_ctrl)
        self.action_mean, self.action_std = torch.Tensor(self.model_cfg['stats']['action_mean'])[None].cuda(), torch.Tensor(self.model_cfg['stats']['action_std'])[None].cuda() # Both shapes: (1, 8)
        self.norm_init_action = (self.init_action - self.action_mean) / self.action_std   # Left shape: (1, 8)
        self.norm_past_action = self.norm_init_action.clone()    # Left shape: (1, 8)
        
        self.action_exec_cnt = 0
        self.action_pred = None
        self.frame_id = 0
        
    def __call__(self, obs_dict, policy = None):
        ctrl_signal = {}
        # There is policy action prediction has not executed completely.
        if self.action_pred is not None and self.action_exec_cnt < self.action_pred.shape[0]:
            action_to_execute = self.action_pred[self.action_exec_cnt][None].clone()   # Left shape: (1, 8)
            self.action_exec_cnt += 1
            self.frame_id += 1
        # The perception model does not work in this step, so keep the last action.
        elif 'handcam_tracktarget_repr' not in obs_dict.keys():
            action_to_execute = self.norm_past_action * self.action_std + self.action_mean  # Left shape: (1, 8)
        else:
            target_repr = torch.cat((obs_dict['handcam_tracktarget_repr'], obs_dict['handcam_grasptarget_repr']), dim = 0)
            batch_reprs, batch_past_action, batch_past_action_is_pad, video_frame_id = self.prepare_policy_input(target_repr)
            norm_action_pred, status_pred = policy(repr = batch_reprs, past_action = batch_past_action, action = None, past_action_is_pad = batch_past_action_is_pad, \
                                action_is_pad = None, status = None, task_instruction_list = None, video_frame_id = video_frame_id, dataset_type = ['put',])  # norm_action_pred shape: (1, chunk_size, 8), status_pred shape: (1,)
            action_pred = norm_action_pred[0] * self.action_std + self.action_mean
            self.action_pred = action_pred[:self.cfg['EVAL']['EXEC_CHUNK']] # Left shape: (exec_chunk, 8)
            self.action_exec_cnt = 0      
            action_to_execute = self.action_pred[self.action_exec_cnt][None].clone()   # Left shape: (1, 8)
            self.action_exec_cnt += 1  
            self.frame_id += 1  
            ctrl_signal['status_pred'] = status_pred
            
        self.norm_past_action = (action_to_execute - self.action_mean) / self.action_std    # Update the last executed action.
        translation_offset = action_to_execute[:, :3]   # Left shape: (1, 3)
        orientation_offset = action_to_execute[:, 3:7]  # Left shape: (1, 4)
        norm_gripper_ctrl = action_to_execute[:, 7:8].expand(-1, 2) # Left shape: (1, 2)
        stable_speed = torch.zeros((1, 3), dtype = torch.float32).cuda()  # Left shape: (1, 3)
        ctrl_signal.update(dict(translation_offset = translation_offset, orientation_offset = orientation_offset, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed))
        return ctrl_signal
        
    def prepare_policy_input(self, repr):
        repr_idxs = []
        for repr_range in self.cfg['DATA']['INPUT_REPR_KEY']:
            repr_idxs += list(range(repr_range[0], repr_range[1]))      
        repr = repr.cpu().numpy()
        repr = repr[:, repr_idxs]   # Left shape: (num_point, state_dim)
        batch_reprs = [repr]
        batch_past_action = self.norm_past_action[None] # Left shape: (1, 1, 8) corresponding to (num_envs, past_action_len, state_dim)
        batch_past_action_is_pad = torch.zeros((1, 1), dtype = torch.bool).cuda()  # Left shape: (1, 1) corresponding to (num_envs, past_action_len)
        video_frame_id = torch.Tensor([self.frame_id,]).long().cuda()   # Left shape: (1, )
        return batch_reprs, batch_past_action, batch_past_action_is_pad, video_frame_id
    
class TargetTrackPutSkill(TargetTrackSkill, TargetPutSkill):
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        TargetTrackSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        TargetPutSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
    def __call__(self, obs_dict, policy = None):
        track_ctrl_signal = TargetTrackSkill.__call__(self, obs_dict, policy)
        put_ctrl_signal = TargetPutSkill.__call__(self, obs_dict, policy)
        # Add the two control signals.
        end_pos = track_ctrl_signal['end_pos'] + put_ctrl_signal['translation_offset']
        end_ori = quaternion_multiply(put_ctrl_signal['orientation_offset'], track_ctrl_signal['end_ori'])
        norm_gripper_ctrl = put_ctrl_signal['norm_gripper_ctrl']
        stable_speed = track_ctrl_signal['stable_speed']
        ctrl_signal = dict(end_pos = end_pos, end_ori = end_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed)
        if 'status_pred' in put_ctrl_signal.keys():
            ctrl_signal['status_pred'] = put_ctrl_signal['status_pred']
        return ctrl_signal
    
    def checkupdate(self, **kwargs):
        pass
    
class TargetTrackTeleSkill(TargetTrackSkill,):
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        TargetTrackSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
        self.translation_offset = None
        self.rotation_offset = None
        self.gripper_ctrl = None
        self.cur_sample_buf = None
        self.exit_to_TargetTrackPickSkill = False
        
        self.init_track_ctrl_signal = None
        
    def __call__(self, obs_dict, policy = None):
        # First obtain the control signal from kinematics based target tracking.
        track_ctrl_signal = TargetTrackSkill.__call__(self, obs_dict, policy)
        
        if self.init_track_ctrl_signal is None:
            self.init_track_ctrl_signal = {key: value.clone() for key, value in track_ctrl_signal.items()}
        
        if self.model_cfg['execution_cfg']['continuous_track_target_flag']:
            end_pos = track_ctrl_signal['end_pos']
            end_ori = track_ctrl_signal['end_ori']
            norm_gripper_ctrl = track_ctrl_signal['norm_gripper_ctrl']
        else:
            end_pos = self.init_track_ctrl_signal['end_pos']
            end_ori = self.init_track_ctrl_signal['end_ori']
            norm_gripper_ctrl = self.init_track_ctrl_signal['norm_gripper_ctrl']
        # Add the two control signals.
        if self.translation_offset is not None:
            end_pos = end_pos + torch.Tensor(self.translation_offset)[None].cuda()
            end_ori = quaternion_multiply(torch.Tensor(self.rotation_offset)[None].cuda(), end_ori)
            norm_gripper_ctrl = torch.Tensor([self.gripper_ctrl, self.gripper_ctrl])[None].cuda()
            self.cur_sample_buf = self.prepare_data_batch(obs_dict,)
        stable_speed = track_ctrl_signal['stable_speed']
        ctrl_signal = dict(end_pos = end_pos, end_ori = end_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed, exit_flag = self.exit_to_TargetTrackPickSkill)
        return ctrl_signal
    
    def prepare_data_batch(self, obs_dict):
        sample_dict = dict(
            top_rgb = obs_dict['top_rgb'][0].cpu().numpy().astype(np.uint8),
            top_depth = obs_dict['top_depth'][0].cpu().numpy(),
            hand_rgb = obs_dict['hand_rgb'][0].cpu().numpy().astype(np.uint8),
            hand_depth = obs_dict['hand_depth'][0].cpu().numpy(),
        )
        sample_dict['tgt_repr'] = obs_dict['handcam_tracktarget_repr'].cpu().numpy()
        sample_dict['grasp_tgt_repr'] = obs_dict['handcam_grasptarget_repr'].cpu().numpy()
        sample_dict['action'] = np.concatenate((self.translation_offset, self.rotation_offset, np.array([self.gripper_ctrl,])), axis = 0)  # self.gripper_ctrl is between 0 and 1
        sample_dict['hand_joint_pose'] = self.envi.dof_pos[0, :, 0].cpu().numpy()   # Left shape: (num_joints,)
        sample_dict['hand_end_pose'] = self.envi.rb_states[self.envi.hand_idxs][0].cpu().numpy()    # Left shape: (13,). 0-2: xyz, 3-6: quaternion, 7-9: linear speed, 10-12: angular speed
        return sample_dict
    
    def checkupdate(self, **kwargs):
        pass
    
class ScriptPutTargetSkill():
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        self.stage = 0
        
        self.last_translation = torch.Tensor([0., 0., 0.])[None].cuda()
        self.norm_gripper_ctrl = torch.Tensor([last_skill_info_dict['norm_gripper_ctrl'], last_skill_info_dict['norm_gripper_ctrl']]).cuda()[None]  # (1, 2)
        self.norm_gripper_ctrl = torch.clamp(self.norm_gripper_ctrl, min = 0, max = 1)
        
    def __call__(self, obs_dict, policy = None):
        container_pose = self.get_container_pose()  # Left shape: (7,). 0-2: xyz, 3-6: quaternion
        obj_pose = self.get_obj_pose()  # Left shape: (7,). 0-2: xyz, 3-6: quaternion
        
        if self.stage == 0:
            target_relative_pos_offset = container_pose[:3] + torch.Tensor(self.model_cfg['execution_cfg']['AlignRelativeOffset']).cuda() - obj_pose[:3]   # Left shape: (3,)
            if torch.norm(target_relative_pos_offset, p = 2) < 0.01:    # The first step of pose alignment has been completed 
                self.stage += 1
                
        if self.stage == 1:
            target_relative_pos_offset = container_pose[:3] + torch.Tensor(self.model_cfg['execution_cfg']['PutRelativeOffset']).cuda() - obj_pose[:3]   # Left shape: (3,)
            if torch.norm(target_relative_pos_offset, p = 2) < 0.01:    # The second step of moving to the releasing position has been completed.
                self.stage += 1
                
        if self.stage == 2:
            target_relative_pos_offset = container_pose[:3] + torch.Tensor(self.model_cfg['execution_cfg']['PutRelativeOffset']).cuda() - obj_pose[:3]
            self.norm_gripper_ctrl = torch.Tensor([1.0, 1.0]).cuda()[None] # Open the gripper.
            
        if torch.norm(target_relative_pos_offset, p = 2) > self.model_cfg['execution_cfg']['translation_max_dist']:
            target_relative_pos_offset = target_relative_pos_offset / torch.norm(target_relative_pos_offset, p = 2) * self.model_cfg['execution_cfg']['translation_max_dist']
        translation_offset = target_relative_pos_offset[None] + self.last_translation   # Left shape: (1, 3)
        self.last_translation = translation_offset.clone()
        orientation_offset = get_quaternion_A2B(obj_pose[3:7], torch.Tensor(self.model_cfg['execution_cfg']['TargetOrientation']).cuda())[None]    # Left shape: (1, 4)
        norm_gripper_ctrl = self.norm_gripper_ctrl.clone() # Left shape: (1, 2)
        ctrl_signal = dict(translation_offset = translation_offset, orientation_offset = orientation_offset, norm_gripper_ctrl = norm_gripper_ctrl,)
        return ctrl_signal
        
    def get_container_pose(self,):
        container_idxs = torch.tensor(np.array(self.envi.container_idxs), dtype = torch.long).to(self.envi.rb_states.device)  # Left shape: (num_envs, num_obj_per_env)
        container_pose = self.envi.rb_states[container_idxs,].clone()[0, 0, :7]  # Left shape: (7,). 0-2: xyz, 3-6: quaternion
        return container_pose
    
    def get_obj_pose(self,):
        obj_idxs = torch.tensor(np.array(self.envi.obj_idxs), dtype = torch.long).to(self.envi.rb_states.device)
        obj_pose = self.envi.rb_states[obj_idxs,].clone()[0, 0, :7]
        return obj_pose
        
    def checkupdate(self, **kwargs):
        pass
    
class TargetTrack_ScriptPutTargetSkill(TargetTrackSkill, ScriptPutTargetSkill):
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        TargetTrackSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        ScriptPutTargetSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        self.cur_sample_buf = None
        
    def __call__(self, obs_dict, policy = None):
        track_ctrl_signal = TargetTrackSkill.__call__(self, obs_dict, policy)
        put_ctrl_signal = ScriptPutTargetSkill.__call__(self, obs_dict, policy)
        self.cur_sample_buf = self.prepare_data_batch(obs_dict, put_ctrl_signal)

        end_pos = track_ctrl_signal['end_pos'] + put_ctrl_signal['translation_offset']
        end_ori = quaternion_multiply(put_ctrl_signal['orientation_offset'], track_ctrl_signal['end_ori'])
        norm_gripper_ctrl = put_ctrl_signal['norm_gripper_ctrl']
        stable_speed = track_ctrl_signal['stable_speed']
        ctrl_signal = dict(end_pos = end_pos, end_ori = end_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed)
        return ctrl_signal
    
    def prepare_data_batch(self, obs_dict, put_ctrl_signal):
        sample_dict = dict(
            top_rgb = obs_dict['top_rgb'][0].cpu().numpy().astype(np.uint8),
            top_depth = obs_dict['top_depth'][0].cpu().numpy(),
            hand_rgb = obs_dict['hand_rgb'][0].cpu().numpy().astype(np.uint8),
            hand_depth = obs_dict['hand_depth'][0].cpu().numpy(),
        )
        sample_dict['tgt_repr'] = obs_dict['handcam_tracktarget_repr'].cpu().numpy()
        sample_dict['grasp_tgt_repr'] = obs_dict['handcam_grasptarget_repr'].cpu().numpy()
        
        translation_offset = put_ctrl_signal['translation_offset'].cpu().numpy()[0]    # Left shape: (3,)
        orientation_offset = put_ctrl_signal['orientation_offset'].cpu().numpy()[0]    # Left shape: (4,)
        norm_gripper_ctrl = put_ctrl_signal['norm_gripper_ctrl'].cpu().numpy()[0, 0:1]  # Left shape: (1,)
        sample_dict['action'] = np.concatenate((translation_offset, orientation_offset, norm_gripper_ctrl), axis = 0)  # self.gripper_ctrl is between 0 and 1
        sample_dict['hand_joint_pose'] = self.envi.dof_pos[0, :, 0].cpu().numpy()   # Left shape: (num_joints,)
        sample_dict['hand_end_pose'] = self.envi.rb_states[self.envi.hand_idxs][0].cpu().numpy()    # Left shape: (13,). 0-2: xyz, 3-6: quaternion, 7-9: linear speed, 10-12: angular speed
        return sample_dict
        
    def checkupdate(self, **kwargs):
        pass
    
class ScriptPostProcessSkill(TargetTrackSkill):
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        TargetTrackSkill.__init__(self, cfg, model_cfg, envi, last_skill_info_dict)
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        self.cur_sample_buf = None
        self.step = 0
        
    def __call__(self, obs_dict, policy = None):
        ctrl_signal = TargetTrackSkill.__call__(self, obs_dict, policy)
        self.step += 1
        self.cur_sample_buf = self.prepare_data_batch(obs_dict)
        return ctrl_signal
    
    def prepare_data_batch(self, obs_dict):
        sample_dict = dict(
            top_rgb = obs_dict['top_rgb'][0].cpu().numpy().astype(np.uint8),
            top_depth = obs_dict['top_depth'][0].cpu().numpy(),
            hand_rgb = obs_dict['hand_rgb'][0].cpu().numpy().astype(np.uint8),
            hand_depth = obs_dict['hand_depth'][0].cpu().numpy(),
        )
        sample_dict['tgt_repr'] = obs_dict['handcam_tracktarget_repr'].cpu().numpy()
        sample_dict['grasp_tgt_repr'] = obs_dict['handcam_grasptarget_repr'].cpu().numpy()
        
        translation_offset = np.array([0, 0, 0], dtype = np.float32)    # Left shape: (3,)
        orientation_offset = np.array([0, 0, 0, 1], dtype = np.float32)    # Left shape: (4,)
        norm_gripper_ctrl = np.array([self.last_skill_info_dict['norm_gripper_ctrl'],], dtype = np.float32)  # Left shape: (1,)
        sample_dict['action'] = np.concatenate((translation_offset, orientation_offset, norm_gripper_ctrl), axis = 0)  # self.gripper_ctrl is between 0 and 1
        sample_dict['hand_joint_pose'] = self.envi.dof_pos[0, :, 0].cpu().numpy()   # Left shape: (num_joints,)
        sample_dict['hand_end_pose'] = self.envi.rb_states[self.envi.hand_idxs][0].cpu().numpy()    # Left shape: (13,). 0-2: xyz, 3-6: quaternion, 7-9: linear speed, 10-12: angular speed
        return sample_dict
    
    def checkupdate(self, **kwargs):
        pass
    
class RotateToTargetPoseSkill:
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
        self.icp_threshold = self.model_cfg['execution_cfg']['icp_threshold']
        if self.last_skill_info_dict['target_ref'].rsplit('.')[-1] == 'obj':
            ref_mesh = o3d.io.read_triangle_mesh(self.last_skill_info_dict['target_ref'])
            self.ref_points = ref_mesh.sample_points_uniformly(number_of_points=self.model_cfg['execution_cfg']['ref_sample_point_num'])
        
    def __call__(self, obs_dict, policy = None):
        trans_init = np.eye(4)
        handcam_grasptarget_mask = obs_dict['handcam_grasptarget_mask'] # Left shape: (h, w)
        handcam_world_coors = self.envi.get_hand_cam_world_coors(obs_dict['hand_depth'])    # Left shape: (1, h, w, 3)
        grasp_target_point3d = handcam_world_coors[0][handcam_grasptarget_mask] # Left shape: (num_point, 3)
        grasp_target_point3d = grasp_target_point3d.cpu().numpy()
        obj_points = o3d.geometry.PointCloud()
        obj_points.points = o3d.utility.Vector3dVector(grasp_target_point3d)
        
        source = obj_points
        target = self.ref_points
        source_fpfh = self.extract_fpfh_features(source)
        target_fpfh = self.extract_fpfh_features(target)
        ransac_result = self.ransac_registration(source, target, source_fpfh, target_fpfh)
        icp_result = self.icp_registration(source, target, ransac_result.transformation)
        obj_points.paint_uniform_color([1, 0.706, 0])
        self.ref_points.paint_uniform_color([0, 0.651, 0.929])
        obj_points.transform(icp_result.transformation)
        pdb.set_trace()
        o3d.visualization.draw_geometries([obj_points, self.ref_points])
        
    def extract_fpfh_features(self, pcd, radius_normal=0.1, radius_feature=0.25):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    def ransac_registration(self, source, target, source_fpfh, target_fpfh, distance_threshold=1):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999)
        )
        return result
    
    def icp_registration(self, source, target, initial_transform, threshold=0.5):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        return result
        
    def checkupdate(self, **kwargs):
        pass
    
class MoveToPosSkill:
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
        self.target_pos = np.array(self.model_cfg['execution_cfg']['move_to_target_pos'], dtype = np.float32)
        
        if self.model_cfg['execution_cfg']['trajectory_planner'] == 'CubicTrajectoryPlanner':
            self.traj_planner = CubicTrajectoryPlanner(time_interval = self.model_cfg['execution_cfg']['plan_step_time_interval'])
        else:
            raise NotImplementedError
        
        self.action_exec_cnt = 0
        self.plan_positions = None
        self.plan_velocities = None
        self.plan_orientations = None
        self.plan_gripper = last_skill_info_dict['norm_gripper_ctrl']
        
    def __call__(self, obs_dict, policy = None):
        if self.plan_positions is None:
            p0 = self.envi.get_robotend_pos().cpu().numpy()[0]
            v0 = self.envi.get_robotend_pos_vel().cpu().numpy()[0]
            pf = self.target_pos    # The position above the center of the container.
            vf = np.array([0, 0, 0,], dtype = np.float32)   # The robot hand is static when opening the gripper.
            self.plan_positions, self.plan_velocities = self.traj_planner.plan_trajectory(p0 = p0, v0 = v0, pf = pf, vf = vf, v_max = self.model_cfg['execution_cfg']['plan_max_velocity'])
            self.plan_orientations = self.envi.get_robotend_ori().expand(self.plan_positions.shape[0], -1).cpu().numpy()
        handends_target_pos = torch.Tensor(self.plan_positions[self.action_exec_cnt])[None].cuda()    # Left shape: (1, 3)
        handends_target_ori = torch.Tensor(self.plan_orientations[self.action_exec_cnt])[None].cuda()    # Left shape: (1, 4)
        norm_gripper_ctrl = self.plan_gripper    # Left shape: (1, 2)
        stable_speed = torch.Tensor(self.plan_velocities[self.action_exec_cnt])[None].cuda()    # Left shape: (1, 3)
        ctrl_signal = dict(end_pos = handends_target_pos, end_ori = handends_target_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed)
        if self.action_exec_cnt < self.plan_positions.shape[0] - 1: self.action_exec_cnt += 1
        return ctrl_signal
    
    def checkupdate(self, **kwargs):
        pass
    
class RotateGripperSkill:
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict
        
        robotend_ori = self.envi.get_robotend_ori()    # Left shape: (1, 4)
        rotate_euler = torch.Tensor(self.model_cfg['execution_cfg']['rotate_euler']).cuda()[None].expand(robotend_ori.shape[0], -1)    # Left shape: (1, 3)
        rotate_quat = euler_zyx_to_quaternion(rotate_euler, degrees = False)    # Left shape: (1, 4)
        
        self.end_pos = self.envi.get_robotend_pos()    # Left shape: (1, 3)
        self.end_ori = quaternion_multiply(rotate_quat, robotend_ori)   # Left shape: (1, 4)
        self.norm_gripper_ctrl = last_skill_info_dict['norm_gripper_ctrl']
        self.stable_speed = torch.Tensor([0.0, 0.0, 0.0])[None].cuda()    # Left shape: (1, 3)
    
    def __call__(self, obs_dict, policy = None):
        ctrl_signal = dict(end_pos = self.end_pos, end_ori = self.end_ori, norm_gripper_ctrl = self.norm_gripper_ctrl, stable_speed = self.stable_speed)
        return ctrl_signal
        
class OpenGripperSkill:
    def __init__(self, cfg, model_cfg, envi, last_skill_info_dict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.last_skill_info_dict = last_skill_info_dict

        self.action_exec_cnt = 0
        self.T = self.model_cfg['execution_cfg']['OpenGripper_T']
        
    def __call__(self, obs_dict, policy = None):
        handends_target_pos = self.envi.get_robotend_pos()
        handends_target_ori = self.envi.get_robotend_ori()
        norm_gripper_ctrl = torch.Tensor([1.0, 1.0])[None].cuda()    # Left shape: (1, 2)
        stable_speed = torch.Tensor([0.0, 0.0, 0.0])[None].cuda()    # Left shape: (1, 3)
        self.action_exec_cnt += 1
        ctrl_signal = dict(end_pos = handends_target_pos, end_ori = handends_target_ori, norm_gripper_ctrl = norm_gripper_ctrl, stable_speed = stable_speed, exec_step = self.action_exec_cnt)
        return ctrl_signal

    def checkupdate(self, **kwargs):
        pass
    
class KinematicsController:
    def __init__(self, cfg, model_cfg, envi):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.envi = envi
        self.init_robot_end_xyz = self.envi.get_robotend_pos()  # Left shpae: (1, 3)
        
        self.pos_pid = PositionPID(kp = self.model_cfg['control_cfg']['Position_PID_kp'], ki = self.model_cfg['control_cfg']['Position_PID_ki'], kd = self.model_cfg['control_cfg']['Position_PID_kd'], output_range = self.model_cfg['control_cfg']['Position_PID_output_range'])
        self.velo_pid = VelocityPID(kp = self.model_cfg['control_cfg']['Velocity_PID_kp'], ki = self.model_cfg['control_cfg']['Velocity_PID_ki'], kd = self.model_cfg['control_cfg']['Velocity_PID_kd'], output_range = self.model_cfg['control_cfg']['Velocity_PID_output_range'])
        self.ang_pid = AngularPID(kp = self.model_cfg['control_cfg']['Angular_PID_kp'], ki = self.model_cfg['control_cfg']['Angular_PID_ki'], kd = self.model_cfg['control_cfg']['Angular_PID_kd'], output_range = self.model_cfg['control_cfg']['Angular_PID_output_range'])
        
    def __call__(self, ctrl_signal, obs_dict):
        end_pos, end_ori, norm_gripper_ctrl, stable_speed = ctrl_signal['end_pos'], ctrl_signal['end_ori'], ctrl_signal['norm_gripper_ctrl'], ctrl_signal['stable_speed']
        cur_time = self.envi.get_time()
        
        gripper_upper_limit, gripper_lower_limit = torch.Tensor(self.envi.robot_upper_limits[7:]).cuda()[None], torch.Tensor(self.envi.robot_lower_limits[7:]).cuda()[None]
        norm_gripper_ctrl = norm_gripper_ctrl.clone()
        norm_gripper_ctrl[norm_gripper_ctrl < 0.5] = 0
        gripper_ctrl = norm_gripper_ctrl * (gripper_upper_limit - gripper_lower_limit) + gripper_lower_limit
        
        end_pos[end_pos.isnan().any(dim = 1)] = self.init_robot_end_xyz
        velo_tgt = self.pos_pid(tgt_pos = end_pos[0], cur_pos = self.envi.get_robotend_pos()[0], time = cur_time, stable_speed = stable_speed[0])    # velo_tgt shape: (3,)
        ctrl_pos = self.velo_pid(tgt_vel = velo_tgt, cur_vel = self.envi.get_robotend_pos_vel()[0], cur_pos = self.envi.get_robotend_pos()[0], time = cur_time)    # ctrl_pos shape: (3,)
        ctrl_ori = self.ang_pid(tgt_ori = end_ori[0], cur_ori = self.envi.get_robotend_ori()[0], time = cur_time)   # ctrl_pos shape: (3,)
        action = torch.cat((ctrl_pos[None], ctrl_ori[None], gripper_ctrl), dim = 1)   # Left shape: (1, 9)
        return action
        
class Obj3DEstimator:
    def __init__(self, cfg, model_cfg):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.deque_len = self.model_cfg['observation_cfg']['obj_track_queue_max_len']
        self.obj_pos_deque = deque(maxlen = self.deque_len)
        self.obj_time_deque = deque(maxlen = self.deque_len)
        
    def append_value(self, obj_pos, time):
        self.obj_pos_deque.append(obj_pos)
        self.obj_time_deque.append(time)
        
    def reset_deque(self,):
        self.obj_pos_deque = deque(maxlen = self.deque_len)
        self.obj_time_deque = deque(maxlen = self.deque_len)
        
    def simple_pred_pos_and_vel(self, pred_time):
        if len(self.obj_pos_deque) < 3: return None, None
        pos_traj = np.array([ele.cpu().numpy() for ele in self.obj_pos_deque])
        t_traj = np.array([ele for ele in self.obj_time_deque])
        
        vel_pred = np.zeros((3,), dtype = np.float32)
        vel_pred[1] = ((pos_traj[-1] - pos_traj[0]) / (t_traj[-1] - t_traj[0]))[1]
        pos_pred = pos_traj[-1] + vel_pred * (pred_time - t_traj[-1])
        return torch.Tensor(pos_pred).cuda(), torch.Tensor(vel_pred).cuda()
        
    
    def predict_pos(self, pred_time):
        if len(self.obj_pos_deque) < 3: return None
        x_traj = np.array([ele[0].cpu().numpy() for ele in self.obj_pos_deque])
        y_traj = np.array([ele[1].cpu().numpy() for ele in self.obj_pos_deque])
        z_traj = np.array([ele[2].cpu().numpy() for ele in self.obj_pos_deque])
        t_traj = np.array([ele for ele in self.obj_time_deque]).reshape(-1, 1)
        kernel = ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp_x = GaussianProcessRegressor(kernel=kernel)
        gp_y = GaussianProcessRegressor(kernel=kernel)
        gp_z = GaussianProcessRegressor(kernel=kernel)
        gp_x.fit(t_traj, x_traj)
        gp_y.fit(t_traj, y_traj)
        gp_z.fit(t_traj, z_traj)
        t_tgt = np.array([[pred_time,],])
        x_pred = gp_x.predict(t_tgt)
        y_pred = gp_y.predict(t_tgt)
        z_pred = gp_z.predict(t_tgt)
        pred_pos3d = torch.Tensor(np.concatenate((x_pred, y_pred, z_pred), axis = -1)).cuda()   # Left shape: (3,)
        return pred_pos3d
    
    def predict_vel(self, pred_time):
        if len(self.obj_pos_deque) < 3: return None
        x_traj = np.array([ele[0].cpu().numpy() for ele in self.obj_pos_deque])
        y_traj = np.array([ele[1].cpu().numpy() for ele in self.obj_pos_deque])
        z_traj = np.array([ele[2].cpu().numpy() for ele in self.obj_pos_deque])
        t_traj = np.array([ele for ele in self.obj_time_deque]).reshape(-1, 1)
        time_diffs = np.diff(t_traj.flatten())
        time_diffs = np.clip(time_diffs, 1e-3, None)
        vel_x_traj = np.diff(x_traj) / time_diffs
        vel_y_traj = np.diff(y_traj) / time_diffs
        vel_z_traj = np.diff(z_traj) / time_diffs
        t_vel_traj = t_traj[:-1]
        kernel = ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp_vel_x = GaussianProcessRegressor(kernel=kernel)
        gp_vel_y = GaussianProcessRegressor(kernel=kernel)
        gp_vel_z = GaussianProcessRegressor(kernel=kernel)
        gp_vel_x.fit(t_vel_traj, vel_x_traj)
        gp_vel_y.fit(t_vel_traj, vel_y_traj)
        gp_vel_z.fit(t_vel_traj, vel_z_traj)
        t_tgt = np.array([[pred_time,]])
        vel_x_pred = gp_vel_x.predict(t_tgt)
        vel_y_pred = gp_vel_y.predict(t_tgt)
        vel_z_pred = gp_vel_z.predict(t_tgt)
        pred_vel3d = torch.Tensor(np.concatenate((vel_x_pred, vel_y_pred, vel_z_pred), axis=-1)).cuda()  # shape: (3,)
        return pred_vel3d
    
    def __len__(self,):
        return len(self.obj_pos_deque)
    
class PositionPID():
    def __init__(self, kp, ki, kd, output_range):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0
        self.output_range = output_range
        
    def __call__(self, tgt_pos, cur_pos, time, stable_speed):
        '''
        Input:
            tgt_pos shape: (3,)
            cur_pos shape: (3,)
            time shape: a float
            stable_speed shape: (3,) 
        Output:
            velo_tgt shape: (3)
        '''
        error = tgt_pos - cur_pos
        time_interval = time - self.prev_time
        self.integral += error * time_interval
        derivative = (error - self.prev_error) / time_interval
        velo_tgt = self.kp * error + self.ki * self.integral + self.kd * derivative
        velo_tgt = velo_tgt + stable_speed
        self.prev_error = error
        self.prev_time = time
        velo_tgt = torch.clamp(velo_tgt, self.output_range[0], self.output_range[1])
        return velo_tgt
        
    def reset(self, time):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time
        
class VelocityPID():
    def __init__(self, kp, ki, kd, output_range):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0
        self.output_range = output_range
        
    def __call__(self, tgt_vel, cur_vel, cur_pos, time):
        '''
        Input:
            tgt_vel shape: (3,)
            cur_vel shape: (3,)
            cur_pos shape: (3,)
            time shape: a float
        Output:
            ctrl shape: (3)
        '''
        error = tgt_vel - cur_vel
        time_interval = time - self.prev_time
        self.integral += error * time_interval
        derivative = (error - self.prev_error) / time_interval
        ctrl = self.kp * error + self.ki * self.integral + self.kd * derivative
        ctrl = torch.clamp(ctrl, self.output_range[0], self.output_range[1])
        ctrl = ctrl + cur_pos
        self.prev_error = error
        self.prev_time = time
        return ctrl
        
    def reset(self, time):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time
        
class AngularPID():
    def __init__(self, kp, ki, kd, output_range):
        """
        Initialize the Angular PID Controller
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param output_range: Output range limit (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_range = output_range  # Output range limit
        self.integral = 0  # Integral term initialized as zero vector
        self.prev_error = 0  # Previous error initialized as zero vector
        self.prev_time = 0.0  # Previous time initialized as zero

    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw)
        :param q: Quaternion, shape (4,)
        :return: Euler angles, shape (3,)
        """   
        x, y, z, w = q
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = torch.asin(2 * (w * y - z * x))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return torch.Tensor([roll, pitch, yaw]).to(q.device)

    def __call__(self, tgt_ori, cur_ori, time):
        """
        Calculate the control signal quaternion
        :param tgt_ori: Target quaternion orientation, shape (4,)
        :param cur_ori: Current quaternion orientation, shape (4,)
        :param time: Current time (float)
        :return: Control signal quaternion, shape (4,)
        """
        # Convert quaternions to Euler angles
        tgt_euler = self.quaternion_to_euler(tgt_ori)
        cur_euler = self.quaternion_to_euler(cur_ori)

        # Calculate Euler angle errors
        error = tgt_euler - cur_euler

        # Normalize errors to [-pi, pi]
        error = torch.remainder(error + torch.pi, 2 * torch.pi) - torch.pi

        # Calculate time interval
        time_interval = time - self.prev_time

        # Update integral term
        self.integral += error * time_interval

        # Calculate derivative term
        derivative = (error - self.prev_error) / time_interval

        # Calculate control signal
        control_delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Clip the control signal to the specified range
        control_delta = torch.clamp(control_delta, self.output_range[0], self.output_range[1])
        control_signal = cur_euler + control_delta
        # Update previous error and time
        self.prev_error = error
        self.prev_time = time
        # Convert control signal to quaternion
        control_quaternion = self.control_signal_to_quaternion(control_signal)
        return control_quaternion

    def control_signal_to_quaternion(self, control_signal):
        """
        Convert the control signal (Euler angles) to a quaternion
        :param control_signal: Control signal (Euler angles), shape (3,)
        :return: Control quaternion, shape (4,)
        """
        roll, pitch, yaw = control_signal
        cr = torch.cos(roll / 2)
        sr = torch.sin(roll / 2)
        cp = torch.cos(pitch / 2)
        sp = torch.sin(pitch / 2)
        cy = torch.cos(yaw / 2)
        sy = torch.sin(yaw / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.tensor([x, y, z, w]).cuda()

    def reset(self, time):
        """
        Reset the PID controller
        :param time: Current time (float)
        """
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time
    
class CubicTrajectoryPlanner:
    def __init__(self, time_interval=0.02):
        self.time_interval = time_interval

    def cubic_interpolation(self, p0, v0, pf, vf, T):
        T = T.item() if isinstance(T, np.ndarray) else T
        a = np.zeros(3)
        b = np.zeros(3)
        c = np.zeros(3)
        d = np.zeros(3)
        for i in range(3):
            A = np.array([
                [T**3, T**2],
                [3*T**2, 2*T]
            ], dtype=np.float64)

            B = np.array([
                pf[i] - p0[i] - v0[i] * T,
                vf[i] - v0[i]
            ], dtype=np.float64).reshape(-1, 1)
            ab = np.linalg.pinv(A) @ B
            a[i], b[i] = ab.flatten()
            c[i] = v0[i]
            d[i] = p0[i]

        return a, b, c, d

    def cubic_velocity_constraint(self, T, p0, v0, pf, vf, v_max):
        T = T.item() if isinstance(T, np.ndarray) else T
        a, b, c, d = self.cubic_interpolation(p0, v0, pf, vf, T)
        t = np.linspace(0, T, 100)
        v = 3 * a * t[:, np.newaxis]**2 + 2 * b * t[:, np.newaxis] + c
        speed = np.linalg.norm(v, axis=1)
        return v_max - np.max(speed)  # Ensure speed <= v_max

    def cubic_velocity_constraint_min(self, T, p0, v0, pf, vf, v_max):
        T = T.item() if isinstance(T, np.ndarray) else T
        a, b, c, d = self.cubic_interpolation(p0, v0, pf, vf, T)
        t = np.linspace(0, T, 100)
        v = 3 * a * t[:, np.newaxis]**2 + 2 * b * t[:, np.newaxis] + c
        speed = np.linalg.norm(v, axis=1)
        return np.min(speed) + v_max  # Ensure speed >= -v_max

    def plan_trajectory(self, p0, v0, pf, vf, v_max):
        distance = np.linalg.norm(pf - p0)
        T_guess = max(distance / v_max, 1e-3)
        result = scipy.optimize.minimize(
            lambda T: T[0],
            T_guess,
            constraints=[
                {'type': 'ineq', 'fun': lambda T: self.cubic_velocity_constraint(T[0], p0, v0, pf, vf, v_max)},
            ],
            bounds=[(1e-6, None)],  # T > 0
            method='SLSQP',
            options={'maxiter': 100}
        )
        T_opt = result.x[0]
        a, b, c, d = self.cubic_interpolation(p0, v0, pf, vf, T_opt)
        num_points = int(T_opt / self.time_interval) + 1
        t = np.linspace(0, T_opt, num_points)
        t = t[:, np.newaxis]
        positions = a * t**3 + b * t**2 + c * t + d  # shape (num_points, 3)
        velocities = 3 * a * t**2 + 2 * b * t + c    # shape (num_points, 3)
        return positions, velocities
    