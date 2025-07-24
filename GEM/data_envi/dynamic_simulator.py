from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_conjugate, quat_mul

import pdb
import math
import numpy as np
import torch
import random
import time
from pathlib import Path
import os
import copy
import h5py
import cv2
import json
from typing import Dict, Optional
from pyquaternion import Quaternion
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from itertools import combinations
import argparse

from GEM.utils.utils import quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply, quaternion_angle_diff, get_quaternion_A2B, quaternion_rotate, normal_from_cross_product

class DynamicSimulator():
    def __init__(self, num_envs = 1, num_obj_per_envi = 5, belt_speed = -0.2, obj_init_range = ((-0.15, 0.15), (0.1, 0.12)), move_mode = 'linear', obj_mode = 'base_mixture', robot_name = 'ur10e', seed = None, vis_marker = False):
        self.num_envs = num_envs
        self.num_obj_per_envi = num_obj_per_envi
        self.belt_speed = belt_speed
        self.obj_init_range = obj_init_range
        self.move_mode = move_mode
        self.obj_mode = obj_mode
        self.robot_name = robot_name
        self.vis_marker = vis_marker
        self.asset3d_root = os.path.join(os.path.abspath(__file__).rsplit('/', 2)[0], 'assets') 

        self.args = self.create_args(headless = True)
        
        self.create_gym_env()
        self.init_simulate_env(seed = seed)
        
        self.objs_vel_PID = BatchVelocityPID(obj_num = num_envs * num_obj_per_envi, kp = 5, ki = 10, kd = 0.0005)

    def create_args(self, headless=False, no_graphics=False, custom_parameters=[]):
        parser = argparse.ArgumentParser(description="Isaac Gym")
        if headless:
            parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
        if no_graphics:
            parser.add_argument('--nographics', action='store_true',
                                help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
        parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
        parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
        parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

        physics_group = parser.add_mutually_exclusive_group()
        physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
        physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

        parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
        parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
        parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

        params = []
        if headless:
            params.append('--headless')
        args = parser.parse_args(params)

        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)
        pipeline = args.pipeline.lower()

        assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
        args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

        if args.sim_device_type != 'cuda' and args.flex:
            print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
            args.sim_device = 'cuda:0'
            args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)

        if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
            print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
            args.pipeline = 'CPU'
            args.use_gpu_pipeline = False

        # Default to PhysX
        args.physics_engine = gymapi.SIM_PHYSX
        args.use_gpu = (args.sim_device_type == 'cuda')

        if args.flex:
            args.physics_engine = gymapi.SIM_FLEX

        # Using --nographics implies --headless
        if no_graphics and args.nographics:
            args.headless = True

        if args.slices is None:
            args.slices = args.subscenes

        return args

    def create_gym_env(self,):
        torch.set_printoptions(precision=4, sci_mode=False)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()

        self.controller = "ik"
        self.sim_type = self.args.physics_engine

        # set torch device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def init_simulate_env(self, seed = None):
        if seed == None:
            self.random_seed = random.randint(0, 1e5)
        else:
            self.random_seed = seed
        self.set_seed(self.random_seed)

        self.damping = 0.05
        self.recoding_data_flag = False

        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # create table asset
        self.x_no_force_margin = 0.025
        table1_block1_dims = gymapi.Vec3(0.4, 4.0, 0.35)
        table1_block2_dims = gymapi.Vec3(0.4 - 2 * self.x_no_force_margin, table1_block1_dims.y, 0.05)
        table1_block3_dims = gymapi.Vec3(self.x_no_force_margin, table1_block1_dims.y, 0.05)
        table1_block4_dims = gymapi.Vec3(self.x_no_force_margin, table1_block1_dims.y, 0.05)
        self.table1_block1_dims = table1_block1_dims
        self.table1_block2_dims = table1_block2_dims
        table2_dims = gymapi.Vec3(0.4, 0.3, 0.2)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table1_block1_asset = self.gym.create_box(self.sim, table1_block1_dims.x, table1_block1_dims.y, table1_block1_dims.z, asset_options)
        table1_block2_asset = self.gym.create_box(self.sim, table1_block2_dims.x, table1_block2_dims.y, table1_block2_dims.z, asset_options)
        table1_block3_asset = self.gym.create_box(self.sim, table1_block3_dims.x, table1_block3_dims.y, table1_block3_dims.z, asset_options)
        table1_block4_asset = self.gym.create_box(self.sim, table1_block4_dims.x, table1_block4_dims.y, table1_block4_dims.z, asset_options)
        table2_asset = self.gym.create_box(self.sim, table2_dims.x, table2_dims.y, table2_dims.z, asset_options)

        # create container asset
        container_bottom_dims = gymapi.Vec3(0.36, 0.26, 0.01)
        container_front_dims  = gymapi.Vec3(0.4, 0.02, 0.2)
        container_back_dims   = gymapi.Vec3(0.4, 0.02, 0.2)
        container_left_dims   = gymapi.Vec3(0.02, 0.26, 0.2)
        container_right_dims  = gymapi.Vec3(0.02, 0.26, 0.2)
        self.container_dims = gymapi.Vec3(container_front_dims.x, container_bottom_dims.y + container_front_dims.y + container_back_dims.y, container_front_dims.z)
        container_front_pose_offset  = gymapi.Vec3(0, -0.14, 0.125)
        container_back_pose_offset   = gymapi.Vec3(0, 0.14, 0.125)
        container_left_pose_offset   = gymapi.Vec3(-0.19, 0, 0.125)
        container_right_pose_offset  = gymapi.Vec3(0.19, 0, 0.125)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        container_bottom_asset = self.gym.create_box(self.sim, container_bottom_dims.x, container_bottom_dims.y, container_bottom_dims.z, asset_options)
        container_front_asset  = self.gym.create_box(self.sim, container_front_dims.x, container_front_dims.y, container_front_dims.z, asset_options)
        container_back_asset   = self.gym.create_box(self.sim, container_back_dims.x, container_back_dims.y, container_back_dims.z, asset_options)
        container_left_asset   = self.gym.create_box(self.sim, container_left_dims.x, container_left_dims.y, container_left_dims.z, asset_options)
        container_right_asset  = self.gym.create_box(self.sim, container_right_dims.x, container_right_dims.y, container_right_dims.z, asset_options)

        obj_assets = []
        for i in range(self.num_obj_per_envi):
            if self.obj_mode == 'box':
                obj_type = 'box'
            elif self.obj_mode == 'capsule':
                obj_type = 'capsule'
            elif self.obj_mode == 'base_mixture':
                obj_type = random.choice(['box', 'capsule', 'cup'])
            else:
                obj_type = self.obj_mode
                
            if obj_type == 'box':
                box_size = 0.045 * random.uniform(0.9, 1.1)
                asset_options = gymapi.AssetOptions()
                obj_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
            elif obj_type == 'capsule':
                capsule_radius = 0.022 * random.uniform(0.9, 1.1)
                asset_options = gymapi.AssetOptions()
                obj_asset = self.gym.create_capsule(self.sim, capsule_radius, capsule_radius, asset_options)
            else:
                asset_options = gymapi.AssetOptions()
                asset_options.vhacd_enabled = True
                asset_options.vhacd_params = gymapi.VhacdParams()
                asset_options.vhacd_params.resolution = 300000
                asset_options.vhacd_params.max_num_vertices_per_ch = 3
                obj_asset = self.gym.load_asset(self.sim, self.asset3d_root, os.path.join('urdf', obj_type + '.urdf'), asset_options)
            obj_assets.append(obj_asset)
        
        if self.vis_marker:
            radius = 0.02
            asset_options = gymapi.AssetOptions()
            asset_options.density = 0
            asset_options.disable_gravity = True 
            marker_asset = self.gym.create_box(self.sim, 0.06, 0.04, 0.02, asset_options)
        
        # load the robot
        asset_root = "/home/cvte/Documents/isaacgym/assets"
        if self.robot_name == 'franka_panda':
            self.finger_name = "panda_hand"
            self.joint_dim = 9
            robot_asset_file = "urdf/franka_description/robots/franka_panda_fast.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
            self.body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
            # {'panda_hand': 8, 'panda_leftfinger': 9, 'panda_link0': 0, 'panda_link1': 1, 'panda_link2': 2, 'panda_link3': 3, 'panda_link4': 4, 'panda_link5': 5, 'panda_link6': 6, 'panda_link7': 7, 'panda_rightfinger': 10}
        elif self.robot_name == 'piper':
            self.finger_name = "link6"
            self.joint_dim = 8
            robot_asset_file = "urdf/piper_description/urdf/piper_description.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
            self.body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
            # {'base_link': 0, 'link1': 1, 'link2': 2, 'link3': 3, 'link4': 4, 'link5': 5, 'link6': 6, 'link7': 7, 'link8': 8}
        elif self.robot_name == 'ur5e':
            self.finger_name = "ee_link"
            self.joint_dim = 6
            robot_asset_file = "urdf/ur_description/urdf/ur5_robot.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
            self.body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
            # 'base_link': 1, 'ee_link': 8, 'forearm_link': 4, 'shoulder_link': 2, 'upper_arm_link': 3, 'world': 0, 'wrist_1_link': 5, 'wrist_2_link': 6, 'wrist_3_link': 7
        elif self.robot_name == 'ur10e':
            self.finger_name = "panda_hand"
            self.joint_dim = 8
            robot_asset_file = "urdf/ur_description/urdf/ur10_joint_limited_robot.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
            self.body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        elif self.robot_name == 'aloha':
            robot_asset_file = "urdf/aloha_description/arx5_description_isaac.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.flip_visual_attachments = True
            robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
            self.body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        else:
            raise NotImplementedError

        # configure robot dofs
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_lower_limits = robot_dof_props["lower"]
        self.robot_upper_limits = robot_dof_props["upper"]
        robot_mids = 0.3 * (self.robot_upper_limits + self.robot_lower_limits)
        # Set position drive for all dofs
        robot_dof_props["driveMode"][:self.joint_dim - 2].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][:self.joint_dim - 2].fill(400.0)
        robot_dof_props["damping"][:self.joint_dim - 2].fill(40.0)
        # Joint 7 and 8 are the gripper open and close joints.
        robot_dof_props["driveMode"][self.joint_dim - 2:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["stiffness"][self.joint_dim - 2:].fill(800.0)
        robot_dof_props["damping"][self.joint_dim - 2:].fill(40.0)
        # default dof states and position targets
        robot_num_dofs = self.gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        if self.robot_name == 'ur10e':
            default_dof_pos[:self.joint_dim - 2] = np.array([-0.2800, -1.7994,  1.6791, -1.8279, -1.2586,  1.3184])
        elif self.robot_name == 'franka_panda':
            default_dof_pos[:self.joint_dim - 2] = np.array([-0.0756, -0.8898,  0.0561, -2.5263,  0.0675,  2.1010,  0.7446])
        else:
            default_dof_pos[:self.joint_dim - 2] = robot_mids[:self.joint_dim - 2]
        # grippers open
        default_dof_pos[self.joint_dim - 2:] = self.robot_upper_limits[self.joint_dim - 2:]

        default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # get link index of panda hand, which we will use as end effector
        robot_link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        robot_hand_index = robot_link_dict[self.finger_name]

        # configure env grid
        num_envs = self.num_envs
        num_per_row = int(math.sqrt(num_envs))
        if num_envs in [2, 3]:
            self.x_row_num = 2
        else:
            self.x_row_num = num_per_row
        self.y_row_num = math.ceil(num_envs / self.x_row_num)
        self.x_spacing, self.y_spacing, self.z_spacing = 1.0, 4.0, 1.0
        env_lower = gymapi.Vec3(-self.x_spacing/2, -self.y_spacing/2, 0.0)
        env_upper = gymapi.Vec3(self.x_spacing/2, self.y_spacing/2, self.z_spacing)
        
        self.robot_pose = gymapi.Transform()
        if self.robot_name == 'piper':
            self.robot_pose.p = gymapi.Vec3(0.25, 0, 0.3)
            # Given a base to this robot to enhance its height
            robot_base_block_dim = gymapi.Vec3(0.4, 0.2, 0.3)
            robot_base_block_pose = gymapi.Transform()
            robot_base_block_pose.p = gymapi.Vec3(0.1, 0.0, 0.5 * robot_base_block_dim.z)
            robot_base_block_asset = self.gym.create_box(self.sim, robot_base_block_dim.x, robot_base_block_dim.y, robot_base_block_dim.z, asset_options)
        elif self.robot_name == 'ur10e':
            self.robot_pose.p = gymapi.Vec3(-0.2, 0.0, 0)
        else:
            self.robot_pose.p = gymapi.Vec3(0, 0, 0)
        
        table1_block1_pose = gymapi.Transform()
        table1_block1_pose.p = gymapi.Vec3(0.5, 1.5, 0.5 * table1_block1_dims.z)
        self.table1_block1_pose = table1_block1_pose
        table1_block2_pose = gymapi.Transform()
        table1_block2_pose.p = gymapi.Vec3(table1_block1_pose.p.x, table1_block1_pose.p.y, table1_block1_dims.z + table1_block2_dims.z / 2)
        self.table1_block2_pose =table1_block2_pose
        table1_block3_pose = gymapi.Transform()
        table1_block3_pose.p = gymapi.Vec3(table1_block2_pose.p.x - (table1_block2_dims.x + table1_block3_dims.x) / 2, table1_block2_pose.p.y, table1_block2_pose.p.z)
        table1_block4_pose = gymapi.Transform()
        table1_block4_pose.p = gymapi.Vec3(table1_block2_pose.p.x + (table1_block2_dims.x + table1_block4_dims.x) / 2, table1_block2_pose.p.y, table1_block2_pose.p.z)
        table2_pose = gymapi.Transform()
        table2_pose.p = gymapi.Vec3(0.1, -0.3, 0.5 * table2_dims.z)
        self.table2_pose = table2_pose

        container_bottom_pose = gymapi.Transform()
        container_front_pose  = gymapi.Transform()
        container_back_pose   = gymapi.Transform()
        container_left_pose   = gymapi.Transform()
        container_right_pose  = gymapi.Transform()
        container_bottom_pose.p = gymapi.Vec3(table2_pose.p.x, table2_pose.p.y, table2_dims.z + 0.5 * container_bottom_dims.z)
        container_front_pose.p  = gymapi.Vec3(container_bottom_pose.p.x + container_front_pose_offset.x, 
                                            container_bottom_pose.p.y + container_front_pose_offset.y,
                                            table2_dims.z + 0.5 * container_front_dims.z)
        container_back_pose.p   = gymapi.Vec3(container_bottom_pose.p.x + container_back_pose_offset.x, 
                                            container_bottom_pose.p.y + container_back_pose_offset.y,
                                            table2_dims.z + 0.5 * container_front_dims.z)
        container_left_pose.p   = gymapi.Vec3(container_bottom_pose.p.x + container_left_pose_offset.x, 
                                            container_bottom_pose.p.y + container_left_pose_offset.y,
                                            table2_dims.z + 0.5 * container_front_dims.z)
        container_right_pose.p  = gymapi.Vec3(container_bottom_pose.p.x + container_right_pose_offset.x, 
                                            container_bottom_pose.p.y + container_right_pose_offset.y,
                                            table2_dims.z + 0.5 * container_front_dims.z)
        
        self.container_bottom_pose = container_bottom_pose
        self.container_pose = gymapi.Transform()
        self.container_pose.p = gymapi.Vec3(container_bottom_pose.p.x, container_bottom_pose.p.y, container_front_pose.p.z)

        objs_pose = []
        for i in range(self.num_obj_per_envi):
            objs_pose.append(gymapi.Transform())

        self.envs = []
        self.envs_id = []
        self.obj_idxs = []
        self.marker_idxs = []
        self.hand_idxs = []
        self.leftfinger_idxs = []
        self.rightfinger_idxs = []
        self.camera_base_idxs = []
        self.task_instruction = []

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        for i in range(num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            self.envs_id.append((i % self.x_row_num, i // self.x_row_num))
            
            # add table
            table1_block1_handle = self.gym.create_actor(env, table1_block1_asset, table1_block1_pose, "table1_block1", i, 0)
            table1_block2_handle = self.gym.create_actor(env, table1_block2_asset, table1_block2_pose, "table1_block2", i, 0)
            table1_block3_handle = self.gym.create_actor(env, table1_block3_asset, table1_block3_pose, "table1_block3", i, 0)
            table1_block4_handle = self.gym.create_actor(env, table1_block4_asset, table1_block4_pose, "table1_block4", i, 0)
            table_prop = self.gym.get_actor_rigid_shape_properties(env, table1_block1_handle)
            table_prop[0].friction = 0.
            table_prop[0].rolling_friction = 0.
            table_prop[0].torsion_friction = 0.
            self.gym.set_actor_rigid_shape_properties(env, table1_block1_handle, table_prop)    # Set the friction of the table as zero.
            self.gym.set_actor_rigid_shape_properties(env, table1_block2_handle, table_prop)
            self.gym.set_actor_rigid_shape_properties(env, table1_block3_handle, table_prop)
            self.gym.set_actor_rigid_shape_properties(env, table1_block4_handle, table_prop)
            self.gym.set_rigid_body_color(env, table1_block2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.4), np.array(0.4), np.array(0.4))) # For debug
            
            '''self.gym.set_rigid_body_color(env, table1_block1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.98), np.array(0.95), np.array(0.65)))
            self.gym.set_rigid_body_color(env, table1_block2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.16), np.array(0.32), np.array(0.24)))
            self.gym.set_rigid_body_color(env, table1_block3_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.98), np.array(0.95), np.array(0.65)))
            self.gym.set_rigid_body_color(env, table1_block4_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.98), np.array(0.95), np.array(0.65)))'''
            
            table2_handle = self.gym.create_actor(env, table2_asset, table2_pose, "table2", i, 0)
            
            # add container
            container_bottom_handle = self.gym.create_actor(env, container_bottom_asset, container_bottom_pose, "container_bottom", i, 0)
            container_front_handle = self.gym.create_actor(env, container_front_asset, container_front_pose, "container_front", i, 0)
            container_back_handle = self.gym.create_actor(env, container_back_asset, container_back_pose, "container_back", i, 0)
            container_left_handle = self.gym.create_actor(env, container_left_asset, container_left_pose, "container_left", i, 0)
            container_right_handle = self.gym.create_actor(env, container_right_asset, container_right_pose, "container_right", i, 0)
            for container_handle in [container_bottom_handle, container_front_handle, container_back_handle, container_left_handle, container_right_handle]:
                self.gym.set_rigid_body_color(env, container_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(np.array(0.2), np.array(0.2), np.array(0.2)))
            
            # add obj
            if self.robot_name == 'franka_panda':
                x_range = (-0.15, 0.15)
            elif self.robot_name == 'piper':
                x_range = (-0.05, 0)
            else:
                x_range = (-0.15, 0.15)
            rerange_flag = True
            max_obj_height = 0.1
            while rerange_flag:
                stack_groups = []
                table_pose_x = self.table1_block1_pose.p.x
                table_dim_z = self.table1_block1_dims.z + self.table1_block2_dims.z
                for cnt, obj_pose in enumerate(objs_pose):
                    obj_pose.p.x = table_pose_x + np.random.uniform(self.obj_init_range[0][0], self.obj_init_range[0][1])
                    obj_pose.p.y = np.random.uniform(self.obj_init_range[1][0], self.obj_init_range[1][1]) + 0.7 * cnt
                    obj_pose.p.z = table_dim_z + 0.5 * max_obj_height
                    # Random yaw orientation
                    '''yaw = np.pi / 2
                    quat = gymapi.Quat.from_euler_zyx(0, 0, yaw)
                    obj_pose.r = quat'''
                    stack_groups.append([[obj_pose, cnt]])
                rerange_flag = False
                for p1, p2, in combinations(objs_pose, 2):
                    if (np.sum(np.square(np.array([p1.p.x - p2.p.x, p1.p.y - p2.p.y]))) < 2 * (max_obj_height + 0.013) ** 2):
                        rerange_flag = True
                        break
            self.obj_handles = [self.gym.create_actor(env, obj_assets[obj_cnt], obj_pose, "obj{}".format(obj_cnt), i, 0) for obj_cnt, obj_pose in enumerate(objs_pose)]
            for obj_handle_id in range(len(self.obj_handles)):
                obj_prop = self.gym.get_actor_rigid_shape_properties(env, self.obj_handles[obj_handle_id])
                obj_prop[0].friction = 1.0
                obj_prop[0].rolling_friction = 1.0
                obj_prop[0].torsion_friction = 1.0
                self.gym.set_actor_rigid_shape_properties(env, self.obj_handles[obj_handle_id], obj_prop)    # Set the friction of the table as zero.
            obj_colors = {
                'red': gymapi.Vec3(1, 0, 0),
                'green': gymapi.Vec3(0, 1, 0),
                'blue': gymapi.Vec3(0, 0, 1),
                'yellow': gymapi.Vec3(1, 1, 0),
                'purple': gymapi.Vec3(1, 0, 1),
                'cyan': gymapi.Vec3(0, 1, 1),
            }   # All potential colors
            assert self.num_obj_per_envi <= len(obj_colors)
            samle_obj_colors = random.sample(obj_colors.keys(), self.num_obj_per_envi)
            env_obj_idxs = []
            for obj_handle, obj_color in zip(self.obj_handles, samle_obj_colors):
                self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_colors[obj_color])
                obj_idx = self.gym.get_actor_rigid_body_index(env, obj_handle, 0, gymapi.DOMAIN_SIM)
                env_obj_idxs.append(obj_idx)
            self.obj_idxs.append(env_obj_idxs)
            
            if self.robot_name == 'piper':
                robot_base_block_handle = self.gym.create_actor(env, robot_base_block_asset, robot_base_block_pose, "robot_base_block", i, 0)
            
            if self.vis_marker:
                marker_pose = gymapi.Transform()
                marker_pose.p.x = 0.0
                marker_pose.p.y = 0.0
                marker_pose.p.z = 0.0
                marker_handle = self.gym.create_actor(env, marker_asset, marker_pose, "sphere", 0, 1)
                color = gymapi.Vec3(0.8, 0.8, 0.8)
                self.gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                marker_idx = self.gym.get_actor_rigid_body_index(env, marker_handle, 0, gymapi.DOMAIN_SIM)
                self.marker_idxs.append(marker_idx)

            # add robot
            robot_handle = self.gym.create_actor(env, robot_asset, self.robot_pose, self.robot_name, i, 2)
            
            # set dof properties
            self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, robot_handle, default_dof_pos)

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, self.finger_name, gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            leftfinger_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, 'panda_leftfinger', gymapi.DOMAIN_SIM)
            self.leftfinger_idxs.append(leftfinger_idx)
            rightfinger_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, 'panda_rightfinger', gymapi.DOMAIN_SIM)
            self.rightfinger_idxs.append(rightfinger_idx)
            if self.robot_name == 'franka_panda':
                self.camera_base_idxs.append(hand_idx)
            elif self.robot_name == 'piper':
                cam_base_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, 'camera', gymapi.DOMAIN_SIM)
                self.camera_base_idxs.append(cam_base_idx)

            # Generate language instruction.
            self.task_instruction.append("Please take all the objects on the moving belt into the container.")

        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.img_width = 320
        self.img_height = 240
        camera_props = copy.deepcopy(gymapi.CameraProperties())
        camera_props.width = self.img_width
        camera_props.height = self.img_height
        top_camera_pos = [0.7, 0.0, 0.9]
        top_camera_look_at = [0.4, 0.0, 0.2]
        self.cameras = []
        self.cameras_idx = []
        
        self.handcam_local_transform = gymapi.Transform()
        if self.robot_name == 'piper':
            self.handcam_local_transform.p = gymapi.Vec3(0.0, 0.0, 0.0)
            self.handcam_local_transform.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
            attach_name = 'camera'
        else:
            # In the gripper coordinate system, z axis is the gripper direction, x axis is perpendicular with the gripper and upwards, y is inwards, right hand coordinate system.
            self.handcam_local_transform.p = gymapi.Vec3(0.1, 0, 0.03)
            self.handcam_local_transform.r = gymapi.Quat.from_euler_zyx(0, -math.pi / 4, math.pi)  # Three inputs: roll, pitch, yaw
            attach_name = self.finger_name
            
        for env in self.envs:
            # Top camera
            top_camera = self.gym.create_camera_sensor(env, camera_props)
            camera_pos = gymapi.Vec3(*top_camera_pos)
            camera_target = gymapi.Vec3(*top_camera_look_at)
            self.gym.set_camera_location(top_camera, env, camera_pos, camera_target)
            top_camera_idx = self.gym.get_actor_rigid_body_index(env, top_camera, 0, gymapi.DOMAIN_SIM)
            
            # Hand camera
            hand_camera = self.gym.create_camera_sensor(env, camera_props)
            attach_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, attach_name)
            self.gym.attach_camera_to_body(hand_camera, env, attach_handle, self.handcam_local_transform, gymapi.FOLLOW_TRANSFORM)
            hand_camera_idx = self.gym.get_actor_rigid_body_index(env, hand_camera, 0, gymapi.DOMAIN_SIM)

            self.cameras.append(dict(top_camera = top_camera, hand_camera = hand_camera))
            self.cameras_idx.append(dict(top_camera_idx = top_camera_idx, hand_camera_idx = hand_camera_idx))
            
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

        # get jacobian tensor
        # for fixed-base robot, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.robot_name)
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to robot
        self.j_eef = jacobian[:, robot_hand_index - 1, :, :self.joint_dim - 2]
        
        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, self.robot_name)
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :self.joint_dim - 2, :self.joint_dim - 2]          # only need elements corresponding to the robot

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        
        _contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(_contact_forces)
        
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(num_envs, self.joint_dim, 1)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)
        
        self.simulation_step = 0

    def update_simulator_before_ctrl(self):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

    def update_simulator_after_ctrl(self):
        if self.belt_speed != 0:
            self.obj_moving_belt_progress()
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        
        self.simulation_step += 1
        
    def get_robotend_pos(self,):
        return self.rb_states[self.hand_idxs, :3]   # Left shape: (num_envs, 3)
    
    def get_robotend_ori(self,):
        return self.rb_states[self.hand_idxs, 3:7]   # Left shape: (num_envs, 4)
    
    def get_robotend_pos_vel(self,):
        return self.rb_states[self.hand_idxs, 7:10] # Left shape: (num_envs, 3)
    
    def get_robotend_ori_vel(self,):
        return self.rb_states[self.hand_idxs, 10:13] # Left shape: (num_envs, 3)
    
    def get_norm_gripper_status(self):
        gripper_status = self.dof_pos[:, -2:, 0]
        gripper_upper_limit = torch.Tensor(self.robot_lower_limits[None, -2:]).to(gripper_status.device)
        gripper_lower_limit = torch.Tensor(self.robot_upper_limits[None, -2:]).to(gripper_status.device)
        norm_gripper_status = (gripper_status - gripper_lower_limit) / (gripper_upper_limit - gripper_lower_limit)
        return norm_gripper_status
    
    def execute_action_ik(self, action):
        '''
        Description:
            Execute the action.
        Input:
            action: shape: (num_envs, 9).
        '''
        goal_pos = action[:, :3]
        goal_rot = action[:, 3:7]
        goal_gripper = action[:, 7:]
        hand_pos = self.rb_states[self.hand_idxs, :3] # Left shape: (num_envs, 3)
        hand_rot = self.rb_states[self.hand_idxs, 3:7]    # Left shape: (num_envs, 4)
        
        # IK based control
        pos_err = goal_pos - hand_pos   # Left shape: (num_envs, 3)
        orn_err = self.orientation_error(goal_rot, hand_rot)   # Left shape: (num_envs, 3)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)    # Left shape: (num_envs, 6, 1)
        arm_ctrl = self.dof_pos.squeeze(-1)[:, :self.joint_dim - 2] + self.control_ik(dpose)   # Control all joints except the gripper.
        self.pos_action[:, :self.joint_dim - 2] = arm_ctrl
        self.pos_action[:, self.joint_dim - 2:] = goal_gripper
        # Update the planned the action to the simulator
        self.update_action_map()
    
    def update_action_map(self):
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

    def get_vision_observations(self):
        images_envs = []
        for cnt, env in enumerate(self.envs):
            top_rgb_image = self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['top_camera'], gymapi.IMAGE_COLOR)
            top_rgb_image = np.ascontiguousarray(top_rgb_image.reshape(240, 320, 4)[:, :, :3])
            top_depth = np.abs(self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['top_camera'], gymapi.IMAGE_DEPTH))   # Left shape: (h, w)
            
            hand_rgb_image = self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['hand_camera'], gymapi.IMAGE_COLOR)
            hand_rgb_image = np.ascontiguousarray(hand_rgb_image.reshape(240, 320, 4)[:, :, :3])
            hand_depth = np.abs(self.gym.get_camera_image(self.sim, env, self.cameras[cnt]['hand_camera'], gymapi.IMAGE_DEPTH))
            
            images = dict(
                top_rgb = top_rgb_image,
                top_depth = top_depth,
                hand_rgb = hand_rgb_image,
                hand_depth = hand_depth,
            )
            images_envs.append(images)
        self.gym.end_access_image_tensors(self.sim)
        return images_envs
    
    def get_time(self):
        return self.gym.get_sim_time(self.sim)
    
    def get_cameras_parameters(self, cam_name):
        cam_projs, cam_views = [], []
        for env, scene_cameras in zip(self.envs, self.cameras):
            cam_proj = np.matrix(self.gym.get_camera_proj_matrix(self.sim, env, scene_cameras[cam_name])).T
            cam_view = np.matrix(self.gym.get_camera_view_matrix(self.sim, env, scene_cameras[cam_name])).T
            cam_projs.append(cam_proj)
            cam_views.append(cam_view)
        cam_projs = np.array(cam_projs) # Left shape: (num_envs, 4, 4)
        cam_views = np.array(cam_views) # Left shape: (num_envs, 4, 4)
        
        # cam_views is in the global world coordinate system. Transform it to the environment world coordinate system.
        view_R = cam_views[0, :3, :3]    # Left shape: (3, 3). The rotation part is the same for different environments.
        envs_coord = np.array(self.envs_id) * np.array((self.x_spacing, self.y_spacing), dtype = np.float32)    # Left shape: (num_envs, 2)
        envs_coord = np.pad(envs_coord, ((0, 0), (0, 1)), mode='constant', constant_values = 0) # Left shape: (num_envs, 3)
        env_T_offset = (view_R[None] @ envs_coord[..., None])   # Left shape: (num_envs, 3, 1)
        cam_views[:, :3, 3:4] += env_T_offset
        return cam_projs, cam_views
    
    def obj_moving_belt_progress(self,):
        obj_idxs = torch.tensor(np.array(self.obj_idxs), dtype = torch.long).to(self.rb_states.device)  # Left shape: (num_envs, num_obj_per_env)
        obj_pos = self.rb_states[obj_idxs, 0:3].clone() # Left shape: (num_envs, num_obj_per_env, 3)
        obj_speed = self.rb_states[obj_idxs, 7:10].view(-1, 3).clone() # Left shape: (num_envs * num_obj_per_env, 3)
        obj_contract_forces = self.contact_forces[obj_idxs] # Left shape: (num_envs, num_obj_per_env, 3)
        on_moving_belt_flags = (obj_pos[..., 0] > self.table1_block2_pose.p.x - self.table1_block2_dims.x / 2) & (obj_pos[..., 0] < self.table1_block2_pose.p.x + self.table1_block2_dims.x / 2) & \
            (obj_pos[..., 1] > self.table1_block2_pose.p.y - self.table1_block2_dims.y / 2 - 0.1) & (obj_pos[..., 1] < self.table1_block2_pose.p.y + self.table1_block2_dims.y / 2 + 0.1) \
            & (obj_contract_forces[..., 2] > 0.2) \
            & (obj_pos[..., 2] > self.table1_block1_dims.z + self.table1_block2_dims.z - 0.05) & (obj_pos[..., 2] < self.table1_block1_dims.z + self.table1_block2_dims.z + 0.1)   # Left shape: (num_envs, num_obj_per_env)

        tgt_speed = torch.zeros((obj_pos.shape[0], obj_pos.shape[1], 3), device=obj_pos.device, dtype=torch.float)  # Left shape: (num_envs, num_obj_per_env, 3)
        tgt_speed[:, :, 1] = self.belt_speed
        if self.move_mode == 'sin':
            pos_kp = 5
            obj_tgt_pos_x = 0.1 * torch.sin(2 * obj_pos[:, :, 1]) + self.table1_block1_pose.p.x # Left shape: (num_envs, num_objs)
            tgt_speed[:, :, 0] = torch.clamp(pos_kp * (obj_tgt_pos_x - obj_pos[:, :, 0]), min = -0.1, max = 0.1)
        tgt_speed = tgt_speed.flatten(0, 1)    # Left shape: (num_envs * num_objs, 3)
        apply_forces = self.objs_vel_PID(tgt_vel = tgt_speed, cur_vel = obj_speed, time = self.get_time())  # Left shape: (num_envs * num_objs, 3)
        apply_forces = apply_forces * on_moving_belt_flags.view(-1, 1)
        forces = torch.zeros((self.rb_states.shape[0], 3), device=obj_pos.device, dtype=torch.float)
        forces[obj_idxs.view(-1),] = apply_forces
        torques = torch.zeros_like(forces)
        
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(self.rb_states[:, 0:3].contiguous()), gymapi.ENV_SPACE)
        #self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
    
    def compute_handcam2handend_transform(self,):
        '''
        Description:
            Get the pose transformation from the hand camera pose to the hand end pose.
        '''
        offset, ori = self.handcam_local_transform.p, self.handcam_local_transform.r
        offset_tensor =  -torch.Tensor([offset.x, offset.y, offset.z]).cuda()
        ori_tensor = quaternion_conjugate(torch.Tensor([ori.x, ori.y, ori.z, ori.w]).cuda())
        return offset_tensor, ori_tensor
    
    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def control_ik(self, dpose):
        # solve damped least squares
        kp = 0.2
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=dpose.device) * (self.damping ** 2)
        # There may raise an error in this line because a bug in Isaac Gym, self.j_eef is fully NaN.
        u = kp * (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, self.joint_dim - 2)
        return u
    
    def clean_up(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
    def get_reward(self,):
        x_range = (self.container_pose.p.x - self.container_dims.x / 2, self.container_pose.p.x + self.container_dims.x / 2)
        y_range = (self.container_pose.p.y - self.container_dims.y / 2, self.container_pose.p.y + self.container_dims.y / 2)
        z_range = (self.container_pose.p.z - self.container_dims.z / 2, self.container_pose.p.z + self.container_dims.z / 2)
        rewards = [] 
        for env_id, env_box_ids in enumerate(self.obj_idxs):
            pos_3d = self.rb_states[env_box_ids, :3]
            in_container_flags = (pos_3d[:, 0] > x_range[0]) & (pos_3d[:, 0] < x_range[1]) & (pos_3d[:, 1] > y_range[0]) & (pos_3d[:, 1] < y_range[1]) & (pos_3d[:, 2] > z_range[0]) & (pos_3d[:, 2] < z_range[1])
            rewards.append(in_container_flags.sum().item())
        rewards = np.array(rewards)
        return rewards
    
    def get_top_cam_world_coors(self, top_depth):
        top_cam_projs, top_cam_views = self.get_cameras_parameters(cam_name = 'top_camera')
        top_cam_projs = torch.Tensor(top_cam_projs).cuda()
        top_cam_views = torch.Tensor(top_cam_views).cuda()
        h_coors, w_coors = np.meshgrid(np.arange(self.img_height), np.arange(self.img_width), indexing='ij')
        ndc_x = (w_coors + 0.5) / self.img_width * 2 - 1
        ndc_y = 1 - (h_coors + 0.5) / self.img_height * 2
        ndc_coors = torch.Tensor(np.stack([ndc_x, ndc_y], axis=-1)).cuda()[None].repeat(top_depth.shape[0], 1, 1, 1)   # ndc_coors shape: (num_envs, img_height, img_width, 2)
        clip_w = top_depth.unsqueeze(3)    # Left shape: (num_envs, img_height, img_width, 1)
        clip_x = ndc_coors[..., 0:1] * clip_w # Left shape: (num_envs, img_height, img_width, 1)
        clip_y = ndc_coors[..., 1:2] * clip_w # Left shape: (num_envs, img_height, img_width, 1)
        B = top_cam_projs[:, 2, 3]  # top_cam_projs[:, 2, 2] is zero.
        clip_z = torch.Tensor(B[:, None, None, None]).cuda().expand(-1, self.img_height, self.img_width, -1) # Left shape: (num_envs, img_height, img_width, 1)
        P_projected = torch.concat([clip_x, clip_y, clip_z, clip_w], dim=-1) # Left shape: (num_envs, img_height, img_width, 4)
        P_camera = top_cam_projs[:, None, None].inverse() @ P_projected.unsqueeze(-1)   # Left shape: (num_envs, img_height, img_width, 4, 1)
        world_coors = (top_cam_views[:, None, None].inverse() @ P_camera).squeeze(-1)[..., :3]   # Left shape: (num_envs, img_height, img_width, 3)
        return world_coors
    
    def get_hand_cam_world_coors(self, hand_depth):
        hand_cam_projs, hand_cam_views = self.get_cameras_parameters(cam_name = 'hand_camera')
        hand_cam_projs = torch.Tensor(hand_cam_projs).cuda()
        hand_cam_views = torch.Tensor(hand_cam_views).cuda()
        h_coors, w_coors = np.meshgrid(np.arange(self.img_height), np.arange(self.img_width), indexing='ij')
        ndc_x = (w_coors + 0.5) / self.img_width * 2 - 1
        ndc_y = 1 - (h_coors + 0.5) / self.img_height * 2
        ndc_coors = torch.Tensor(np.stack([ndc_x, ndc_y], axis=-1)).cuda()[None].repeat(hand_depth.shape[0], 1, 1, 1)   # ndc_coors shape: (num_envs, img_height, img_width, 2)
        clip_w = hand_depth.unsqueeze(3)    # Left shape: (num_envs, img_height, img_width, 1)
        clip_x = ndc_coors[..., 0:1] * clip_w # Left shape: (num_envs, img_height, img_width, 1)
        clip_y = ndc_coors[..., 1:2] * clip_w # Left shape: (num_envs, img_height, img_width, 1)
        B = hand_cam_projs[:, 2, 3]  # top_cam_projs[:, 2, 2] is zero.
        clip_z = torch.Tensor(B[:, None, None, None]).cuda().expand(-1, self.img_height, self.img_width, -1) # Left shape: (num_envs, img_height, img_width, 1)
        P_projected = torch.concat([clip_x, clip_y, clip_z, clip_w], dim=-1) # Left shape: (num_envs, img_height, img_width, 4)
        P_camera = hand_cam_projs[:, None, None].inverse() @ P_projected.unsqueeze(-1)   # Left shape: (num_envs, img_height, img_width, 4, 1)
        world_coors = (hand_cam_views[:, None, None].inverse() @ P_camera).squeeze(-1)[..., :3]   # Left shape: (num_envs, img_height, img_width, 3)
        return world_coors
    
    def get_hand_cam_cam_coors(self, hand_depth):
        hand_cam_projs, _ = self.get_cameras_parameters(cam_name = 'hand_camera')
        hand_cam_projs = torch.Tensor(hand_cam_projs).cuda()
        h_coors, w_coors = np.meshgrid(np.arange(self.img_height), np.arange(self.img_width), indexing='ij')
        ndc_x = (w_coors + 0.5) / self.img_width * 2 - 1
        ndc_y = 1 - (h_coors + 0.5) / self.img_height * 2
        ndc_coors = torch.Tensor(np.stack([ndc_x, ndc_y], axis=-1)).cuda()[None].repeat(hand_depth.shape[0], 1, 1, 1)   # ndc_coors shape: (num_envs, img_height, img_width, 2)
        clip_w = hand_depth.unsqueeze(3)    # Left shape: (num_envs, img_height, img_width, 1)
        clip_x = ndc_coors[..., 0:1] * clip_w # Left shape: (num_envs, img_height, img_width, 1)
        clip_y = ndc_coors[..., 1:2] * clip_w # Left shape: (num_envs, img_height, img_width, 1)
        B = hand_cam_projs[:, 2, 3]  # top_cam_projs[:, 2, 2] is zero.
        clip_z = torch.Tensor(B[:, None, None, None]).cuda().expand(-1, self.img_height, self.img_width, -1) # Left shape: (num_envs, img_height, img_width, 1)
        P_projected = torch.concat([clip_x, clip_y, clip_z, clip_w], dim=-1) # Left shape: (num_envs, img_height, img_width, 4)
        P_camera = (hand_cam_projs[:, None, None].inverse() @ P_projected.unsqueeze(-1)).squeeze(-1)[..., :3]   # Left shape: (num_envs, img_height, img_width, 3)
        return P_camera
    
    def project_target_world3d_to_hand_camuv(self, world3d):
        '''              
        Description:
            Project 3D points in the environment 3D coordinate system of different environments (one point per environment) to the hand camera image coordinate system.
        Input:
            world3d shape: (num_env, 3)
        '''
        hand_cam_projs, hand_cam_views = self.get_cameras_parameters(cam_name = 'hand_camera')
        hand_cam_projs = torch.Tensor(hand_cam_projs).cuda()    # Left shape: (num_envs, 4, 4)
        hand_cam_views = torch.Tensor(hand_cam_views).cuda()    # Left shape: (num_envs, 4, 4)
        i_world3d = torch.cat((world3d, torch.ones((world3d.shape[0], 1), dtype = torch.float32).cuda()), dim = -1)   # Left shape: (num_envs, 4)
        
        P_camera = hand_cam_views @ i_world3d.unsqueeze(-1)   # Left shape: (num_envs, 4, 1)
        P_projected = (hand_cam_projs @ P_camera).squeeze(-1) # Left shape: (num_envs, 4)
        uv_ndc = P_projected[:, 0:2] / P_projected[:, 3:4]   # Left shape: (num_envs, 2)
        u_pixel = (uv_ndc[:, 0:1] + 1) * self.img_width / 2   # Left shape: (num_envs, 1)
        v_pixel = (1 - uv_ndc[:, 1:2]) * self.img_height / 2  # Left shape: (num_envs, 1)
        uv_pixel = torch.cat((u_pixel, v_pixel), dim = -1) # Left shape: (num_envs, 2)
        return uv_pixel
    
    def get_grasp_pos3d(self,):
        self.rb_states[self.leftfinger_idxs]
        leftfinger_pos3d = self.rb_states[self.leftfinger_idxs, 0:3]
        rightfinger_pos3d = self.rb_states[self.rightfinger_idxs, 0:3]
        finger_center = (leftfinger_pos3d + rightfinger_pos3d) / 2  # Left shape: (num_envs, 3).
        # finger_center position is at the root of the gripper, so we still need to add an offset.
        offset_dist = 0.06
        hand_ori = self.rb_states[self.hand_idxs, 3:7]    # Left shape: (num_envs, 4).
        rotation_matrices = quaternion_to_rotation_matrix(hand_ori) # Left shape: (num_envs, 3, 3)
        direction_vector = torch.Tensor([1.0, 0.0, 0.0]).cuda().repeat(self.num_envs, 1)    # Left shape: (num_envs, 3)
        direction_vector_world = torch.bmm(rotation_matrices, direction_vector.unsqueeze(2)).squeeze(2)
        direction_vector_world = direction_vector_world / direction_vector_world.norm(dim=1, keepdim=True)  # Left shape: (num_envs, 3)
        offset3d = torch.Tensor([offset_dist,]).cuda().expand(self.num_envs,)   # Left shape: (num_envs,)
        grasp_pos3d = finger_center + direction_vector_world * offset3d.unsqueeze(1)    # Left shape: (num_envs, 3)
        return grasp_pos3d
        
class BatchVelocityPID():
    def __init__(self, obj_num, kp, ki, kd):
        self.obj_num = obj_num
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = torch.zeros(obj_num, 3).cuda()
        self.prev_error = torch.zeros(obj_num, 3).cuda()
        self.prev_time = 0

    def __call__(self, tgt_vel, cur_vel, time):
        '''
        Input:
            tgt_vel shape: (obj_num, 3)
            cur_vel shape: (obj_num, 3)
            time shape: a float
        Output:
            ctrl shape: (3)
        '''
        error = tgt_vel - cur_vel
        time_interval = time - self.prev_time
        self.integral += error * time_interval
        derivative = (error - self.prev_error) / max(time_interval, 1e-5)
        ctrl = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        self.prev_time = time
        return ctrl
    
def map_to_uniform_bell_shape(x):
    if x <= 0.5:
        return 2 * x
    else:
        return 2 * (1 - x)
    

if __name__ == '__main__':
    envi = DynamicSimulator(num_envs = 1, num_obj_per_envi = 5, belt_speed = -0.0, obj_mode = 'daily_objs', robot_name = 'ur10e',)
    step = 0
    T = 1000
    test_joint_range = (4, 5)
    while True:
        envi.update_simulator_before_ctrl()
        '''ctrl_ratio = map_to_uniform_bell_shape((step % T) / T)
        arm_ctrl = (ctrl_ratio * (envi.robot_upper_limits - envi.robot_lower_limits) + envi.robot_lower_limits)[test_joint_range[0] : test_joint_range[1]]
        envi.pos_action[:, test_joint_range[0] : test_joint_range[1]] = torch.Tensor(arm_ctrl).cuda()[None].expand(envi.pos_action.shape[0], -1)
        print(f"step: {step}, {envi.pos_action[0, test_joint_range[0] : test_joint_range[1]]}")
        step += 1
        envi.update_action_map()'''
        envi.update_simulator_after_ctrl()
    envi.clean_up()