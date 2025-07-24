import numpy as np
import os
import pdb
import time
import pickle
import datetime
import logging
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import IPython
e = IPython.embed

def main(args):
    # Initialize logger
    if comm.get_rank() == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    exp_start_time = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    rank = comm.get_rank()
    logger = setup_logger(args.save_dir, rank, file_name="log_{}.txt".format(exp_start_time))
    if comm.is_main_process():
        logger.info("Using {} GPUs".format(comm.get_world_size()))
        logger.info("Collecting environment info")
        logger.info(args) 
        logger.info("Loaded configuration file {}".format(args.config_name+'.yaml'))
    
    # Initialize cfg
    cfg = load_yaml_with_base(os.path.join('configs', args.config_name+'.yaml'))
    cfg['IS_EVAL'] = args.eval
    cfg['CKPT_DIR'] = args.save_dir
    cfg['IS_DEBUG'] = args.debug
    cfg['NUM_NODES'] = args.num_nodes
    cfg['EVAL']['REAL_ROBOT'] = args.real_robot
    
    if cfg['SEED'] >= 0:
        set_seed(cfg['SEED'])

    if cfg['IS_EVAL']:
        if args.load_dir != '':
            ckpt_paths = [args.load_dir]
        else:
            ckpt_paths = [os.path.join(cfg['CKPT_DIR'], 'policy_latest.ckpt')]
        results = []
        for ckpt_path in ckpt_paths:
            success_rate, avg_return = eval_bc(cfg, ckpt_path)
            results.append([ckpt_path.split('/')[-1], success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        exit()

def eval_bc(cfg, ckpt_path):
    ckpt_dir = cfg['CKPT_DIR']
    ckpt_name = ckpt_path.split('/')[-1]
    policy_class = cfg['POLICY']['POLICY_NAME']

    policy, stats = load_policy_and_stats(ckpt_path, policy_class, cfg)
    if cfg['TASK_NAME'] in ['isaac_move_manipulation']:
        from GEM.utils.inference.isaac_move_manipulation import PointMoveManipulationTestEnviManager
        envi_manager = PointMoveManipulationTestEnviManager(cfg, policy, stats)
    else:
        raise NotImplementedError

    reward_info = envi_manager.inference()

    if reward_info != None:
        success_rate = reward_info['success_rate']
        avg_return = reward_info['average_reward']
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'

        print(summary_str)

        # save success rate to txt
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)

        return success_rate, avg_return
    else:
        return 0.0, 0.0

def forward_pass(data, policy, cfg, iter_cnt):
    repr, past_action, action_data, past_action_is_pad, action_is_pad, task_instruction, status, video_frame_id, dataset_type = data
    past_action, action_data, past_action_is_pad, action_is_pad, status, video_frame_id = past_action.cuda(), action_data.cuda(), past_action_is_pad.cuda(), action_is_pad.cuda(), status.cuda(), video_frame_id.cuda()
    return policy(repr = repr, past_action = past_action, action = action_data, past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, \
        status = status, task_instruction_list = task_instruction, video_frame_id = video_frame_id, dataset_type = dataset_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', action='store', type=str, help='configuration file name', required=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_dir', action='store', type=str, help='saving directory', required=True)
    parser.add_argument('--load_dir', action='store', type=str, default = '', help='The path to weight',)
    parser.add_argument('--load_pretrain', action='store', type=str, default = '', help='The path to pre-trained weight')
    parser.add_argument('--real_robot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_episode', action='store_true')
    parser.add_argument('--num_nodes', default = 1, type = int, help = "The number of nodes.")
    args = parser.parse_args()

    if not args.real_robot:
        from isaacgym import gymapi
        from isaacgym import gymutil
        from isaacgym import gymtorch
        from isaacgym.torch_utils import quat_rotate, quat_conjugate, quat_mul
    import torch    # torch must be imported after isaacgym
    from torch.utils.tensorboard import SummaryWriter
    
    from utils.utils import set_seed
    from utils.engine import launch
    from utils import comm
    from utils.logger import setup_logger
    from configs.utils import load_yaml_with_base
    from utils.model_zoo import make_policy, load_policy_and_stats
    
    launch(main, args)
