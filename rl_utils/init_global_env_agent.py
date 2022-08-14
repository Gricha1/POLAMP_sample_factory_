from asyncio import tasks
import sys
import time
import json
from collections import deque
from pickle import TRUE
import multiprocessing
from argparse import Namespace
import gym
import yaml
from torch import nn
import numpy as np
import torch
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
from EnvLib.ObstGeomEnvSampleFactory import *
from utils_SF.residual_net import ResnetEncoder

use_wandb = False

with open("configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

def custom_parse_args(argv=None, evaluation=False):

    parser = arg_parser(argv, evaluation=evaluation)

    # add custom args here
    parser.add_argument('--my_custom_arg', type=int, 
                    default=42, help='Any custom arguments users might define')
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    cfg.evaluation = evaluation
    
    cfg.rollout = 512
    cfg.recurrence = 128
    #cfg.rollout = 32
    #cfg.recurrence = 32
    
    cfg.encoder_type = 'conv'
    cfg.encoder_subtype = 'convnet_simple'

    cfg.encoder_extra_fc_layers = 0
    cfg.train_for_env_steps = 2000000000
    
    cfg.use_rnn = True
    cfg.rnn_num_layers = 1
    cfg.batch_size = 2048
    
    cfg.ppo_epochs = 1
    cfg.ppo_clip_value = 5.0
    cfg.reward_scale = 0.1
    cfg.reward_clip = 5.0
    cfg.with_vtrace = False
    
    cfg.nonlinearity = 'relu'
    
    cfg.exploration_loss_coeff = 0.005
    
    cfg.kl_loss_coeff = 0.3
    cfg.value_loss_coeff = 0.5
    
    cfg.use_wandb = use_wandb
    cfg.with_wandb = use_wandb
    
    return cfg

def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    maps = {}
    trainTask = {}
    valTasks = {}
    second_goal = []
  
    if our_env_config["union"]:
        environment_config = {
            'car_config': car_config,
            'tasks': trainTask,
            'valTasks': valTasks,
            'maps': maps,
            'our_env_config' : our_env_config,
            'reward_config' : reward_config,
            'evaluation': cfg.evaluation,
            "second_goal" : second_goal
        }
    else:
        environment_config = {
            'car_config': car_config,
            'tasks': trainTask,
            'valTasks': valTasks,
            'maps': maps,
            'our_env_config' : our_env_config,
            'reward_config' : reward_config,
            'evaluation': cfg.evaluation
        }

    cfg.other_keys = environment_config

    return ObsEnvironment(full_env_name, cfg['other_keys'])

def add_extra_params_func(env, parser):
    p = parser

def override_default_params_func(env, parser):
    parser.set_defaults(
        encoder_custom='custom_env_encoder',
        hidden_size=1024,
    )

def polamp_extra_summaries(policy_id, policy_avg_stats, env_steps, 
                           summary_writer, cfg):
    print(f"policy_id : {policy_id}")

def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='polamp_env',
        make_env_func=make_custom_env_func,
        add_extra_params_func=add_extra_params_func,
        override_default_params_func=override_default_params_func,
    )
    EXTRA_PER_POLICY_SUMMARIES.append(polamp_extra_summaries)
    register_custom_encoder('custom_env_encoder', ResnetEncoder)

def init_global_env_agent():
    register_custom_components()
    
    cfg = custom_parse_args(evaluation=True)

    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)
    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, 
                                            env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    return env, actor_critic, cfg