"""
From the root of Sample Factory repo this can be run as:
python training_sample_factory.py --algo=APPO --env=polamp_env --experiment=example
After training for a desired period of time, evaluate the policy by running:
python enjoy_sample_factory.py --algo=APPO --env=polamp_env --experiment=example
"""

from pickle import TRUE
import sys
import multiprocessing
from argparse import Namespace
import gym
import wandb
import yaml
import numpy as np
import json
from torch import nn
sys.path.insert(0, "sample-factory/")
sys.path.insert(0, "../")
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

from policy_gradient.utlis import generateDataSet
from policy_gradient.utlis import getTestTasks
from EnvLib.ObstGeomEnvSampleFactory import *
from utils_SF.residual_net import ResnetEncoder

use_wandb = True

with open("configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

with open("configs/dataset_configs.json", 'r') as f:
    dataset_config = json.load(f)



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
    if our_env_config["validate_env"] and our_env_config["validate_test_dataset"]:
        generated_tasks = getTestTasks(car_config)
    elif our_env_config["validate_env"]:
        tasks_config = {}
        tasks_config["union"] = dataset_config["union"]
        tasks_config["static"] = dataset_config["static"]
        tasks_config["dynamic"] = dataset_config["dynamic"]
        generated_tasks = getTrainValidateTasks(tasks_config, car_config, train=False)
    else:
        tasks_config = {}
        tasks_config["union"] = dataset_config["union"]
        tasks_config["static"] = dataset_config["static"]
        tasks_config["dynamic"] = dataset_config["dynamic"]
        generated_tasks = getTrainValidateTasks(tasks_config, car_config, train=True)
    maps, generated_tasks = ChangeTaskFormat(generated_tasks)
  
    environment_config = {
        'car_config': car_config,
        'Tasks': generated_tasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': cfg.evaluation,
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


def main():
    register_custom_components()
    cfg = custom_parse_args()
    if use_wandb:
        wandb.init(config=cfg, project='sample-factory-POLAMP', entity='grisha1', 
                    save_code=False, sync_tensorboard=True)
    
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
