import json
from policy_gradient.curriculum_train import trainCurriculum
import ray.rllib.agents.ppo as ppo
import argparse
import os

parser = argparse.ArgumentParser()
#save folder data
parser.add_argument('-run', "--run_num", help='number of the run', type=int, default=0)
parser.add_argument('-ex', "--ex_num", help='number of the experement', type=int, default=0)
parser.add_argument('-conf', "--conf_num", help='number of the config', type=int, default=0)

parser.add_argument('-l_run', "--loaded_run_num", help='number of the run', type=int, default=0)
parser.add_argument('-l_ex', "--loaded_ex_num", help='number of the experement', type=int, default=0)
parser.add_argument('-l_conf', "--loaded_conf_num", help='number of the config', type=int, default=0)
parser.add_argument('-l_check', "--loaded_check_num", help='number of the config', type=int, default=0)

#run data
'''
parser.add_argument('-term_h', "--hard_constraints", help='number of the run', type=int, default = 0)
parser.add_argument('-term_m', "--medium_constraints", help='number of the run', type=int, default = 0)
parser.add_argument('-term_e', "--soft_constraints", help='number of the run', type=int, default = 0)
parser.add_argument('-type_u', "--union", help='number of the run', type=int, default = 0)
parser.add_argument('-type_d', "--dynamic", help='number of the run', type=int, default = 0)
parser.add_argument('-type_s', "--static", help='number of the run', type=int, default = 0)
parser.add_argument('-map_e', "--easy_map_constraints", help='number of the run', type=int, default = 0)
parser.add_argument('-map_m', "--medium_map_constraints", help='number of the run', type=int, default = 0)
parser.add_argument('-map_h', "--hard_map_constraints", help='number of the run', type=int, default = 0)
parser.add_argument('-task_e', "--easy_task_difficulty", help='number of the run', type=int, default = 0)
parser.add_argument('-task_m', "--medium_task_difficulty", help='number of the run', type=int, default = 0)
parser.add_argument('-task_h', "--hard_task_difficulty", help='number of the run', type=int, default = 0)

parser.add_argument('-task_e', "--HARD_EPS", help='number of the run', type=int, default = 0)
parser.add_argument('-task_m', "--MEDIUM_EPS", help='number of the run', type=int, default = 0)
parser.add_argument('-task_h', "--SOFT_EPS", help='number of the run', type=int, default = 0)
parser.add_argument('-task_h', "--ANGLE_EPS", help='number of the run', type=int, default = 0)
parser.add_argument('-task_h', "--SPEED_EPS", help='number of the run', type=int, default = 0)
'''

#logging
parser.add_argument('-show', "--only_show", help='only show', type=int, default = 0)
parser.add_argument('-type', "--type_show", help='type show', type=int, default = 1)
parser.add_argument('-val', "--only_validate", help='only validate', type=int, default = 0)
args = parser.parse_args()

if not args.ex_num:
    args.ex_num = 1
if not args.conf_num:
    args.conf_num = 1

#if not args.only_show:
#    args.only_show = False
#else:
#    args.only_show = True
#if not args.only_validate:
#    args.only_validate = False
#else:
#    args.only_validate = True   


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

with open("configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)


config = ppo.DEFAULT_CONFIG.copy()
config['framework'] = train_config['framework']
config['train_batch_size'] = train_config['train_batch_size']
config['lambda'] = train_config['lambda']
config['use_critic'] = train_config['use_critic']
config['use_gae'] = train_config['use_gae']
config['horizon'] = train_config['horizon']
config['rollout_fragment_length'] = train_config['rollout_fragment_length']
config['num_gpus'] = train_config['num_gpus']
config['num_workers'] = train_config['num_workers']
config['lr'] = train_config['lr']
config['sgd_minibatch_size'] = train_config['sgd_minibatch_size']
config['num_sgd_iter'] = train_config['num_sgd_iter']
config['clip_param'] = train_config['clip_param']
config["optimizer"] = train_config["optimizer"]
config['entropy_coeff'] = train_config['entropy_coeff']
config['vf_clip_param'] = train_config['vf_clip_param']
config['normalize_actions'] = train_config['normalize_actions']
config['batch_mode'] = train_config['batch_mode']
config['_fake_gpus'] = train_config['_fake_gpus']
config['vf_loss_coeff'] =  train_config['vf_loss_coeff']
config["model"]['fcnet_hiddens'] = [512, 512]
config["model"]["use_lstm"] = train_config['use_lstm']
config["model"]["lstm_use_prev_action"] = train_config['lstm_use_prev_action']
config["model"]["free_log_std"] = train_config["free_log_std"]

#custom changes
#train_config["run_num"] = args.run_num 
#train_config["ex_num"] = args.ex_num
#train_config["conf_num"] = args.conf_num

#train_config["loaded_run_num"] = args.loaded_run_num 
#train_config["loaded_ex_num"] = args.loaded_ex_num
#train_config["loaded_conf_num"] = args.loaded_conf_num
#train_config["loaded_check_num"] = args.loaded_check_num

'''
our_env_config["hard_constraints"] = args.hard_constraints
our_env_config["medium_constraints"] = args.medium_constraints
our_env_config["soft_constraints"] = args.soft_constraints
our_env_config["union"] = args.union
our_env_config["dynamic"] = args.dynamic
our_env_config["static"] = args.static
our_env_config["easy_map_constraints"] = args.easy_map_constraints
our_env_config["medium_map_constraints"] = args.medium_map_constraints
our_env_config["hard_map_constraints"] = args.hard_map_constraints
#our_env_config["validate_custom_case"] = args.validate_custom_case
our_env_config["easy_task_difficulty"] = args.easy_task_difficulty
our_env_config["medium_task_difficulty"] = args.medium_task_difficulty
our_env_config["hard_task_difficulty"] = args.hard_task_difficulty

our_env_config["HARD_EPS"] = args.HARD_EPS
our_env_config["MEDIUM_EPS"] = args.MEDIUM_EPS
our_env_config["SOFT_EPS"] = args.SOFT_EPS
our_env_config["ANGLE_EPS"] = args.ANGLE_EPS
our_env_config["SPEED_EPS"] = args.SPEED_EPS
'''

train_config["only_show"] = args.only_show
train_config["type_show"] = args.type_show
train_config["only_validate"] = args.only_validate

trainCurriculum(config, train_config, our_env_config, reward_config, car_config)