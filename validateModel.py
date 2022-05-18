import json
import ray.rllib.agents.ppo as ppo
#import wandb
import matplotlib.pyplot as plt
import torch
from EnvLib.ObstGeomEnv import *
from planning.generateMap import *
from policy_gradient.utlis import *
from generate_trajectory import create_task
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

start = time.time()

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


with open("configs/train_configs.json", 'r') as f:
    # train mode config
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    # config for our env
    our_env_config = json.load(f)

with open('configs/reward_weight_configs.json', 'r') as f:
    reward_config = json.load(f)

with open('configs/car_configs.json', 'r') as f:
    car_config = json.load(f)

rrt = True


vehicle_config = VehicleConfig(car_config)
roi_boundaries = [point(0, 0), point(10, 0),
                      point(10, -5), point(14, -5), 
                      point(14, 0), point(24, 0),
                      point(24, 8), point(0, 8)]

vehicle_pos = point(0, 2.5)
parking_pos = point(12, -1.5)

maps, trainTask, valTasks = create_task(roi_boundaries, vehicle_pos, 
                                            parking_pos)


if not rrt:
    maps_obst = {}
    valTasks_obst = {}
    trainTask_obst = {}
    for index in range(12):
        maps_obst["map" + str(index)] = readObstacleMap("maps/obstacle_map" + str(index)+ ".txt")
        valTasks_obst["map" + str(index)] = readTasks("maps/val_map" + str(index)+ ".txt")
        trainTask_obst["map" + str(index)] = readTasks("maps/train_map" + str(index)+ ".txt")

    maps_dyn_obst = {}
    valTasks_dyn_obst = {}
    trainTask_dyn_obst = {}
    for index in range(12):
        maps_dyn_obst["map" + str(index)] = readObstacleMap("maps/obstacle_map" + str(index)+ ".txt")
        valTasks_dyn_obst["map" + str(index)] = readDynamicTasks("maps/dyn_val_map" + str(index)+ ".txt")
        trainTask_dyn_obst["map" + str(index)] = readDynamicTasks("maps/dyn_train_map" + str(index)+ ".txt")

    obstacle_map = []

    if our_env_config["obstacles"]:
        maps = maps_obst
        trainTask = trainTask_obst
        valTasks = valTasks_obst

    # if our_env_config["dyn_obstacles"]:
    #     maps = maps_dyn_obst
    #     trainTask = valTasks_dyn_obst
    #     valTasks = trainTask_dyn_obst

else:
    maps = maps
    trainTask = valTasks
    valTasks = trainTask

#print(f"our_env_config {our_env_config}")
if our_env_config["union"]:
    environment_config = {
        'vehicle_config': vehicle_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        "second_goal" : [parking_pos.x, parking_pos.y, 
                        degToRad(90), 0, 0]
    }
else:
    environment_config = {
        'vehicle_config': vehicle_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config
    }

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
config['num_workers'] = 1
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
config['env_config'] = environment_config
config["explore"] = False

config["model"]['fcnet_hiddens'] = [512, 512]
config["model"]["use_lstm"] = train_config['use_lstm']
config["model"]["lstm_use_prev_action"] = train_config['lstm_use_prev_action']
config["model"]["free_log_std"] = train_config["free_log_std"]

trainer = ppo.PPOTrainer(config=config, env=ObsEnvironment)
val_env = ObsEnvironment(environment_config)


folder_path = "./myModelWeight1"
train_config['curriculum_name'] = "rllib_ppoWithOrientation"
curriculum_name = train_config['curriculum_name']
print(train_config["curriculum_name"])
folder_path = os.path.join(folder_path, train_config['curriculum_name'])
#trainer.restore("./myModelWeight1/new_dynamic_steps_7/6_step_3/checkpoint_003700/checkpoint-3700")
#trainer.restore("./myModelWeight1/new_dynamic_steps_7/4_step_2/checkpoint_002760/checkpoint-2760")
trainer.restore("./myModelWeight1/new_dynamic_steps_10/1_step_16/checkpoint_003130/checkpoint-3130")
    
ANGLE_EPS = float(our_env_config['ANGLE_EPS']) / 2.
SPEED_EPS = float(our_env_config['SPEED_EPS']) / 2.
STEERING_EPS = float(our_env_config['STEERING_EPS']) / 2.
agent = trainer
env = val_env


##using wandb
#wandb_config = dict(car_config, **train_config, **our_env_config, **reward_config)
#wandb.init(config=wandb_config, project=train_config['project_name'], entity='grisha1')


end = time.time()
print("installation_time: ", end - start)

start = time.time()

#val_done_rate, min_total_val_dist, val_counter_collision, val_traj = validation(val_env, trainer)
#_, val_traj, _, _ = validate_task(val_env, trainer, save_image=True, val_key=key)
#wandb.log({f"Val_trajectory_{1}": wandb.Video(val_traj, fps=10, format="gif")})
for key in val_env.valTasks:
    _, val_traj, _, _ = validate_task(val_env, trainer, save_image=True, val_key=key)
    
    ##using wandb
    #wandb.log({f"Val_trajectory_{0}": wandb.Video(val_traj, fps=10, format="gif")})
    
print("get video")
#print([(state.x, state.y) for state in states])


end = time.time()

print("algorithm_time: ", end - start)


plt.plot([state.x for state in states], 
				[state.y for state in states])
plt.show()

#print(f"val_done_rate: {val_done_rate}")
#print(f"val_counter_collision: {val_counter_collision}")
#print(f"min_total_val_dist: {min_total_val_dist}")

