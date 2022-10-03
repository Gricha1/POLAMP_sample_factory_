import json
import time
import sys
import argparse
from bottle import run, post, request, response
from dataset_generation.utlis import ChangeTaskFormat
sys.path.insert(0, "sample-factory/")
from rl_utils.init_global_env_agent import init_global_env_agent
from rl_utils.rl_algorithm import run_algorithm
from rl_utils.utils import *

use_wandb = True
if use_wandb:
  import wandb
  wandb.init(project='validate_polamp', entity='grisha1')
else:
  wandb = None

env, agent, cfg = init_global_env_agent()

@post('/process')
def my_process():
  request_ = json.loads(request.body.read())
  request_task = request_["task"]

  generated_tasks = transformHTTPRequestToTask(request_task)
  print("debug count task:", len(generated_tasks))
  print("debug task:", generated_tasks)

  maps, generated_tasks = ChangeTaskFormat(generated_tasks)

  ####DEBUG
  #with open("configs/environment_configs.json", 'r') as f:
  #    our_env_config = json.load(f)
  #with open("configs/car_configs.json", 'r') as f:
  #    car_config = json.load(f)
  #dataSet, second_goal = generateTestDataSet(our_env_config, car_config)
  #maps, trainTasks, valTasks = dataSet["empty"]
  #print("correct maps:", maps)
  #print("correct tasks:", trainTasks)
  ######

  env.update_task(maps, generated_tasks)

  trajectory_info, run_info = run_algorithm(cfg, env, agent, 
                        max_steps=env.max_episode_steps, wandb=wandb)

  respond_ = {"trajectory": trajectory_info, "run_info": run_info}
  
  return(json.dumps(respond_))
  
run(host='172.17.0.2', port=8080, debug=True)