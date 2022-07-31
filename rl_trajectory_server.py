import json
import time
import sys
import argparse
from bottle import run, post, request, response
from policy_gradient.utlis import generateTestDataSet

sys.path.insert(0, "sample-factory/")

from rl_utils.init_global_env_agent import init_global_env_agent
from rl_utils.rl_algorithm import run_algorithm

env, agent, cfg = init_global_env_agent()

@post('/process')
def my_process():
  request_ = json.loads(request.body.read())
  map_ = request_["map"]
  trainTask = request_["task"]
  valTask = request_["task"]
  second_goal = request_["second_goal"]
  print("debug count of tasks:", "train:", 
          len(trainTask), "val:", len(valTask))
  print("debug count of maps:", len(map_))

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

  env.update_task(map_, trainTask, valTask, second_goal)

  trajectory_info = run_algorithm(cfg, env, agent, max_steps=600)

  return(json.dumps(trajectory_info))

run(host='172.17.0.2', port=8080, debug=True)