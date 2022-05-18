import json
import wandb
import torch
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
from EnvLib.ObstGeomEnv import *
from planning.generateMap import *
from policy_gradient.utlis import *


def create_task(roi_boundaries, vehicle_pos, parking_pos):
    '''
        roi_boundaries[0] = (roi_boundaries[0].x, roi_boundaries[0].y)
        vehicle_pos = (vehicle_pos.x, vehicle_pos.y)

        create horzontal task:

                            top
        -----------------------------------------------------

                                            first_goal
          start                               
        -------------------      -----------------------------   
         left_bottom        |   |     right_bottom
                            |   |
                            -----
    '''
    left_bottom = [(roi_boundaries[1].x + roi_boundaries[0].x) / 2, 
                         roi_boundaries[1].y - 2,
                         0,
                         2,
                         (roi_boundaries[1].x - roi_boundaries[0].x) / 2]

    right_bottom = [(roi_boundaries[5].x + roi_boundaries[4].x) / 2, 
                         roi_boundaries[5].y - 2,
                         0,
                         2,
                         (roi_boundaries[5].x - roi_boundaries[4].x) / 2]

    top = [(roi_boundaries[7].x + roi_boundaries[6].x) / 2, 
            roi_boundaries[7].y + 2,
            0,
            2,
            (roi_boundaries[7].x - roi_boundaries[6].x) / 2]

    start = [vehicle_pos.x, vehicle_pos.y, 
            0, 0., 0]
            
    first_goal = [(roi_boundaries[5].x + roi_boundaries[4].x) / 2, 
                (roi_boundaries[7].y + roi_boundaries[0].y) / 2 + (roi_boundaries[7].y - roi_boundaries[0].y) / 4, 
                0, 0., 0]

    #set task
    maps = {}
    trainTask = {}
    valTasks = {}
    maps["map0"] = [left_bottom, right_bottom, top]
    trainTask["map0"] = [[start, first_goal]]
    valTasks["map0"] = [[start, first_goal]]
    
    
    return maps, trainTask, valTasks

