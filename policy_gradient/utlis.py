import torch
import json
import os
import numpy as np
#from EnvLib.ObstGeomEnv import *
from EnvLib.ObstGeomEnvSampleFactory import *
#from ray.rllib.utils.spaces import space_utils

def save_configs(config, folder_path, name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(f'{folder_path}/{name}', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def save_weights(folder_path, model_weights):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model_weights, f'{folder_path}/policy.pkl')
 

def validate_task(env, agent, max_steps=300, idx=None, save_image=False, val_key=None):
    agent.config["explore"] = False
    observation = env.reset(idx=idx, fromTrain=False, val_key=val_key)
    images = []
    #states = []

    if agent.config["model"]["use_lstm"]:
        prev_action = list(torch.zeros((2)))
        state = list(torch.zeros((2,256)))
    collision = False
    sum_reward = 0
    min_distance = float('inf')
    
    if save_image:
        images.append(env.render(sum_reward))

    isDone = False
    t = 0

    while not isDone and t < max_steps:


        if agent.config["model"]["use_lstm"]:
            action, state, logits = agent.compute_single_action(
                                                                    observation, 
                                                                    state=state, 
                                                                    prev_action=prev_action
                                                                )
            prev_action = action
        else:
            action = agent.compute_single_action(
                                                    observation
                                                )
        #action[0] = 1.5
        #action[1] = 1.5

        observation, reward, isDone, info = env.step(action)
        # print(f'info: {info}')
        if "EuclideanDistance" in info:
            if min_distance >= info["EuclideanDistance"]:
                min_distance = info["EuclideanDistance"]

        sum_reward += reward
        if save_image:
            images.append(env.render(sum_reward))
        t += 1

        if "SoftEps" in info or "Collision" in info:
            if "Collision" in info:
                collision = True
                isDone = False
            break

    if save_image:
        images = np.transpose(np.array(images), axes=[0, 3, 1, 2])

    agent.config["explore"] = True
    #return isDone, images, min_distance, collision, states
    return isDone, images, min_distance, collision

def validation(env, agent, max_steps):
    val_done_rate = 0
    min_total_val_dist = 0
    val_counter_collision = 0
    n_vals = 0

    #print("DEBUG:", "env val task:", len(env.valTasks))

    for key in env.valTasks:
        # print(f'len(env.valTasks[key]): {len(env.valTasks[key])}')
        #print("DEBUG:", "val one task:", len(env.valTasks[key]))
        for i in range(len(env.valTasks[key])):
            # print(key)
            isDone, val_traj, min_distance, collision = \
                        validate_task(env, agent, max_steps=max_steps, 
                                      idx=i, save_image=False, val_key=key)
            # print(f'collision: {collision}')
            # print(f'isDone: {isDone}')
            val_counter_collision += int(collision)
            val_done_rate += int(isDone)
            min_total_val_dist += min_distance 
        n_vals += len(env.valTasks[key])

    if n_vals < 1:
        n_vals = 1
    val_done_rate /= n_vals
    val_done_rate *= 100
    min_total_val_dist /= n_vals
    val_counter_collision /= n_vals
    val_counter_collision *= 100

    return val_done_rate, min_total_val_dist, val_counter_collision, val_traj

def generateValidateTasks(config):
    min_dist = config['min_dist']
    max_dist = config['max_dist']
    max_val_vel = config['max_vel']
    alpha = config['alpha']
    discrete_alpha = config['discrete_alpha']
    max_steer = config['max_steer']
    alpha = degToRad(alpha)
    alphas = np.linspace(-alpha, alpha, discrete_alpha)
    valTasks = []

    for angle in alphas:
        for angle1 in alphas:
            valTasks.append(([
                0., 0., angle, 0., 
                degToRad(np.random.randint(-max_steer, max_steer + 1))
                ], 
                [
                np.random.randint(min_dist, max_dist + 1), 0., angle1,
                 kmToM(np.random.randint(0, max_val_vel + 1)), 0.
                ]))

    return valTasks

def getDynamicTask(dataset_info, temp_info):
    forward_start_x = dataset_info["forward_start_x"]
    forward_end_x = dataset_info["forward_end_x"]
    forward_start_y = dataset_info["forward_start_y"]
    forward_end_y = dataset_info["forward_end_y"]
    buttom_road_edge_y = dataset_info["buttom_road_edge_y"]
    road_width = dataset_info["road_width"]
    samples_theta_eps_ego = dataset_info["samples_theta_eps_ego"]
    union_without_forward_task = dataset_info["union_without_forward_task"]

    forward_dyn_movement = np.random.choice([True, False])
    theta_angle = np.random.choice(samples_theta_eps_ego)
    dynamic_tasks = []
    if temp_info["forward_task"]:
        forward_start_x_ = temp_info["forward_start_x_"]
        forward_end_x_ = temp_info["forward_end_x_"]
        forward_start_y_ = temp_info["forward_start_y_"]
        forward_end_y_ = temp_info["forward_end_y_"]

        if road_width <= 3:
            dyn_obst_y = [buttom_road_edge_y + road_width + 0.5 * road_width,
                            buttom_road_edge_y + 0.5 * road_width]
        else:
            dyn_obst_y = [buttom_road_edge_y + road_width + 0.5 * road_width,
                        buttom_road_edge_y + road_width, 
                        buttom_road_edge_y + 0.5 * road_width]
    
        if not forward_dyn_movement:              
            if forward_start_x_ == forward_start_x[2]:
                dyn_obst_x = [forward_end_x[1] + 6, forward_end_x[1] + 8]
                dynamic_speed = [0.9, 1, 1.1]
            elif forward_start_x_ == forward_start_x[0]\
                and forward_end_x_ == forward_end_x[1]:
                dyn_obst_x = [forward_end_x[1] + 2, forward_end_x[1] + 4]
                dynamic_speed = [0.9, 1.1, 1.3, 1.5]
            elif forward_start_x_ == forward_start_x[0]\
                and forward_end_x_ == forward_end_x[0]:
                dyn_obst_x = [forward_end_x[0] + 3, forward_end_x[0] + 5]
                dynamic_speed = [0.9, 1, 1.2]
            elif forward_start_x_ == forward_start_x[1]:
                dyn_obst_x = [forward_end_x[0] + 8, 
                            forward_end_x[1] + 6,
                            forward_end_x[1] + 8]
                dynamic_speed = [0.9, 1, 1.2]
            else:
                dyn_obst_x = np.linspace(forward_start_x_ + 10, 
                            forward_start_x_ + 18, 3)
                dynamic_speed = [0.9, 1, 1.2]
            
        else:
            dyn_obst_x = np.linspace(forward_start_x_ - 18, forward_start_x_ - 10, 3)
            dynamic_speed = [1.5, 1.1, 0.9]
            
        dyn_obst_x_ = np.random.choice(dyn_obst_x)
        dyn_obst_y_ = np.random.choice(dyn_obst_y)
        dynamic_speed_ = np.random.choice(dynamic_speed)
        if forward_dyn_movement:
            dyn_theta = 0
        else:
            dyn_theta = degToRad(180)
        dyn_obs = [dyn_obst_x_, 
                    dyn_obst_y_, 
                    dyn_theta, dynamic_speed_, 0]
        if union_without_forward_task:
            dynamic_tasks.append(([forward_start_x_, forward_start_y_, 
                                theta_angle, 0., 0], 
                                [0, 0, 0, 0, 0], 
                                [dyn_obs]))
            if forward_dyn_movement:
                dyn_obst_x_second_ = np.random.choice(dyn_obst_x)
                dyn_obst_y_second_ = np.random.choice(dyn_obst_y)
                dynamic_speed_second_ = np.random.choice(dynamic_speed)
            else:
                dyn_obst_x_second = np.linspace(forward_end_x_ + 10, forward_end_x_ + 18, 3)
                dynamic_speed_second = [1.5, 1.1, 0.9]
                dyn_obst_x_second_ = np.random.choice(dyn_obst_x_second)
                dyn_obst_y_second_ = np.random.choice(dyn_obst_y)
                dynamic_speed_second_ = np.random.choice(dynamic_speed_second)
            dyn_obs_second = [dyn_obst_x_second_, 
                    dyn_obst_y_second_, 
                    dyn_theta, dynamic_speed_second_, 0]
                    
            dynamic_tasks.append([forward_end_x_, forward_end_y_, 
                                theta_angle, 0., 0], 
                                [0, 0, 0, 0, 0], 
                                [dyn_obs_second])
        else:
            dynamic_tasks.append(([forward_start_x_, forward_start_y_, 
                                0, 0., 0], 
                                [forward_end_x_, forward_end_y_, 
                                theta_angle, 0, 0], 
                                [dyn_obs]))
                
    
    else:
        backward_start_x_ = temp_info["backward_start_x_"]
        backward_start_y_ = temp_info["backward_start_y_"]

        if road_width <= 3:
            dyn_obst_y = [buttom_road_edge_y + road_width + 0.5 * road_width, 
                                buttom_road_edge_y + 0.5 * road_width]
        else:
            dyn_obst_y = [buttom_road_edge_y + road_width + 0.5 * road_width,
                        buttom_road_edge_y + road_width, 
                        buttom_road_edge_y + 0.5 * road_width]
        if not forward_dyn_movement:
            dyn_obst_x = np.linspace(backward_start_x_ + 17, 
                                        backward_start_x_ + 7, 3)
            dynamic_speed = [1.5, 1.1, 0.9]
        else: 
            dyn_obst_x = np.linspace(backward_start_x_ - 17, 
                                        backward_start_x_ - 7, 3)
            dynamic_speed = [1.5, 1.1, 0.9]
            
        dyn_obst_x_ = np.random.choice(dyn_obst_x)
        dyn_obst_y_ = np.random.choice(dyn_obst_y)
        dynamic_speed_ = np.random.choice(dynamic_speed)
        if forward_dyn_movement:
            dyn_theta = 0
        else:
            dyn_theta = degToRad(180)
        dyn_obs = [dyn_obst_x_, 
                    dyn_obst_y_, 
                    dyn_theta, dynamic_speed_, 0]

        dynamic_tasks.append(([backward_start_x_, backward_start_y_, 
                            theta_angle, 0., 0], 
                            [0, 0, 0, 0, 0], 
                            [dyn_obs]))

    return dynamic_tasks
        
def generateTasks(config, 
                bottom_left_boundary_center_x,
                bottom_left_boundary_center_y, 
                bottom_left_boundary_height, 
                parking_height, 
                buttom_road_edge_y,
                road_width, second_goal, task_difficulty,
                dynamic, union, union_without_forward_task,
                validate_on_train=False):
                                 
    valTasks = []

    EASY_TASK = False # static positions
    MEDIUM_TASK = False # position OX + OY
    HARD_TASK = False # rotation + position
    if task_difficulty == "easy":
        EASY_TASK = True
    elif task_difficulty == "medium":
        MEDIUM_TASK = True
    else:
        HARD_TASK = True
    
    assert (EASY_TASK + MEDIUM_TASK + HARD_TASK) == 1, \
        "custom assert: pick only one task"

    #generate tasks
    forward_start_x = np.array(
                                [
                                bottom_left_boundary_center_x \
                                - bottom_left_boundary_height \
                                + config["length"] / 2 \
                                - config["wheel_base"] / 2,
                                bottom_left_boundary_center_x \
                                - config["wheel_base"] / 2,
                                bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                - config["length"] / 2 \
                                - config["wheel_base"] / 2
                                ]
                              )
    forward_start_y = np.array(
                                [
                                buttom_road_edge_y + config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + road_width,
                                buttom_road_edge_y + 2 * road_width \
                                - config["width"] / 2 - 1.2 #d for boundary
                                ]
                              )
    forward_end_x = (forward_start_x + (2 * bottom_left_boundary_height + \
                                                    parking_height) + 2)[:2]
    forward_end_y = [forward_start_y[0], forward_start_y[2]]
    backward_start_x = np.linspace(forward_end_x[0], forward_end_x[1], 5)
    backward_start_y = np.linspace(forward_end_y[0], forward_end_y[1], 5)
    dynamic_speed = np.linspace(0.5, 2, 30)
    theta_eps_ego = degToRad(15)
    samples_theta_eps_ego = np.linspace(-theta_eps_ego, theta_eps_ego, 30)

    dataset_info = {
        "forward_start_x": forward_start_x,
        "forward_start_y": forward_start_y,
        "forward_end_x": forward_end_x,
        "forward_end_y": forward_end_y,
        "backward_start_x": backward_start_x,
        "backward_start_y": backward_start_y,
        "buttom_road_edge_y": buttom_road_edge_y,
        "samples_theta_eps_ego": samples_theta_eps_ego,
        "road_width": road_width,
        "union_without_forward_task": union_without_forward_task
    }


    if not validate_on_train: #if validate dataset
        val_start_forward_x = []
        for i in range(len(forward_start_x) - 1):
            val_start_forward_x.append(
                (forward_start_x[i] + forward_start_x[i + 1]) / 2
                                        )
        val_start_forward_x.append(forward_start_x[-1])

        val_start_forward_y = []
        for i in range(len(forward_start_y) - 1):
            val_start_forward_y.append(
                (forward_start_y[i] + forward_start_y[i + 1]) / 2
                                        )
        val_start_forward_y.append(forward_start_y[-1])

        val_end_forward_x = []
        for i in range(len(forward_end_x) - 1):
            val_end_forward_x.append(
                (forward_end_x[i] + forward_end_x[i + 1]) / 2
                                        )
        val_end_forward_x.append(forward_end_x[-1])

        val_end_forward_y = []
        for i in range(len(forward_end_y) - 1):
            val_end_forward_y.append(
                (forward_end_y[i] + forward_end_y[i + 1]) / 2
                                        )
        val_end_forward_y.append(forward_end_y[-1])

        forward_start_x = val_start_forward_x
        forward_start_y = val_start_forward_y
        forward_end_x = val_end_forward_x
        forward_end_y = val_end_forward_y
        backward_start_x = np.linspace(forward_end_x[0], forward_end_x[1], 5)
        backward_start_y = np.linspace(forward_end_y[0], forward_end_y[1], 5)


    dataset_info = {
        "forward_start_x": forward_start_x,
        "forward_start_y": forward_start_y,
        "forward_end_x": forward_end_x,
        "forward_end_y": forward_end_y,
        "backward_start_x": backward_start_x,
        "backward_start_y": backward_start_y,
        "buttom_road_edge_y": buttom_road_edge_y,
        "samples_theta_eps_ego": samples_theta_eps_ego,
        "road_width": road_width,
        "union_without_forward_task": union_without_forward_task
    }

    #print("debug utils:", "validate forward:", not validate_on_train, 
    #     len(forward_start_y) * len(forward_end_x) * len(forward_end_y) * len(forward_start_x))

    #forward_tasks(3 * 2 * 3 * 3 = 48 tasks)
    for forward_start_y_ in forward_start_y:
        for forward_end_x_ in forward_end_x:
            for forward_end_y_ in forward_end_y:  
                for forward_start_x_ in forward_start_x:
                    temp_info = {}
                    temp_info["forward_task"] = True
                    temp_info["forward_start_x_"] = forward_start_x_
                    temp_info["forward_end_x_"] = forward_end_x_
                    temp_info["forward_start_y_"] = forward_start_y_
                    temp_info["forward_end_y_"] = forward_end_y_

                    if not union: # not union DIDNT COMPLETE
                        if dynamic:
                            #dynamic_speed_ = np.random.choice(dynamic_speed)
                            #dyn_obs = [forward_end_x_, buttom_road_edge_y + road_width + 0.5 * road_width, 
                            #        degToRad(180), dynamic_speed_, 0]        

                            #valTasks.append(([forward_start_x_, forward_start_y_, 0, 0., 0], 
                            #                [forward_end_x_, forward_end_y_, 0, 0, 0], 
                            #                [dyn_obs]))  
                            valTasks.extend(getDynamicTask(dataset_info, 
                                                                temp_info))                        
                        else:
                            if EASY_TASK:
                                #valTasks.append(([forward_start_x[2], forward_start_y[2], 0, 0., 0], 
                                #                [forward_end_x[1], forward_end_y[2], 0, 0, 0]))
                                valTasks.append(([forward_start_x[2], forward_start_y[2], 0, 0., 0], 
                                                [forward_end_x[1], forward_end_y[1], 0, 0, 0]))

                            elif MEDIUM_TASK:
                                valTasks.append(([forward_start_x_, forward_start_y_, 0, 0., 0], 
                                                [forward_end_x_, forward_end_y_, 0, 0, 0]))

                            elif HARD_TASK:   
                                theta_angle = np.random.choice(samples_theta_eps_ego)          
                                valTasks.append(([forward_start_x_, forward_start_y_, 0, 0., 0], 
                                                [forward_end_x_, forward_end_y_, theta_angle, 0, 0]))

                    else: #union tasks
                        forward_dyn_movement = np.random.choice([True, False])
                        theta_angle = np.random.choice(samples_theta_eps_ego) 
                        if dynamic:
                            valTasks.extend(getDynamicTask(dataset_info, 
                                            temp_info))

                        else:        
                            if EASY_TASK:
                                component_1 = forward_start_x_[2]
                                component_2 = forward_start_y_[2]
                                component_3 = 0
                                component_4 = 0
                                component_5 = 0
                                component_6 = forward_end_x_[1]
                                component_7 = forward_end_y_[2]
                                component_8 = 0
                                component_9 = 0
                                component_10 = 0

                            elif MEDIUM_TASK:
                                component_1 = forward_start_x_
                                component_2 = forward_start_y_
                                component_3 = 0
                                component_4 = 0
                                component_5 = 0
                                component_6 = forward_end_x_
                                component_7 = forward_end_y_
                                component_8 = 0
                                component_9 = 0
                                component_10 = 0

                            elif HARD_TASK:
                                component_1 = forward_start_x_
                                component_2 = forward_start_y_
                                component_3 = 0
                                component_4 = 0
                                component_5 = 0
                                component_6 = forward_end_x_
                                component_7 = forward_end_y_
                                component_8 = 0
                                component_9 = 0
                                component_10 = 0    
                                theta_angle = np.random.choice(samples_theta_eps_ego)   
                                component_8 = theta_angle  
                            if union_without_forward_task:
                                valTasks.append(([component_1, component_2, 
                                                    component_3, component_4, component_5], 
                                                    [0, 0, 0, 0, 0]))
                                valTasks.append(([component_6, component_7, 
                                                    component_8, component_9, component_10], 
                                                    [0, 0, 0, 0, 0]))                  
                            else:
                                valTasks.append(([component_1, component_2, 
                                                    component_3, component_4, component_5], 
                                                    [component_6, component_7, component_8, 
                                                        component_9, component_10]))

    if not union: # backward tasks
        #print("debug utils:", "validate backward:", not validate_on_train, 
        # len(backward_start_x) * len(backward_start_y))
        #backward_tasks(4 * 3 = 12 tasks)
        for backward_start_x_ in backward_start_x:
            for backward_start_y_ in backward_start_y:
                temp_info = {}
                temp_info["forward_task"] = False
                temp_info["backward_start_x_"] = backward_start_x_
                temp_info["backward_start_y_"] = backward_start_y_

                if dynamic:
                    '''
                    dynamic_speed_ = np.random.choice(dynamic_speed)
                    dyn_obs = [forward_end_x_, buttom_road_edge_y + road_width + 0.5 * road_width, 
                            degToRad(180), dynamic_speed_, 0]        

                    valTasks.append(([forward_start_x_, forward_start_y_, 0, 0., 0], 
                                    [forward_end_x_, forward_end_y_, 0, 0, 0], 
                                    [dyn_obs]))        
                    '''                  
                    valTasks.extend(getDynamicTask(dataset_info, 
                                            temp_info))
                else:
                    if EASY_TASK:
                        valTasks.append(([backward_start_x[2], backward_start_y[2], 0, 0., 0], 
                                        second_goal))

                    elif MEDIUM_TASK:
                        valTasks.append(([backward_start_x_, backward_start_y_, 0, 0., 0], 
                                        second_goal))

                    elif HARD_TASK:  
                        theta_angle = np.random.choice(samples_theta_eps_ego)           
                        valTasks.append(([backward_start_x_, backward_start_y_, theta_angle, 0., 0], 
                                        second_goal))


    return valTasks
