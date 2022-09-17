from turtle import forward
import torch
import json
import os
import numpy as np
from EnvLib.utils import *

def getTrainValidateTasks(tasks_config, car_config, train):
    """
    return:
        [task1, task2, task3, ...]
        where task1 = ([stat_obst_1, ...], [start, goal], [dyn_obst_1, ...])
    """
    tasks = [] 
    count_task_per_case = 1000 if train else 10
    for case_num in range(1, 12 + 1):
        for i in range(count_task_per_case):
            train_case_task = getCaseTask(case_num,
                                          tasks_config, 
                                          car_config)
            tasks.extend(train_case_task)

    return tasks

def ChangeTaskFormat(generated_tasks):
    """
    this function is used for changing task format for current environment
    task format.
    this function will be deleted in the future
    return:
        maps["map1"] = [[stat_obst_1, stat_obst_2, ..., stat_obst_4]]
        generated_tasks["map1"] = [
                                    [start, goal, [dyn_obst_1, dyn_obst_2, ...]],
                                    ...
                                  ]
    """
    maps = {}
    generated_tasks_map = {}
    
    index = 0
    for static_obsts, start_goal, dynamic_obsts in generated_tasks:
        maps[f"map{index}"] = static_obsts
        if len(dynamic_obsts) != 0:
            generated_tasks_map[f"map{index}"] = [[start_goal[0], 
                                                   start_goal[1], 
                                                   dynamic_obsts]]
        else:
            generated_tasks_map[f"map{index}"] = [[start_goal[0],
                                                   start_goal[1]]]
        index += 1
    
        
    return maps, generated_tasks_map

def getCaseTask(case_num, tasks_config, car_config):
    """
    return:
        [
            ([stat_obst_1, stat_obst_2, ..., stat_obst_4], 
             [start, goal], 
             [dyn_obst_1, dyn_obst_2, ...]),
            ([stat_obst_1, stat_obst_2, ..., stat_obst_5],
             [start, goal], 
             [dyn_obst_1, dyn_obst_2, ...]),
            ...
        ]
    """
    case_task = []
    static = tasks_config['static']
    dynamic = tasks_config['dynamic']
    union = tasks_config['union']
    
    # static obsts params:
    parking_heights = np.linspace(2.7 + 0.2, 2.7 + 5, 20)
    parking_widths = np.linspace(4.5, 7, 20)
    road_widths = np.linspace(3, 6, 20)
    bottom_left_boundary_heights = np.linspace(4.5, 6, 20)
    upper_boundary_widths = np.linspace(0.5, 3, 20)
    upper_boundary_heights = np.linspace(14, 16, 20)
    bottom_left_boundary_center_x = 5 # any value
    bottom_left_boundary_center_y = -5.5 # init value
    bottom_left_boundary_height = np.random.choice(bottom_left_boundary_heights)
    upper_boundary_width = np.random.choice(upper_boundary_widths)
    upper_boundary_height = np.random.choice(upper_boundary_heights)
    parking_height = np.random.choice(parking_heights)
    parking_width = np.random.choice(parking_widths)
    road_width = np.random.choice(road_widths)
    staticObstsInfo = {}
    if static:
        static_obsts = getValetStaticObstsAndUpdateInfo(
            parking_height, parking_width, bottom_left_boundary_height, 
            upper_boundary_width, upper_boundary_height,
            bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
            road_width,
            staticObstsInfo
        )
    else:
        bottom_left_boundary_height = bottom_left_boundary_heights[-1]
        upper_boundary_width = upper_boundary_widths[0]
        upper_boundary_height = upper_boundary_heights[-1]
        parking_height = parking_heights[-1]
        parking_width = parking_widths[-1]
        road_width = road_widths[-1]
        getValetStaticObstsAndUpdateInfo(
            parking_height, parking_width, bottom_left_boundary_height, 
            upper_boundary_width, upper_boundary_height,
            bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
            road_width,
            staticObstsInfo
        )
        static_obsts = []

    assert (static and len(static_obsts) == 4) or \
            (not static and len(static_obsts) == 0), \
            f"incorrect static task, static is {static} " + \
            f"but len static is {len(static_obsts)}"
    
    if union:
        forward_task = None
    else:
        forward_task = np.random.choice([True, False])
    start, goal = getValetStartGoalPose(staticObstsInfo, 
                                        car_config, 
                                        union,
                                        forward_task)
    agent_task = [start, goal]
    assert len(start) == 5 and len(goal) == 5, \
        f"start and goal len must be 5 but len start is: {len(start)} " + \
        f"and len goal is: {len(goal)}"

    if dynamic:
        dynamic_obsts = getCaseValetDynamicObst(staticObstsInfo, 
                                                car_config, 
                                                case_num)
    else:
        dynamic_obsts = []

    assert len(dynamic_obsts) == 1 or len(dynamic_obsts) == 0,\
        "incorrent count of dynamics in env, should be <= 1 " \
        + f"but {len(dynamic_obsts)} given"
    static_obsts, agent_task, dynamic_obsts = trySatisfyCollisionConditions(
                                                        staticObstsInfo,
                                                        car_config,
                                                        static_obsts, 
                                                        agent_task, 
                                                        dynamic_obsts
                                                        )
    case_task.append((static_obsts, agent_task, dynamic_obsts))


    return case_task

def trySatisfyCollisionConditions(staticObstsInfo,
                                  car_config,
                                  static_obsts, 
                                  agent_task, 
                                  dynamic_obsts):
    if len(dynamic_obsts) == 0:
        return static_obsts, agent_task, dynamic_obsts
    
    agent_x, agent_y, agent_theta, agent_v, agent_steer = agent_task[0]
    goal_x, goal_y, goal_theta, goal_v, goal_steer = agent_task[1]
    dyn_x, dyn_y, dyn_theta, dyn_v, dyn_steer, dynamic_config = dynamic_obsts[0]

    case_num = dynamic_config["case"]
    road_width = staticObstsInfo["road_width"]
    buttom_road_edge_y = staticObstsInfo["buttom_road_edge_y"]
    wheel_base = car_config['wheel_base']
    dyn_width = dynamic_config["width"]
    dyn_length = dynamic_config["length"]
    dyn_wheel_base = dynamic_config["wheel_base"]
    agent_width = car_config["width"]
    agent_length = car_config["length"]

    # check collision when agent any dynamic obst appears
    self_d_radius = 1
    # get agent safe circle
    agent_x_center = agent_x + (wheel_base / 2) * np.cos(agent_theta)
    agent_y_center = agent_y + (wheel_base / 2) * np.sin(agent_theta)
    agent_conor_x = agent_x_center - (agent_length / 2) * np.cos(agent_theta) \
        + (agent_width / 2) * np.cos(agent_theta + degToRad(90))
    agent_conor_y = agent_y_center - (agent_length / 2) * np.sin(agent_theta) \
        + (agent_width / 2) * np.sin(agent_theta + degToRad(90))
    agent_radius = np.sqrt(abs(agent_x_center - agent_conor_x) ** 2 \
                           + abs(agent_y_center - agent_conor_y) ** 2)

    # get dynamic safe circle
    dyn_x_center = dyn_x + (dyn_wheel_base / 2) * np.cos(dyn_theta)
    dyn_y_center = dyn_y + (dyn_wheel_base / 2) * np.sin(dyn_theta)
    dyn_conor_x = dyn_x_center - (dyn_length / 2) * np.cos(dyn_theta) \
        + (dyn_width / 2) * np.cos(dyn_theta + degToRad(90))
    dyn_conor_y = dyn_y_center - (dyn_length / 2) * np.sin(dyn_theta) \
        + (dyn_width / 2) * np.sin(dyn_theta + degToRad(90))
    dyn_radius = np.sqrt(abs(dyn_x_center - dyn_conor_x) ** 2 \
                           + abs(dyn_y_center - dyn_conor_y) ** 2)
    
    safe_distance_between_centers = agent_radius + self_d_radius \
                                    + dyn_radius + self_d_radius                           
    
    distance_between_centers = np.sqrt((agent_x_center - dyn_x_center) ** 2 \
                                        + (agent_y_center - dyn_y_center) ** 2)
    if distance_between_centers <= safe_distance_between_centers:
        if dyn_x < agent_x and (case_num == 5 or case_num == 6):
            dyn_x = agent_x - safe_distance_between_centers
        elif case_num == 1 or case_num == 2 or \
                case_num == 3 or case_num == 4 or case_num == 5 or \
                case_num == 6:
            dynamic_new_dxs = np.linspace(10, 20, 20)
            dynamic_new_dx = np.random.choice(dynamic_new_dxs)
            dyn_x = agent_x + safe_distance_between_centers \
                    + dynamic_new_dx
        elif (dyn_x <= agent_x and (case_num == 7 or case_num == 8)) or \
                (case_num == 9 or case_num == 10):
            dynamic_new_dxs = np.linspace(5, 10, 20)
            dynamic_new_dx = np.random.choice(dynamic_new_dxs)
            dyn_x = agent_x - safe_distance_between_centers - dynamic_new_dx
        elif case_num == 7 or case_num == 8:
            dyn_x = agent_x + safe_distance_between_centers
        if dyn_y <= agent_y and (case_num == 11 or case_num == 12):
            dynamic_new_dxs = np.linspace(10, 15, 20)
            dynamic_new_dys = np.linspace(10, 15, 20)
            dynamic_new_dx = np.random.choice(dynamic_new_dxs)
            dynamic_new_dy = np.random.choice(dynamic_new_dys)
            if dyn_theta >= degToRad(90):
                dyn_x = agent_x + safe_distance_between_centers + dynamic_new_dx
            else:
                dyn_x = agent_x - safe_distance_between_centers - dynamic_new_dx
            dyn_y = agent_y - safe_distance_between_centers - dynamic_new_dy
        elif dyn_y > agent_y and (case_num == 11 or case_num == 12):
            dynamic_new_dxs = np.linspace(10, 15, 20)
            dynamic_new_dys = np.linspace(10, 15, 20)
            dynamic_new_dx = np.random.choice(dynamic_new_dxs)
            dynamic_new_dy = np.random.choice(dynamic_new_dys)
            if dyn_theta <= degToRad(270):
                dyn_x = agent_x + safe_distance_between_centers + dynamic_new_dx
            else:
                dyn_x = agent_x - safe_distance_between_centers - dynamic_new_dx
            dyn_y = agent_y + safe_distance_between_centers + dynamic_new_dy

    agent = [agent_x, agent_y, agent_theta, agent_v, agent_steer]
    goal = [goal_x, goal_y, goal_theta, goal_v, goal_steer]
    agent_task = [agent, goal]
    dynamic_obsts = [[dyn_x, dyn_y, dyn_theta, dyn_v, dyn_steer, dynamic_config]]

    return static_obsts, agent_task, dynamic_obsts

def getCaseValetDynamicObst(staticObstsInfo, car_config, case_num):
    """
    return:
        [dynamic] where dynamic = [x, y, theta, v, steer, dynamic_config]
                        or dynamic = [] if no dynamic obst
    """
    dynamic_config = {}
    dynamic_config["case"] = case_num
    dynamic_config["movement_func_params"] = {}
    parking_height = staticObstsInfo["parking_height"]
    parking_width = staticObstsInfo["parking_width"]
    road_width = staticObstsInfo["road_width"]
    buttom_road_edge_y = staticObstsInfo["buttom_road_edge_y"]
    bottom_left_boundary_center_x = staticObstsInfo["bottom_left_boundary_center_x"]
    bottom_left_boundary_center_y = staticObstsInfo["bottom_left_boundary_center_y"]
    bottom_left_boundary_height = staticObstsInfo["bottom_left_boundary_height"]
    bottom_left_boundary_width = staticObstsInfo["bottom_left_boundary_width"]
    bottom_right_boundary_height = staticObstsInfo["bottom_right_boundary_height"]
    upper_boundary_width = staticObstsInfo["upper_boundary_width"]
    wheel_base = car_config['wheel_base']
    car_width = car_config["width"]
    shift_for_dynamic_dx = 0.7 # for learning purpose 
    shift_for_dynamic_dy = 0.7 # for learning purpose 
    shift_from_static_boundary = 1.2

    if case_num == 1:
        x_min = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height / 2 \
                + wheel_base / 2 - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height \
                + wheel_base / 2 + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + road_width + shift_for_dynamic_dy
        y_max = buttom_road_edge_y + 2 * road_width \
                - car_width / 2 + shift_from_static_boundary
        theta_min = degToRad(178)
        theta_max = degToRad(182)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    elif case_num == 2:
        x_min = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height / 2 \
                + wheel_base / 2 - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height \
                + wheel_base / 2 + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + road_width + shift_for_dynamic_dy
        y_max = buttom_road_edge_y + 2 * road_width \
                - car_width / 2 + shift_from_static_boundary
        theta_min = degToRad(178)
        theta_max = degToRad(182)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0
 
    elif case_num == 3:
        x_min = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height / 2 \
                + wheel_base / 2 - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height \
                + wheel_base / 2 + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + car_width / 2 \
                + shift_from_static_boundary - shift_for_dynamic_dy
        y_max = buttom_road_edge_y + road_width
        theta_min = degToRad(178)
        theta_max = degToRad(182)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    elif case_num == 4:
        x_min = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height / 2 \
                + wheel_base / 2 - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height \
                + wheel_base / 2 + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + car_width / 2 \
                + shift_from_static_boundary - shift_for_dynamic_dy
        y_max = buttom_road_edge_y + road_width
        theta_min = degToRad(178)
        theta_max = degToRad(182)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0    

    elif case_num == 5:
        x_min = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height / 2 \
                + wheel_base / 2 - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height \
                + wheel_base / 2 + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + road_width + shift_for_dynamic_dy
        y_max = buttom_road_edge_y + 2 * road_width \
                - car_width / 2 + shift_from_static_boundary
        theta_min = degToRad(178)
        theta_max = degToRad(182)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    elif case_num == 6:
        x_min = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height / 2 \
                + wheel_base / 2 - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height \
                + wheel_base / 2 + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + car_width / 2 \
                + shift_from_static_boundary - shift_for_dynamic_dy
        y_max = buttom_road_edge_y + road_width
        theta_min = degToRad(178)
        theta_max = degToRad(182)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    elif case_num == 7:
        x_min = bottom_left_boundary_center_x \
                - bottom_left_boundary_height \
                + car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height \
                - car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + car_width / 2 \
                + shift_from_static_boundary - shift_for_dynamic_dy
        y_max = buttom_road_edge_y + road_width
        theta_min = degToRad(0 - 2)
        theta_max = degToRad(0 + 2)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    elif case_num == 8:
        x_min = bottom_left_boundary_center_x \
                - bottom_left_boundary_height \
                + car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height \
                - car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + road_width + shift_for_dynamic_dy
        y_max = buttom_road_edge_y + 2 * road_width \
                - car_width / 2 + shift_from_static_boundary
        theta_min = degToRad(0 - 2)
        theta_max = degToRad(0 + 2)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0
            
    elif case_num == 9:
        x_min = bottom_left_boundary_center_x \
                - bottom_left_boundary_height \
                + car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height \
                - car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + road_width + shift_for_dynamic_dy
        y_max = buttom_road_edge_y + 2 * road_width \
                - car_width / 2 + shift_from_static_boundary
        theta_min = degToRad(0 - 2)
        theta_max = degToRad(0 + 2)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0
         
    elif case_num == 10:
        x_min = bottom_left_boundary_center_x \
                - bottom_left_boundary_height \
                + car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                - shift_for_dynamic_dx 
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height \
                - car_config["length"] / 2 \
                - car_config["wheel_base"] / 2 \
                + shift_for_dynamic_dx 
        y_min = buttom_road_edge_y + road_width + shift_for_dynamic_dy
        y_max = buttom_road_edge_y + 2 * road_width \
                - car_width / 2 + shift_from_static_boundary
        theta_min = degToRad(0 - 2)
        theta_max = degToRad(0 + 2)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0
        
    elif case_num == 11:
        from_bottom_parking_place = np.random.choice([True, False])
        x_min = bottom_left_boundary_center_x - shift_for_dynamic_dx
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + bottom_right_boundary_height + shift_for_dynamic_dx
        if from_bottom_parking_place:
            y_min = bottom_left_boundary_center_y - bottom_left_boundary_width \
                    - shift_for_dynamic_dx
            y_max = buttom_road_edge_y + road_width - shift_for_dynamic_dx
            theta_min = degToRad(90 - 40)
            theta_max = degToRad(90 + 40)
        else:
            y_min = buttom_road_edge_y + road_width \
                    + shift_for_dynamic_dx
            y_max = buttom_road_edge_y + 2 * road_width \
                + 2 * upper_boundary_width \
                + shift_for_dynamic_dx
            theta_min = degToRad(270 - 40)
            theta_max = degToRad(270 + 40)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    elif case_num == 12:
        from_bottom_parking_place = np.random.choice([True, False])
        x_min = bottom_left_boundary_center_x \
                - bottom_left_boundary_height - shift_for_dynamic_dx
        x_max = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + 2 * bottom_left_boundary_height + shift_for_dynamic_dx
        if from_bottom_parking_place:
            y_min = bottom_left_boundary_center_y - bottom_left_boundary_width \
                    - shift_for_dynamic_dx
            y_max = buttom_road_edge_y + road_width - shift_for_dynamic_dx
            theta_min = degToRad(88)
            theta_max = degToRad(92)
        else:
            y_min = buttom_road_edge_y + road_width \
                    + shift_for_dynamic_dx
            y_max = buttom_road_edge_y + 2 * road_width \
                + 2 * upper_boundary_width \
                + shift_for_dynamic_dx
            theta_min = degToRad(268)
            theta_max = degToRad(272)
        init_v_min = 0
        init_v_max = 0.7
        steer_min = 0
        steer_max = 0

    dynamic = generateCar(x_min, x_max, 
                          y_min, y_max,
                          theta_min, theta_max, 
                          init_v_min, init_v_max,
                          steer_min, steer_max)
    if len(dynamic) == 0:
        return []

    min_width = 1.8 - 0.5
    max_width = 1.8 + 0.4
    dynamic_widths = np.linspace(min_width, max_width, 20)
    min_length = 4.140 - 0.4
    max_length = 4.140 + 0.4
    dynamic_lengths = np.linspace(min_length, max_length, 20)
    safe_buffer = 0.6
    dynamic_config["width"] = np.random.choice(dynamic_widths) + safe_buffer
    dynamic_config["length"] = np.random.choice(dynamic_lengths) + safe_buffer
    dynamic_config["wheel_base"] = wheel_base

    boundary_v_max = 1
    boundary_v_min = 0.2
    dynamic_config["boundary_v"] = np.random.choice(
                        np.linspace(boundary_v_min, boundary_v_max, 10))

    # movement function for dynamic obst
    a_min_possible = 0
    a_max_possible = 1
    a_max = np.random.choice(np.linspace(a_min_possible, a_max_possible, 20))
    Eps_min_possible = 0
    Eps_max_possible = 0
    Eps_max = np.random.choice(np.linspace(Eps_min_possible, Eps_max_possible, 20))
    boundary_action = [a_max, Eps_max]
    dynamic_config["movement_func_params"] = {"boundary_action": boundary_action}
    
    if case_num == 1 or case_num == 2 or case_num == 3 or \
            case_num == 4 or case_num == 9 or case_num == 10:
        if case_num == 2 or case_num == 4 or case_num == 10:
            stop_steps = list(range(90, 200))
        else:
            stop_steps = list(range(30, 70))
        stop_step = np.random.choice(stop_steps)
        dynamic_config["movement_func_params"]["stop_step"] = int(stop_step)

        dynamic_do_reverse_after_stop = np.random.choice([True, False, False])
        dynamic_config["movement_func_params"]["dynamic_do_reverse_after_stop"] \
                                                    = bool(dynamic_do_reverse_after_stop)
        if dynamic_do_reverse_after_stop:
            reverse_steps = list(range(10, 40))
            reverse_step = np.random.choice(stop_steps)
            dynamic_config["movement_func_params"]["reverse_step"] \
                                                    = int(reverse_step)

        def move(last_state, current_steps, time_step=0.1, **args):
            a = np.random.choice(
                    np.linspace(0, boundary_action[0], 10))
            Eps = np.random.choice(
                    np.linspace(0, boundary_action[1], 10))
            if current_steps >= stop_step:
                if last_state[3] > 0:
                    a = -min(boundary_action[0], last_state[3] / time_step)
                elif last_state[3] == 0:
                    a = 0
            if dynamic_do_reverse_after_stop and current_steps >= stop_step and \
                    last_state[3] <= 0 and current_steps < stop_step + reverse_step:
                a = -np.random.choice(
                    np.linspace(0, boundary_action[0], 10))
            if dynamic_do_reverse_after_stop and current_steps >= stop_step + reverse_step:
                if last_state[3] < 0:
                    a = min(boundary_action[0], abs(last_state[3]) / time_step)
                elif last_state[3] == 0:
                    a = 0
            action = [a, Eps]

            return action
        dynamic_config["movement_func"] = move
    elif case_num == 5 or case_num == 6 or case_num == 7 or \
            case_num == 8 or case_num == 11 or case_num == 12:
        a_min_possible = 0
        a_max_possible = 1
        a_max = np.random.choice(np.linspace(a_min_possible, a_max_possible, 20))
        Eps_min_possible = 0
        Eps_max_possible = 0
        Eps_max = np.random.choice(np.linspace(Eps_min_possible, Eps_max_possible, 20))
        boundary_action = [a_max, Eps_max]
        def move(last_state, current_steps, time_step=0.1, **args):
            a = np.random.choice(
                    np.linspace(0, boundary_action[0], 10))
            Eps = np.random.choice(
                    np.linspace(0, boundary_action[1], 10))
            action = [a, Eps]

            return action
        dynamic_config["movement_func"] = move

    dynamic = trySatisfyDynamicConditions(staticObstsInfo,
                                          car_config,
                                          dynamic,
                                          dynamic_config)
    dynamic.append(dynamic_config)

    return [dynamic]

def trySatisfyDynamicConditions(staticObstsInfo, car_config, dynamic, dynamic_config):
    case_num = dynamic_config["case"]
    road_width = staticObstsInfo["road_width"]
    buttom_road_edge_y = staticObstsInfo["buttom_road_edge_y"]
    wheel_base = car_config['wheel_base']
    width = dynamic_config["width"]
    length = dynamic_config["length"]
    agent_width = car_config["width"]
    shift_from_static_boundary = 1.2

    x, y, theta, init_v, steer = dynamic

    # if narrow road, dynamic obst can move only on straigh line
    if buttom_road_edge_y <= y and \
       y <= buttom_road_edge_y + 2 * road_width and \
       road_width <= 4:
            if degToRad(178) <= theta and theta <= degToRad(182):
                theta = degToRad(180)
            elif degToRad(-2) <= theta and theta <= degToRad(2):
                theta = 0
            if y <= buttom_road_edge_y + road_width:
                y = buttom_road_edge_y + width / 2 \
                    + shift_from_static_boundary
            else:
                y = buttom_road_edge_y + 2 * road_width \
                    - width / 2 - shift_from_static_boundary 

    init_v = min(dynamic_config["boundary_v"], init_v)

    # dynamic obst which is moving straigh parallel to OX, might fills all space
    # on the road
    if (case_num == 1 or case_num == 2 or \
            case_num == 5 or case_num == 8 or case_num == 10):
        safe_dy = 1.2
        if (y - width / 2 - buttom_road_edge_y) <= agent_width + safe_dy:
            dy = 1.2
            y += dy
    if (case_num == 3 or case_num == 4 or \
            case_num == 6 or case_num == 7 or case_num == 9):
        safe_dy = 1.2
        if (buttom_road_edge_y + 2 * road_width - (y + width / 2)) <= agent_width + safe_dy:
            dy = 2
            y -= dy
        
    return [x, y, theta, init_v, steer]


def generateCar(x_min, x_max, 
                y_min, y_max,
                theta_min, theta_max, 
                init_v_min, init_v_max,
                steer_min, steer_max):
    x_s = np.linspace(x_min, x_max, 10)
    y_s = np.linspace(y_min, y_max, 10)
    theta_s = np.linspace(theta_min, theta_max, 10)
    v_s = np.linspace(init_v_min, init_v_max, 10)
    steer_s = np.linspace(steer_min, steer_max, 10)
    
    x = np.random.choice(x_s)
    y = np.random.choice(y_s)
    theta = np.random.choice(theta_s)
    v = np.random.choice(v_s)
    steer = np.random.choice(steer_s)
    
    return [x, y, theta, v, steer]

def getValetStaticObstsAndUpdateInfo(
        parking_height, parking_width, 
        bottom_left_boundary_height, upper_boundary_width, upper_boundary_height,
        bottom_left_boundary_center_x, bottom_left_boundary_center_y, road_width_,
        staticObstsInfo):
    '''

                            top
        -----------------------------------------------------

                                            
                                       
        -------------------      -----------------------------   
         left_bottom        |   |     right_bottom
                            |   |
                            -----
                          down_bottom
    '''         
    # parking_height * parking_width = parking area
    bottom_left_boundary_width = parking_width / 2
    bottom_right_boundary_width = parking_width / 2
    bottom_right_boundary_height = bottom_left_boundary_height
    bottom_right_boundary_center_x = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + bottom_right_boundary_height
    bottom_right_boundary_center_y = bottom_left_boundary_center_y
    bottom_road_edge_y = bottom_left_boundary_center_y + \
                    bottom_left_boundary_width 
    bottom_down_width = upper_boundary_width 
    bottom_down_height = parking_height / 2
    upper_boundary_center_x = bottom_left_boundary_center_x \
                    + bottom_left_boundary_height + parking_height / 2
    bottom_down_center_x = upper_boundary_center_x
    bottom_down_center_y = bottom_left_boundary_center_y \
                - bottom_left_boundary_width - bottom_down_width
    road_center_y = bottom_road_edge_y + road_width_
    upper_boundary_center_y = road_center_y + road_width_ + \
                                upper_boundary_width

    staticObstsInfo["parking_height"] = parking_height 
    staticObstsInfo["parking_width"] = parking_width
    staticObstsInfo["road_width"] = road_width_
    staticObstsInfo["buttom_road_edge_y"] = bottom_left_boundary_center_y + \
                                            bottom_left_boundary_width
    staticObstsInfo["bottom_left_boundary_center_x"] = bottom_left_boundary_center_x
    staticObstsInfo["bottom_left_boundary_center_y"] = bottom_left_boundary_center_y
    staticObstsInfo["bottom_left_boundary_height"] = bottom_left_boundary_height
    staticObstsInfo["bottom_left_boundary_width"] = bottom_left_boundary_width
    staticObstsInfo["upper_boundary_width"] = upper_boundary_width
    staticObstsInfo["bottom_right_boundary_height"] = bottom_right_boundary_height

    left_bottom = [
                    bottom_left_boundary_center_x, 
                    bottom_left_boundary_center_y, 0, bottom_left_boundary_width,
                    bottom_left_boundary_height
                  ]
    down_bottom = [
                    bottom_down_center_x, bottom_down_center_y, 
                    0, bottom_down_width, bottom_down_height
                  ]
    right_bottom = [
                    bottom_right_boundary_center_x, 
                    bottom_right_boundary_center_y, 0, bottom_right_boundary_width, 
                    bottom_right_boundary_height
                   ]
    top = [
            upper_boundary_center_x, upper_boundary_center_y, 
            0, upper_boundary_width, upper_boundary_height
          ]

    return [left_bottom, down_bottom, right_bottom, top]

def getTestTasks(car_config):
    parking_height = 2.7
    parking_width = 4.5
    bottom_left_boundary_width = parking_width / 2
    bottom_right_boundary_width = parking_width / 2
    bottom_left_boundary_height = 6 # any value
    bottom_right_boundary_height = bottom_left_boundary_height
    bottom_left_boundary_center_x = 5 # any value
    bottom_right_boundary_center_x = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + bottom_right_boundary_height
    bottom_left_boundary_center_y = -5.5 # init value
    bottom_right_boundary_center_y = bottom_left_boundary_center_y
    bottom_road_edge_y = bottom_left_boundary_center_y + \
                    bottom_left_boundary_width
    upper_boundary_width = 0.5 # any value
    upper_boundary_height = 17 
    bottom_down_width = upper_boundary_width 
    bottom_down_height = parking_height / 2 
    upper_boundary_center_x = bottom_left_boundary_center_x \
                    + bottom_left_boundary_height + parking_height / 2
    bottom_down_center_x = upper_boundary_center_x
    bottom_down_center_y = bottom_left_boundary_center_y \
                - bottom_left_boundary_width - bottom_down_width \
                - 0.2 # dheight
    #get second goal
    second_goal_x = bottom_left_boundary_center_x \
                  + bottom_left_boundary_height \
                  + parking_height / 2
    second_goal_y = bottom_left_boundary_center_y - \
                        car_config["wheel_base"] / 2
    second_goal = [second_goal_x, second_goal_y, degToRad(90), 0, 0]


    # parameters variation:
    bottom_left_right_dx = np.linspace(-1, -0.2, 3)
    road_width = np.linspace(3, 6, 3)
    bottom_left_right_dx_ = -1.5
    road_width_ = 5
    
    road_center_y = bottom_road_edge_y + road_width_
    upper_boundary_center_y = road_center_y + road_width_ + \
                        upper_boundary_width
    map_ = [
            [upper_boundary_center_x, upper_boundary_center_y, 
            0, upper_boundary_width, upper_boundary_height], 
            [bottom_left_boundary_center_x + bottom_left_right_dx_, 
        bottom_left_boundary_center_y, 0, bottom_left_boundary_width,
            bottom_left_boundary_height],
            [bottom_right_boundary_center_x - bottom_left_right_dx_, 
        bottom_right_boundary_center_y, 0, bottom_right_boundary_width, 
            bottom_right_boundary_height], 
            [bottom_down_center_x, bottom_down_center_y, 
            0, bottom_down_width, bottom_down_height]
            ]
    
    forward_start_x_ = bottom_left_boundary_center_x \
                                - bottom_left_boundary_height \
                                + car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
    start_y_on_bottom_lane = bottom_road_edge_y + car_config["width"] / 2 \
                                + 1.2
    start_y_on_center = bottom_road_edge_y + road_width_


    start_pose_1 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_1 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_1 = (
                map_,
                [start_pose_1,
                second_goal],
                [dyn_obst_1]
             )
             
    start_pose_2 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_2 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_2 = (
                map_,
                [start_pose_2,
                second_goal],
                [dyn_obst_2]
             )

    start_pose_3 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_3 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_3 = (
                map_,
                [start_pose_3,
                second_goal],
                [dyn_obst_3]
             )

    start_pose_4 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_4 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_4 = (
                map_,
                [start_pose_4,
                second_goal],
                [dyn_obst_4]
             )
        
    start_pose_5 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_5 = [forward_start_x_ + 20, 
                  bottom_road_edge_y + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_5 = (
                map_,
                [start_pose_5,
                second_goal],
                [dyn_obst_5]
             )

    start_pose_6 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_6 = [forward_start_x_ + 20, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_6 = (
                map_,
                [start_pose_6,
                second_goal],
                [dyn_obst_6]
             )

    start_pose_7 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_7 = [forward_start_x_ - 7, 
                  start_y_on_center, 
                  0, 0.5, 0]
    task_7 = (
                map_,
                [start_pose_7,
                second_goal],
                [dyn_obst_7]
             )

    start_pose_8 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_8 = [second_goal[0] + 5, 
                  second_goal[1], 
                  degToRad(130), 0.35, 0]
    task_8 = (
                map_,
                [start_pose_8,
                second_goal],
                [dyn_obst_8]
             )

    start_pose_9 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_9 = [second_goal[0] - 3.2, 
                  second_goal[1], 
                  degToRad(90), 0.3, 0]
    task_9 = (
                map_,
                [start_pose_9,
                second_goal],
                [dyn_obst_9]
             )


    tasks = [task_1, task_2, task_3, task_4, 
             task_5, task_6, task_7, task_8, task_9]

    return tasks

def getValetStartGoalPose(staticObstsInfo, car_config, union, forward_task):
    parking_height = staticObstsInfo["parking_height"]
    parking_width = staticObstsInfo["parking_width"]
    road_width = staticObstsInfo["road_width"]
    buttom_road_edge_y = staticObstsInfo["buttom_road_edge_y"]
    bottom_left_boundary_center_x = staticObstsInfo["bottom_left_boundary_center_x"]
    bottom_left_boundary_center_y = staticObstsInfo["bottom_left_boundary_center_y"]
    bottom_left_boundary_height = staticObstsInfo["bottom_left_boundary_height"]

    assert (union and (forward_task is None)) or \
            (not union and not (forward_task is None)), \
            f"not correct task config: union is {union} " + \
            f" but forward task is {forward_task}"

    shift_for_start_dx = 0.7
    shift_from_static_boundary = 1.2

    if union and (forward_task is None):
        start_xs = np.linspace(bottom_left_boundary_center_x \
                                - bottom_left_boundary_height \
                                + car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2, 
                                bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                - car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2 \
                                + 2 * bottom_left_boundary_height + \
                                parking_height,
                                20)
        start_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + shift_from_static_boundary,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - shift_from_static_boundary,
                                20
                                )
        start_thetas = np.linspace(-degToRad(20), degToRad(20), 20)
        start_vs = np.linspace(-0.5, 0.5, 20)

    elif forward_task:
        start_xs = np.linspace(bottom_left_boundary_center_x \
                                - bottom_left_boundary_height \
                                + car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
                                - shift_for_start_dx, 
                                bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                - car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
                                + shift_for_start_dx,
                                20)
        start_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - 1.2,
                                20
                                )
        start_thetas = np.linspace(-degToRad(20), degToRad(20), 20)
        start_vs = np.linspace(0, 0.5, 20)

    else:
        start_xs = np.linspace(bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                + parking_height \
                                + car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
                                - shift_for_start_dx, 
                                bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                + parking_height \
                                + 2 * bottom_left_boundary_height \
                                - car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
                                + shift_for_start_dx,
                                20)
        start_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - 1.2,
                                20
                                )
        start_thetas = np.linspace(-degToRad(20), degToRad(20), 20)
        start_vs = np.linspace(-0.5, 0, 20)

    safe_dx = 0.4 # d from left boundary to vehicle (image observation resolution)
    safe_dy = 0.4 # d from bottom boundary to vehicle (image observation resolution)
    shift_goal_dx = 0.7
    shift_goal_dy = 0.7
    if union and (forward_task is None) or not (forward_task is None) and \
        not forward_task:
        goal_xs = np.linspace(
            bottom_left_boundary_center_x \
            + bottom_left_boundary_height + safe_dx + car_config["width"] / 2,
            bottom_left_boundary_center_x \
            + bottom_left_boundary_height + parking_height - safe_dx \
            - car_config["width"] / 2,
            20
        )
        goal_ys = np.linspace(
            bottom_left_boundary_center_y - parking_width / 2 + safe_dy \
            + car_config["length"] / 2 - car_config["wheel_base"] / 2,
            bottom_left_boundary_center_y + parking_width / 2 \
            - car_config["length"] / 2 - car_config["wheel_base"] / 2 \
            + shift_goal_dy,
            20
        )
        goal_thetas = np.linspace(degToRad(85), degToRad(95), 20)

    elif forward_task:
        goal_xs = np.linspace(bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                + parking_height \
                                + car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
                                - shift_goal_dx, 
                                bottom_left_boundary_center_x \
                                + bottom_left_boundary_height \
                                + parking_height \
                                + 2 * bottom_left_boundary_height \
                                - car_config["length"] / 2 \
                                - car_config["wheel_base"] / 2
                                + shift_goal_dx,
                                20)
        goal_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - 1.2,
                                20
                                )
        goal_thetas = np.linspace(-degToRad(20), degToRad(20), 20)

    start_steers = np.linspace(-degToRad(28), degToRad(28), 20)

    start_x = np.random.choice(start_xs)
    start_y = np.random.choice(start_ys)    
    goal_x = np.random.choice(goal_xs)
    goal_y = np.random.choice(goal_ys)
    start_theta = np.random.choice(start_thetas)
    goal_theta = np.random.choice(goal_thetas)
    start_v = np.random.choice(start_vs)
    start_steer = np.random.choice(start_steers)

    start_x, start_y, start_theta, goal_x, goal_y, goal_theta \
                        = trySatisfyStartGoalConditions(start_x, start_y,
                                                        start_theta,
                                                        goal_x, goal_y,
                                                        goal_theta,
                                                        union,
                                                        forward_task,
                                                        staticObstsInfo,
                                                        car_config)

    return [start_x, start_y, start_theta, start_v, start_steer], \
           [goal_x, goal_y, goal_theta, 0, 0]

def trySatisfyStartGoalConditions(start_x, start_y,
                                  start_theta,
                                  goal_x, goal_y,
                                  goal_theta,
                                  union,
                                  forward_task,
                                  staticObstsInfo,
                                  car_config):
    parking_height = staticObstsInfo["parking_height"]
    parking_width = staticObstsInfo["parking_width"]
    road_width = staticObstsInfo["road_width"]
    buttom_road_edge_y = staticObstsInfo["buttom_road_edge_y"]
    bottom_left_boundary_center_x = staticObstsInfo["bottom_left_boundary_center_x"]
    bottom_left_boundary_center_y = staticObstsInfo["bottom_left_boundary_center_y"]
    bottom_left_boundary_height = staticObstsInfo["bottom_left_boundary_height"]
    car_width = car_config['width']
    shift_from_static_boundary_dy = 1.2
    safe_dx = 0.4 + 0.1 # 0.1 constrain for angle

    if not (forward_task is None) and forward_task:
        assert start_x < goal_x, \
            f"not correct start{start_x} and goal{goal_x} for task forward"
        if goal_x - start_x < 10:
            goal_x = start_x + 10
        if abs(start_y - goal_y) > 3 and (goal_x - start_x) < 15:
            start_x = start_x - 5
            if np.random.random() > 0.5:
                if start_y > goal_y:
                    goal_y = start_y - 3
                else:
                    goal_y = start_y + 3
            else:
                if start_y > goal_y:
                    start_y = goal_y + 3
                else:
                    start_y = goal_y - 3

    if not (forward_task is None) or union:
        if buttom_road_edge_y + 2 * road_width - start_y \
                < 2 * shift_from_static_boundary_dy + car_width / 2 and \
                start_theta > degToRad(5):
            start_y -= shift_from_static_boundary_dy
        if start_y  - buttom_road_edge_y \
                < 2 * shift_from_static_boundary_dy + car_width / 2 and \
                start_theta < -degToRad(5):
            start_y += shift_from_static_boundary_dy

    if not (forward_task is None) and forward_task:
        if buttom_road_edge_y + 2 * road_width - goal_y \
                < 2 * shift_from_static_boundary_dy + car_width / 2 and \
                goal_theta > degToRad(5):
            goal_y -= shift_from_static_boundary_dy
        if goal_y  - buttom_road_edge_y \
                < 2 * shift_from_static_boundary_dy + car_width / 2 and \
                    goal_theta < -degToRad(5):
            goal_y += shift_from_static_boundary_dy

    if union or not (forward_task is None) and not forward_task:
        if bottom_left_boundary_center_x + bottom_left_boundary_height \
                + parking_height - car_width / 2 - goal_x \
                <= safe_dx and goal_theta <= degToRad(88):
            goal_theta = degToRad(90)
        if goal_x - bottom_left_boundary_center_x \
                - bottom_left_boundary_height - car_width / 2 \
                <= safe_dx and goal_theta >= degToRad(92):
            goal_theta = degToRad(90)

    return start_x, start_y, start_theta, goal_x, goal_y, goal_theta
