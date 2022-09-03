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
    dynamic_cases = createDynamicCases() 
    tasks = [] 
    for dyn_case in dynamic_cases:
        train_case_tasks = getCaseTasks(dyn_case, 
                                        tasks_config, 
                                        car_config, 
                                        train=train)
        tasks.extend(train_case_tasks)

    return tasks

def createDynamicCases():

    return [0]

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

def getCaseTasks(dyn_case, tasks_config, car_config, train=True):
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
    case_tasks = []
    static = tasks_config['static']
    dynamic = tasks_config['dynamic']
    union = tasks_config['union']
    # static obsts params:
    if train:
        parking_heights = np.linspace(2.7 + 0.2, 2.7 + 5, 3)
        parking_widths = np.linspace(4.5, 7, 3)
        road_widths = np.linspace(3, 6, 3)
    else:
        parking_heights = np.linspace(2.7 + 0.2, 2.7 + 5, 2)
        parking_widths = np.linspace(4.5, 7, 2)
        road_widths = np.linspace(3, 6, 2)
    bottom_left_boundary_height = 6 # any value
    bottom_left_boundary_center_x = 5 # any value
    bottom_left_boundary_center_y = -5.5 # init value
    upper_boundary_width = 0.5 # any value
    upper_boundary_height = 17 

    task_number = 1
    for parking_height in parking_heights:
        for parking_width in parking_widths:
            for road_width in road_widths:               
                staticObstsInfo = {}
                static_obsts = generateValetStaticObstsAndUpdateInfo(
                    parking_height, parking_width, bottom_left_boundary_height, 
                    upper_boundary_width, upper_boundary_height,
                    bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
                    road_width,
                    staticObstsInfo
                )
                assert (static and len(static_obsts) == 4) or \
                       (not static and len(static_obsts) == 0), \
                        f"incorrect static task, static is {static} " + \
                        f"but len static is {len(static_obsts)}"
                
                if not union:
                    forward_task = None
                elif task_number % 2 == 0:
                    forward_task = False
                else:
                    forward_task = True
                start, goal = getStartGoalPose(staticObstsInfo, 
                                               car_config, 
                                               union,
                                               forward_task)
                #test_start_x = staticObstsInfo["bottom_left_boundary_center_x"]
                #test_start_y = staticObstsInfo["buttom_road_edge_y"] + 2
                #test_goal_x = test_start_x + 15
                #test_goal_y = test_start_y
                #start, goal = [test_start_x, test_start_y, 0, 0, 0], \
                #              [test_goal_x, test_goal_y, 0, 0, 0]
                agent_task = [start, goal]
                assert len(start) == 5 and len(goal) == 5, \
                    f"start and goal len must be 5 but len start is: {len(start)} " + \
                    f"and len goal is: {len(goal)}"
                dynamic_obsts = []
                case_tasks.append((static_obsts, agent_task, dynamic_obsts))
                task_number += 1

    return case_tasks

def isAppropiateTask():
    pass

def changeToApprotiateTask():
    pass

def generateValetStaticObstsAndUpdateInfo(parking_height, parking_width, 
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

def generateTestDataSet(our_env_config, car_config):
    dataSet = {}
    maps = {} 
    trainTask = {}
    valTasks = {}
    #boundaries
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

    #bottom_left_right_dx_ = -0.2
    #road_width_ = 3
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
    task_1 = [(
                start_pose_1,
                second_goal,
                [dyn_obst_1]
             )]
             
    start_pose_2 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_2 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_2 = [(
                start_pose_2,
                second_goal,
                [dyn_obst_2]
             )]

    start_pose_3 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_3 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_3 = [(
                start_pose_3,
                second_goal,
                [dyn_obst_3]
             )]

    start_pose_4 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_4 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_4 = [(
                start_pose_4,
                second_goal,
                [dyn_obst_4]
             )]
        
    start_pose_5 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_5 = [forward_start_x_ + 20, 
                  bottom_road_edge_y + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_5 = [(
                start_pose_5,
                second_goal,
                [dyn_obst_5]
             )]

    start_pose_6 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_6 = [forward_start_x_ + 20, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    task_6 = [(
                start_pose_6,
                second_goal,
                [dyn_obst_6]
             )]

    start_pose_7 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_7 = [forward_start_x_ - 7, 
                  start_y_on_center, 
                  0, 0.5, 0]
    task_7 = [(
                start_pose_7,
                second_goal,
                [dyn_obst_7]
             )]

    start_pose_8 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_8 = [second_goal[0] + 5, 
                  second_goal[1], 
                  degToRad(130), 0.35, 0]
    task_8 = [(
                start_pose_8,
                second_goal,
                [dyn_obst_8]
             )]

    start_pose_9 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_9 = [second_goal[0] - 3.2, 
                  second_goal[1], 
                  degToRad(90), 0.3, 0]
    task_9 = [(
                start_pose_9,
                second_goal,
                [dyn_obst_9]
             )]


    tasks = [task_1, task_2, task_3, task_4, 
             task_5, task_6, task_7, task_8, task_9]
    for index, task_ in enumerate(tasks):
        maps["map" + str(index)] = map_
        trainTask["map" + str(index)] = task_
        valTasks["map" + str(index)] = task_

    dataSet["empty"] = (maps, trainTask, valTasks)

    return dataSet, second_goal

def generateDataSet(our_env_config, car_config):
    dataSet = {}
    maps = {} 
    trainTask = {}
    valTasks = {}
    if our_env_config["dynamic"]: 
        dynamic = True
    else: 
        dynamic = False
    if our_env_config["union"]: 
        union = True
    else: 
        union = False
    if our_env_config["union_without_forward_task"]:
        union_without_forward_task = True
    else:
        union_without_forward_task = False

    # static obsts params:
    parking_heights = np.linspace(2.7 + 0.2, 2.7 + 5, 3)
    parking_widths = np.linspace(4.5, 7, 3)
    road_widths = np.linspace(3, 6, 3)
    bottom_left_boundary_height = 6 # any value
    bottom_left_boundary_center_x = 5 # any value
    bottom_left_boundary_center_y = -5.5 # init value
    upper_boundary_width = 0.5 # any value
    upper_boundary_height = 17 

    #set tasks
    index = 0
    for parking_height in parking_heights:
        for parking_width in parking_widths:
            for road_width in road_widths:
                if our_env_config['static']:
                    maps["map" + str(index)] = generateValetStaticObstsAndUpdateInfo(
                        parking_height, parking_width, bottom_left_boundary_height, 
                        upper_boundary_width, upper_boundary_height,
                        bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
                        road_width)
                else:
                    maps["map" + str(index)] = []

                second_goal_x = bottom_left_boundary_center_x \
                    + bottom_left_boundary_height \
                    + parking_height / 2
                second_goal_y = bottom_left_boundary_center_y - \
                                    car_config["wheel_base"] / 2
                second_goal = [second_goal_x, second_goal_y, degToRad(90), 0, 0]
                bottom_road_edge_y = bottom_left_boundary_center_y + parking_width / 2
                trainTask["map" + str(index)] = generateTasks(
                        car_config, 
                        bottom_left_boundary_center_x,
                        bottom_left_boundary_center_y,
                        bottom_left_boundary_height,
                        parking_height,
                        bottom_road_edge_y, 
                        road_width, second_goal,
                        dynamic=dynamic, union=union, 
                        union_without_forward_task = union_without_forward_task,
                        validate_on_train=True)
                        
                valTasks["map" + str(index)] = generateTasks(car_config, 
                        bottom_left_boundary_center_x,
                        bottom_left_boundary_center_y,
                        bottom_left_boundary_height,
                        parking_height,
                        bottom_road_edge_y, 
                        road_width, second_goal,
                        dynamic=dynamic, union=union,
                        union_without_forward_task = union_without_forward_task,
                        validate_on_train=our_env_config["validate_on_train"])

            index += 1

    dataSet["empty"] = (maps, trainTask, valTasks)

    return dataSet, second_goal

def getStartGoalPose(staticObstsInfo, car_config, union, forward_task):
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
                                10)
        start_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + shift_from_static_boundary,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - shift_from_static_boundary,
                                5
                                )
        start_thetas = np.linspace(-degToRad(20), degToRad(20), 10)

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
                                5)
        start_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - 1.2,
                                5
                                )
        start_thetas = np.linspace(-degToRad(20), degToRad(20), 10)

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
                                10)
        start_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - 1.2,
                                5
                                )
        start_thetas = np.linspace(-degToRad(20), degToRad(20), 10)

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
            5
        )
        goal_ys = np.linspace(
            bottom_left_boundary_center_y - parking_width / 2 + safe_dy \
            + car_config["length"] / 2 - car_config["wheel_base"] / 2,
            bottom_left_boundary_center_y + parking_width / 2 \
            - car_config["length"] / 2 - car_config["wheel_base"] / 2 \
            + shift_goal_dy,
            5
        )
        goal_thetas = np.linspace(degToRad(85), degToRad(95), 5)

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
                                10)
        goal_ys = np.linspace(buttom_road_edge_y + car_config["width"] / 2 
                                + 1.2,
                                buttom_road_edge_y + 2 * road_width \
                                - car_config["width"] / 2 - 1.2,
                                5
                                )
        goal_thetas = np.linspace(-degToRad(20), degToRad(20), 10)

    start_x = np.random.choice(start_xs)
    start_y = np.random.choice(start_ys)    
    goal_x = np.random.choice(goal_xs)
    goal_y = np.random.choice(goal_ys)
    start_theta = np.random.choice(start_thetas)
    start_theta = np.random.choice(start_thetas)
    goal_theta = np.random.choice(goal_thetas)

    start_x, start_y, start_theta, goal_x, goal_y, goal_theta \
                        = trySatisfyStartGoalConditions(start_x, start_y,
                                                        start_theta,
                                                        goal_x, goal_y,
                                                        goal_theta,
                                                        union,
                                                        forward_task,
                                                        staticObstsInfo,
                                                        car_config)

    return [start_x, start_y, start_theta, 0., 0], [goal_x, goal_y, goal_theta, 0, 0]


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

def getDynamicObsts(start_x, start_y, goal_x, goal_y):
    theta_angle = 0

    return [start_x, start_y, theta_angle, 0., 0], [goal_x, goal_y, 0, 0, 0]

def getDynamicTask(dataset_info, temp_info):
    forward_start_x = dataset_info["forward_start_x"]
    forward_end_x = dataset_info["forward_end_x"]
    forward_start_y = dataset_info["forward_start_y"]
    forward_end_y = dataset_info["forward_end_y"]
    second_goal = dataset_info["second_goal"]
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
                dynamic_speed = [0.2, 0.5, 0.6]
            elif forward_start_x_ == forward_start_x[0]\
                and forward_end_x_ == forward_end_x[1]:
                dyn_obst_x = [forward_end_x[1] + 2, forward_end_x[1] + 4]
                dynamic_speed = [0.2, 0.4, 0.5, 0.6]
            elif forward_start_x_ == forward_start_x[0]\
                and forward_end_x_ == forward_end_x[0]:
                dyn_obst_x = [forward_end_x[0] + 3, forward_end_x[0] + 5]
                dynamic_speed = [0.2, 0.5, 0.6]
            elif forward_start_x_ == forward_start_x[1]:
                dyn_obst_x = [forward_end_x[0] + 8, 
                            forward_end_x[1] + 6,
                            forward_end_x[1] + 8]
                dynamic_speed = [0.2, 0.5, 0.6]
            else:
                dyn_obst_x = np.linspace(forward_start_x_ + 10, 
                            forward_start_x_ + 18, 3)
                dynamic_speed = [0.2, 0.5, 0.6]
            
        else:
            dyn_obst_x = np.linspace(forward_start_x_ - 18, 
                                     forward_start_x_ - 10, 3)
            dynamic_speed = [0.6, 0.5, 0.2]
            
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
                dyn_obst_x_second = np.linspace(forward_end_x_ + 10, 
                                                forward_end_x_ + 18, 3)
                dynamic_speed_second = [0.6, 0.5, 0.2]
                dyn_obst_x_second_ = np.random.choice(dyn_obst_x_second)
                dyn_obst_y_second_ = np.random.choice(dyn_obst_y)
                dynamic_speed_second_ = np.random.choice(dynamic_speed_second)
            dyn_obs_second = [dyn_obst_x_second_, 
                    dyn_obst_y_second_, 
                    dyn_theta, dynamic_speed_second_, 0]
                    
            dynamic_tasks.append(([forward_end_x_, forward_end_y_, 
                                theta_angle, 0., 0], 
                                [0, 0, 0, 0, 0], 
                                [dyn_obs_second]))
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
            dyn_obst_x = np.linspace(backward_start_x_ + 18, 
                                        backward_start_x_ + 14, 3)
            dynamic_speed = [0.6, 0.5, 0.2]
        else: 
            dyn_obst_x = np.linspace(backward_start_x_ - 18, 
                                        backward_start_x_ - 14, 3)
            dynamic_speed = [0.6, 0.5, 0.2]
        
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
                             second_goal, 
                             [dyn_obs]))

    return dynamic_tasks
        

def generateTasks(config, 
                bottom_left_boundary_center_x,
                bottom_left_boundary_center_y, 
                bottom_left_boundary_height, 
                parking_height, 
                buttom_road_edge_y,
                road_width, second_goal,
                dynamic, union, union_without_forward_task,
                validate_on_train=False):
                                 
    Tasks = []

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
        backward_start_x = np.linspace(forward_end_x[0], 
                                       forward_end_x[1], 5)
        backward_start_y = np.linspace(forward_end_y[0], 
                                       forward_end_y[1], 5)

    dataset_info = {
        "forward_start_x": forward_start_x,
        "forward_start_y": forward_start_y,
        "forward_end_x": forward_end_x,
        "forward_end_y": forward_end_y,
        "backward_start_x": backward_start_x,
        "backward_start_y": backward_start_y,
        "second_goal": second_goal,
        "buttom_road_edge_y": buttom_road_edge_y,
        "samples_theta_eps_ego": samples_theta_eps_ego,
        "road_width": road_width,
        "union_without_forward_task": union_without_forward_task
    }

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

                    if not union:
                        if dynamic:
                            Tasks.extend(getDynamicTask(dataset_info, 
                                                                temp_info))                        
                        else: 
                            theta_angle = np.random.choice(samples_theta_eps_ego)          
                            Tasks.append(([forward_start_x_, forward_start_y_, 
                                              0, 0., 0], 
                                             [forward_end_x_, forward_end_y_, 
                                              theta_angle, 0, 0]))

                    else: #union tasks
                        forward_dyn_movement = np.random.choice([True, False])
                        theta_angle = np.random.choice(samples_theta_eps_ego) 
                        if dynamic:
                            Tasks.extend(getDynamicTask(dataset_info, 
                                            temp_info))

                        else:        
                            theta_angle = np.random.choice(samples_theta_eps_ego)   
                            if union_without_forward_task:
                                Tasks.append(([forward_start_x_, forward_start_y_, 
                                                  0, 0, 0], 
                                                 [0, 0, 0, 0, 0]))
                                Tasks.append(([forward_end_x_, forward_end_y_, 
                                                  theta_angle, 0, 0], 
                                                 [0, 0, 0, 0, 0]))                  
                            else:
                                Tasks.append(([forward_start_x_, forward_start_y_, 
                                                    0, 0, 0], 
                                                 [forward_end_x_, forward_end_y_, 
                                                    theta_angle, 0, 0]))

    if not union: # backward tasks
        #print("debug utils:", "validate backward:", not validate_on_train, 
        # len(backward_start_x) * len(backward_start_y))
        for backward_start_x_ in backward_start_x:
            for backward_start_y_ in backward_start_y:
                temp_info = {}
                temp_info["forward_task"] = False
                temp_info["backward_start_x_"] = backward_start_x_
                temp_info["backward_start_y_"] = backward_start_y_

                if dynamic:                
                    Tasks.extend(getDynamicTask(dataset_info, 
                                            temp_info))
                else: 
                    theta_angle = np.random.choice(samples_theta_eps_ego)           
                    Tasks.append(([backward_start_x_, backward_start_y_, 
                                        theta_angle, 0., 0], 
                                        second_goal))

    return Tasks
