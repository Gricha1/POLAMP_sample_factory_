from dataset_generation.utlis import *

def getApolloTestTasks(car_config):

    # test
    #parking_height = 2.7
    parking_width = 4.5
    parking_height = 2.7 + 0.5

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
    goal_x = bottom_left_boundary_center_x \
                  + bottom_left_boundary_height \
                  + parking_height / 2
    #goal_y = bottom_left_boundary_center_y - \
    #                    car_config["wheel_base"] / 2
    # test
    goal_y = bottom_left_boundary_center_y \
             + parking_width / 2 - car_config["length"] / 2 \
             - car_config["wheel_base"] / 2

    goal = [goal_x, goal_y, degToRad(90), 0, 0]
    
    # parameters variation:
    bottom_left_right_dxs = np.linspace(-1, -0.2, 3)
    road_widths = np.linspace(3, 6, 3)
    bottom_left_right_dx = -1.5
    road_width_ = 5
    
    road_center_y = bottom_road_edge_y + road_width_
    upper_boundary_center_y = road_center_y + road_width_ + \
                        upper_boundary_width
    
    staticObstsInfo = {}
    
    map_ = getValetStaticObstsAndUpdateInfo(
        parking_height, parking_width, bottom_left_boundary_height, 
        upper_boundary_width, upper_boundary_height,
        bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
        road_width_,
        staticObstsInfo
    )

    dynamic_config = {}
    dynamic_config["movement_func_params"] = {}
    wheel_base = car_config["wheel_base"]
    min_width = 1.8 - 0.5
    max_width = 1.8 + 0.4
    dynamic_widths = np.linspace(min_width, max_width, 20)
    min_length = 4.140 - 0.4
    max_length = 4.140 + 0.4
    dynamic_lengths = np.linspace(min_length, max_length, 20)
    safe_buffer = 0.6
    # test
    #dynamic_config["width"] = np.random.choice(dynamic_widths) + safe_buffer
    dynamic_config["width"] = car_config["width"] + 0.3
    #dynamic_config["length"] = np.random.choice(dynamic_lengths) + safe_buffer
    dynamic_config["length"] = car_config["length"] + 0.1
    dynamic_config["wheel_base"] = wheel_base

    boundary_v_max = 1
    boundary_v_min = 0.2
    dynamic_config["boundary_v"] = np.random.choice(
                        np.linspace(boundary_v_min, boundary_v_max, 10))

    # movement function for dynamic obst
    def move(last_state, current_steps, time_step=0.1, dynamic_config=None):
        action = [0, 0]

        return action
    dynamic_config["movement_func"] = move

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
    dyn_obst_1.append(dynamic_config)
    task_1 = (
                map_,
                [start_pose_1,
                goal],
                [dyn_obst_1]
             )
             
    start_pose_2 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]

    # test
    #dyn_obst_2 = [forward_start_x_ + 28, 
    #              bottom_road_edge_y + 0.5 * road_width_, 
    #              degToRad(180), 0.5, 0]
    
    dyn_obst_2 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + road_width_ + 0.1 * road_width_, 
                  degToRad(180), 0.5, 0]

    dyn_obst_2.append(dynamic_config)
    task_2 = (
                map_,
                [start_pose_2,
                goal],
                [dyn_obst_2]
             )
    
    start_pose_3 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_3 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    dyn_obst_3.append(dynamic_config)
    task_3 = (
                map_,
                [start_pose_3,
                goal],
                [dyn_obst_3]
             )

    start_pose_4 = [forward_start_x_, start_y_on_center, 0, 0, 0]

    # test
    #dyn_obst_4 = [forward_start_x_ + 28, 
    #              bottom_road_edge_y + 0.5 * road_width_, 
    #              degToRad(180), 0.5, 0]
    dyn_obst_4 = [forward_start_x_ + 28, 
                  bottom_road_edge_y + road_width_ + 0.3 * road_width_, 
                  degToRad(180), 0.5, 0]

    dyn_obst_4.append(dynamic_config)
    task_4 = (
                map_,
                [start_pose_4,
                goal],
                [dyn_obst_4]
             )
        
    start_pose_5 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    # test
    #dyn_obst_5 = [forward_start_x_ + 20, 
    #              bottom_road_edge_y + 0.5 * road_width_, 
    #              degToRad(180), 0.5, 0]
    dyn_obst_5 = [forward_start_x_ + 20, 
                  bottom_road_edge_y + 0.18 * road_width_, 
                  degToRad(180), 0.5, 0]
    dyn_obst_5.append(dynamic_config)
    task_5 = (
                map_,
                [start_pose_5,
                goal],
                [dyn_obst_5]
             )

    start_pose_6 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    dyn_obst_6 = [forward_start_x_ + 20, 
                  bottom_road_edge_y + road_width_ + 0.5 * road_width_, 
                  degToRad(180), 0.5, 0]
    dyn_obst_6.append(dynamic_config)
    task_6 = (
                map_,
                [start_pose_6,
                goal],
                [dyn_obst_6]
             )

    start_pose_7 = [forward_start_x_, start_y_on_center, 0, 0, 0]
    # test
    #dyn_obst_7 = [forward_start_x_ - 7, 
    #              start_y_on_center, 
    #              0, 0.5, 0]
    dyn_obst_7 = [forward_start_x_ - 9, 
                  start_y_on_center + 0.3 * road_width_, 
                  0, 0.7, 0]

    dyn_obst_7.append(dynamic_config)
    task_7 = (
                map_,
                [start_pose_7,
                goal],
                [dyn_obst_7]
             )

    start_pose_8 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_8 = [goal[0] + 5, 
                  goal[1], 
                  degToRad(130), 0.35, 0]
    dyn_obst_8.append(dynamic_config)
    task_8 = (
                map_,
                [start_pose_8,
                goal],
                [dyn_obst_8]
             )

    start_pose_9 = [forward_start_x_, start_y_on_bottom_lane, 0, 0, 0]
    dyn_obst_9 = [goal[0] - 3.2, 
                  goal[1], 
                  degToRad(90), 0.3, 0]
    dyn_obst_9.append(dynamic_config)
    task_9 = (
                map_,
                [start_pose_9,
                goal],
                [dyn_obst_9]
             )


    tasks = [task_1, task_2, task_3, task_4, 
             task_5, task_6, task_7, task_8, 
             task_9]

    return tasks