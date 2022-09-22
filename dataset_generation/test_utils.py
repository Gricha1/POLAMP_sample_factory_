from dataset_generation.utlis import *


def testingDataSetModule(car_config):

    tasks = []


    parking_heights = np.linspace(2.7 - 0.3, 2.7 + 5, 20)
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


    static_obsts = getValetStaticObstsAndUpdateInfo(
                parking_height, parking_width, bottom_left_boundary_height, 
                upper_boundary_width, upper_boundary_height,
                bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
                road_width,
                staticObstsInfo
            )


    start, goal = getValetStartGoalPose(staticObstsInfo, 
                                        car_config, 
                                        True,
                                        None)

    agent_task = [start, goal]

    dynamic_config = {}
    dynamic_config["case"] = 1
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

    x = bottom_left_boundary_center_x + bottom_left_boundary_height \
        + parking_height + 5
    y = buttom_road_edge_y + road_width

    theta = degToRad(180)
    v = 0.6
    steer = 0 

    dynamic = [x, y, theta, v, steer]

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
    dynamic_config["boundary_v"] = 0.6


    a_min_possible = 0
    a_max_possible = 1
    a_max = np.random.choice(np.linspace(a_min_possible, a_max_possible, 20))
    Eps_min_possible = 0
    Eps_max_possible = 0
    Eps_max = np.random.choice(np.linspace(Eps_min_possible, Eps_max_possible, 20))
    boundary_action = [a_max, Eps_max]
    dynamic_config["movement_func_params"] = {"boundary_action": boundary_action}

    stop_step = 50
    dynamic_config["movement_func_params"]["stop_step"] = int(stop_step)

    dynamic_do_reverse_after_stop = np.random.choice([True, False, False])
    dynamic_config["movement_func_params"]["dynamic_do_reverse_after_stop"] \
                                                = False
    if dynamic_do_reverse_after_stop:
        reverse_steps = list(range(10, 40))
        reverse_step = 50
        dynamic_config["movement_func_params"]["reverse_step"] \
                                                = int(reverse_step)

    def move(last_state, current_steps, time_step=0.1, dynamic_config=None):
        x = last_state[0]
        y = last_state[1]
        theta = last_state[2]
        v = last_state[3]
        steer = last_state[4]
        boundary_action = dynamic_config["boundary_action"]
        stop_step = dynamic_config["stop_step"]
        dynamic_do_reverse_after_stop = dynamic_config["dynamic_do_reverse_after_stop"]
        if dynamic_do_reverse_after_stop:
            reverse_step = dynamic_config["reverse_step"]

        a = np.random.choice(
                np.linspace(0, boundary_action[0], 10))
        Eps = np.random.choice(
                np.linspace(0, boundary_action[1], 10))
        stop_step_condition = False
        if current_steps >= stop_step:
            stop_step_condition = True
            if v > 0:
                a = -min(boundary_action[0], v / time_step)
            elif v == 0:
                a = 0
        if dynamic_do_reverse_after_stop and current_steps >= stop_step and \
                v <= 0 and current_steps < stop_step + reverse_step:
            a = -np.random.choice(
                np.linspace(0, boundary_action[0], 10))
        if dynamic_do_reverse_after_stop and current_steps >= stop_step + reverse_step:
            if v < 0:
                a = min(boundary_action[0], abs(v) / time_step)
            elif v == 0:
                a = 0
        if "collision_condition" in dynamic_config:
            #print("########")
            #print("stop step condition:", stop_step_condition, 
            #      "current steps", current_steps)
            #print("dynamic_do_reverse_after_stop:", dynamic_do_reverse_after_stop)
            #print("agent v:", v)
            #print("x - parking_place_x:", abs(x - dynamic_config["parking_place_x"]))
            #print("safe_rad:", dynamic_config["safe_radius"])

            if (v == 0 and not dynamic_do_reverse_after_stop) or \
                (v == 0 and dynamic_do_reverse_after_stop and \
                    current_steps >= stop_step + reverse_step):
                if abs(x - dynamic_config["parking_place_x"]) <= \
                        dynamic_config["safe_radius"]:
                    dynamic_config["try_to_fix_collision"] = True
            if "try_to_fix_collision" in dynamic_config:
                if dynamic_config["forward_move_prevent_collision"]:
                    a = np.random.choice(
                        np.linspace(0, boundary_action[0], 10))
                else:
                    a = -np.random.choice(
                        np.linspace(0, boundary_action[0], 10))
            #print("try to fix collision:", "try_to_fix_collision" in dynamic_config)
            #print("########")
        action = [a, Eps]

        return action
    dynamic_config["movement_func"] = move


    dynamic = trySatisfyDynamicConditions(staticObstsInfo,
                                          car_config,
                                          dynamic,
                                          dynamic_config)

    dynamic.append(dynamic_config)

    dynamic_obsts = [dynamic]

    static_obsts, agent_task, dynamic_obsts = trySatisfyCollisionConditions(
                                                        staticObstsInfo,
                                                        car_config,
                                                        static_obsts, 
                                                        agent_task, 
                                                        dynamic_obsts
                                                        )


    tasks.append((static_obsts, agent_task, dynamic_obsts))


    return tasks





