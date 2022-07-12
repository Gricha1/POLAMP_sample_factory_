import json
import wandb
import torch
import pickle
#import ray.rllib.agents.ppo as ppo
#import ray.rllib.agents.ddpg as ddpg
#from EnvLib.ObstGeomEnv import *
from EnvLib.ObstGeomEnvSampleFactory import *
from planning.generateMap import *
from policy_gradient.utlis import *
#import sys
#sys.path.insert(0, "../")

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
with open("configs/car_configs.json", 'r') as f:
    car_configs = json.load(f)
#with open("../configs/environment_configs.json", 'r') as f:
#    our_env_config = json.load(f)

#with open("../configs/car_configs.json", 'r') as f:
#    car_configs = json.load(f)

ANGLE_EPS = float(our_env_config['ANGLE_EPS'])
SPEED_EPS = float(our_env_config['SPEED_EPS'])
STEERING_EPS = float(our_env_config['STEERING_EPS'])


def trasitionCurriculum(trainer, val_env, environment_config, config, current_counter):
    model_weights = trainer.get_policy().get_weights()
    environment_config['our_env_config']['ANGLE_EPS'] = ANGLE_EPS / (current_counter + 1)
    environment_config['our_env_config']['SPEED_EPS'] = SPEED_EPS / (current_counter + 1)
    environment_config['our_env_config']['STEERING_EPS'] = STEERING_EPS / (current_counter + 1)
    config['env_config'] = environment_config
    trainer = ppo.PPOTrainer(config=config, env=ObsEnvironment)
    trainer.get_policy().set_weights(model_weights)
    val_env = ObsEnvironment(environment_config)
    print("current_counter: ", current_counter)
    print(trainer.config['env_config']['our_env_config'])
    
    return trainer, val_env


def generateDataSet(our_env_config, car_config):
    #CUSTOM DATASET
    dataSet = {}
    maps = {} 
    trainTask = {}
    valTasks = {}
    if our_env_config["easy_task_difficulty"]:
        task_difficulty = "easy"
    elif our_env_config["medium_task_difficulty"]:
        task_difficulty = "medium"
    else:
        task_difficulty = "hard"
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

    '''
    ------> height

         width
           ^
           |
           |
           |


                        upper_boundary
    ------------------------------------------------------------

                             road
                             
    -------------------------     -----------------------------
             bottom_left    |     |        bottom_right
                            |_____|
                           bottom_down
    '''
    if our_env_config['validate_custom_case']:
        bottom_left_right_dx_ = 0.1
        road_width_ = 5
        # parameters variation:
        #bottom_left_right_dx = np.linspace(0.1, 0.25, 3)
        #road_width = np.linspace(2.2, 3, 3)
        #bottom_left_right_dx = np.linspace(-0.8, 0, 3)
        #road_width = np.linspace(3, 5, 3)
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
        bottom_road_edge_y = bottom_left_boundary_center_y + bottom_left_boundary_width
        upper_boundary_width = 2 # any value
        upper_boundary_height = 17 # any value
        bottom_down_width = upper_boundary_width 
        bottom_down_height = upper_boundary_height 
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
        second_goal_y = bottom_left_boundary_center_y - car_config["wheel_base"] / 2
        second_goal = [second_goal_x, second_goal_y, degToRad(90), 0, 0]



        index = 0
        road_center_y = bottom_road_edge_y + road_width_
        upper_boundary_center_y = road_center_y + road_width_ + upper_boundary_width
        if our_env_config['static']:
            maps["map" + str(index)] = [
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
        else:
            maps["map" + str(index)] = []

        trainTask["map" + str(index)] = generateTasks(car_config, 
                                        bottom_left_boundary_center_x,
                                        bottom_left_boundary_center_y,
                                        bottom_left_boundary_height,
                                        parking_height,
                                        bottom_road_edge_y, 
                                        road_width_, second_goal, task_difficulty,
                                        dynamic=dynamic, union=union,
                                        union_without_forward_task = union_without_forward_task,
                                        validate_on_train=True)
        valTasks["map" + str(index)] = generateTasks(car_config, 
                                        bottom_left_boundary_center_x,
                                        bottom_left_boundary_center_y,
                                        bottom_left_boundary_height,
                                        parking_height,
                                        bottom_road_edge_y, 
                                        road_width_, second_goal, task_difficulty,
                                        dynamic=dynamic, union=union,
                                        union_without_forward_task = union_without_forward_task,
                                        validate_on_train=our_env_config["validate_on_train"])

        dataSet["empty"] = (maps, trainTask, valTasks)

        return dataSet, second_goal



    # parameters variation:
    if our_env_config["easy_map_constraints"]:
        bottom_left_right_dx = np.linspace(-0.8, 0, 3)
        road_width = np.linspace(3, 5, 3)
    elif our_env_config["medium_map_constraints"]:
        bottom_left_right_dx = np.linspace(-1, 0.25, 3)
        road_width = np.linspace(2.2, 3, 3)
    elif our_env_config["hard_map_constraints"]:
        #bottom_left_right_dx = np.linspace(0.1, 0.25, 3)
        #road_width = np.linspace(2.6, 6, 3)
        bottom_left_right_dx = np.linspace(-1, -0.2, 3)
        road_width = np.linspace(3, 6, 3)
        
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
    bottom_road_edge_y = bottom_left_boundary_center_y + bottom_left_boundary_width
    #upper_boundary_width = 2 # any value
    upper_boundary_width = 0.5 # any value
    upper_boundary_height = 17 
    bottom_down_width = upper_boundary_width 
    #bottom_down_height = upper_boundary_height 
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
    second_goal_y = bottom_left_boundary_center_y - car_config["wheel_base"] / 2
    second_goal = [second_goal_x, second_goal_y, degToRad(90), 0, 0]
    #set tasks
    index = 0
    for bottom_left_right_dx_ in bottom_left_right_dx:
        for road_width_ in road_width:

            road_center_y = bottom_road_edge_y + road_width_
            upper_boundary_center_y = road_center_y + road_width_ + upper_boundary_width
            
            if our_env_config['static']:
                maps["map" + str(index)] = [
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
            else:
                maps["map" + str(index)] = []

            trainTask["map" + str(index)] = generateTasks(car_config, 
                                            bottom_left_boundary_center_x,
                                            bottom_left_boundary_center_y,
                                            bottom_left_boundary_height,
                                            parking_height,
                                            bottom_road_edge_y, 
                                            road_width_, second_goal, task_difficulty,
                                            dynamic=dynamic, union=union, 
                                            union_without_forward_task = union_without_forward_task,
                                            validate_on_train=True)
            valTasks["map" + str(index)] = generateTasks(car_config, 
                                            bottom_left_boundary_center_x,
                                            bottom_left_boundary_center_y,
                                            bottom_left_boundary_height,
                                            parking_height,
                                            bottom_road_edge_y, 
                                            road_width_, second_goal, task_difficulty,
                                            dynamic=dynamic, union=union,
                                            union_without_forward_task = union_without_forward_task,
                                            validate_on_train=our_env_config["validate_on_train"])

            index += 1

    dataSet["empty"] = (maps, trainTask, valTasks)

    return dataSet, second_goal


def trainCurriculum(
                    config, 
                    train_config, 
                    our_env_config, 
                    reward_config, 
                    car_config, 
                    ppo_algorithm=True
                    ):
    
    dataSet, second_goal = generateDataSet(our_env_config, car_config)
    maps, trainTask, valTasks = dataSet["empty"] 
    vehicle_config = VehicleConfig(car_config)

    if our_env_config["union"]:
        environment_config = {
            'vehicle_config': vehicle_config,
            'tasks': trainTask,
            'valTasks': valTasks,
            'maps': maps,
            'our_env_config' : our_env_config,
            'reward_config' : reward_config,
            "second_goal" : second_goal
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

    config['env_config'] = environment_config


    if ppo_algorithm:
        trainer = ppo.PPOTrainer(config=config, env=ObsEnvironment)
    else:
        trainer = ddpg.DDPGTrainer(config=config, env=ObsEnvironment)
    val_env = ObsEnvironment(environment_config)

    #DEBUG
    trainer.config["clip_actions"] = True

    print(trainer.config["model"])
    print(trainer.config['env_config']['our_env_config'])

    folder_path = "./myModelWeight1"

    #load model
    no_weights = train_config["no_initial_weights"]

    loaded_conf_num = train_config["loaded_conf_num"]
    loaded_ex_num = train_config["loaded_ex_num"]
    loaded_run_num = train_config["loaded_run_num"]
    load_check_num = train_config["loaded_check_num"]
    count_of_digits = len(str(load_check_num))
    pre_checkpoint_digit = "0" * (6 - count_of_digits)

    load_folder = f"./myModelWeight1/configuration_{loaded_conf_num}/{loaded_ex_num}_step_{loaded_run_num}"
    load_exp = f"checkpoint_{pre_checkpoint_digit}{load_check_num}/checkpoint-{load_check_num}"

    
    #trainer.restore("./myModelWeight1/new_dynamic_steps_10/2_step_19/checkpoint_004830/checkpoint-4830")
    #trainer.restore("./myModelWeight1/new_dynamic_steps_10/2_step_20/checkpoint_004910/checkpoint-4910")
    #trainer.restore("./myModelWeight1/new_dynamic_steps_10/2_step_15/checkpoint_004030/checkpoint-4030")
    #trainer.restore(load_folder + f"/checkpoint_{pre_checkpoint_digit}{load_check_num}/checkpoint-{load_check_num}")

    if not no_weights:
        trainer.restore(load_folder + "/" + load_exp)

    #with open('myModelWeight1/new_weights/conf_1_ex_2_run_67/weights_1_2_67.pickle', 'rb') as handle:
    #    weights = pickle.load(handle)
    #    trainer.get_policy().set_weights(weights)
    
    

    if train_config["curriculum"]:
        print("curriculum")
        curr_folder_path = os.path.join(folder_path, train_config['curriculum_name'])
        #trainer.get_policy().set_weights(torch.load(f'{curr_folder_path}/policy.pkl'))
        print("get folder_path: ", curr_folder_path)

    if train_config["curriculum"]:
        folder_path = os.path.join(folder_path, train_config['name_save'] + "Curriculum")
    else:
        folder_path = os.path.join(folder_path, train_config['name_save'])
    print("save folder_path: ", folder_path)

    save_configs(car_config, folder_path, "car_configs.json")
    save_configs(reward_config, folder_path, "reward_weight_configs.json")
    save_configs(train_config, folder_path, "train_configs.json")
    save_configs(our_env_config, folder_path, "environment_configs.json")

    t_max = train_config["max_episodes"]
    t_val = train_config["val_time_steps"]
    acceptable_success = train_config["acceptable_success"]
    flag_wandb = train_config["wandb"]
    if flag_wandb:
        wandb_config = dict(car_config, **train_config, **our_env_config, **reward_config)
        wandb.init(config=wandb_config, project=train_config['project_name'], entity='grisha1')
    t = 0
    old_val_rate = -1
    current_counter = 0
    old_step = 0

    show_example = train_config["only_show"]
    type_show_example =  train_config["type_show"]
    validate_examples = train_config["only_validate"]

    conf_num = train_config["conf_num"]
    ex_num = train_config["ex_num"]
    run_num = train_config["run_num"]	



    weight = trainer.get_policy().get_weights()
    with open(f'./myModelWeight1/weights_{loaded_conf_num}_{loaded_ex_num}_{loaded_run_num}.pickle', 'wb') as handle:
        pickle.dump(weight, handle, protocol=pickle.HIGHEST_PROTOCOL)




    #time parametets
    if validate_examples:
        assert show_example == False, "custom assertion: only show == True, \
                                        and only validate == True"
        t_validate = 1
    if show_example: 
        t_show = 1
        assert validate_examples == False, "custom assertion: only show = True \
                                        and only validate = True"
    if not validate_examples and not show_example:        
        t_show = 500 #100
        t_save = 250 #10
        t_first_save = 250
        t_validate = 250 #5
        print("t_save:", t_save, "t_first_save", t_first_save)

    #save_folder_name = "new_dynamic_steps_10/2_step_21"

    #conf_num = train_config["conf_num"]
    #ex_num = train_config["ex_num"]
    #run_num = train_config["run_num"]
    save_folder_name = f"configuration_{conf_num}/{ex_num}_step_{run_num}"

    #logging
    print("load folder:", load_folder + "/" + load_exp, "no init weights", no_weights)
    print("save folder:", save_folder_name)
    print("settings:")
    print("jerk:", car_config["jerk"])
    print("task type--", "union:", our_env_config["union"], 
          "dynamic:", our_env_config["dynamic"],
          "static:", our_env_config["static"],
           )
    print("normalize actions:", train_config["normalize_actions"])
    print("reward penalty--", "linear_acc:", reward_config["a_penalty"],"angle_acc:", reward_config["Eps_penalty"])
    print("terminal constraints--", "hard_constraints:", our_env_config["hard_constraints"], 
          "medium_constraints:", our_env_config["medium_constraints"],
          "soft_constraints:", our_env_config["soft_constraints"],
           )
    print("terminal constraints--", "HARD_EPS:", our_env_config["HARD_EPS"], 
          "MEDIUM_EPS:", our_env_config["MEDIUM_EPS"],
          "SOFT_EPS:", our_env_config["SOFT_EPS"],
          "ANGLE_EPS:", our_env_config["ANGLE_EPS"],
          "SPEED_EPS:", our_env_config["SPEED_EPS"]
           )
    print("map constraints--", "easy_map_constraints:", our_env_config["easy_map_constraints"],
        "medium_map_constraints:", our_env_config["medium_map_constraints"],
        "hard_map_constraints:", our_env_config["hard_map_constraints"])
    print("task difficult--", "easy_task_difficulty:", our_env_config["easy_task_difficulty"],
         "medium_task_difficulty:", our_env_config["medium_task_difficulty"],
         "hard_task_difficulty:", our_env_config["hard_task_difficulty"])
    print("only_show:", show_example, "type_show", type_show_example, 
          "only_validate:", validate_examples)

    last_saved_val = 0
    while t <= (t_max * 250):
        print(f"t {t}", end=" ")
        t += 1

        if show_example:
            if t % t_show == 0:
                for key in val_env.valTasks:
                    isDone, val_traj, _, _ = validate_task(val_env, 
                                                trainer, max_steps=train_config["horizon"], save_image=True, val_key=key)
                    if type_show_example and flag_wandb and isDone \
                       or not type_show_example and flag_wandb and not isDone:
                        print("get video", end=" ")
                        wandb.log({f"Val_trajectory_{t}": wandb.Video(val_traj,
                                                             fps=10, format="gif")})
            print()

        else:
            #train
            if not validate_examples:
                res = trainer.train()
                log = res['episode_reward_mean']
                info = res['info']['learner']['default_policy']['learner_stats']

                #get video
                if t % t_show == 0:
                    for key in val_env.valTasks:
                        _, val_traj, _, _ = validate_task(val_env, 
                                                trainer, max_steps=train_config["horizon"], 
                                                save_image=True, val_key=key)
                        if flag_wandb:
                            print("get video", end=" ")
                            wandb.log({f"Val_trajectory_{t}": wandb.Video(val_traj, 
                                                            fps=10, format="gif")})

                if flag_wandb:
                    if ppo_algorithm:
                        wandb.log(
                                {
                                'episode_reward_mean': log,
                                "total_loss": info['total_loss'],
                                "vf_loss": info['vf_loss'],
                                "kl" : info['kl'],
                                "policy_loss": info['policy_loss'],
                                "entropy": info['entropy']
                                },
                                step = (res['info']['num_agent_steps_trained'] + old_step)
                                )
                    else:
                        wandb.log(
                                {
                                'episode_reward_mean': log,
                                "actor_loss": info['actor_loss'],
                                "critic_loss": info['critic_loss'],
                                "mean_q" : info['mean_q']
                                },
                                step = (res['info']['num_agent_steps_trained'] + old_step)
                                )

                    #get validate result
                    if (t % t_validate == 0):
                        val_done_rate, min_total_val_dist, val_counter_collision, _ \
                                    = validation(val_env, trainer, max_steps=train_config["horizon"])
                        print(f"val_done_rate {val_done_rate}", end=" ")
                        if flag_wandb:
                            wandb.log(
                                {
                                'val_done_rate': val_done_rate,
                                "min_val_dist": min_total_val_dist,
                                "val_counter_collision": val_counter_collision
                                }, 
                                step = (res['info']['num_agent_steps_trained'] + old_step))
                    
                    #save the model
                    name_of_adding_folder = "./myModelWeight1/" + save_folder_name
                    if t % t_save == 0 and t >= t_first_save:
                        print("save model", 
                            val_done_rate >= last_saved_val, 
                            end = " ")
                        if val_done_rate >= last_saved_val:
                            if not os.path.exists(name_of_adding_folder):
                                os.makedirs(name_of_adding_folder)
                            trainer.save(name_of_adding_folder)
                            last_saved_val = val_done_rate
                    
                print()

            else:
                #get validate result
                if (t % t_validate == 0):
                    val_done_rate, min_total_val_dist, val_counter_collision, _ \
                                                = validation(val_env, trainer, max_steps=train_config["horizon"])
                    print(f"val_done_rate {val_done_rate}", end=" ")

                    #if flag_wandb:
                    #    print("get video", end=" ")
                    #    wandb.log({f"Val_trajectory_{t}": wandb.Video(val_traj,
                    #                                         fps=10, format="gif")})
                    
                    print()
