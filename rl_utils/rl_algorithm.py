from asyncio import tasks
import sys
import time
from collections import deque
import numpy as np
import torch
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.utils.utils import log, AttrDict

def run_algorithm(cfg, env, agent, max_steps=30):
    trajectory_info = {}
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    print("run info:")
    print("steps for episode", max_steps)
    last_render_start = time.time()
    for val_key in env.valTasks:
        trajectory_info["x"] = []
        trajectory_info["y"] = []
        trajectory_info["v"] = []
        trajectory_info["steer"] = []
        trajectory_info["accelerations"] = []
        trajectory_info["heading"] = []

        id = 0
        start_time = time.time()
        obs = env.reset(idx=id, fromTrain=False, val_key=val_key)
        rnn_states = torch.zeros([env.num_agents, 
                get_hidden_size(cfg)], dtype=torch.float32, device=device)
        episode_reward = np.zeros(env.num_agents)

        trajectory_info["x"].append(np.float64(env.current_state.x))
        trajectory_info["y"].append(np.float64(env.current_state.y))
        trajectory_info["v"].append(np.float64(env.current_state.v))
        trajectory_info["steer"].append(np.float64(env.current_state.steer))
        trajectory_info["heading"].append(np.float64(env.current_state.theta))

        images = []
        images_observ = []
        cumul_reward = 0 
        num_frames = 0
        done = [False]
        run_info = "OK"
        with torch.no_grad():
            while not done[0]:
                #if num_frames != 0:
                    #Dx = trajectory_info["x"][num_frames] - \
                    #            trajectory_info["x"][num_frames - 1]
                    #Dy = trajectory_info["y"][num_frames] - \
                    #            trajectory_info["y"][num_frames - 1]
                    #if Dx != 0:
                    #    trajectory_info["heading"].append(np.arctan(Dy / Dx))
                    #else:
                    #    trajectory_info["heading"].append(1.5)

                obs_torch = AttrDict(transform_dict_observations(obs))
                for key, x in obs_torch.items():
                    obs_torch[key] = torch.from_numpy(x).to(device).float()

                policy_outputs = agent(obs_torch, rnn_states, 
                            with_action_distribution=True)
                actions = policy_outputs.actions
                action_distribution = policy_outputs.action_distribution
                if not cfg.continuous_actions_sample:  
                    actions = action_distribution.means
                actions = actions.cpu().numpy()
                rnn_states = policy_outputs.rnn_states
                render_action_repeat = 1
                for _ in range(render_action_repeat):
                    if not cfg.no_render:
                        target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                        current_delay = time.time() - last_render_start
                        time_wait = target_delay - current_delay

                        if time_wait > 0:
                            time.sleep(time_wait)

                        last_render_start = time.time()

                    obs, rew, done, info = env.step(actions)

                    episode_reward += rew
                    num_frames += 1
                    # append agent action to the previous state
                    if not done[0]:
                        trajectory_info["accelerations"].append(np.float64(env.last_action[0]))
                        trajectory_info["x"].append(np.float64(env.current_state.x))
                        trajectory_info["y"].append(np.float64(env.current_state.y))
                        trajectory_info["v"].append(np.float64(env.current_state.v))
                        trajectory_info["steer"].append(np.float64(env.current_state.steer))
                        trajectory_info["heading"].append(np.float64(env.current_state.theta))
                    else: # add terminal state, because after env.step, env resets in wrapper
                        trajectory_info["accelerations"].append(np.float64(info[0]["terminal_last_action"][0]))
                        trajectory_info["x"].append(np.float64(info[0]["terminal_x"]))
                        trajectory_info["y"].append(np.float64(info[0]["terminal_y"]))
                        trajectory_info["v"].append(np.float64(info[0]["terminal_v"]))
                        trajectory_info["steer"].append(np.float64(info[0]["terminal_steer"]))
                        trajectory_info["heading"].append(np.float64(info[0]["terminal_heading"]))

            if "Collision" in info[0]:
                run_info = "Collision"
            elif "Max steps reached" in info[0]:
                run_info = "Max steps reached"

            #DEBUG
            #print("terminal v:", env.current_state.v)
            #print("terminal prev a:", env.last_action[0])
            ######
            
            env.close()
        # append agent action to the previous state
        trajectory_info["accelerations"].append(0)

        end_time = time.time()
        print("run spended time:", abs(end_time - start_time))
        print("run trajectory len:", len(trajectory_info["x"]))
        print("run status:", run_info)
    return trajectory_info, run_info