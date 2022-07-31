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
    last_render_start = time.time()
    for val_key in env.valTasks:
        trajectory_info["states"] = [] # x, y, v
        trajectory_info["accelerations"] = [] # a
        trajectory_info["headings"] = [] # phi

        id = 0
        start_time = time.time()
        obs = env.reset(idx=id, fromTrain=False, val_key=val_key)
        rnn_states = torch.zeros([env.num_agents, 
                get_hidden_size(cfg)], dtype=torch.float32, device=device)
        episode_reward = np.zeros(env.num_agents)
        done = [False]
        images = []
        images_observ = []
        cumul_reward = 0 
        num_frames = 0
        with torch.no_grad():
            while not done[0]:
                if num_frames >= max_steps:
                    break

                trajectory_info["states"].append((np.float64(env.current_state.x), 
                                                np.float64(env.current_state.y),
                                                np.float64(env.current_state.v),
                                                np.float64(env.current_state.steer)))
                if num_frames != 0:
                    Dx = trajectory_info["states"][num_frames][0] - \
                                trajectory_info["states"][num_frames - 1][0]
                    Dy = trajectory_info["states"][num_frames][1] - \
                                trajectory_info["states"][num_frames - 1][1]
                    if Dx != 0:
                        trajectory_info["headings"].append(np.arctan(Dy / Dx))
                    else:
                        trajectory_info["headings"].append(1.5)

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

                    obs, rew, done, infos = env.step(actions)
                    episode_reward += rew
                    num_frames += 1
                    episode_done = False
                    if "Collision" in infos[0]:
                        print("$$ Collision $$")
                    elif "SoftEps" in infos[0]:
                        print("$$ SoftEps $$")
                    elif num_frames >= max_steps:
                        episode_done = True

                    trajectory_info["accelerations"].append(np.float64(env.last_action[0]))

            done = False
            if not ("Collision" in infos[0]) and not ("SoftEps" in infos[0]):
                done = True 
            env.close()

        end_time = time.time()
        print("spended time:", abs(end_time - start_time))
    
    return trajectory_info