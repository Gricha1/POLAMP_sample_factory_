from asyncio import tasks
import sys
import time
from collections import deque
import numpy as np
import json
import torch
import wandb
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict

with open("configs/validate_config.json", 'r') as f:
    validate_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

def enjoy(init_cfg, max_num_frames=1200, use_wandb=True):
    if use_wandb:
        wandb.init(project='validate_polamp', entity='grisha1')
    i = 0
    while i < 1:
        cfg = load_from_checkpoint(init_cfg)
        print("debug env keys:", our_env_config.keys())
        render_environment = our_env_config["validate_with_render"]
        get_observation = our_env_config["validate_with_observation"]
        done_save_img = False
        debug_not_done_save_img = False
        debug_forward_move = None
        debug_dataset = False
        debug_testdataset = False
        render_action_repeat = cfg.render_action_repeat if \
            cfg.render_action_repeat is not None else cfg.env_frameskip
        if render_action_repeat is None:
            log.warning('Not using action repeat!')
            render_action_repeat = 1
        log.debug('Using action repeat %d during evaluation', render_action_repeat)
        cfg.env_frameskip = 1 
        cfg.num_envs = 1
        def make_env_func(env_config):
            return create_env(cfg.env, cfg=cfg, env_config=env_config)
        env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
        is_multiagent = is_multiagent_env(env)
        if not is_multiagent:
            env = MultiAgentWrapper(env)
        if hasattr(env.unwrapped, 'reset_on_init'):
            env.unwrapped.reset_on_init = False
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
        actor_critic.model_to_device(device)
        policy_id = cfg.policy_index
        checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        num_frames = 0
        last_render_start = time.time()

        def max_frames_reached(frames):
            return max_num_frames is not None and frames > max_num_frames
        successed_tasks = 0
        max_steps_tasks = 0
        collision_tasks = 0
        total_tasks = 0
        start_time = time.time()
        count_map = 0
        print("validation environment:")
        print("dataset size:", env.Tasks.keys())
        for val_key in env.Tasks:
            total_tasks += 1
            count_map += 1
            eval_tasks = len(env.Tasks[val_key])
            print("Num map:", count_map, "out of", len(env.Tasks), 
                    "count task:", eval_tasks)
            id = 0
            if debug_not_done_save_img:
                render_environment = False
                get_observation = False
                if not episode_done and not saved_last_image:
                    id = id - 1
                    render_environment = True
                    get_observation = True
                    saved_last_image = True
                else:
                    saved_last_image = False
            start_time = time.time()
            obs = env.reset(val_key=val_key)
            rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], 
                                        dtype=torch.float32, device=device)
            episode_reward = np.zeros(env.num_agents)
            finished_episode = [False] * env.num_agents
            done = [False]
            images = []
            images_observ = []
            cumul_reward = 0 
            num_frames = 0
            with torch.no_grad():
                while not done[0]:
                    if max_frames_reached(num_frames):
                        break
                    obs_torch = AttrDict(transform_dict_observations(obs))
                    for key, x in obs_torch.items():
                        obs_torch[key] = torch.from_numpy(x).to(device).float()
                    policy_outputs = actor_critic(obs_torch, rnn_states, 
                                                    with_action_distribution=True)

                    actions = policy_outputs.actions

                    action_distribution = policy_outputs.action_distribution
                    if isinstance(action_distribution, ContinuousActionDistribution):
                        if not cfg.continuous_actions_sample:  
                            actions = action_distribution.means

                    actions = actions.cpu().numpy()

                    rnn_states = policy_outputs.rnn_states

                    for _ in range(render_action_repeat):
                        if not cfg.no_render:
                            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                            current_delay = time.time() - last_render_start
                            time_wait = target_delay - current_delay

                            if time_wait > 0:
                                time.sleep(time_wait)

                            last_render_start = time.time()

                        obs, rew, done, infos = env.step(actions)

                        if render_environment and not done[0]:
                            cumul_reward += rew[0]
                            image = env.render(reward=cumul_reward)
                            images.append(image)
                        if render_environment and done[0]:
                            image = infos[0]["terminal_render"]
                            images.append(image)
                        if get_observation and not done[0]:
                            obs_img = np.expand_dims(
                                env.grid_static_obst * 255, axis=2)
                            ag_img = np.expand_dims(
                                env.grid_agent * 255, axis=2)
                            dyn_img = np.expand_dims(
                                env.grid_dynamic_obst * 255, axis=2)
                            image = np.concatenate((obs_img, ag_img, dyn_img), 
                                                    axis=2)
                            images_observ.append(image)
                        if get_observation and done[0]:
                            obs_img = np.expand_dims(
                                infos[0]["terminal_grid_static_obst"] * 255, axis=2)
                            ag_img = np.expand_dims(
                                infos[0]["terminal_grid_agent"] * 255, axis=2)
                            dyn_img = np.expand_dims(
                                infos[0]["terminal_grid_dynamic_obst"] * 255, axis=2)
                            image = np.concatenate((obs_img, ag_img, dyn_img), 
                                                    axis=2)
                            images_observ.append(image)

                        episode_reward += rew
                        num_frames += 1

                        for agent_i, done_flag in enumerate(done):
                            if done_flag:
                                finished_episode[agent_i] = True
                                episode_rewards[agent_i].append(episode_reward[agent_i])
                                true_rewards[agent_i].append(
                                    infos[agent_i].get('true_reward', 
                                    episode_reward[agent_i]))
                                log.info('Episode finished for agent %d at %d frames. '\
                                            + 'Reward: %.3f, true_reward: %.3f', \
                                            agent_i, num_frames, episode_reward[agent_i], 
                                            true_rewards[agent_i][-1])
                                rnn_states[agent_i] = torch.zeros(
                                                    [get_hidden_size(cfg)], 
                                                    dtype=torch.float32, device=device)
                                episode_reward[agent_i] = 0

                        episode_done = False
                        if all(finished_episode):
                            print(f"finished_episode: {finished_episode}")
                            if "Collision" in infos[0]:
                                print("$$ Collision $$")
                                collision_tasks += 1
                            elif "SoftEps" in infos[0]:
                                print("$$ SoftEps $$")
                            elif num_frames != max_num_frames:
                                successed_tasks += 1
                                episode_done = True
                            else:
                                max_steps_tasks += 1
                            finished_episode = [False] * env.num_agents
                            avg_episode_rewards_str, avg_true_reward_str = '', ''
                            for agent_i in range(env.num_agents):
                                avg_rew = np.mean(episode_rewards[agent_i])
                                avg_true_rew = np.mean(true_rewards[agent_i])
                                if not np.isnan(avg_rew):
                                    if avg_episode_rewards_str:
                                        avg_episode_rewards_str += ', '
                                    avg_episode_rewards_str += \
                                        f'#{agent_i}: {avg_rew:.3f}'
                                if not np.isnan(avg_true_rew):
                                    if avg_true_reward_str:
                                        avg_true_reward_str += ', '
                                    avg_true_reward_str += \
                                        f'#{agent_i}: {avg_true_rew:.3f}'
                                    
                    # test
                    #interested_episode = False
                    #if num_frames == 1 and "Collision" in infos[0]:
                    #    interested_episode = True
                    #else:
                    #    break

            # test                
            #if not interested_episode:
            #    continue

            done = False
            if not ("Collision" in infos[0]) and not ("SoftEps" in infos[0]) \
                and num_frames != max_num_frames:
                done = True 
            if render_environment and use_wandb and ((done_save_img and done) or \
                not done_save_img):
                print("num frames:", num_frames)
                images = np.transpose(np.array(images), axes=[0, 3, 1, 2])
                print("DEBUG img:", images.shape)
                print("get video", end=" ")
                wandb.log({f"Val_trajectory_": wandb.Video(images, 
                                            fps=10, format="gif")})
            if get_observation and use_wandb and ((done_save_img and done) or \
                                                        not done_save_img):
                images_observ = np.transpose(np.array(images_observ), 
                                                axes=[0, 3, 1, 2])
                print("DEBUG obs:", images_observ.shape)
                wandb.log({f"Obs_trajectory_": wandb.Video(images_observ, 
                                                fps=10, format="gif")})
            end_time = time.time()
            print("spended time:", abs(end_time - start_time))
            env.close()
        
        end_time = time.time()
        print("final time: ", end_time - start_time)
        success_rate = successed_tasks / total_tasks * 100
        max_step_rate = max_steps_tasks / total_tasks * 100
        collision_rate = collision_tasks / total_tasks * 100
        print (f"success_rate: {success_rate}")
        print (f"max_step_rate: {max_step_rate}")
        print (f"collision_rate: {collision_rate}")
        if use_wandb:
            wandb.log(
                    {
                    'success_rate': success_rate,
                    'max_step_rate': max_step_rate,
                    'collision_rate': collision_rate,
                    })

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)
    
