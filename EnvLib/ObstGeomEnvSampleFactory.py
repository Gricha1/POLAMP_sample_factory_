from tkinter import E
import gym
import matplotlib.pyplot as plt
from .line import *
from math import pi
import numpy as np
from .Vec2d import Vec2d
from .utils import *
from math import cos, sin, tan
from scipy.spatial import cKDTree
import time
import cv2 as cv
from planning.reedShepp import *
from dataset_generation.utlis import *

class State:
    def __init__(self, x, y, theta, v, steer, 
                 v_s=0, w=0, gear=None, last_a=0, last_Eps=0,
                 centred_x=0, centred_y=0, width=0, length=0, wheel_base=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.steer = steer
        self.v_s = v_s
        self.w = w
        self.gear = gear
        self.last_a = last_a
        self.last_Eps = last_Eps
        self.centred_x = centred_x
        self.centred_y = centred_y
        self.width = width
        self.length = length
        self.wheel_base = wheel_base

class Vehicle:
    def __init__(self, car_config, ego_car):
        self.ego_car = ego_car
        self.wheel_base = car_config["wheel_base"]
        if self.ego_car:
            self.length = car_config["length"]
            self.width = car_config["width"]
            self.safe_eps = car_config["safe_eps"]
        self.max_steer = degToRad(car_config["max_steer"])
        self.max_vel = car_config["max_vel"]
        self.min_vel = car_config["min_vel"]
        self.max_acc = car_config["max_acc"]
        self.max_ang_vel = car_config["max_ang_vel"]
        self.max_ang_acc = car_config["max_ang_acc"]
        self.delta_t = car_config["delta_t"]
        self.use_clip = car_config["use_clip"]
        if "movement_function" not in car_config:
            self.movement_function = None
        else:
            self.movement_function = car_config["movement_function"]

    def get_action(self):
        assert not self.ego_car, "this function for dynamic obsts in env only"
        dt = self.delta_t
        x = self.shifted_x
        y = self.shifted_y
        theta = self.theta
        v = self.v
        steer = self.steer
        v_s = self.v_s
        gear = self.gear
        a, Eps = self.movement_func([x, y, theta, v, steer], 
                                     self.current_steps, 
                                     **self.movement_func_params)

        return [a, Eps]

    def reset(self, state, dynamic_config=None):
        self.last_a = 0
        self.last_Eps = 0
        self.shifted_x = state.x
        self.shifted_y = state.y
        self.theta = state.theta
        self.v = state.v
        self.steer = state.steer
        self.v_s = 0
        self.w = 0
        self.gear = None
        if not self.ego_car:
            self.movement_func = dynamic_config["movement_func"]
            self.movement_func_params = dynamic_config["movement_func_params"]
            self.max_vel = dynamic_config["boundary_v"]
            self.min_vel = -dynamic_config["boundary_v"]
            self.width = dynamic_config["width"]
            self.length = dynamic_config["length"]
        self.current_steps = 0

    def step(self, action=None):
        dt = self.delta_t
        x = self.shifted_x
        y = self.shifted_y
        theta = self.theta
        v = self.v
        steer = self.steer
        v_s = self.v_s
        gear = self.gear

        assert not(action is None), \
            "action must exist for the vehicle in the step function"
        assert len(action) == 2, "action len should be 2" + \
            f"but given {len(action)}"
        a = action[0]
        Eps = action[1]
        if self.use_clip:
            a = np.clip(a, -self.max_acc, self.max_acc)
            Eps = np.clip(Eps, -self.max_ang_acc, self.max_ang_acc)

        dV = a * dt
        new_v = v + dV
        overSpeeding = new_v > self.max_vel or new_v < self.min_vel
        new_v = np.clip(new_v, self.min_vel, self.max_vel)

        dv_s = Eps * dt
        new_v_s = v_s + dv_s
        new_v_s = np.clip(new_v_s, -self.max_ang_vel, self.max_ang_vel)
        dsteer = new_v_s * dt
        new_steer = steer + dsteer
        new_steer = normalizeAngle(new_steer)
        overSteering = abs(new_steer) > self.max_steer
        new_steer = np.clip(new_steer, -self.max_steer, self.max_steer)

        w = (new_v * np.tan(new_steer) / self.wheel_base)

        dtheta = w * dt
        new_theta = theta + dtheta
        new_theta = normalizeAngle(new_theta)

        dx = new_v * np.cos(theta) * dt
        dy = new_v * np.sin(theta) * dt
        new_x = x + dx
        new_y = y + dy

        new_gear = gear
        if gear is None:
            if new_v > 0:
                new_gear = True
            elif new_v < 0:
                new_gear = False
        else:
            if gear:
                if new_v < 0:
                    new_gear = False
            else:
                if new_v > 0:
                    new_gear = True

        self.last_a = a
        self.last_Eps = Eps
        self.prev_gear = gear
        self.shifted_x = new_x
        self.shifted_y = new_y
        self.theta = new_theta
        self.v = new_v
        self.steer = new_steer
        self.v_s = new_v_s
        self.w = w
        self.gear = new_gear

        new_state = State(new_x, new_y, new_theta, new_v, new_steer)

        self.current_steps += 1

        return new_state, overSpeeding, overSteering
    
    def getCurrentState(self):
        centred_x, centred_y = self.getCentredCoordinates()

        return State(self.shifted_x, self.shifted_y, 
                     self.theta, self.v, self.steer,
                     v_s=self.v_s, w=self.w, gear=self.gear,
                     last_a=self.last_a, last_Eps=self.last_Eps,
                     centred_x=centred_x, centred_y=centred_y, 
                     width=self.width, length=self.length, 
                     wheel_base=self.wheel_base)

    def getCentredCoordinates(self):
        shift = self.wheel_base / 2
        centred_x = self.shifted_x + shift * cos(self.theta)
        centred_y = self.shifted_y + shift * sin(self.theta)

        return centred_x, centred_y
    
class ObsEnvironment(gym.Env):
    def __init__(self, full_env_name, config):
        self.adding_ego_features = True
        self.adding_dynamic_features = True
        self.gridCount = 4
        self.grid_resolution = 4
        self.grid_shape = (120, 120)
        assert self.grid_shape[0] % self.grid_resolution == 0 \
                   and self.grid_shape[1] % self.grid_resolution == 0, \
                   "incorrect grid shape"

        self.name = full_env_name
        env_config = config["our_env_config"]
        self.reward_config = config["reward_config"]
        self.old_state = None
        self.obstacle_segments = []
        self.dyn_obstacle_segments = []
        self.last_observations = []
        self.car_config = config['car_config']
        self.vehicle = Vehicle(self.car_config, ego_car=True)
        self.Tasks = config['Tasks']
        self.maps_init = config['maps']
        self.maps = dict(config['maps'])
        self.HARD_EPS = env_config['HARD_EPS']
        self.ANGLE_EPS = degToRad(env_config['ANGLE_EPS'])
        self.SPEED_EPS = env_config['SPEED_EPS']
        self.STEERING_EPS = degToRad(env_config['STEERING_EPS'])
        self.MAX_DIST_LIDAR = env_config['MAX_DIST_LIDAR']
        self.UPDATE_SPARSE = env_config['UPDATE_SPARSE']
        self.frame_stack = env_config['frame_stack']
        self.max_episode_steps = env_config['max_episode_steps']
        self.dyn_acc = 0
        self.dyn_ang_vel = 0
        self.collision_time = 0
        self.gear_switch_reward = env_config['gear_switch_reward']
        self.RS_reward = env_config['redshape_reward']
        self.reward_weights = [
            self.reward_config["collision"],
            self.reward_config["goal"],
            self.reward_config["timeStep"],
            self.reward_config["redshapeDistance"],
            self.reward_config["overSpeeding"],
            self.reward_config["overSteering"],
            self.reward_config["gearSwitchPenalty"]
        ]

        self.validate_env = env_config["validate_env"]
        self.validate_test_dataset = env_config["validate_test_dataset"]
        if self.validate_env:
            print("DEBUG: VALIDATE ENV")
            if self.validate_test_dataset:
                print("DEBUG: VALIDATE TEST DATASET")
        self.validate_with_reward = env_config["validate_with_reward"]
        self.validate_with_render = env_config["validate_with_render"]
        
        state_min_box = [[[-np.inf for j in range(self.grid_shape[1])] 
                for i in range(self.grid_shape[0])] for _ in range(self.gridCount)]
        state_max_box = [[[np.inf for j in range(self.grid_shape[1])] 
                for i in range(self.grid_shape[0])] for _ in range(self.gridCount)]
        obs_min_box = np.array(state_min_box)
        obs_max_box = np.array(state_max_box)
        self.observation_space = gym.spaces.Box(obs_min_box, obs_max_box, 
                                            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), 
                            high=np.array([1, 1]), dtype=np.float32)
        if len(self.maps) != 0:
            self.lst_keys = list(self.maps.keys())
            index = np.random.randint(len(self.lst_keys))
            self.map_key = self.lst_keys[index]
            self.obstacle_map = self.maps[self.map_key]

    def update_task(self, maps, Tasks, 
                    valTasks, second_goal):
        self.Tasks = Tasks
        self.maps_init = maps
        self.maps = dict(maps)

        self.lst_keys = list(self.maps.keys())
        index = np.random.randint(len(self.lst_keys))
        self.map_key = self.lst_keys[index]
        self.obstacle_map = self.maps[self.map_key]

        assert 1 == 0, "second goal not deleted in update task"
        if self.unionTask:
            if len(second_goal) != 0:
                self.second_goal = State(second_goal[0], 
                                         second_goal[1],
                                         second_goal[2],
                                         second_goal[3],
                                         second_goal[4])
            else:
                self.second_goal = None

    def _getBB(self, state):
        x = state.centred_x
        y = state.centred_y
        angle = state.theta
        w = state.width / 2
        l = state.length / 2
        BBPoints = [(-l, -w), (l, -w), (l, w), (-l, w)]
        vertices = []
        sinAngle = math.sin(angle)
        cosAngle = math.cos(angle)
        for i in range(len(BBPoints)):
            new_x = cosAngle * (BBPoints[i][0]) - sinAngle * (BBPoints[i][1])
            new_y = sinAngle * (BBPoints[i][0]) + cosAngle * (BBPoints[i][1])
            vertices.append(Point(new_x + x, new_y + y))
            
        segments = [(vertices[(i) % len(vertices)], \
                    vertices[(i + 1) % len(vertices)]) 
                    for i in range(len(vertices))]
        
        return segments

    def _getDiff(self, state):
        if self.goal is None:
            self.goal = state
        delta = []
        dx = self.goal.x - state.x
        dy = self.goal.y - state.y
        dtheta = self.goal.theta - state.theta
        dv = self.goal.v - state.v
        dsteer = self.goal.steer - state.steer
        theta = state.theta
        v = state.v
        steer = state.steer
        v_s = state.v_s
        delta.extend([dx, dy, dtheta, dv, dsteer, theta, v, steer, v_s])
        
        return delta

    def transformTask(self, from_state, goal_state, 
                      obstacles):
        start_transform = list(from_state)
        goal_transform = list(goal_state)
        new_obstacle_map = []
        for index in range(len(obstacles)):
            state = obstacles[index]
            new_obstacle_map.append([state[0], state[1],  
                        state[2], state[3], state[4]])
        self.obstacle_map = new_obstacle_map

        start = State(start_transform[0], start_transform[1], 
                      start_transform[2], start_transform[3], 
                      start_transform[4])
        goal = State(goal_transform[0], goal_transform[1], 
                     goal_transform[2], goal_transform[3], 
                     goal_transform[4])
        
        return start, goal 

    def _setTask(self, task, obstacles, set_task_without_dynamic_obsts):
        assert len(task) > 0, \
            "incorrect count of tasks must be len(tasks) > 0 but given 0"
        current_task = tuple(task[0])
        if(len(current_task) == 2):
            current, goal = current_task
        else:
            current, goal, dynamic_obstacles = current_task
            if not set_task_without_dynamic_obsts:
                for dyn_obst in dynamic_obstacles:
                    vehicle = Vehicle(self.car_config, ego_car=False)
                    state = State(dyn_obst[0],
                                  dyn_obst[1],
                                  dyn_obst[2],
                                  dyn_obst[3],
                                  dyn_obst[4])
                    vehicle.reset(state, dyn_obst[5])
                    self.vehicles.append(vehicle)
                    
        self.current_state, self.goal = self.transformTask(
                            current, goal, obstacles)
        self.old_state = self.current_state
        self.vehicle.reset(self.current_state)

    def reset(self, val_key=None):
        # add this to prevent reset when step returns DONE = TRUE
        # when step returns DONE = TRUE, sample factory wrapper do reset
        if self.validate_env and val_key is None:
            return None

        self.maps = dict(self.maps_init)
        set_task_without_dynamic_obsts = np.random.randint(5) == 0
        self.stepCounter = 0
        self.last_observations = []
        self.last_action = [0., 0.]
        self.obstacle_segments = []
        self.dyn_obstacle_segments = []
        self.vehicles = []
        self.dyn_acc = 0
        self.dyn_ang_vel = 0
        self.dyn_ang_acc = 0
        self.collision_time = 0
        if self.RS_reward:
            self.new_RS = None
        if self.validate_env:
            set_task_without_dynamic_obsts = False
            # test
        #    if self.validate_test_dataset:
        #        self.stop_dynamic_step = 1200
        #        if val_key == "map0" or val_key == "map2":
        #            self.stop_dynamic_step = 100
        #        elif val_key == "map1" or val_key == "map3":
        #            self.stop_dynamic_step = 110
        #        elif val_key == "map6":
        #            self.stop_dynamic_step = 200

        if not self.validate_env:
            index = np.random.randint(len(self.lst_keys))
            self.map_key = self.lst_keys[index]
        else:
            assert not(val_key is None), \
                "in reset func: val_key must be set in validate env" \
                + f"but val_key is: {val_key}"
            self.map_key = val_key
        self.obstacle_map = self.maps[self.map_key]
        task = self.Tasks[self.map_key]

        self._setTask(task, self.obstacle_map, 
                      set_task_without_dynamic_obsts)

        for obstacle in self.obstacle_map:
            static_obst = State(None, None, theta=obstacle[2], v=0, steer=0, 
                                centred_x=obstacle[0], centred_y=obstacle[1], 
                                width=2*obstacle[3], length=2*obstacle[4])
            self.obstacle_segments.append(
                self._getBB(static_obst))
                
        self.start_dist = self._goalDist(self.current_state)
        self.last_images = []
        self.grid_static_obst = None
        self.grid_agent = None
        self.grid_with_adding_features = None
        observation = self._getObservation(first_obs=True)
    
        return observation
    
    def _reward(self, current_state, new_state, goalReached, 
                collision, overSpeeding, overSteering):
        if not self.RS_reward:
            previous_delta = self._goalDist(current_state)
            new_delta = self._goalDist(new_state)
        reward = []

        reward.append(-1 if collision else 0)

        if goalReached:
            reward.append(1)
        else:
            reward.append(0)
        if not (self.stepCounter % self.UPDATE_SPARSE):
            reward.append(-1)

            if self.RS_reward:
                if self.new_RS is None:
                    self.prev_RS = reedsSheppSteer(current_state, self.goal)
                else:
                    self.prev_RS = (self.new_RS[0].copy(), 
                                    self.new_RS[1].copy(), 
                                    self.new_RS[2].copy())
                self.new_RS = reedsSheppSteer(new_state, self.goal)

                if self.new_RS[2] is None or self.prev_RS[2] is None:
                    self.new_RS = None
                    self.prev_RS = None
                    reward.append(0)
                else:
                    RS_L_prev = abs(self.prev_RS[2][0]) + \
                            abs(self.prev_RS[2][1]) + abs(self.prev_RS[2][2])
                    RS_L_new = abs(self.new_RS[2][0]) + \
                            abs(self.new_RS[2][1]) + abs(self.new_RS[2][2])
                    self.RS_diff = RS_L_prev - RS_L_new
                    reward.append(RS_L_prev - RS_L_new)
            else:
                if (new_delta < 0.5):
                    new_delta = 0.5
                reward.append(previous_delta - new_delta)
            reward.append(-1 if overSpeeding else 0)
            reward.append(-1 if overSteering else 0)
        else:
            reward.append(0)
            reward.append(0)
            reward.append(0)
            reward.append(0)
        if self.gear_switch_reward:
            if not(self.vehicle.prev_gear is None) and \
               self.vehicle.prev_gear != self.vehicle.gear:
                reward.append(-1)
            else:
                reward.append(0)
        else:
            reward.append(0)

        return np.matmul(self.reward_weights, reward)

    def _goalDist(self, state):
        return math.hypot(self.goal.x - state.x, self.goal.y - state.y)

    def step(self, action):        
        info = {}
        isDone = False

        new_state, overSpeeding, overSteering = \
                self.vehicle.step(action)

        for vehicle in self.vehicles:
            action = vehicle.get_action()
            vehicle.step(action)
                        
        self.current_state = new_state
        self.last_action = action

        observation = self._getObservation()

        #collision
        start_time = time.time()
        temp_grid_obst = self.grid_static_obst + self.grid_dynamic_obst
        collision = temp_grid_obst[self.grid_agent == 1].sum() > 0
        collision = collision or (self.grid_agent.sum() == 0)
        end_time = time.time()
        self.collision_time += (end_time - start_time)
        
        distanceToGoal = self._goalDist(new_state)
        info["EuclideanDistance"] = distanceToGoal
        goalReached = distanceToGoal < self.HARD_EPS and abs(
            normalizeAngle(new_state.theta - self.goal.theta)) < self.ANGLE_EPS \
            and abs(new_state.v - self.goal.v) <= self.SPEED_EPS
    
        if self.validate_env and not self.validate_with_reward:
            reward = 0
        else:
            reward = self._reward(self.old_state, new_state, 
                                goalReached, collision, overSpeeding, 
                                overSteering)
        
        if not (self.stepCounter % self.UPDATE_SPARSE):
            self.old_state = self.current_state
        self.stepCounter += 1

        if goalReached or collision or (self.max_episode_steps == self.stepCounter):
            isDone = True
            if goalReached:
                info["OK"] = True
            else:
                if collision:
                    info["Collision"] = True
                else:
                    info["Max episode reached"] = True
            info["terminal_last_action"] = self.last_action
            info["terminal_x"] = self.current_state.x
            info["terminal_y"] = self.current_state.y
            info["terminal_v"] = self.current_state.v
            info["terminal_steer"] = self.current_state.steer
            info["terminal_heading"] = self.current_state.theta
            info["terminal_w"] = self.vehicle.w
            info["terminal_v_s"] = self.vehicle.v_s

            if self.validate_env and self.validate_with_render:
                info["terminal_grid_agent"] = self.grid_agent
                info["terminal_grid_static_obst"] = self.grid_static_obst
                info["terminal_grid_dynamic_obst"] = self.grid_dynamic_obst
                info["terminal_render"] = self.render(0)
            
        return observation, reward, isDone, info

    def drawObstacles(self, vertices, color="-b"):
        a = vertices
        plt.plot([a[(i + 1) % len(a)][0].x for i in range(len(a) + 1)], 
                 [a[(i + 1) % len(a)][0].y for i in range(len(a) + 1)], color)

    def drawState(self, state, ax, color="-b", 
                  with_heading=False, heading_color='red'):
        vertices = self._getBB(state)
        a = vertices
        plt.plot([a[(i + 1) % len(a)][0].x for i in range(len(a) + 1)], 
                 [a[(i + 1) % len(a)][0].y for i in range(len(a) + 1)], color)
        if with_heading:
            vehicle_heading = Vec2d(cos(state.theta),
                                sin(state.theta)) * 2
            ax.arrow(state.x, state.y,
                    vehicle_heading.x, vehicle_heading.y, 
                    width=0.1, head_width=0.3,
                    color=heading_color)
    
    def getCentredCoordinates(self, state):
        shift = state.wheel_base / 2
        centred_x = state.x + shift * cos(state.theta)
        centred_y = state.y + shift * sin(state.theta)

        return centred_x, centred_y

    def render(self, reward, figsize=(8, 8), save_image=True):
        fig, ax = plt.subplots(figsize=figsize)

        x_delta = self.MAX_DIST_LIDAR
        y_delta = self.MAX_DIST_LIDAR
        x_min = self.current_state.x - x_delta
        x_max = self.current_state.x + x_delta
        ax.set_xlim(x_min, x_max)
        y_min = self.current_state.y - y_delta
        y_max = self.current_state.y + y_delta
        ax.set_ylim(y_min, y_max)

        for vehicle in self.vehicles:
            self.drawState(vehicle.getCurrentState(), ax, 
                           color="-b", with_heading=True, heading_color="magenta")
        
        for obstacle in self.obstacle_segments:
            self.drawObstacles(obstacle)
        
        agent_state = self.vehicle.getCurrentState()
        self.drawState(agent_state, ax, color="-g", with_heading=True)

        self.goal.width = agent_state.width
        self.goal.length = agent_state.length
        self.goal.wheel_base = agent_state.wheel_base
        cented_x, centred_y = self.getCentredCoordinates(self.goal)
        self.goal.centred_x, self.goal.centred_y = cented_x, centred_y
        self.drawState(self.goal, ax, color="cyan", with_heading=True)

        dx = self.goal.x - agent_state.x
        dy = self.goal.y - agent_state.y
        ds = math.hypot(dx, dy)
        step_count = self.stepCounter
        theta = radToDeg(agent_state.theta)
        v = agent_state.v
        delta = radToDeg(agent_state.steer)
        v_s = agent_state.v_s
        a = agent_state.last_a
        Eps = agent_state.last_Eps
        gear = agent_state.gear

        reeshep_dist = 0
        if self.RS_reward and not self.new_RS is None:
            if self.new_RS[2] is None:
               reeshep_dist = 0
            else:
                reeshep_dist = abs(self.new_RS[2][0]) + \
                    abs(self.new_RS[2][1]) + abs(self.new_RS[2][2])
                ax.plot([st[0] for st in self.new_RS[0]], 
                    [st[1] for st in self.new_RS[0]])
        else:
            reeshep_dist = 0
        
        ax.set_title(f'$step = {step_count:.0f}, ' \
                     + f'ds = {ds:.1f}, dx = {dx:.1f}, dy = {dy:.1f}, ' \
                     + f'gear = {gear}, ' \
                     + f'steer={delta:.0f}$ \n $\\theta = {theta:.0f}^\\circ, ' \
                     + f'a = {a:.2f}, E={Eps:.2f}, v = {v:.2f}, v_s={v_s:.2f}, ' \
                     + f'r={reward:.0f}, RS_d={reeshep_dist:.1f}$')
        
        if save_image:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image
        else:
            plt.pause(0.1)
            plt.show()

    def close(self):
        pass

    def _getObservation(self, first_obs=False):
        fake_static_obstacles = False
        if len(self.obstacle_segments) == 0:
            fake_static_obstacles = True
            parking_heights = np.linspace(2.7 + 0.2, 2.7 + 5, 20)
            parking_widths = np.linspace(4.5, 7, 20)
            road_widths = np.linspace(3, 6, 20)
            bottom_left_boundary_heights = np.linspace(4.5, 6, 20)
            upper_boundary_widths = np.linspace(0.5, 3, 20)
            upper_boundary_heights = np.linspace(14, 16, 20)
            bottom_left_boundary_center_x = 5 # any value
            bottom_left_boundary_center_y = -5.5 # init value
            bottom_left_boundary_height = bottom_left_boundary_heights[-1]
            upper_boundary_width = upper_boundary_widths[0]
            upper_boundary_height = upper_boundary_heights[-1]
            parking_height = parking_heights[-1]
            parking_width = parking_widths[-1]
            road_width = road_widths[-1]
            staticObstsInfo = {}
            self.obstacle_map = getValetStaticObstsAndUpdateInfo(
                parking_height, parking_width, bottom_left_boundary_height, 
                upper_boundary_width, upper_boundary_height,
                bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
                road_width,
                staticObstsInfo
            )
            for obstacle in self.obstacle_map:
                static_obst = State(None, None, theta=obstacle[2], v=0, steer=0, 
                                    centred_x=obstacle[0], centred_y=obstacle[1], 
                                    width=2*obstacle[3], length=2*obstacle[4])
                self.obstacle_segments.append(
                    self._getBB(static_obst))
            
        assert len(self.obstacle_segments) > 0, "not static env"

        grid_resolution = self.grid_resolution
        self.grid_static_obst = np.zeros(self.grid_shape)
        self.grid_dynamic_obst = np.zeros(self.grid_shape)
        self.grid_agent = np.zeros(self.grid_shape)
        self.grid_with_adding_features = np.zeros(self.grid_shape)
        
        #find static init point
        if first_obs:
            static_obsts_points = []
            for obstacle in self.obstacle_segments:
                static_obsts_points.extend(obstacle)

            x_min, y_min = static_obsts_points[0][0].x, static_obsts_points[0][0].y
            for point_ in static_obsts_points:         
                if point_[0].x < x_min:
                    x_min = point_[0].x
                if point_[0].y < y_min:
                    y_min = point_[0].y

            self.normalized_x_init = x_min
            self.normalized_y_init = y_min

        #get normalized static boxes
        if first_obs:
            self.normalized_static_boxes = []
            for obstacle in self.obstacle_segments:
                self.normalized_static_boxes.append(
                    [Point(pair_[0].x - self.normalized_x_init, 
                     pair_[0].y - self.normalized_y_init) 
                     for pair_ in obstacle])

        #get normalized dynamic boxes
        self.normalized_dynamic_boxes = []
        for vehicle in self.vehicles:
            vertices = self._getBB(vehicle.getCurrentState())
            self.normalized_dynamic_boxes.append(
                    [Point(max(0, pair_[0].x - self.normalized_x_init), 
                     max(0, pair_[0].y - self.normalized_y_init)) 
                     for pair_ in vertices])

        vertices = self._getBB(self.vehicle.getCurrentState())
        self.normalized_agent_box = [Point(max(0, pair_[0].x - self.normalized_x_init), 
                                     max(0, pair_[0].y - self.normalized_y_init)) 
                                     for pair_ in vertices]

        #choice grid indexes
        self.all_normilized_boxes = self.normalized_static_boxes.copy()
        self.all_normilized_boxes.extend(self.normalized_dynamic_boxes)
        self.all_normilized_boxes.append(self.normalized_agent_box)

        x_shape, y_shape = self.grid_static_obst.shape
        self.cv_index_boxes = []
        for box_ in self.all_normilized_boxes:
            box_cv_indexes = []
            for i in range(len(box_)):
                prev_x, prev_y = box_[i - 1].x, box_[i - 1].y
                curr_x, curr_y = box_[i].x, box_[i].y
                next_x, next_y = box_[(i + 1) % len(box_)].x, \
                                 box_[(i + 1) % len(box_)].y
                x_f, x_ceil = np.modf(curr_x)
                y_f, y_ceil = np.modf(curr_y)
                one_x_one = (int(x_ceil * grid_resolution), 
                             int(y_ceil * grid_resolution))
                one_x_one_x_ind = 0
                one_x_one_y_ind = 0
                rx, lx, ry, ly = 1.0, 0.0, 1.0, 0.0
                curr_ind_add = grid_resolution
                while rx - lx > 1 / grid_resolution:
                    curr_ind_add = curr_ind_add // 2
                    mx = (lx + rx) / 2
                    if x_f < mx:
                        rx = mx
                    else:
                        lx = mx
                        one_x_one_x_ind += curr_ind_add
                    my = (ly + ry) / 2
                    if y_f < my:
                        ry = my
                    else:
                        ly = my
                        one_x_one_y_ind += curr_ind_add
                if x_f == 0:
                    if prev_x <= curr_x and next_x <= curr_x:
                        x_ceil -= 1
                        one_x_one = (int(x_ceil * grid_resolution), 
                                     int(y_ceil * grid_resolution))
                        one_x_one_x_ind = grid_resolution - 1
                if y_f == 0:
                    if prev_y <= curr_y and next_y <= curr_y:
                        y_ceil -= 1
                        one_x_one = (int(x_ceil * grid_resolution), 
                                     int(y_ceil * grid_resolution))
                        one_x_one_y_ind = grid_resolution - 1
                index_grid_rev_x = one_x_one[0] + one_x_one_x_ind
                index_grid_rev_y = one_x_one[1] + one_x_one_y_ind
                cv_index_x = index_grid_rev_x
                cv_index_y = y_shape - index_grid_rev_y 
                box_cv_indexes.append(Point(cv_index_x, cv_index_y))
            self.cv_index_boxes.append(box_cv_indexes)

        self.cv_index_agent_box = self.cv_index_boxes.pop(-1)
        
        #CV draw
        for ind_box, cv_box in enumerate(self.cv_index_boxes):
            contours = np.array([[cv_box[3].x, cv_box[3].y], 
                                 [cv_box[2].x, cv_box[2].y], 
                                 [cv_box[1].x, cv_box[1].y], 
                                 [cv_box[0].x, cv_box[0].y]])
            color = 1
            if ind_box >= len(self.normalized_static_boxes):
                self.grid_dynamic_obst = cv.fillPoly(self.grid_dynamic_obst, 
                                                     pts = [contours], color=color)    
            self.grid_static_obst = cv.fillPoly(self.grid_static_obst, 
                                                pts = [contours], color=color)

        cv_box = self.cv_index_agent_box
        contours = np.array([[cv_box[3].x, cv_box[3].y], [cv_box[2].x, cv_box[2].y], 
                            [cv_box[1].x, cv_box[1].y], [cv_box[0].x, cv_box[0].y]])
        self.grid_agent = cv.fillPoly(self.grid_agent, pts = [contours], color=1)

        if not self.adding_ego_features:
            cv_box = self.cv_index_goal_box
            contours = np.array([[cv_box[3].x, cv_box[3].y], 
                                 [cv_box[2].x, cv_box[2].y], 
                                 [cv_box[1].x, cv_box[1].y], 
                                 [cv_box[0].x, cv_box[0].y]])
            self.grid_with_adding_features = cv.fillPoly(
                            self.grid_with_adding_features, 
                            pts = [contours], color=1)
        else:
            agent_state = self.vehicle.getCurrentState()
            adding_features = self._getDiff(agent_state)
            self.grid_with_adding_features[0, 0:len(adding_features)] = adding_features
            if self.adding_dynamic_features:
                assert len(self.vehicles) <= 2, "dynamic objects more than 2"
                for ind, vehicle in enumerate(self.vehicles):
                    dyn_state = vehicle.getCurrentState()
                    self.grid_with_adding_features[ind + 1, 0:4] \
                                = [dyn_state.centred_x - self.normalized_x_init, 
                                   dyn_state.centred_y - self.normalized_y_init,
                                   dyn_state.theta,
                                   dyn_state.v]

        if fake_static_obstacles:
            self.grid_static_obst = np.zeros(self.grid_shape)
        dim_images = []
        dim_images.append(np.expand_dims(self.grid_static_obst, 0))
        dim_images.append(np.expand_dims(self.grid_dynamic_obst, 0))
        dim_images.append(np.expand_dims(self.grid_agent, 0))
        dim_images.append(np.expand_dims(self.grid_with_adding_features, 0))
        image = np.concatenate(dim_images, axis = 0)
        self.last_images.append(image)
        if first_obs:
            assert len(self.last_images) == 1, "incorrect init images"
            for _ in range(self.frame_stack - 1):
                self.last_images.append(image)
        else:
            self.last_images.pop(0)
        frames_images = np.concatenate(self.last_images, axis = 0)
        
        if fake_static_obstacles:
            self.obstacle_map = []
            self.obstacle_segments = []

        return frames_images