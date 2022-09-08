from tkinter import E
import gym
import matplotlib.pyplot as plt
from planning.generateMap import generateTasks
from .line import *
from math import pi
import numpy as np
from .Vec2d import Vec2d
from .utils import *
from math import cos, sin, tan
from scipy.spatial import cKDTree
from planning.utilsPlanning import *
import time
import cv2 as cv
from planning.reedShepp import *
from policy_gradient.utlis import *

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
        if ego_car:
            self.length = car_config["length"]
            self.width = car_config["width"]
        else:
            safe_buffer = 0.6
            self.length = car_config["length"] + safe_buffer
            self.width = car_config["width"] + safe_buffer
        self.wheel_base = car_config["wheel_base"]
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
        
    def reset(self, state):
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

    def step(self, action=None):
        dt = self.delta_t
        x = self.shifted_x
        y = self.shifted_y
        theta = self.theta
        v = self.v
        steer = self.steer
        v_s = self.v_s
        gear = self.gear

        if not self.ego_car:
            a = 0
            Eps = 0
        else:
            assert not(action is None), \
                "action cant be None for ego_car in step function"
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

        w = (new_v * np.tan(steer) / self.wheel_base)
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
        self.v_s = v_s
        self.w = w
        self.gear = new_gear
        new_state = State(new_x, new_y, new_theta, new_v, steer)

        return new_state, overSpeeding, overSteering
    
    def getCurrentState(self):
        centred_x, centred_y = self.getCentredCoordinates()

        return State(self.shifted_x, self.shifted_y, 
                     self.theta, self.v, self.steer,
                     self.v_s, self.w, self.gear,
                     self.last_a, self.last_Eps,
                     centred_x, centred_y, 
                     self.width, self.length, self.wheel_base)

    def getCentredCoordinates(self):
        shift = self.wheel_base / 2
        centred_x = self.shifted_x + shift * cos(self.theta)
        centred_y = self.shifted_y + shift * sin(self.theta)

        return centred_x, centred_y
    
class ObsEnvironment(gym.Env):
    def __init__(self, full_env_name, config):
        self.gear_switch_penalty = True
        self.RS_reward = True
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
        self.validate_env = env_config["validate_env"]
        self.validateTestDataset = env_config["validateTestDataset"]
        if self.validate_env:
            print("DEBUG: VALIDATE ENV")
            if self.validateTestDataset:
                print("DEBUG: VALIDATE TEST DATASET")
        self.reward_config = config["reward_config"]
        self.goal = None
        self.current_state = None
        self.old_state = None
        self.last_action = [0., 0.]
        self.obstacle_segments = []
        self.dyn_obstacle_segments = []
        self.last_observations = []
        self.car_config = config['car_config']
        self.vehicle = Vehicle(self.car_config, ego_car=True)
        self.Tasks = config['Tasks']
        self.maps_init = config['maps']
        self.maps = dict(config['maps'])
        self.max_steer = env_config['max_steer']
        self.max_dist = env_config['max_dist']
        self.min_dist = env_config['min_dist']
        self.min_vel = env_config['min_vel']
        self.max_vel = env_config['max_vel']
        self.min_obs_v = env_config['min_obs_v']
        self.max_obs_v = env_config['max_obs_v']
        self.HARD_EPS = env_config['HARD_EPS']
        self.MEDIUM_EPS = env_config['MEDIUM_EPS']
        self.SOFT_EPS = env_config['SOFT_EPS']
        self.ANGLE_EPS = degToRad(env_config['ANGLE_EPS'])
        self.SPEED_EPS = env_config['SPEED_EPS']
        self.STEERING_EPS = degToRad(env_config['STEERING_EPS'])
        self.MAX_DIST_LIDAR = env_config['MAX_DIST_LIDAR']
        self.UPDATE_SPARSE = env_config['UPDATE_SPARSE']
        self.view_angle = degToRad(env_config['view_angle'])
        self.hard_constraints = env_config['hard_constraints']
        self.medium_constraints = env_config['medium_constraints']
        self.soft_constraints = env_config['soft_constraints']
        self.with_potential = env_config['reward_with_potential']
        self.frame_stack = env_config['frame_stack']
        self.use_acceleration_penalties = env_config['use_acceleration_penalties']
        self.use_velocity_goal_penalty = env_config['use_velocity_goal_penalty']
        self.use_different_acc_penalty = env_config['use_different_acc_penalty']
        self.max_episode_steps = env_config['max_polamp_steps']
        self.dyn_acc = 0
        self.dyn_ang_vel = 0
        self.collision_time = 0
        self.reward_weights = [
            self.reward_config["collision"],
            self.reward_config["goal"],
            self.reward_config["timeStep"],
            self.reward_config["distance"],
            self.reward_config["overSpeeding"],
            self.reward_config["overSteering"],
            self.reward_config["gearSwitchPenalty"]
        ]
        if self.use_acceleration_penalties:
            self.reward_weights.append(self.reward_config["Eps_penalty"])
            self.reward_weights.append(self.reward_config["a_penalty"])
        if self.use_velocity_goal_penalty:
            self.reward_weights.append(self.reward_config["v_goal_penalty"])
        if self.use_different_acc_penalty:
            self.reward_weights.append(self.reward_config["differ_a"])
            self.reward_weights.append(self.reward_config["differ_Eps"])
        assert self.hard_constraints \
            + self.medium_constraints \
            + self.soft_constraints == 1, \
            "only one constraint is acceptable"

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

    def getBB(self, state):
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

    def getRelevantSegments(self, state, with_angles=False):
        relevant_obstacles = []
        obstacles = list(self.obstacle_segments)
        obstacles.extend(self.dyn_obstacle_segments)
        for obst in obstacles:
            new_segments = []
            for segment in obst:
                d1 = math.hypot(state.x - segment[0].x, state.y - segment[0].y)
                d2 = math.hypot(state.x - segment[1].x, state.y - segment[1].y)
                new_segments.append((min(d1, d2), segment)) 
            new_segments.sort(key=lambda s: s[0])
            new_segments = [pair[1] for pair in new_segments[:2]]
            if not with_angles:
                relevant_obstacles.append(new_segments)
            else:
                new_segments_with_angle = []
                angles = []
                for segment in new_segments:
                    angle1 = math.atan2(segment[0].y - state.y, segment[0].x - state.x)
                    angle2 = math.atan2(segment[1].y - state.y, segment[1].x - state.x)
                    min_angle = min(angle1, angle2)
                    max_angle = max(angle1, angle2)
                    new_segments_with_angle.append(((min_angle, max_angle), segment))
                    angles.append((min_angle, max_angle))
                if angleIntersection(angles[0][0], angles[0][1], angles[1][0]) and \
                    angleIntersection(angles[0][0], angles[0][1], angles[1][1]):
                    relevant_obstacles.append([new_segments_with_angle[0]])
                elif angleIntersection(angles[1][0], angles[1][1], angles[0][0]) and \
                    angleIntersection(angles[1][0], angles[1][1], angles[0][1]):
                    relevant_obstacles.append([new_segments_with_angle[1]])
                else:
                    relevant_obstacles.append(new_segments_with_angle)
                    
        return relevant_obstacles

    def getDiff(self, state):
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
        v_s = self.vehicle.v_s
        w = self.vehicle.w
        a = self.vehicle.a
        Eps = self.vehicle.Eps
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

    def setTask(self, tasks, idx, obstacles, set_task_without_dynamic_obsts):
        assert len(tasks) > 0, \
            "incorrect count of tasks must be len(tasks) > 0 but given 0"
        i = np.random.randint(len(tasks)) if idx is None else idx
        current_task = tuple(tasks[i])
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
                    vehicle.reset(state)
                    self.vehicles.append(vehicle)
                    
        self.current_state, self.goal = self.transformTask(
                            current, goal, obstacles)
        self.old_state = self.current_state
        self.vehicle.reset(self.current_state)

    def reset(self, idx=None, val_key=None):
        self.maps = dict(self.maps_init)
        if not self.validate_env:
            self.obst_random_actions = np.random.choice([
                                        True, False, False, False, False])
        else:
            self.obst_random_actions = False
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
        self.vehicle.v_s = 0
        self.vehicle.w = 0
        self.vehicle.Eps = 0
        self.vehicle.a = 0
        self.vehicle.j_a = 0
        self.vehicle.j_Eps = 0
        self.vehicle.prev_a = 0
        self.vehicle.prev_Eps = 0
        self.collision_time = 0
        if self.RS_reward:
            self.new_RS = None
        self.vehicle.gear = None
        self.vehicle.prev_gear = None
        if self.validate_env:
            set_task_without_dynamic_obsts = False
            if self.validateTestDataset:
                self.stop_dynamic_step = 1200
                if val_key == "map0" or val_key == "map2":
                    self.stop_dynamic_step = 100
                elif val_key == "map1" or val_key == "map3":
                    self.stop_dynamic_step = 110
                elif val_key == "map6":
                    self.stop_dynamic_step = 200

        index = np.random.randint(len(self.lst_keys))
        self.map_key = self.lst_keys[index]
        self.obstacle_map = self.maps[self.map_key]
        tasks = self.Tasks[self.map_key]
        self.setTask(tasks, idx, self.obstacle_map, 
                     set_task_without_dynamic_obsts)

        for obstacle in self.obstacle_map:
            obs = State(None, None, theta=obstacle[2], v=0, steer=0, 
                        centred_x=obstacle[0], centred_y=obstacle[1], 
                        width=2*obstacle[3], length=2*obstacle[4])
            self.obstacle_segments.append(
                self.getBB(obs))
                
        self.start_dist = self.__goalDist(self.current_state)
        self.last_images = []
        self.grid_static_obst = None
        self.grid_agent = None
        self.grid_with_adding_features = None
        observation = self._getObservation(first_obs=True)
    
        return observation
    
    def __reward(self, current_state, new_state, goalReached, 
                collision, overSpeeding, overSteering):
        if not self.RS_reward:
            previous_delta = self.__goalDist(current_state)
            new_delta = self.__goalDist(new_state)
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
                if self.with_potential:
                    #reward.append((previous_delta - new_delta) / new_delta)
                    reward.append(previous_delta - new_delta)
                else:
                    reward.append(previous_delta - new_delta)
            reward.append(-1 if overSpeeding else 0)
            reward.append(-1 if overSteering else 0)
            if self.use_acceleration_penalties:
                reward.append(-abs(self.vehicle.Eps))
                reward.append(-abs(self.vehicle.a))
            if self.use_velocity_goal_penalty:
                if goalReached:
                    reward.append(-abs(new_state.v))
                else:
                    reward.append(0)
            if self.use_different_acc_penalty:
                reward.append(-abs(self.vehicle.a - self.vehicle.prev_a))
                reward.append(-abs(self.vehicle.Eps - self.vehicle.prev_Eps))
        else:
            reward.append(0)
            reward.append(0)
            reward.append(0)
            reward.append(0)
            if self.use_acceleration_penalties:
                reward.append(0)
                reward.append(0)
            if self.use_velocity_goal_penalty:
                if goalReached:
                    reward.append(0)
                else:
                    reward.append(0)
            if self.use_different_acc_penalty:
                reward.append(0)
                reward.append(0)
        if self.gear_switch_penalty:
            if not(self.vehicle.prev_gear is None) and \
               self.vehicle.prev_gear != self.vehicle.gear:
                reward.append(-1)
            else:
                reward.append(0)
        else:
            reward.append(0)

        return np.matmul(self.reward_weights, reward)

    def isCollision(self, state, min_beam, lst_indexes=[]):
        
        if (self.vehicle.min_dist_to_check_collision < min_beam):
            return False

        if len(self.obstacle_segments) > 0 or len(self.dyn_obstacle_segments) > 0:
            bounding_box = self.getBB(state)
            for i, obstacle in enumerate(self.obstacle_segments):
                if (intersectPolygons(obstacle, bounding_box)):
                    return True
                    
            for obstacle in self.dyn_obstacle_segments:
                mid_x = (obstacle[0][0].x + obstacle[1][1].x) / 2.
                mid_y = (obstacle[0][0].y + obstacle[1][1].y) / 2.
                distance = math.hypot(mid_x - state.x, mid_y - state.y)    
                #if (distance > (self.vehicle.min_dist_to_check_collision)):
                dyn_obst_corner_x = obstacle[0][0].x
                dyn_obst_corner_y = obstacle[0][0].y
                dyn_obst_radius = math.hypot(mid_x - dyn_obst_corner_x, 
                                             mid_y - dyn_obst_corner_y)
                #if (distance > (self.vehicle.car_radius + dyn_obst_radius)):
                #    continue
                if (distance > (self.vehicle.min_dist_to_check_collision + dyn_obst_radius)):
                    continue
                if (intersectPolygons(obstacle, bounding_box)):
                    return True
            
        return False

    def __goalDist(self, state):
        return math.hypot(self.goal.x - state.x, self.goal.y - state.y)

    def obst_dynamic(self, state, action, previous_v_s, constant_forward=True):
        a = action[0]
        Eps = action[1]
        dt = self.vehicle.delta_t
        if constant_forward:
            a = 0
            Eps = 0
        if self.validate_env and self.validateTestDataset and \
                self.stepCounter >= self.stop_dynamic_step:
            a = 0
            Eps = 0
        elif self.validate_env and self.validateTestDataset and \
                (self.stepCounter + 5) >= self.stop_dynamic_step:
            a = -1
            Eps = 0

        dV = a * dt
        V = state.v + dV
        overSpeeding = V > self.vehicle.max_vel or V < self.vehicle.min_vel
        
        if not constant_forward:
            V = np.clip(V, self.vehicle.min_vel, self.vehicle.max_vel)

        dv_s = Eps * dt
        v_s = previous_v_s + dv_s
        v_s = np.clip(v_s, -self.vehicle.max_ang_vel, self.vehicle.max_ang_vel)
        dsteer = v_s * dt
        steer = normalizeAngle(state.steer + dsteer)
        overSteering = abs(steer) > self.vehicle.max_steer
        steer = np.clip(steer, -self.vehicle.max_steer, self.vehicle.max_steer)

        w = (V * np.tan(steer) / self.vehicle.wheel_base)
        dtheta = w * dt
        theta = normalizeAngle(state.theta + dtheta)

        dx = V * np.cos(theta) * dt
        dy = V * np.sin(theta) * dt
        x = state.x + dx
        y = state.y + dy

        new_state = State(x, y, theta, V, steer)

        return new_state, overSpeeding, overSteering, v_s

    def step(self, action):        
        info = {}
        isDone = False
        new_state, overSpeeding, overSteering = \
                self.vehicle.step(action)

        for vehicle in self.vehicles:
            vehicle.step()
                        
        self.current_state = new_state
        self.last_action = [self.vehicle.a, self.vehicle.Eps]

        observation = self._getObservation()

        #collision
        start_time = time.time()
        temp_grid_obst = self.grid_static_obst + self.grid_dynamic_obst
        collision = temp_grid_obst[self.grid_agent == 1].sum() > 0
        collision = collision or (self.grid_agent.sum() == 0)
        end_time = time.time()
        self.collision_time += (end_time - start_time)
        
        distanceToGoal = self.__goalDist(new_state)
        info["EuclideanDistance"] = distanceToGoal
        if self.hard_constraints:
            goalReached = distanceToGoal < self.HARD_EPS and abs(
                normalizeAngle(new_state.theta - self.goal.theta)) < self.ANGLE_EPS \
                and abs(new_state.v - self.goal.v) <= self.SPEED_EPS
        elif self.medium_constraints:
            goalReached = distanceToGoal < self.MEDIUM_EPS and abs(
                normalizeAngle(new_state.theta - self.goal.theta)) < self.ANGLE_EPS
        elif self.soft_constraints:
            goalReached = distanceToGoal < self.SOFT_EPS

        if not self.validate_env:
            reward = self.__reward(self.old_state, new_state, 
                                goalReached, collision, overSpeeding, 
                                overSteering)
        else:
            reward = 0
        
        if not (self.stepCounter % self.UPDATE_SPARSE):
            self.old_state = self.current_state
        self.stepCounter += 1
        if goalReached or collision or (self.max_episode_steps == self.stepCounter):

            #test
            if self.stepCounter == 1:
                print("DEBUG 1 step episode:")
                print("agent pixels:", self.grid_agent.sum())
                temp_grid_obst = self.grid_static_obst + self.grid_dynamic_obst
                print("intersect pixels:", temp_grid_obst[self.grid_agent == 1].sum())

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

        self.vehicle.prev_a = self.vehicle.a
        self.vehicle.prev_Eps = self.vehicle.Eps
        
        return observation, reward, isDone, info

    def drawObstacles(self, vertices, color="-b"):
        a = vertices
        plt.plot([a[(i + 1) % len(a)][0].x for i in range(len(a) + 1)], 
                 [a[(i + 1) % len(a)][0].y for i in range(len(a) + 1)], color)

    def drawState(self, state, ax, color="-b", 
                  with_heading=False, heading_color='red'):
        vertices = self.getBB(state)
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
            '''
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
            road_width_ = 6
            bottom_left_right_dx_ = 0.25
            road_center_y = bottom_road_edge_y + road_width_
            upper_boundary_center_y = road_center_y + road_width_ + upper_boundary_width
            
            self.obstacle_map = [
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
            '''
            parking_height = 2.7
            parking_width = 4.5
            bottom_left_boundary_height = 6
            upper_boundary_height = 17
            upper_boundary_width = 0.5
            bottom_left_boundary_center_x = 5
            bottom_left_boundary_center_y = -5.5
            road_width_ = 6
            bottom_left_right_dx_ = 0.25
            self.obstacle_map = getValetStaticObstsAndUpdateInfo(
                        parking_height, parking_width, 
                        bottom_left_boundary_height, 
                        upper_boundary_width, upper_boundary_height,
                        bottom_left_boundary_center_x, 
                        bottom_left_boundary_center_y, road_width_,
                        bottom_left_right_dx_)
            for obstacle in self.obstacle_map:
                obs = State(obstacle[0], obstacle[1], obstacle[2], 0, 0)
                width = obstacle[3]
                length = obstacle[4]
                self.obstacle_segments.append(
                    self.getBB(obs, width=width, length=length, ego=False))
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
            vertices = self.getBB(vehicle.getCurrentState())
            self.normalized_dynamic_boxes.append(
                    [Point(max(0, pair_[0].x - self.normalized_x_init), 
                     max(0, pair_[0].y - self.normalized_y_init)) 
                     for pair_ in vertices])

        vertices = self.getBB(self.vehicle.getCurrentState())
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
            adding_features = self.getDiff(self.current_state)
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