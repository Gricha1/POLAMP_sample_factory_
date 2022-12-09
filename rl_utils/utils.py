from EnvLib.utils import *
import numpy as np
from math import cos, sin, tan
from math import pi

# test
from rl_utils.rl_utils import create_task
import json

class apollo_roi_boundary_data:
    def __init__(self):
        pass
class Vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def parse_cpp_request(request):
    request_ = {}
    print("debug cpp request", request)
    
    # create apollo like data
    data = apollo_roi_boundary_data()
    points = []
    for key in request:
        val = request[key]
        if key[0] == "x":
            x = float(val)
        elif key[0] == "y":
            y = float(val)
            points.append(Vec2d(x, y))
    data.point = points

    task = create_task(data)
    print("debug generated task")
    print("task:", task)
    request_ = {}
    request_["task"] = task
    
    return request_

def transformFuncImageToFunc(movement_func_image):
    def movement_func(last_state, current_steps, time_step=0.1, **args):
        a = 0
        Eps = 0

        action = [0, 0]
        return action 

    return movement_func

def transformHTTPRequestToTask(request_task):
    static_obsts, start_goal, dynamic_obsts = request_task

    new_dynamic_obsts = []
    for dynamic_obst in dynamic_obsts:
        new_dynamic = [dynamic_obst[0], dynamic_obst[1], 
                       dynamic_obst[2], dynamic_obst[3], 
                       dynamic_obst[4]]
        new_dynamic_config = dynamic_obst[5]
        new_dynamic_config["movement_func"] = transformFuncImageToFunc(
                                                new_dynamic_config["movement_func"])
        new_dynamic.append(new_dynamic_config)
        new_dynamic_obsts.append(new_dynamic)
    
    task = (static_obsts, start_goal, new_dynamic_obsts)

    return [task]

def normalizeFromZeroTo2Pi(theta):
    if theta > 2 * pi:
        theta = theta % (2 * pi)
    if theta < 0:
        theta = 2 * pi - theta % (2 * pi)

    return theta

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
                                     dynamic_config = self.movement_func_params)

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