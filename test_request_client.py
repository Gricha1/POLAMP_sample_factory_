import json
import http.client
from urllib import request

def main():
    #map_ = {"map0" : [[1, 2, 3, 5, 6]]}
    #task = {"map0" : [([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])]}
    #request_ = json.dumps({"map": map_, "task" : task})

    test_static = [1, 2, 3, 5, 6]
    test_start, test_goal = [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]
    dynamic_config = {}
    test_movement_func_image = {}
    movement_func_params = {}
    max_vel = 1
    width = 2.3
    length = 4
    dynamic_config["movement_func"] = test_movement_func_image
    dynamic_config["movement_func_params"] = movement_func_params
    dynamic_config["boundary_v"] = max_vel
    dynamic_config["width"] = width
    dynamic_config["length"] = length

    test_dynamic = [1, 2, 3, 4, 5, dynamic_config]
    #task = ([test_static], [test_start, test_goal], [test_dynamic])
    task = ([test_static], [test_start, test_goal], [])
    request_ = json.dumps({"task" : task})

    c = http.client.HTTPConnection('172.17.0.2', 8080)
    c.request('POST', '/process', request_)
    respond_ = c.getresponse().read()
    respond_ = json.loads(respond_)

    rl_trajectory_info = respond_["trajectory"]
    rl_run_info = respond_["run_info"]
    count_of_points = len(rl_trajectory_info["x"])
    output_count = min(5, count_of_points)

    print("x:", rl_trajectory_info["x"][0:output_count])
    print("y:", rl_trajectory_info["y"][0:output_count])
    print("v:", rl_trajectory_info["v"][0:output_count])
    print("steer:", rl_trajectory_info["steer"][0:output_count])
    print("headin:", rl_trajectory_info["heading"][0:output_count])
    print("a:", rl_trajectory_info["accelerations"][0:output_count])

if __name__ == "__main__":
    main()