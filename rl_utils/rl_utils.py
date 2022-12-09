from math import pi
from numpy import sqrt
import json
from dataset_generation.utlis import *

# test previous 
#with open("modules/tools/valet_parking_rl/POLAMP_sample_factory_/configs/car_configs.json", 'r') as f:
#    car_config = json.load(f)
with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

ONE_RAD_GRAD = pi / 180

class point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def degToRad(deg):
    return deg * ONE_RAD_GRAD

def generate_static_obsts_from_roi_points(roi_boundaries, car_config):
    '''
        roi_boundaries[0] = (roi_boundaries[0].x, roi_boundaries[0].y)
        vehicle_pos = (vehicle_pos.x, vehicle_pos.y)

        create horzontal task:

                            top
        -----------------------------------------------------

                                            first_goal
          start                             
        -------------------      -----------------------------   
         left_bottom        |   |     right_bottom
                            |   |
                            -----
                          down_bottom
    '''         
    # parking_height * parking_width = parking area
    assert roi_boundaries[3].x > roi_boundaries[2].x, \
        "parking height is not corrent from roi boundary points"
    parking_height = roi_boundaries[3].x - roi_boundaries[2].x
    assert roi_boundaries[1].y > roi_boundaries[2].y, \
        "parking width is not corrent from roi boundary points"
    parking_width = roi_boundaries[1].y - roi_boundaries[2].y
    trained_parking_height = 2.7 + 0.6
    trained_parking_width = 4.5
    shift_left_boundary_x = 0 # shift because use trained_parking_height
    if trained_parking_height < parking_height:
        shift_left_boundary_x = (parking_height - trained_parking_height) / 2
        parking_height = trained_parking_height

    parking_height = trained_parking_height
    #if parking_width < car_config["width"] + 0.1:
    #    parking_width = trained_parking_width

    bottom_left_boundary_height = 6 # any value
    upper_boundary_width = 0.5 # any value
    upper_boundary_height = 17
    bottom_left_boundary_width = parking_width / 2
    bottom_left_boundary_center_x = roi_boundaries[1].x - 6 + shift_left_boundary_x
    bottom_left_boundary_center_y = roi_boundaries[0].y - bottom_left_boundary_width
    assert roi_boundaries[7].y > roi_boundaries[0].y, \
            "road width is not corrent from roi boundary points"
    road_width_ = (roi_boundaries[7].y - roi_boundaries[0].y) / 2

    staticObstsInfo = {}
    map_ = getValetStaticObstsAndUpdateInfo(
        parking_height, parking_width, bottom_left_boundary_height, 
        upper_boundary_width, upper_boundary_height,
        bottom_left_boundary_center_x, bottom_left_boundary_center_y, 
        road_width_,
        staticObstsInfo
    )

    return map_, staticObstsInfo

def create_task(data):
    data = list(data.point)
    obsts = data[:-12]
    data = data[-12:]
    data.pop(2)
    data.pop(3)
    roi_boundary_points = [point(data[i].x, data[i].y) 
                    for i in range(len(data) - 2)]
    parking_space = [point_ for point_ in data[1:5]]
    vehicle_pos = point(data[-2].x, data[-2].y)
    parking_pos = [(data[4].x + data[1].x) / 2, 
                (data[1].y + data[2].y) / 2 + 1]
    origin_point_x = data[-1].x
    origin_point_y = data[-1].y
    dynamic_obsts = []
    for obst in obsts:
        x = obst.x
        y = obst.y
        normalized_x = x - origin_point_x
        normalized_y = y - origin_point_y
        theta = obst.theta
        v_x = obst.v_x
        v_y = obst.v_y
        assert theta >= 0 and theta <= 2 * pi, \
            f"theta for dynamic obst isn't corrent, \
                theta is {theta} must be 0 <= <= 2pi" 
        v = sqrt(v_x * v_x + v_y * v_y)
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
        dynamic_obsts.append([normalized_x, 
                              normalized_y, 
                              theta, v, 0, 
                              dynamic_config])

    print("****************************************")
    print("NEW MASSAGE")
    print("dynamic obstacles:", dynamic_obsts)
    print(f"vehicle pose is: {vehicle_pos.x}, {vehicle_pos.y}")
    print("roi boundary points:")
    for p in roi_boundary_points:
        print("x:", p.x, "y:", p.y)

    generated_map, staticObstsInfo = generate_static_obsts_from_roi_points(
                                                            roi_boundary_points,
                                                            car_config)

    generated_start = [vehicle_pos.x, vehicle_pos.y, 0, 0., 0]
    #generated_goal = [parking_pos[0], parking_pos[1] - car_config["wheel_base"] / 2, 
    #                  90 * (pi / 180), 0, 0]
    generated_goal = [parking_pos[0], 
                      staticObstsInfo["buttom_road_edge_y"] \
                      - car_config["width"] / 2
                      - car_config["wheel_base"] / 2, 
                      90 * (pi / 180), 0, 0]
    generated_task = [generated_start, generated_goal]

    # test task conditions
    y_left_bottom_obst_edge = staticObstsInfo["bottom_left_boundary_center_y"] \
                              + staticObstsInfo["bottom_left_boundary_width"]
    y_upper_obst_edge = staticObstsInfo["upper_boundary_center_y"] \
                              - staticObstsInfo["upper_boundary_width"]
    assert vehicle_pos.y > y_left_bottom_obst_edge, \
                    f"incorrect task: vehicle y is: {vehicle_pos.y} but " \
                    + f"left bottom obst edge y is: {y_left_bottom_obst_edge}"
    assert vehicle_pos.y < y_upper_obst_edge, \
                    f"incorrect task: vehicle y is: {vehicle_pos.y} but " \
                    + f"left bottom obst edge y is: {y_upper_obst_edge}"
    
    return (generated_map, generated_task, dynamic_obsts)
