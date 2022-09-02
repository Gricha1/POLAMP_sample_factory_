from math import pi
from numpy import sqrt
import json

with open("modules/tools/valet_parking_rl/POLAMP_sample_factory_/configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

ONE_RAD_GRAD = pi / 180

class point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def degToRad(deg):
    return deg * ONE_RAD_GRAD

def generate_map(roi_boundaries, vehicle_pos, parking_pos):
    '''
        return list of [left_bottom, right_bottom, top]
    '''
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
    trained_parking_height = 2.7
    trained_parking_width = 4.5
    shift_left_boundary_x = 0 # shift because use trained_parking_height
    if trained_parking_height < parking_height:
        shift_left_boundary_x = (parking_height - trained_parking_height) / 2
        parking_height = trained_parking_height
    shift_left_boundary_width = 0
    if trained_parking_width < parking_width:
        parking_width = trained_parking_width
    bottom_left_boundary_height = 6 # any value
    upper_boundary_width = 0.5 # any value
    upper_boundary_height = 17
    bottom_left_boundary_width = parking_width / 2
    bottom_right_boundary_width = parking_width / 2
    bottom_right_boundary_height = bottom_left_boundary_height
    bottom_left_boundary_center_x = roi_boundaries[1].x - 6 + shift_left_boundary_x
    bottom_left_boundary_center_y = roi_boundaries[0].y - bottom_left_boundary_width
    bottom_right_boundary_center_x = bottom_left_boundary_center_x \
                + bottom_left_boundary_height + parking_height \
                + bottom_right_boundary_height
    bottom_right_boundary_center_y = bottom_left_boundary_center_y
    bottom_road_edge_y = bottom_left_boundary_center_y + \
                    bottom_left_boundary_width 
    bottom_down_width = upper_boundary_width 
    bottom_down_height = parking_height / 2 
    upper_boundary_center_x = bottom_left_boundary_center_x \
                    + bottom_left_boundary_height + parking_height / 2
    bottom_down_center_x = upper_boundary_center_x
    bottom_down_center_y = bottom_left_boundary_center_y \
                - bottom_left_boundary_width - bottom_down_width \
                - 0.2 # dheight
    assert roi_boundaries[7].y > roi_boundaries[0].y, \
            "road width is not corrent from roi boundary points"
    road_width_ = (roi_boundaries[7].y - roi_boundaries[0].y) / 2
    road_center_y = bottom_road_edge_y + road_width_
    upper_boundary_center_y = road_center_y + road_width_ + \
                                upper_boundary_width
    bottom_left_right_dx_ = -0.3

    left_bottom = [
                    bottom_left_boundary_center_x + bottom_left_right_dx_, 
                    bottom_left_boundary_center_y, 0, bottom_left_boundary_width,
                    bottom_left_boundary_height
                  ]
    down_bottom = [
                    bottom_down_center_x, bottom_down_center_y, 
                    0, bottom_down_width, bottom_down_height
                  ]
    right_bottom = [
                    bottom_right_boundary_center_x - bottom_left_right_dx_, 
                    bottom_right_boundary_center_y, 0, bottom_right_boundary_width, 
                    bottom_right_boundary_height
                   ]
    top = [
            upper_boundary_center_x, upper_boundary_center_y, 
            0, upper_boundary_width, upper_boundary_height
          ]
  
    start = [vehicle_pos.x, vehicle_pos.y, 
            0, 0., 0]

    print("DEBUG contrains:")
    print("left_bottom: ", "x_c:", left_bottom[0], 
          "y_c", left_bottom[1], "theta:", left_bottom[2],
          "width:", left_bottom[3], "height:", left_bottom[4])
    print("down_bottom: ", "x_c:", down_bottom[0], 
          "y_c", down_bottom[1], "theta:", down_bottom[2],
          "width:", down_bottom[3], "height:", down_bottom[4])
    print("right_bottom: ", "x_c:", right_bottom[0], 
          "y_c", right_bottom[1], "theta:", right_bottom[2],
          "width:", right_bottom[3], "height:", right_bottom[4])
    print("top: ", "x_c:", top[0], 
          "y_c", top[1], "theta:", top[2],
          "width:", top[3], "height:", top[4])

    return [left_bottom, down_bottom, right_bottom, top]

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
        dynamic_obsts.append([normalized_x, 
                              normalized_y, 
                              theta, v, 0])

    print("****************************************")
    print("NEW MASSAGE")
    print("dynamic obstacles:", dynamic_obsts)
    print(f"vehicle pose is: {vehicle_pos.x}, {vehicle_pos.y}")
    print("roi boundary points:")
    for p in roi_boundary_points:
        print("x:", p.x, "y:", p.y)

    generated_map = generate_map(roi_boundary_points, vehicle_pos, parking_pos)
    generated_start = [vehicle_pos.x, vehicle_pos.y, 0, 0., 0]
    generated_goal = [parking_pos[0], parking_pos[1] - car_config["wheel_base"] / 2, 
                        90 * (pi / 180), 0, 0]
    generated_task = [generated_start, generated_goal]
    if len(dynamic_obsts) != 0:
        generated_task.append(dynamic_obsts)
    map_ = {}
    task = {}
    map_["map0"] = generated_map
    task["map0"] = [generated_task]
 
    return map_, task, generated_goal