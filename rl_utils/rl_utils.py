from math import pi
from numpy import sqrt

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

    left_bottom = [
                    (roi_boundaries[1].x + roi_boundaries[0].x) / 2, 
                    (roi_boundaries[1].y + roi_boundaries[2].y) / 2,
                    0,
                    (roi_boundaries[1].y - roi_boundaries[2].y) / 2,
                    (roi_boundaries[1].x - roi_boundaries[0].x) / 2
                  ]

    down_bottom = [
                    (roi_boundaries[3].x + roi_boundaries[2].x) / 2, 
                    roi_boundaries[2].y - 2,
                    0,
                    2,
                    (roi_boundaries[3].x - roi_boundaries[2].x) / 2 \
                        + roi_boundaries[5].x - roi_boundaries[4].x
                  ]

    right_bottom = [
                    (roi_boundaries[5].x + roi_boundaries[4].x) / 2, 
                    (roi_boundaries[3].y + roi_boundaries[4].y) / 2,
                    0,
                    (roi_boundaries[4].y - roi_boundaries[3].y) / 2,
                    (roi_boundaries[5].x - roi_boundaries[4].x) / 2
                   ]
    #DEBUG
    top = [(roi_boundaries[7].x + roi_boundaries[6].x) / 2, 
            roi_boundaries[7].y,
            0,
            2,
            (roi_boundaries[6].x - roi_boundaries[7].x) / 2]

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
        # correct vector of dynamic velocity sign
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
    generated_goal = [parking_pos[0], parking_pos[1] - 2.2, 90 * (pi / 180), 0, 0]
    generated_task = [generated_start, generated_goal]
    if len(dynamic_obsts) != 0:
        generated_task.append(dynamic_obsts)
    map_ = {}
    task = {}
    map_["map0"] = generated_map
    task["map0"] = [generated_task]
 
    return map_, task, generated_goal