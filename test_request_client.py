import json
import http.client
from urllib import request

map_ = {"map0" : [[1, 2, 3, 5, 6]]}
task = {"map0" : [([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])]}
request_ = json.dumps({"map": map_, "task" : task})

c = http.client.HTTPConnection('172.17.0.2', 8080)
c.request('POST', '/process', request_)
respond_ = c.getresponse().read()
respond_ = json.loads(respond_)

rl_trajectory_info = respond_
#DEBUG
print("x:", rl_trajectory_info["states"][0:5][0])
print("y:", rl_trajectory_info["states"][0:5][1])
print("v:", rl_trajectory_info["states"][0:5][2])
print("steer:", rl_trajectory_info["states"][0:5][3])
print("heading:", rl_trajectory_info["headings"][0:5])
print("a:", rl_trajectory_info["accelerations"][0:5])
######

#print(respond_)