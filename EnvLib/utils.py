from math import pi
from math import fmod
from .line import Segment
from .line import Point

ONE_RAD_GRAD = pi / 180
ONE_GRAD_RAD = 180. / pi

def normalizeAngle(angle, symmetric=True):
    begin = pi if symmetric else 0
    norm_angle = fmod(angle + begin, 2*pi)
    if norm_angle < 0:
        norm_angle += 2*pi
    return norm_angle - begin

def angleIntersection(angle1, angle2, angle):
    if angle1 == angle or angle2 == angle:
        return True

    if angle1 * angle2 > 0:
        if (angle <= angle1 or angle >= angle2):
            return False
    else:
        dist1 = angle2 - angle1
        if dist1 > math.pi:
            if (angle < 0 and angle >= angle1):
                return False
            if (angle > 0 and angle <= angle2):
                return False
        else:
            if (angle <= angle1 or angle >= angle2):
                return False
    return True

def fromVerticesToSegments(vertices):
    n = len(vertices)
    segments = []
    for i in range(n):
        if i < n-1:
            segments.append(Segment(vertices[i], vertices[i+1]))
        elif n > 2:
            segments.append(Segment(vertices[i], vertices[0]))
    return segments


def degToRad(deg):
    return deg * ONE_RAD_GRAD


def radToDeg(rad):
    return rad * ONE_GRAD_RAD


def mToKm(vel):
    return vel * 3.6


def kmToM(vel):
    return vel / 3.6

import math
import numpy as np
import matplotlib.pylab as plt

def separatingAxes(a, axes):
    for i in range(len(a)):
        current = a[i]
        next = a[(i + 1) % len(a)]
        edge = np.array(next) - np.array(current) 
        new_edge = edge / (np.sqrt(np.sum(edge ** 2)) + 1e-6)
        axes.append([-new_edge[1], new_edge[0]])

def project(a, axis):
    maxProj = -math.inf
    minProj = math.inf
    for v in a:
        proj = np.dot(axis, v)
        if proj < minProj:
            minProj = proj
        if proj > maxProj:
            maxProj = proj
    
    return minProj, maxProj

def intersectPolygons(a, b, rl=True):
    axes = []
    new_a = []
    if rl:
        for pair in a:
            new_a.append((pair[0].x, pair[0].y))
        a = new_a
        new_b = []
        for pair in b:
            new_b.append((pair[0].x, pair[0].y))
        b = new_b 
    separatingAxes(a, axes)
    separatingAxes(b, axes)
    for axis in axes:
        aMinProj, aMaxProj, bMinProj, bMaxProj = 0., 0., 0., 0.
        aMinProj, aMaxProj = project(a, axis)
        bMinProj, bMaxProj = project(b, axis)
        if (aMinProj > bMaxProj) or (bMinProj > aMaxProj):
            return False 
    return True

def intersectPoint(point, polygon, epsilon=1e-4):
    result = False
    i = 0
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if ((polygon[i][1] > point[1]) != (polygon[j][1] > point[1]) and (point[0] < (polygon[j][0]\
            - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1] + epsilon) + polygon[i][0])):
            result = not result
        j = i
        i += 1

    return result
