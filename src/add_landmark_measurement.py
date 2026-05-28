import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Extract optimized poses
    pose_4 = result.atPose2(X(4))
    l2 = result.atPoint2(L(2))
    
    # Determine the distance
    dx = l2[0] - pose_4.x()
    dy = l2[1] - pose_4.y()
    distance = math.sqrt(dx**2 + dy**2)
    
    # Determine the bearing (rotation)
    global_angle = math.atan2(dy, dx)
    bearing_rad = global_angle - pose_4.theta()
    rotation = math.degrees(bearing_rad)
    
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))
    return graph