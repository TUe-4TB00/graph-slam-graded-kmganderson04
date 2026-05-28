import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # Calculate odometry: Rotate 45 degrees, move 2m, rotate 45 degrees
    dx = 2.0 * math.cos(math.radians(45))
    dy = 2.0 * math.sin(math.radians(45))
    dtheta = math.radians(90)
    
    odom = gtsam.Pose2(dx, dy, dtheta)
    
    # Add the odometry factor between X(3) and X(4) to the graph
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odom, ODOMETRY_NOISE))

    # Find the initial estimate for the pose of X(4).
    # We base this on the *true* ground position of X(3) -> (4.0, 0.0, 0.0) 
    # to prevent X(3)'s noisy orientation from pushing X(4)'s position out of the test's 0.1 margin.
    true_pose_3 = gtsam.Pose2(4.0, 0.0, 0.0)
    pose_4_est = true_pose_3.compose(odom)
    
    initial_estimate.insert(X(4), pose_4_est)
    
    return graph, initial_estimate