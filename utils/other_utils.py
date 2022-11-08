import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

from utils.graph_utils import *
from utils.gnss_utils import *

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def get_current_partial_traj(prev_st, path_traj, path_progress, margin_before=10, margin_after=50):
    prev_progress = prev_st[0]

    cond1 = path_progress >= prev_progress - margin_before
    cond2 = path_progress < prev_progress + margin_after
    margin_filter = np.logical_and(cond1, cond2)

    partial_traj = path_traj[margin_filter]
    start_progress = path_progress[margin_filter][0]

    return partial_traj, start_progress

def route_completion_test(global_plan_pos, traj, path_progress, min_dist_s=20, min_dist_t=5):
    plan_idx = 0
    prev_pos_st = np.zeros(2) #global_plan_pos[plan_idx]

    for n, pos in enumerate(traj[1:]):
        partial_traj, start_progress = get_current_partial_traj(prev_pos_st, traj, path_progress)
        vehicle_st_partial = cartesian_to_frenet_approx(partial_traj, pos.reshape(1, 2)).squeeze()
        plan_pos_st_partial = cartesian_to_frenet_approx(partial_traj, global_plan_pos[plan_idx].reshape(1, 2)).squeeze()

        dist_s = abs(vehicle_st_partial[0] - plan_pos_st_partial[0])
        dist_t = abs(vehicle_st_partial[1] - plan_pos_st_partial[1])
        if dist_s <= min_dist_s and dist_t <= min_dist_t:
            plan_idx += 1
            if plan_idx == len(global_plan_pos):
                break

        prev_pos_st = vehicle_st_partial.copy()
        prev_pos_st[0] += start_progress

    completion = plan_idx / len(global_plan_pos)
    return completion

def get_waypoint_command(global_plan_pos, vehicle_pos, traj, path_progress, plan_start_idx):
    plan_idx = 0
    prev_pos_st = np.array([path_progress[plan_start_idx], 0])

    target_point_list = []

    for n, pos in enumerate(vehicle_pos):
        partial_traj, start_progress = get_current_partial_traj(prev_pos_st, traj, path_progress)
        vehicle_st_partial = cartesian_to_frenet_approx(partial_traj, pos.reshape(1, 2)).squeeze()
        plan_pos_st_partial = cartesian_to_frenet_approx(partial_traj,
                                                         global_plan_pos[plan_idx].reshape(1, 2)).squeeze()

        if vehicle_st_partial[0] >= plan_pos_st_partial[0]:
            plan_idx += 1

            if plan_idx == len(global_plan_pos):
                target_point_list.append(global_plan_pos[plan_idx - 1])
                break

        target_point_list.append(global_plan_pos[plan_idx])

        prev_pos_st = vehicle_st_partial.copy()
        prev_pos_st[0] += start_progress

    if len(vehicle_pos) > len(target_point_list):
        last_plan_pos = target_point_list[-1].copy()
        target_point_list += [last_plan_pos] * (len(vehicle_pos) - len(target_point_list))

    target_point_list = np.array(target_point_list)

    return target_point_list