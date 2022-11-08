import math
import numpy as np
import networkx as nx

from utils.graph_utils import *

def get_pixel_location(pixel, points, normalized=True):
    focus_length1 = 400 / (2 * math.tan(100 * math.pi / 360))
    focus_length2 = 300 / (2 * math.tan(100 * math.pi / 360))
    intrinsic_mat = np.array([[focus_length1, 0, 200], [0, focus_length2, 150], [0, 0, 1]])
    extrinsic_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    p3d = np.dot(np.linalg.inv(intrinsic_mat), np.array([[pixel[0]], [pixel[1]], [1.0]]))
    p3d[1] -= -0.2  # camera is below lidar sensor

    points = np.matmul(extrinsic_mat, points.T).T
    points = points[np.where(points[:, 2] > 0.1)]

    if len(points) == 0:
        return None

    # get closest point
    pc_xy = points[:, 0:2]
    xy = p3d[:2].transpose()

    if normalized:
        pc_z = points[:, 2]
        normalized_pc = pc_xy / pc_z[:, None]
        dist = np.sum((normalized_pc - xy) ** 2, axis=1)
        closest_index = np.argmin(dist)
        location = points[closest_index]
        p3d *= location[2]
        pixel_location = np.matmul(np.linalg.inv(extrinsic_mat), p3d).T.reshape(-1)
    else:
        dist = np.sum((pc_xy - xy) ** 2, axis=1)
        closest_index = np.argmin(dist)
        location = points[closest_index]
        p3d[2] = location[2]
        pixel_location = np.matmul(np.linalg.inv(extrinsic_mat), p3d).T.reshape(-1)

    return pixel_location

def should_stop(result_td, result_od, point_cloud, vehicle_pos, compass,
                H, traj_node_id, partial_traj, vehicle_st_partial, obs_dist=5.5):
    stop = is_red_light(result_td, vehicle_pos, H) or \
           is_obstacle(result_od, point_cloud, vehicle_pos, compass,
                       H, traj_node_id, partial_traj, vehicle_st_partial, obs_dist=obs_dist)
    return stop

def is_red_light(obs_result, vehicle_pos, H):
    # find the nearest node to the vehicle
    pos_list_key = np.stack(list(dict(H.nodes('pos')).keys()))
    pos_list = np.stack(list(dict(H.nodes('pos')).values()))
    v_nearest_idx = np.argmin(np.sum((pos_list - vehicle_pos.reshape([1,-1]))**2, axis=1))
    v_nearest_node_id = pos_list_key[v_nearest_idx]
    if H.nodes[v_nearest_node_id]['on_traffic_signal'] is not None:
        return False

    is_red, dist = False, np.inf
    for obj in obs_result:
        bbox = obj.bounding_box_2D
        x = (bbox.x_min + bbox.x_max) // 2
        y = (bbox.y_min + bbox.y_max) // 2
        new_dist = (x - 200) ** 2 + (y - 150) ** 2
        if new_dist < dist:
            is_red = (obj.state.name == 'RED' or obj.state.name == 'YELLOW')
            dist = new_dist
    return is_red

def is_obstacle(obs_result, point_cloud, vehicle_pos, compass,
                H, traj_node_id, partial_traj, vehicle_st_partial, obs_dist=5.5):
    # find the nearest node to the vehicle
    cos_theta = np.cos(compass).reshape([-1, 1])
    sin_theta = np.sin(compass).reshape([-1, 1])
    R = np.concatenate([cos_theta, -sin_theta, sin_theta, cos_theta], axis=1).reshape(
        [2, 2])

    pos_list_key = np.stack(list(dict(H.nodes('pos')).keys()))
    pos_list = np.stack(list(dict(H.nodes('pos')).values()))

    # get obstacle locations
    obs_locations = []
    for obj in obs_result:
        bbox = obj.bounding_box_2D
        x = (bbox.x_min + bbox.x_max) // 2
        y = (bbox.y_min + bbox.y_max) // 2

        location = get_pixel_location([x, y], point_cloud)
        if location is not None:
            obs_locations.append(location)

    # check collision
    collision = False
    for ob_pos in obs_locations:
        ob_pos_xy = ob_pos[:2]
        ob_pos_rot = vehicle_pos + np.matmul(ob_pos_xy, R)  # apply diff_rot

        nearest_idx = np.argmin(np.sum((pos_list - ob_pos_rot.reshape([1,-1]))**2, axis=1))
        if np.sum((pos_list[nearest_idx] - ob_pos_rot)**2, axis=0) > 10:
            continue
        nearest_node_id = pos_list_key[nearest_idx]

        on_path = (nearest_node_id in traj_node_id)
        if 'contraction' in H.nodes[nearest_node_id]:
            on_path = on_path or any([(contraction in traj_node_id) for contraction in H.nodes[nearest_node_id]['contraction']])

        ob_st_partial = cartesian_to_frenet_approx(partial_traj, ob_pos_rot.reshape(1, 2)).squeeze()

        s_dist = ob_st_partial.squeeze()[0] - vehicle_st_partial.squeeze()[0]
        if (0 < s_dist < obs_dist) and on_path:
            collision = True
            break
    return collision