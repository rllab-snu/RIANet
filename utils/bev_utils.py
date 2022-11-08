import numpy as np
import math
import carla
import cv2

from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.special import comb
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import linear_sum_assignment
from scipy import interpolate

from sklearn.metrics import average_precision_score, precision_recall_curve


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import networkx as nx

from .graph_utils import *

#####################################
#    point cloud (RIAN) :           #
#              ^                    #
#              | y                  #
#         <-----      upper:z       #
#            x                      #
#####################################
#      world  :                     #
#              ^                    #
#              | x                  #
#         <-----      upper:z       #
#            y                      #
#####################################
#      cam  :                       #
#            u                      #
#         <-----                    #
#              | v    depth:z       #
#              v                    #
#####################################
#   cam (RIAN) :                    #
#                u                  #
#              ----->               #
#            v |      depth:z       #
#              v                    #
#####################################
color_code = {}
color_code['Others'] = mcolors.hex2color(mcolors.CSS4_COLORS['black'])  # Other
color_code['LM-W'] = mcolors.hex2color(mcolors.CSS4_COLORS['white'])  # lane marker white
color_code['LM-Y'] = mcolors.hex2color(mcolors.CSS4_COLORS['yellow'])  # lane marker yellow (center)
color_code['DA-O'] = mcolors.hex2color(mcolors.CSS4_COLORS['purple'])  # other lane
color_code['DA-LL'] = mcolors.hex2color(mcolors.CSS4_COLORS['red'])  # left left lane
color_code['DA-L'] = mcolors.hex2color(mcolors.CSS4_COLORS['orange'])  # left lane
color_code['DA-C'] = mcolors.hex2color(mcolors.CSS4_COLORS['yellowgreen'])  # center lane
color_code['DA-R'] = mcolors.hex2color(mcolors.CSS4_COLORS['green'])  # right lane
color_code['DA-RR'] = mcolors.hex2color(mcolors.CSS4_COLORS['blue'])  # right right lane
color_code['DA-Int'] = mcolors.hex2color(mcolors.CSS4_COLORS['grey'])  # intersection
color_code['DA-LO'] = mcolors.hex2color(mcolors.CSS4_COLORS['pink'])  # left over intersection
color_code['DA-RO'] = mcolors.hex2color(mcolors.CSS4_COLORS['cyan'])  # right over intersection
for n, c in enumerate(color_code.keys()):
    color_code[c] = (list((255 * np.array(color_code[c])).astype('uint8')), n)

color_code_DA = {}
color_code_DA['Others'] = mcolors.hex2color(mcolors.CSS4_COLORS['black'])  # Other
color_code_DA['LM-W'] = mcolors.hex2color(mcolors.CSS4_COLORS['white'])  # lane marker white
color_code_DA['LM-Y'] = mcolors.hex2color(mcolors.CSS4_COLORS['yellow'])  # lane marker yellow (center)
color_code_DA['DA'] = mcolors.hex2color(mcolors.CSS4_COLORS['grey'])  # driving area
for n, c in enumerate(color_code_DA.keys()):
    color_code_DA[c] = (list((255 * np.array(color_code_DA[c])).astype('uint8')), n)

NUM_COLOR = len(color_code)

def graph_to_bev_encoder(H, v_pos, theta, resolution=256, x_max=24, y_max=48, cutoff_graph=True):
    H = H.copy()
    for key in H.nodes():
        rot = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        node_pos = (rot @ (H.nodes[key]['pos'] - v_pos).T)
        node_pos = np.stack([node_pos[1], node_pos[0]], axis=0)
        node_pos = np.array([-resolution / (2 * x_max), -resolution / y_max]) * node_pos \
                   + np.array([resolution / 2, resolution])
        H.nodes[key]['pos'] = node_pos

    in_img_range = []
    feature_map = np.zeros([resolution, resolution, 19])
    for key in H.nodes():
        p = H.nodes[key]['pos']
        if p[0] >= 0 and p[0] < resolution and p[1] >= 0 and p[1] < resolution:
            p_uint = p.astype('int')
            feature_map[p_uint[1], p_uint[0], 0] = 1
            in_img_range.append(key)

            out_edges = list(H.out_edges(key))
            out_edges_theta = [H.nodes[e2]['pos'] - H.nodes[key]['pos'] for _, e2 in out_edges]
            out_edges_theta = [np.arctan2(t[1], t[0]) for t in out_edges_theta]
            out_edges_dict = {e: t for e, t in zip(out_edges, out_edges_theta)}
            out_edges = [k for k, _ in sorted(out_edges_dict.items(), key=lambda item: item[1])]
            for n, (_, e2) in enumerate(out_edges[:6]):
                k = 1 + 3 * n
                feature_map[p_uint[1], p_uint[0], k] = 1
                direc = (H.nodes[e2]['pos'] - H.nodes[key]['pos'])
                feature_map[p_uint[1], p_uint[0], k + 1:k + 3] = direc / (resolution / y_max)
    if cutoff_graph:
        H = H.subgraph(in_img_range)
    return H, feature_map

def bev_to_graph_decoder(feature_map, resolution=256, x_max=24, y_max=48, p_thr=0.5, weight=100, d_thr=0.15, world_coord=True, v_pos=None, theta=None):
    w_list, h_list = np.where(feature_map[:, :, 0] > p_thr)

    H = nx.DiGraph()
    pos_list = []
    for n, (w, h) in enumerate(zip(w_list, h_list)):
        H.add_node(n, pos=np.array([h, w]))
        pos_list.append([h, w])
    pos_list = np.array(pos_list) / (resolution / y_max)

    for n, (w, h) in enumerate(zip(w_list, h_list)):
        for ii in range(6):
            k = 1 + 3 * ii
            if feature_map[w, h, k] > p_thr:
                direc = feature_map[w, h, k + 1:k + 3]
                dist = np.linalg.norm(pos_list[n] + direc - pos_list, ord=1, axis=1)
                cos_dist = (pos_list - pos_list[n]).dot(direc)
                cos_dist /= (1e-6 + np.linalg.norm(direc) * np.linalg.norm(pos_list - pos_list[n], axis=1))
                cos_dist = 1 - cos_dist
                condition = np.where(dist + weight * cos_dist < d_thr)[0]
                for c in condition:
                    if c != n:
                        H.add_edge(n, c)
    if world_coord:
        H = graph_pixel_to_world_coord(H, v_pos, theta, \
                                         resolution=resolution, x_max=x_max, y_max=y_max)
    else:
        for edge in H.edges():
            edge_direc = H.nodes[edge[1]]['pos'] - H.nodes[edge[0]]['pos']
            H.edges[edge]['direc'] = edge_direc
            H.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    return H


def graph_world_to_pixel_coord(H, v_pos, theta, resolution=256, x_max=24, y_max=48):
    if H.number_of_nodes() == 0:
        return H.copy()

    cos_theta = np.cos(-0.5 * math.pi + theta).reshape([-1, 1])
    sin_theta = np.sin(-0.5 * math.pi + theta).reshape([-1, 1])
    R = np.concatenate([cos_theta, -sin_theta, sin_theta, cos_theta], axis=1).reshape([2, 2])

    H_copy = H.copy()
    for key, value in H_copy.nodes().items():
        pos = value['pos'] - v_pos
        pos = np.matmul(pos, R)
        new_pos = np.array([resolution / (2 * x_max), -resolution / y_max]) * pos \
                  + np.array([resolution / 2, resolution])
        H_copy.nodes[key]['pos'] = new_pos

    for edge in H_copy.edges():
        edge_direc = H_copy.nodes[edge[1]]['pos'] - H_copy.nodes[edge[0]]['pos']
        H_copy.edges[edge]['direc'] = edge_direc
        H_copy.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    return H_copy


def graph_pixel_to_world_coord(H, v_pos, theta, resolution=256, x_max=24, y_max=48):
    if H.number_of_nodes() == 0:
        return H.copy()

    H_copy = H.copy()
    for key, value in H_copy.nodes().items():
        node_pos = value['pos']
        node_pos = np.array([(-2 * x_max) / resolution, -y_max / resolution]) * node_pos \
                   + np.array([x_max, y_max])
        node_pos = node_pos[::-1]

        rot = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        node_pos = (rot @ node_pos.T) + v_pos
        H_copy.nodes[key]['pos'] = node_pos

    for edge in H_copy.edges():
        edge_direc = H_copy.nodes[edge[1]]['pos'] - H_copy.nodes[edge[0]]['pos']
        H_copy.edges[edge]['direc'] = edge_direc
        H_copy.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    return H_copy

def sem_to_rgb(sem):
    # sem : w x h x 12
    indices = sem.argmax(2)
    color_map = np.array([color for color, _ in color_code.values()])
    pred_imgs = [color_map[p] for p in indices]
    return np.array(pred_imgs).reshape([sem.shape[0], sem.shape[1], 3])

def sem_to_rgb_DA(sem):
    # sem : w x h x 4
    indices = sem.argmax(2)
    color_map = np.array([color for color, _ in color_code_DA.values()])
    pred_imgs = [color_map[p] for p in indices]
    return np.array(pred_imgs).reshape([sem.shape[0], sem.shape[1], 3])

def lane_type_finder(sub_seg_G, center_seg_name):
    center_road_name = '_'.join(center_seg_name.split('_')[:-1])
    center_seg_num = int(center_seg_name.split('_')[-1])

    sub_seg_list = list(sub_seg_G.nodes.keys())
    lane_type_dict = {}

    if not sub_seg_G.has_node(center_seg_name):
        for seg_name in sub_seg_list:
            lane_type_dict[seg_name] = 'DA-O'
        return lane_type_dict
    elif sub_seg_G.nodes[center_seg_name]['on_traffic_signal'] is not None:
        next_seg_queue = [center_seg_name]
        next_lane_type_queue = ['DA-Int']
    else:
        next_seg_queue = [center_seg_name]
        next_lane_type_queue = ['DA-C']

    while (len(list(lane_type_dict)) < len(sub_seg_list) and len(next_seg_queue) > 0):
        seg_name = next_seg_queue[0]
        seg_num = int(seg_name.split('_')[-1])
        road_name = '_'.join(seg_name.split('_')[:-1])
        lane_type = next_lane_type_queue[0]
        lane_type_dict[seg_name] = lane_type

        next_seg_queue = next_seg_queue[1:]
        next_lane_type_queue = next_lane_type_queue[1:]

        if lane_type == 'DA-C':
            a = (np.arange(-2, 3, 1) * np.sign(seg_num) + seg_num)
            a = (np.sign(a) != np.sign(seg_num)) * np.sign(seg_num) * -1 + a
            a_left = np.arange(-10 * np.sign(seg_num), a[0], np.sign(seg_num))
            a_right = np.arange(a[-1] + np.sign(seg_num), 11 * np.sign(seg_num), np.sign(seg_num))
            type_name = ['DA-LL', 'DA-L', 'DA-C', 'DA-R', 'DA-RR']
            type_name = ['DA-O'] * len(a_left) + type_name + ['DA-O'] * len(a_right)
            pp = np.concatenate([a_left, a, a_right])
            for k, p in enumerate(pp):
                near_seg_name = road_name + '_' + str(p)
                if near_seg_name not in lane_type_dict and near_seg_name in sub_seg_list:
                    if sub_seg_G.nodes[near_seg_name]['on_traffic_signal'] is not None:
                        pass
                    else:
                        lane_type_dict[near_seg_name] = type_name[k]

        for n, _ in list(sub_seg_G.in_edges(seg_name)):
            if n in list(lane_type_dict):
                pass
            elif sub_seg_G.edges[n, seg_name]['edge_type'] == 'successor':
                if sub_seg_G.nodes[n]['on_traffic_signal'] is not None or lane_type == 'DA-Int':
                    pass
                else:
                    next_seg_queue.append(n)
                    next_lane_type_queue.append(lane_type)

        for _, n in list(sub_seg_G.out_edges(seg_name)):
            if n in list(lane_type_dict):
                pass
            elif sub_seg_G.edges[seg_name, n]['edge_type'] == 'successor':
                if sub_seg_G.nodes[n]['on_traffic_signal'] is not None or lane_type == 'DA-Int':
                    pass
                else:
                    next_seg_queue.append(n)
                    next_lane_type_queue.append(lane_type)

    center_seg_pos = sub_seg_G.nodes[center_seg_name]['lane_center'][-1]

    for seg_name in sub_seg_list:
        if seg_name not in lane_type_dict:
            if sub_seg_G.nodes[seg_name]['on_traffic_signal'] is not None:
                lane_type_dict[seg_name] = 'DA-Int'
            else:
                d1 = np.sum(np.abs(sub_seg_G.nodes[seg_name]['lane_center'][0] - center_seg_pos))
                d2 = np.sum(np.abs(sub_seg_G.nodes[seg_name]['lane_center'][-1] - center_seg_pos))
                if d1 > d2:
                    lane_type_dict[seg_name] = 'DA-LO'
                else:
                    lane_type_dict[seg_name] = 'DA-RO'

    return lane_type_dict

class PC_CoordConverter:
    def __init__(self, cam_yaw, lidar_xyz=[1.3, 0, 2.5], cam_xyz=[1.3, 0, 2.3], \
                 rgb_h=300, rgb_w=400, fov=100):
        self.rgb_w = rgb_w
        self.rgb_h = rgb_h
        self.lidar_xyz = lidar_xyz
        self.cam_xyz = cam_xyz
        self.cam_yaw = cam_yaw
        self.lidar_cam_height_diff = lidar_xyz[2] - cam_xyz[2]

        self.fov = fov
        focus_length1 = rgb_w / (2 * math.tan(self.fov * math.pi / 360))
        focus_length2 = rgb_w / (2 * math.tan(self.fov * math.pi / 360))
        self.intrinsic_mat = np.array([[focus_length1, 0, rgb_w / 2], \
                                       [0, focus_length2, rgb_h / 2], [0, 0, 1]])
        self.intrinsic_mat_inv = np.linalg.inv(self.intrinsic_mat)

        self.pc_to_lidar = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.lidar_to_world = np.array(carla.Transform(
            carla.Location(*lidar_xyz)
        ).get_matrix())

        self.world_to_cam = np.array(carla.Transform(
            carla.Location(*cam_xyz),
            carla.Rotation(yaw=cam_yaw),
        ).get_inverse_matrix())

        self.cam_to_uv = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        self.pc_to_cam = self.cam_to_uv @ self.world_to_cam @ self.lidar_to_world @ self.pc_to_lidar
        self.pc_to_cam_inv = np.linalg.inv(self.pc_to_cam)

        self.camera_view_front_margin = 0

    def lidar_to_cam_2d(self, lidar):
        lidar_xyz = lidar[:, :3].T
        lidar_xyz1 = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]

        cam_2d = (self.pc_to_cam @ lidar_xyz1)
        cam_2d = np.array([
            cam_2d[0, :] / (1e-5 + cam_2d[2, :]),
            cam_2d[1, :] / (1e-5 + cam_2d[2, :]),
            np.ones_like(cam_2d[2, :])]).T

        return (self.intrinsic_mat @ cam_2d.T).T

    def lidar_to_cam_2d_cutoff(self, lidar):
        lidar_xyz = lidar[:, :3].T
        lidar_xyz1 = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]

        cam_2d = (self.pc_to_cam @ lidar_xyz1)
        cutoff_condition = cam_2d[2, :] > self.camera_view_front_margin
        cam_2d = cam_2d[:, cutoff_condition]
        cam_2d = np.array([
            cam_2d[0, :] / (1e-5 + cam_2d[2, :]),
            cam_2d[1, :] / (1e-5 + cam_2d[2, :]),
            np.ones_like(cam_2d[2, :])]).T

        return (self.intrinsic_mat @ cam_2d.T).T, cutoff_condition

    def cam_2d_to_lidar_xyz(self, sem):
        rgb_h = sem.shape[1]
        if sem.shape[0] == 3:
            cam_pos = np.stack(np.where(sem[0, :, :] >= 0), axis=1)[:, ::-1]  # no cutoff
        else:
            cam_pos = np.stack(np.where(sem[0, :, :] != 1), axis=1)[:, ::-1]  # cutoff

        condition = cam_pos[:, 1] > (rgb_h / 2)
        cam_pos = cam_pos[condition]
        cam_pos = np.concatenate([cam_pos, np.ones_like(cam_pos[:, :1])], axis=1)

        img_cam_2d = (self.intrinsic_mat_inv @ cam_pos.T).T

        height = self.lidar_xyz[2]
        cam_2d = np.stack([
            img_cam_2d[:, 0] * height / (1e-5 + img_cam_2d[:, 1]),
            height * np.ones_like(img_cam_2d[:, 1]),
            height / (1e-5 + img_cam_2d[:, 1]),
            np.ones_like(img_cam_2d[:, 1])], axis=1).T
        return (self.pc_to_cam_inv @ cam_2d).T, cam_pos

    def get_corner_xyz(self, sem):
        rgb_h, rgb_w = sem.shape[1], sem.shape[2]
        cam_pos = np.array([[0, rgb_h - 1, 1], [rgb_w - 1, rgb_h - 1, 1]])
        img_cam_2d = (self.intrinsic_mat_inv @ cam_pos.T).T

        height = self.lidar_xyz[2]
        cam_2d = np.stack([
            img_cam_2d[:, 0] * height / (1e-5 + img_cam_2d[:, 1]),
            height * np.ones_like(img_cam_2d[:, 1]),
            height / (1e-5 + img_cam_2d[:, 1]),
            np.ones_like(img_cam_2d[:, 1])], axis=1).T
        return (self.pc_to_cam_inv @ cam_2d).T, cam_pos

    def pos_map_to_ego(self, pos_array, v_pos, theta):
        rot = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        node_pos = (rot @ (pos_array - v_pos).T).T
        node_pos_array = np.stack([node_pos[:, 1], node_pos[:, 0], \
                                   -self.lidar_xyz[2] * np.ones(len(node_pos))], axis=1)
        return node_pos_array

    def camera_cutoff_ego_coord(self, n1_ego_pos, n2_ego_pos):
        margin = self.camera_view_front_margin
        theta = self.cam_yaw * math.pi / 180
        rot = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))

        n1_ego_rot = (rot @ (n1_ego_pos[:2]).T).T
        n2_ego_rot = (rot @ (n2_ego_pos[:2]).T).T

        if n1_ego_rot[1] > margin and n2_ego_rot[1] > margin:
            return n1_ego_pos, n2_ego_pos
        elif n1_ego_rot[1] <= margin and n2_ego_rot[1] <= margin:
            return None, None
        else:
            x1, x2, y1, y2 = n1_ego_rot[0], n2_ego_rot[0], n1_ego_rot[1], n2_ego_rot[1]
            x = x1 + (y1 - margin) * (x2 - x1) / (y1 - y2)
            rot_inv = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
            new_node_pos_array = np.append((rot_inv @ np.array([x, margin]).T).T, -self.lidar_xyz[2])

            if y1 > margin:
                return n1_ego_pos, new_node_pos_array
            else:
                return new_node_pos_array, n2_ego_pos

    def visualize_road_graph(self, img, H, v_pos, theta):
        background_img = img.copy()
        # visualize road graph
        plt.figure()
        plt.imshow(background_img)
        plt.xlim(0, self.rgb_w - 1)
        plt.ylim(self.rgb_h - 1, 0)

        node_pos = np.stack([node_v['pos'] for node_v in H.nodes().values()])
        node_pos_ego = self.pos_map_to_ego(node_pos, v_pos, theta)
        node_pos_ego_dict = {node: node_pos_ego[n] for n, node in enumerate(H.nodes())}

        node_pos_2d_cutoff, _ = self.lidar_to_cam_2d_cutoff(node_pos_ego)
        for n_2d_pos in node_pos_2d_cutoff:
            plt.plot(n_2d_pos[0], n_2d_pos[1], 'ro', markersize=3)

        for n1, n2 in H.edges():
            n1_ego_pos, n2_ego_pos = node_pos_ego_dict[n1], node_pos_ego_dict[n2]
            n1_ego_pos, n2_ego_pos = self.camera_cutoff_ego_coord(n1_ego_pos, n2_ego_pos)
            if n1_ego_pos is not None:
                edge_2d_pos = self.lidar_to_cam_2d(np.stack([n1_ego_pos, n2_ego_pos]))
                edge_direc = (edge_2d_pos[1, :] - edge_2d_pos[0, :])
                edge_direc_norm = np.linalg.norm(edge_direc)
                if edge_direc_norm > 40:
                    edge_direc -= 20 * edge_direc / edge_direc_norm
                else:
                    edge_direc *= 0.5
                plt.arrow(edge_2d_pos[0, 0], edge_2d_pos[0, 1], edge_direc[0], edge_direc[1],
                          color='k', width=0.3, head_width=3)

    def visualize_sem_image(self, img, seg_G, H, graph_feature, v_pos, theta, stride=30):
        ss_img = np.zeros([img.shape[0], img.shape[1], NUM_COLOR]).astype('uint8')

        most_nearest_node_id = graph_feature['most_nearest_node_id']
        most_nearest_seg_id = '_'.join(most_nearest_node_id.split('_')[:-1])

        # visualize segmentation image
        ss = list(set([node['seg'] for node in H.nodes().values()]))
        sub_seg_G = seg_G.subgraph(ss)
        lane_type_dict = lane_type_finder(sub_seg_G, most_nearest_seg_id)

        for seg_name, seg in sub_seg_G.nodes().items():
            interval = np.arange(len(seg['left_border'])) % stride == 0
            interval[-1] = True
            left_border = seg['left_border'][interval]
            right_border = seg['right_border'][interval]
            road_center = seg['road_center'][interval]
            lane_roadmark = seg['lane_roadmark'][interval]
            center_roadmark = seg['center_roadmark'][interval]
            is_inverted = seg['is_inverted']
            if 'hole_pos' in seg:
                hole_pos = np.stack(seg['hole_pos'])
                left_kdtree = KDTree(left_border)
                _, hole_idxs = left_kdtree.query(hole_pos, k=2)
            else:
                hole_idxs = None

            left_border_ego = self.pos_map_to_ego(left_border, v_pos, theta)
            right_border_ego = self.pos_map_to_ego(right_border, v_pos, theta)
            road_center_ego = self.pos_map_to_ego(road_center, v_pos, theta)

            # show lane
            for n in range(len(left_border_ego) - 1):
                if hole_idxs is not None:
                    if any([(n == h.min() and n + 1 == h.max()) for h in hole_idxs]):
                        continue

                left1, left2 = left_border_ego[n], left_border_ego[n + 1]
                right1, right2 = right_border_ego[n], right_border_ego[n + 1]

                point = [left1, left2, right2, right1]
                point_cutoff = []
                for i in range(3):
                    p1, p2 = self.camera_cutoff_ego_coord(point[i], point[i + 1])
                    if p1 is not None:
                        point_cutoff.append(p1);
                        point_cutoff.append(p2)

                if len(point_cutoff) > 0:
                    point_cutoff = np.stack(point_cutoff)
                    point_cutoff_2d = self.lidar_to_cam_2d(point_cutoff)[:, :2].astype(int)

                    w_check = np.logical_or(point_cutoff_2d[:, 0] < 0, point_cutoff_2d[:, 0] >= self.rgb_w)
                    h_check = np.logical_or(point_cutoff_2d[:, 1] < 0, point_cutoff_2d[:, 1] >= self.rgb_h)
                    check = np.logical_or(w_check, h_check)
                    if not check.all():
                        lane_type = lane_type_dict[seg_name]
                        _, c = color_code[lane_type]
                        ss_img[:, :, c] = cv2.fillConvexPoly(ss_img[:, :, c].copy(), \
                                                             point_cutoff_2d, 255, cv2.LINE_AA, 0)

            # show road marker
            for n in range(len(left_border_ego) - 1):
                if is_inverted:
                    right1, right2 = left_border_ego[n], left_border_ego[n + 1]
                else:
                    right1, right2 = right_border_ego[n], right_border_ego[n + 1]

                right_norm = np.linalg.norm(right2[:2] - right1[:2])
                if right_norm == 0:
                    continue
                width_vec = (right2[:2] - right1[:2]) / right_norm
                width_vec = np.array([-width_vec[1], width_vec[0], 0])

                lane_roadmark_part = lane_roadmark[n].split('_')
                width, color, m_type = lane_roadmark_part[0], lane_roadmark_part[1], lane_roadmark_part[2]
                if width in ['None', 'none'] or color in ['None', 'none'] or m_type in ['None', 'none']:
                    continue
                width = float(width)
                wf = width / 2

                point = [right1 + wf * width_vec, right2 + wf * width_vec, \
                         right2 - wf * width_vec, right1 - wf * width_vec]
                point_cutoff = []
                for i in range(3):
                    p1, p2 = self.camera_cutoff_ego_coord(point[i], point[i + 1])
                    if p1 is not None:
                        point_cutoff.append(p1);
                        point_cutoff.append(p2)

                if len(point_cutoff) > 0:
                    point_cutoff = np.stack(point_cutoff)
                    point_cutoff_2d = self.lidar_to_cam_2d(point_cutoff)[:, :2].astype(int)

                    w_check = np.logical_or(point_cutoff_2d[:, 0] < 0, point_cutoff_2d[:, 0] >= self.rgb_w)
                    h_check = np.logical_or(point_cutoff_2d[:, 1] < 0, point_cutoff_2d[:, 1] >= self.rgb_h)
                    check = np.logical_or(w_check, h_check)
                    if not check.all():
                        _, c = color_code['LM-W']
                        ss_img[:, :, c] = cv2.fillConvexPoly(ss_img[:, :, c].copy(), \
                                                             point_cutoff_2d, 255, cv2.LINE_AA, 0)

            # show center marker
            for n in range(len(left_border_ego) - 1):
                center1, center2 = road_center_ego[n], road_center_ego[n + 1]

                center_norm = np.linalg.norm(center2[:2] - center1[:2])
                if center_norm == 0:
                    continue
                width_vec = (center2[:2] - center1[:2]) / center_norm
                width_vec = np.array([-width_vec[1], width_vec[0], 0])

                center_roadmark_part = center_roadmark[n].split('_')
                width, color, m_type = center_roadmark_part[0], center_roadmark_part[1], \
                                       center_roadmark_part[2]
                if width in ['None', 'none'] or color in ['None', 'none'] \
                        or m_type in ['None', 'none']:
                    continue

                width = float(width)
                wf = width / 2

                point = [center1 + wf * width_vec, center2 + wf * width_vec, \
                         center2 - wf * width_vec, center1 - wf * width_vec]
                point_cutoff = []
                for i in range(3):
                    p1, p2 = self.camera_cutoff_ego_coord(point[i], point[i + 1])
                    if p1 is not None:
                        point_cutoff.append(p1);
                        point_cutoff.append(p2)

                if len(point_cutoff) > 0:
                    point_cutoff = np.stack(point_cutoff)
                    point_cutoff_2d = self.lidar_to_cam_2d(point_cutoff)[:, :2].astype(int)

                    w_check = np.logical_or(point_cutoff_2d[:, 0] < 0, point_cutoff_2d[:, 0] >= self.rgb_w)
                    h_check = np.logical_or(point_cutoff_2d[:, 1] < 0, point_cutoff_2d[:, 1] >= self.rgb_h)
                    check = np.logical_or(w_check, h_check)
                    if not check.all():
                        _, c = color_code['LM-Y']
                        ss_img[:, :, c] = cv2.fillConvexPoly(ss_img[:, :, c].copy(), \
                                                             point_cutoff_2d, 255, cv2.LINE_AA, 0)

        return ss_img

def point_painting_lidar_colorize(lidar, sems, coord_converters, front_first=True):
    assert len(sems) == len(coord_converters)

    if len(sems) == 3 and front_first:
        sems = sems[[0,2,1]]
        coord_converters = [coord_converters[c] for c in [0,2,1]]

    _, lidar_d = lidar.shape
    sem_c, sem_h, sem_w = sems[0].shape
    rgb_h, rgb_w = coord_converters[0].rgb_h, coord_converters[0].rgb_w

    lidar_color = np.zeros((len(lidar), sem_c))
    lidar_painted_condition = np.zeros(len(lidar), dtype=bool)
    for sem, coord_converter in zip(sems, coord_converters):
        lidar_cam, lidar_idx = coord_converter.lidar_to_cam_2d_cutoff(lidar)
        lidar_cam = lidar_cam.astype(int)
        lidar_cam[:, 0] -= (rgb_w // 2 - sem_w // 2)
        lidar_cam[:, 1] -= (rgb_h // 2 - sem_h // 2)

        cam_condition1 = np.logical_and(lidar_cam[:, 0] >= 0, lidar_cam[:, 0] < sem_w)
        cam_condition2 = np.logical_and(lidar_cam[:, 1] >= 0, lidar_cam[:, 1] < sem_h)
        cam_condition = np.logical_and(cam_condition1, cam_condition2)
        lidar_cam = lidar_cam[cam_condition][:, :2]

        pixel_sem = sem[:, lidar_cam[:, 1], lidar_cam[:, 0]]

        lidar_color_valid = lidar_color[lidar_idx]
        lidar_color_valid[cam_condition] = pixel_sem.T
        lidar_color[lidar_idx] = lidar_color_valid

        lidar_painted_condition_valid = lidar_painted_condition[lidar_idx]
        lidar_painted_condition_valid[cam_condition] = True
        lidar_painted_condition[lidar_idx] = lidar_painted_condition_valid

    return lidar_color[lidar_painted_condition].astype('float64'), lidar[lidar_painted_condition].astype('float64')

def point_painting_plot_pixel(lidar, sems, coord_converters, front_first=True):
    assert len(sems) == len(coord_converters)

    if len(sems) == 3 and front_first:
        sems = sems[[0,2,1]]
        coord_converters = [coord_converters[c] for c in [0,2,1]]

    _, lidar_d = lidar.shape
    sem_c, sem_h, sem_w = sems[0].shape
    rgb_h, rgb_w = coord_converters[0].rgb_h, coord_converters[0].rgb_w

    pixel_xyz_list = []
    pixel_color_list = []
    for sem, coord_converter in zip(sems, coord_converters):
        pixel_xyz, cam_pos = coord_converter.cam_2d_to_lidar_xyz(sem)
        pixel_color = sem[:, cam_pos[:, 1], cam_pos[:, 0]].T
        pixel_xyz_list.append(pixel_xyz)
        pixel_color_list.append(pixel_color)

    return np.concatenate(pixel_color_list).astype('float64'), np.concatenate(pixel_xyz_list).astype('float64')

def get_bev_img(lidar, sems, coord_converters, resolution=256, x_max=24, y_max=48, \
                plot_method='plot_pixel', \
                color_type=None, padding='same', d=4, z=-2.0):
    if plot_method == 'lidar_colorize':
        lp, lpc = point_painting_lidar_colorize(lidar, sems, coord_converters)
    elif plot_method == 'plot_pixel':
        lp, lpc = point_painting_plot_pixel(lidar, sems, coord_converters)
    pos_condition1 = np.logical_and(lpc[:, 0] < x_max, lpc[:, 0] > -x_max)
    pos_condition2 = np.logical_and(lpc[:, 1] < y_max, lpc[:, 1] > 0)
    pos_condition = np.logical_and(pos_condition1, pos_condition2)
    pos_condition = np.logical_and(pos_condition, lpc[:, 2] < z)
    pos_condition = np.logical_and(pos_condition, np.any(lp != 0, axis=1))
    lp, lpc = lp[pos_condition], lpc[pos_condition]
    lpc = np.array([[-resolution / (2 * x_max), -resolution / y_max]]) * lpc[:, :2] \
          + np.array([[resolution / 2, resolution]])

    img = np.zeros([resolution, resolution, 3], dtype=np.uint8)

    if color_type == 'Delaunay':
        d_tri = Delaunay(lpc)
        tri_set = lpc[d_tri.simplices].astype(int)
        tri_color = np.mean(lp[d_tri.simplices], axis=1)

        for convex, color in zip(tri_set, tri_color):
            cv2.fillConvexPoly(img, convex, color, cv2.LINE_AA, 0)
    elif color_type == 'Voronoi':
        vor = Voronoi(lpc)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                convex = np.array([vor.vertices[i] for i in region]).astype(int)
                cv2.fillConvexPoly(img, convex, lp[r], cv2.LINE_AA, 0)
    else:
        for p, color in zip(lpc.astype(int), lp):
            cv2.circle(img, p, 1, color, -1)

    if padding is None:
        pass
    else:
        if plot_method == 'lidar_colorize':
            np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
            np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
            np_pos = np.stack([np_pos_x, np_pos_y], axis=2)

            if padding == 'zero':
                radius_condition = np.sum(np_pos ** 2, axis=2) <= (d + 3 * y_max / resolution) ** 2
                img[radius_condition] = np.zeros(sem_c)
            elif padding == 'same':
                radius_condition = np.sum(np_pos ** 2, axis=2) <= (d + 4 * y_max / resolution) ** 2
                radius_condition2 = np.argmax(radius_condition, axis=0)
                for i in range(resolution):
                    if radius_condition2[i] > 0:
                        img[radius_condition2[i]:, i] = img[radius_condition2[i - 1], i]
        elif plot_method == 'plot_pixel':
            np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
            np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
            np_pos = np.stack([np_pos_x, np_pos_y], axis=2)

            cond, _ = coord_converters[len(coord_converters) // 2].get_corner_xyz(sems[len(coord_converters) // 2])
            radius_condition = np.logical_and(np_pos[:, :, 0] < cond[0][0] + 2 * x_max / resolution, \
                                              np_pos[:, :, 0] > cond[1][0] - 2 * x_max / resolution)
            radius_condition = np.logical_and(radius_condition, np_pos[:, :, 1] < cond[0][1] + y_max / resolution)
            if padding == 'zero':
                img[radius_condition] = np.zeros(sem_c)
            elif padding == 'same':
                radius_condition2 = np.argmax(radius_condition, axis=0)
                for i in range(resolution):
                    if radius_condition2[i] > 0:
                        img[radius_condition2[i]:, i] = img[radius_condition2[i - 1], i]
    return img

def get_bev_sem_img(lidar, sems, coord_converters, resolution=256, x_max=24, y_max=48, \
                plot_method='plot_pixel', \
                color_type=None, padding='same', d=4):
    if plot_method == 'lidar_colorize':
        lp, lpc = point_painting_lidar_colorize(lidar, sems, coord_converters)
    elif plot_method == 'plot_pixel':
        lp, lpc = point_painting_plot_pixel(lidar, sems, coord_converters)
    pos_condition1 = np.logical_and(lpc[:, 0] < x_max, lpc[:, 0] > -x_max)
    pos_condition2 = np.logical_and(lpc[:, 1] < y_max, lpc[:, 1] > 0)
    pos_condition = np.logical_and(pos_condition1, pos_condition2)
    pos_condition = np.logical_and(pos_condition, lpc[:, 2] < -2.0)
    pos_condition = np.logical_and(pos_condition, lp.argmax(1) != 0)
    lp, lpc = lp[pos_condition], lpc[pos_condition]
    lpc = np.array([[-resolution / (2 * x_max), -resolution / y_max]]) * lpc[:, :2] \
          + np.array([[resolution / 2, resolution]])

    sem_c = sems.shape[1]
    img = np.zeros([resolution, resolution, sem_c], dtype=np.uint8)

    if color_type == 'Delaunay':
        d_tri = Delaunay(lpc)
        tri_set = lpc[d_tri.simplices].astype(int)
        tri_color = np.mean(lp[d_tri.simplices], axis=1)
        for convex, color in zip(tri_set, tri_color):
            img[:, :, color.argmax()] = cv2.fillConvexPoly(img[:, :, color.argmax()].copy(), \
                                                           convex, 255, cv2.LINE_AA, 0)
    elif color_type == 'Voronoi':
        vor = Voronoi(lpc)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                color = lp[r]
                convex = np.array([vor.vertices[i] for i in region]).astype(int)
                img[:, :, color.argmax()] = cv2.fillConvexPoly(img[:, :, color.argmax()].copy(), \
                                                               convex, 255, cv2.LINE_AA, 0)
    else:
        lpc_int = lpc.astype(int)
        img[lpc_int[:, 1], lpc_int[:, 0]] = lp

    if padding is None:
        pass
    else:
        if plot_method == 'lidar_colorize':
            np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
            np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
            np_pos = np.stack([np_pos_x, np_pos_y], axis=2)

            if padding == 'zero':
                radius_condition = np.sum(np_pos ** 2, axis=2) <= (d + 3 * y_max / resolution) ** 2
                img[radius_condition] = np.zeros(sem_c)
            elif padding == 'same':
                radius_condition = np.sum(np_pos ** 2, axis=2) <= (d + 4 * y_max / resolution) ** 2
                radius_condition2 = np.argmax(radius_condition, axis=0)
                for i in range(resolution):
                    if radius_condition2[i] > 0:
                        img[radius_condition2[i]:, i] = img[radius_condition2[i - 1], i]
        elif plot_method == 'plot_pixel':
            np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
            np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
            np_pos = np.stack([np_pos_x, np_pos_y], axis=2)

            cond, _ = coord_converters[len(coord_converters) // 2].get_corner_xyz(sems[len(coord_converters) // 2])
            radius_condition = np.logical_and(np_pos[:, :, 0] < cond[0][0] + 2 * x_max / resolution, \
                                              np_pos[:, :, 0] > cond[1][0] - 2 * x_max / resolution)
            radius_condition = np.logical_and(radius_condition, np_pos[:, :, 1] < cond[0][1] + y_max / resolution)
            if padding == 'zero':
                img[radius_condition] = np.zeros(sem_c)
            elif padding == 'same':
                radius_condition2 = np.argmax(radius_condition, axis=0)
                for i in range(resolution):
                    if radius_condition2[i] > 0:
                        img[radius_condition2[i]:, i] = img[radius_condition2[i - 1], i]
    return img

def get_interpolate(img, color_index, resolution=256, x_max=24, y_max=48, s=18000):
    np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
    np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
    np_pos = np.stack([np_pos_x, np_pos_y], axis=2)
    if type(color_index) == list:
        argmax_color = img.reshape([-1, NUM_COLOR]).argmax(1)
        cond = False
        for c in color_index:
            cond = np.logical_or(cond, argmax_color == c)
        a = np_pos.reshape([-1, 2])[cond][:, :2]
    else:
        a = np_pos.reshape([-1, 2])[img.reshape([-1, NUM_COLOR]).argmax(1) == color_index][:, :2]
    #a = lpc[lp.argmax(1) == color_index]
    x, y = a[:, 1] + np.random.normal(0, 1e-6, a[:, 1].shape), a[:, 0]  # for uniqueness of x
    x_argsort = x.argsort()
    x, y = x[x_argsort], y[x_argsort]

    tck = interpolate.splrep(x, y, k=2, s=s)
    x_new = np.unique(x)
    y_new = interpolate.splev(x_new, tck, der=0)

    xy_new = np.stack([y_new, x_new], axis=1)
    return a, xy_new

def get_division_interp(img, color_index, div_lines, resolution=256, x_max=24, y_max=48, s=18000):
    np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
    np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
    np_pos = np.stack([np_pos_x, np_pos_y], axis=2)
    if type(color_index) == list:
        argmax_color = img.reshape([-1, NUM_COLOR]).argmax(1)
        cond = False
        for c in color_index:
            cond = np.logical_or(cond, argmax_color == c)
        a = np_pos.reshape([-1, 2])[cond][:, :2]
    else:
        a = np_pos.reshape([-1, 2])[img.reshape([-1, NUM_COLOR]).argmax(1) == color_index][:, :2]

    container = []
    for line in div_lines:
        frenet_by_l = cartesian_to_frenet_approx(line, a)
        a_left = a[frenet_by_l[:,1] < 0]
        a_right = a[frenet_by_l[:,1] >= 0]
        container.append(a_left)
        a = a_right
    if len(a_right) > 10:
        container.append(a_right)
    container = [ct for ct in container if len(ct) > 10]

    interp_list = []
    for c in container:
        x, y = c[:, 1] + np.random.normal(0, 1e-6, c[:, 1].shape), c[:, 0]  # for uniqueness of x
        x_argsort = x.argsort()
        x, y = x[x_argsort], y[x_argsort]

        tck = interpolate.splrep(x, y, k=2, s=s)
        x_new = np.unique(x)
        y_new = interpolate.splev(x_new, tck, der=0)

        xy_new = np.stack([y_new, x_new], axis=1)
        interp_list.append(xy_new)

    return container, interp_list

def get_dbscan_max_interpolate(img, color_index, resolution=256, x_max=24, y_max=48, s=18000):
    np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
    np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
    np_pos = np.stack([np_pos_x, np_pos_y], axis=2)
    if type(color_index) == list:
        argmax_color = img.reshape([-1, NUM_COLOR]).argmax(1)
        cond = False
        for c in color_index:
            cond = np.logical_or(cond, argmax_color == c)
        a = np_pos.reshape([-1, 2])[cond][:, :2]
    else:
        a = np_pos.reshape([-1, 2])[img.reshape([-1, NUM_COLOR]).argmax(1) == color_index][:, :2]
    #a = lpc[lp.argmax(1) == color_index]
    x = np.stack([a[:, 1], a[:, 0]], axis=1)

    db = DBSCAN(eps=2.0, min_samples=5).fit(x)
    labels = db.labels_

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    unique_labels = set(labels)
    cluster_result = [x[(labels == k) & core_samples_mask] for k in unique_labels if k != -1]
    cc = max(cluster_result, key=lambda tt: len(tt))

    x, y = cc[:, 0] + np.random.normal(0, 1e-6, cc[:, 0].shape), cc[:, 1]  # for uniqueness of x
    x_argsort = x.argsort()
    x, y = x[x_argsort], y[x_argsort]

    tck = interpolate.splrep(x, y, k=2, s=s)
    x_new = np.unique(x)
    y_new = interpolate.splev(x_new, tck, der=0)

    xy_new = np.stack([y_new, x_new], axis=1)
    return cc[:,::-1], xy_new

def get_dbscan(img, color_index, resolution=256, x_max=24, y_max=48, bev_center=np.array([0, 24]), s=18000,
               min_pixel_num=20):
    np_pos_x = np.tile(np.arange(x_max, -x_max, -2 * x_max / resolution), [resolution, 1])
    np_pos_y = np.tile(np.arange(y_max, 0, -y_max / resolution), [resolution, 1]).T
    np_pos = np.stack([np_pos_x, np_pos_y], axis=2)
    if type(color_index) == list:
        argmax_color = img.reshape([-1, NUM_COLOR]).argmax(1)
        cond = False
        for c in color_index:
            cond = np.logical_or(cond, argmax_color == c)
        a = np_pos.reshape([-1, 2])[cond][:, :2]
    else:
        a = np_pos.reshape([-1, 2])[img.reshape([-1, NUM_COLOR]).argmax(1) == color_index][:, :2]
    # a = lpc[lp.argmax(1) == color_index]
    x = np.stack([a[:, 1], a[:, 0]], axis=1)

    if len(x) == 0:
        return [], []

    db = DBSCAN(eps=2.0, min_samples=5).fit(x)
    labels = db.labels_

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    img_center_pos = bev_center[::-1]
    unique_labels = set(labels)
    cluster_result = [x[(labels == k) & core_samples_mask] for k in unique_labels if k != -1]
    cluster_result = [cc for cc in cluster_result if len(cc) > min_pixel_num]
    interpolated_result = []
    for cc in cluster_result:
        x_mean = (cc.mean(axis=0) - img_center_pos)
        theta = math.atan2(x_mean[1], x_mean[0])
        rot = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
        rot_inv = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        x_rot = (rot @ (cc - img_center_pos).T).T

        x_c, y_c = x_rot[:, 0] + np.random.normal(0, 1e-6, x_rot[:, 0].shape), x_rot[:, 1]  # for uniqueness of x
        x_c_argsort = x_c.argsort()
        x_c, y_c = x_c[x_c_argsort], y_c[x_c_argsort]

        tck = interpolate.splrep(x_c, y_c, k=2, s=s)
        x_c_new = np.unique(x_c)
        y_c_new = interpolate.splev(x_c_new, tck, der=0)
        xy_c_new = (rot_inv @ (np.stack([x_c_new, y_c_new], axis=1)).T).T + img_center_pos

        interpolated_result.append(xy_c_new)

    cluster_result = [cr[:, ::-1] for cr in cluster_result]
    interpolated_result = [ir[:, ::-1] for ir in interpolated_result]
    return cluster_result, interpolated_result

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    xyvals = np.stack([xvals, yvals], axis=1)

    return xyvals

def get_intersection_point(pos1, pos2, pos3, pos4):
    x1, y1, x2, y2, x3, y3, x4, y4 = pos1[0], pos1[1], pos2[0], pos2[1], pos3[0], pos3[1], pos4[0], pos4[1]
    if (y4 - y3) * (x2 - x1) == (y2 - y1) * (x4 - x3):  # parallel
        return None
    elif x2 == x1:
        x = x1
        y = ((y4 - y3) / (x4 - x3)) * (x - x3) + y3
    elif x4 == x3:
        x = x3
        y = ((y2 - y1) / (x2 - x1)) * (x - x2) + y2
    else:
        slope_diff = (y4 - y3) / (x4 - x3) - (y2 - y1) / (x2 - x1)
        a = x4 * (y4 - y3) / (x4 - x3) - x2 * (y2 - y1) / (x2 - x1) + y2 - y4
        x = a / slope_diff
        y = ((y2 - y1) / (x2 - x1)) * (x - x2) + y2

    return np.array([x, y])


def get_graph_from_sem_img(lidar, sems, coord_converters, resolution=256, x_max=24, y_max=48, s=18000, \
                           min_pixel_num=20, division_method='seg', \
                           plot_method='plot_pixel', color_type=None, padding='same', d=4, \
                           world_coord=True, v_pos=None, theta=None):
    img = get_bev_sem_img(lidar, sems, coord_converters, plot_method=plot_method, color_type=color_type,
                          padding=padding, d=d)
    img_argmax = img.argmax(2)

    new_seg_G = nx.DiGraph()
    new_G = nx.DiGraph()

    # center_line
    lm_y_cluster, lm_y_line = get_dbscan(img, color_code['LM-Y'][1], s=s, min_pixel_num=min_pixel_num)
    if len(lm_y_cluster) > 0:
        lm_y_pos = [np.sum(np.abs(lmc.mean(0))) for lmc in lm_y_cluster]
        center_line = lm_y_line[np.argmin(lm_y_pos)][::-1]
    else:
        center_line = np.arange(0, y_max, 1)
        center_line = np.stack([x_max * np.ones_like(center_line), center_line], axis=1)

    # division line
    color_index_list = [color_code['DA-LL'][1], color_code['DA-L'][1], color_code['DA-C'][1], \
                        color_code['DA-R'][1], color_code['DA-RR'][1]]

    if division_method == 'line':
        lm_w_cluster, lm_w_line = get_dbscan(img, [color_code['LM-Y'][1], color_code['LM-W'][1]], s=s)
        if len(lm_w_cluster) > 0:
            lm_w_pos_cond = [lmc[:, 1].min() < 10 for lmc in lm_w_cluster]
            div_lines = [lml[::-1] for lml, lmpc in zip(lm_w_line, lm_w_pos_cond) if lmpc]
            div_lines = sorted(div_lines, key=lambda l: -l[:, 0].min())
            div_cluster, div_interp = get_division_interp(img, color_index_list, div_lines)
        else:
            div_cluster, div_interp = None, None
    else:
        div_cluster, div_interp = [], []
        for color in color_index_list:
            if np.sum(img.argmax(2) == color) > min_pixel_num:
                a, xy_new = get_dbscan_max_interpolate(img, color, s=s)
                if len(a) > min_pixel_num * 10:
                    div_cluster.append(a)
                    div_interp.append(xy_new)

    # get lane segment
    if len(div_cluster) == 0:  # pseudo cluster
        frenet_cl_start_list = [np.zeros([1, 2])]
        lane_num_list = np.array([1])
        lane_name_list = ['DA_1']
        num_left_lane = 0
        on_intersection = True

        nodes_pixel = np.array([[resolution / 2, resolution], [resolution / 2, resolution - 0.5]])
        lane_name = 'DA_1'
        new_seg_G.add_node(lane_name, pos=nodes_pixel.mean(0), on_same_direc=True)
        new_seg_G.nodes[lane_name]['point_line'] = nodes_pixel
        new_seg_G.nodes[lane_name]['frenet'] = np.array([[0, 0], [0.5, 0]])
        new_seg_G.nodes[lane_name]['point_names'] = ['{}_{}'.format(lane_name, k) \
                                                     for k in range(len(nodes_pixel))]

        for k, k_pos in enumerate(new_seg_G.nodes[lane_name]['point_line']):
            new_G.add_node('{}_{}'.format(lane_name, k), pos=k_pos, seg=lane_name, \
                           frenet=new_seg_G.nodes[lane_name]['frenet'][k])
        for k in range(len(nodes_pixel) - 1):
            new_G.add_edge('{}_{}'.format(lane_name, k), '{}_{}'.format(lane_name, k + 1), \
                           edge_type='sequential')
    else:
        cl_left, cl_right = 0, 0
        frenet_cl_start_list = []
        for l_cluster, l_line in zip(div_cluster, div_interp):
            frenet_by_cl = cartesian_to_frenet_approx(center_line, l_cluster.mean(0).reshape([-1, 2]))
            frenet_cl_start_list.append(frenet_by_cl)
            on_same_direc = frenet_by_cl[:, 1].mean() > 0
            if on_same_direc:
                cl_right += 1
            else:
                cl_left += 1

        lane_num_list = np.concatenate([np.arange(-cl_left, 0), np.arange(1, cl_right + 1)])
        lane_name_list = ['DA_' + str(n) for n in lane_num_list]
        num_left_lane = len([l for l in lane_num_list if l < 0])
        on_intersection = False

    for n, (l_cluster, l_line) in enumerate(zip(div_cluster, div_interp)):
        if len(l_cluster) > min_pixel_num:
            l_max = l_cluster[:, 1].max()
            frenet_s = np.concatenate([np.arange(0, l_max, 3), [l_max]])
            frenet_s = np.stack([frenet_s, np.zeros_like(frenet_s)]).T

            nodes = frenet_to_cartesian_approx(l_line, frenet_s)
            nodes_pixel = np.array([-resolution / (2 * x_max), -resolution / y_max]) * nodes \
                          + np.array([resolution / 2, resolution])
            on_same_direc = lane_num_list[n] > 0
            lane_name = lane_name_list[n]
            frenet_by_cl = cartesian_to_frenet_approx(center_line, nodes)

            new_seg_G.add_node(lane_name, pos=nodes_pixel.mean(0), on_same_direc=on_same_direc)
            new_seg_G.nodes[lane_name]['point_line'] = nodes_pixel if on_same_direc else nodes_pixel[::-1]
            new_seg_G.nodes[lane_name]['frenet'] = frenet_by_cl if on_same_direc else frenet_by_cl[::-1]
            new_seg_G.nodes[lane_name]['point_names'] = ['{}_{}'.format(lane_name, k) \
                                                         for k in range(len(nodes_pixel))]

            for k, k_pos in enumerate(new_seg_G.nodes[lane_name]['point_line']):
                new_G.add_node('{}_{}'.format(lane_name, k), pos=k_pos, seg=lane_name, \
                               frenet=new_seg_G.nodes[lane_name]['frenet'][k])
            for k in range(len(nodes_pixel) - 1):
                new_G.add_edge('{}_{}'.format(lane_name, k), '{}_{}'.format(lane_name, k + 1), \
                               edge_type='sequential')

    # connect lane segment
    for n in range(len(div_interp) - 1):
        direc1 = lane_num_list[n] > 0
        direc2 = lane_num_list[n + 1] > 0
        if direc1 == direc2:
            edge_type1 = 'right' if new_seg_G.nodes[lane_name_list[n]]['on_same_direc'] else 'left'
            edge_type2 = 'left' if new_seg_G.nodes[lane_name_list[n]]['on_same_direc'] else 'right'
            new_seg_G.add_edge(lane_name_list[n], lane_name_list[n + 1], edge_type=edge_type1)
            new_seg_G.add_edge(lane_name_list[n + 1], lane_name_list[n], edge_type=edge_type2)

    lane_change_progress_min_dist = 0.0
    lane_change_progress_max_dist = 6.0
    for seg_edge in new_seg_G.edges():
        if new_seg_G.edges[seg_edge]['edge_type'] in ['left', 'right']:
            lane_id = seg_edge[0]
            changed_lane_id = seg_edge[1]

            lane_points = new_seg_G.nodes[lane_id]['point_names']
            changed_lane_points = new_seg_G.nodes[changed_lane_id]['point_names']
            is_inverted = not new_seg_G.nodes[lane_id]['on_same_direc']

            changed_lane_idx = 0
            for p_id in lane_points:
                while (changed_lane_idx < len(changed_lane_points)):
                    changed_p_id = changed_lane_points[changed_lane_idx]

                    if is_inverted:
                        lane_s = -new_G.nodes[p_id]['frenet'][0]
                        changed_lane_s = -new_G.nodes[changed_p_id]['frenet'][0]
                    else:
                        lane_s = new_G.nodes[p_id]['frenet'][0]
                        changed_lane_s = new_G.nodes[changed_p_id]['frenet'][0]

                    if changed_lane_s < lane_s + lane_change_progress_min_dist:
                        changed_lane_idx += 1
                    else:
                        if changed_lane_s < lane_s + lane_change_progress_max_dist:
                            new_G.add_edge(p_id, changed_p_id, edge_type=new_seg_G.edges[seg_edge]['edge_type'])
                        break

    # add lane segment over intersection
    if np.sum(img_argmax == color_code['DA-Int'][1]) > min_pixel_num:
        if np.sum(img_argmax == color_code['DA-LO'][1]) > min_pixel_num:
            if on_intersection:
                da_lo_cluster, da_lo_line = get_dbscan(img, color_code['DA-LO'][1], bev_center=np.zeros(2), s=s)
            else:
                da_lo_cluster, da_lo_line = get_dbscan(img, color_code['DA-LO'][1], s=s)

            for n, (cluster, l_line) in enumerate(zip(da_lo_cluster, da_lo_line)):
                if len(cluster) < min_pixel_num:
                    continue
                else:
                    num_left_lanes = [l for l in lane_num_list if l < 0]
                    left_frenet_starts = [p[0, 1] for l, p in zip(lane_num_list, frenet_cl_start_list) if l < 0]
                    if len(num_left_lanes) == 0:
                        continue

                    left_frenet_starts_mean = np.mean(left_frenet_starts)
                    left_frenet_starts = [ls - left_frenet_starts_mean for ls in left_frenet_starts]

                    road_name = 'DA-LO-{}'.format(n)
                    for lane_num, start in zip(num_left_lanes, left_frenet_starts):
                        lane_name = road_name + '_' + str(lane_num)
                        frenet_s = np.arange(0, 10, 3)
                        frenet_s = np.stack([frenet_s, start * np.ones_like(frenet_s)]).T

                        nodes = frenet_to_cartesian_approx(l_line, frenet_s)
                        nodes_pixel = np.array([-resolution / (2 * x_max), -resolution / y_max]) * nodes \
                                      + np.array([resolution / 2, resolution])
                        on_same_direc = False
                        new_seg_G.add_node(lane_name, pos=nodes_pixel.mean(0), on_same_direc=on_same_direc)
                        new_seg_G.nodes[lane_name]['point_line'] = nodes_pixel if on_same_direc else nodes_pixel[::-1]
                        # new_seg_G.nodes[lane_name]['frenet'] = frenet_by_cl if on_same_direc else frenet_by_cl[::-1]
                        new_seg_G.nodes[lane_name]['point_names'] = ['{}_{}'.format(lane_name, k) \
                                                                     for k in range(len(nodes_pixel))]

                        for k, k_pos in enumerate(new_seg_G.nodes[lane_name]['point_line']):
                            new_G.add_node('{}_{}'.format(lane_name, k), pos=k_pos, seg=lane_name)  # , \
                            #              frenet=new_seg_G.nodes[lane_name]['frenet'][n])
                        for k in range(len(nodes_pixel) - 1):
                            new_G.add_edge('{}_{}'.format(lane_name, k), '{}_{}'.format(lane_name, k + 1), \
                                           edge_type='sequential')

        if np.sum(img_argmax == color_code['DA-RO'][1]) > min_pixel_num:
            if on_intersection:
                da_ro_cluster, da_ro_line = get_dbscan(img, color_code['DA-RO'][1], bev_center=np.zeros(2), s=s)
            else:
                da_ro_cluster, da_ro_line = get_dbscan(img, color_code['DA-RO'][1], s=s)
            for n, (cluster, l_line) in enumerate(zip(da_ro_cluster, da_ro_line)):
                if len(cluster) < min_pixel_num:
                    continue
                else:
                    num_right_lanes = [l for l in lane_num_list if l > 0]
                    right_frenet_starts = [p[0, 1] for l, p in zip(lane_num_list, frenet_cl_start_list) if l > 0]
                    if len(num_right_lanes) == 0:
                        continue
                    right_frenet_starts_mean = np.mean(right_frenet_starts)
                    right_frenet_starts = [rs - right_frenet_starts_mean for rs in right_frenet_starts]

                    road_name = 'DA-RO-{}'.format(n)
                    for lane_num, start in zip(num_right_lanes, right_frenet_starts):
                        lane_name = road_name + '_' + str(lane_num)
                        frenet_s = np.arange(0, 10, 3)
                        frenet_s = np.stack([frenet_s, start * np.ones_like(frenet_s)]).T

                        nodes = frenet_to_cartesian_approx(l_line, frenet_s)
                        nodes_pixel = np.array([-resolution / (2 * x_max), -resolution / y_max]) * nodes \
                                      + np.array([resolution / 2, resolution])
                        on_same_direc = True
                        new_seg_G.add_node(lane_name, pos=nodes_pixel.mean(0), on_same_direc=on_same_direc)
                        new_seg_G.nodes[lane_name]['point_line'] = nodes_pixel if on_same_direc else nodes_pixel[::-1]
                        # new_seg_G.nodes[lane_name]['frenet'] = frenet_by_cl if on_same_direc else frenet_by_cl[::-1]
                        new_seg_G.nodes[lane_name]['point_names'] = ['{}_{}'.format(lane_name, k) \
                                                                     for k in range(len(nodes_pixel))]

                        for k, k_pos in enumerate(new_seg_G.nodes[lane_name]['point_line']):
                            new_G.add_node('{}_{}'.format(lane_name, k), pos=k_pos, seg=lane_name)  # , \
                            #              frenet=new_seg_G.nodes[lane_name]['frenet'][n])
                        for k in range(len(nodes_pixel) - 1):
                            new_G.add_edge('{}_{}'.format(lane_name, k), '{}_{}'.format(lane_name, k + 1), \
                                           edge_type='sequential')

        # TODO
        # 1. lane의 위치에 따라 연결 하고 안하고 선택 (가장 오른쪽만 우회전 가능 이라던가)
        # 2. lane의 위치에 따라 연결 위치 변경
        # 3. frenet 거리 기반으로 graph cutoff

        # connect lane segment over intersection
        node_list = list(new_seg_G.nodes().keys())
        for seg_node in node_list:
            for lane_num, lane_name in zip(lane_num_list, lane_name_list):
                if seg_node.startswith('DA-LO') and seg_node.endswith(str(lane_num)):
                    if new_seg_G.has_node(lane_name) and not new_seg_G.nodes[lane_name]['on_same_direc']:
                        start_node = new_seg_G.nodes[seg_node]['point_names'][-1]
                        start_node_before = new_seg_G.nodes[seg_node]['point_names'][-2]
                        end_node = new_seg_G.nodes[lane_name]['point_names'][0]
                        end_node_after = new_seg_G.nodes[lane_name]['point_names'][1]

                        pos1 = new_G.nodes[start_node_before]['pos']
                        pos2 = new_G.nodes[start_node]['pos']
                        pos3 = new_G.nodes[end_node]['pos']
                        pos4 = new_G.nodes[end_node_after]['pos']
                        intersect_point = get_intersection_point(pos1, pos2, pos3, pos4)
                        if intersect_point[0] < 0 or intersect_point[0] > resolution \
                                or intersect_point[1] < 0 or intersect_point[1] > 0:
                            intersect_point = [resolution / 2, resolution / 2]

                        points = np.stack([pos2, intersect_point, pos3])
                        bc_line = bezier_curve(points)[::-1]
                        min_max = cartesian_to_frenet_approx(bc_line, bc_line[-1:])
                        frenet_s = np.arange(0, min_max[0, 0], 3 * 8)[1:]
                        frenet_s = np.stack([frenet_s, np.zeros_like(frenet_s)]).T

                        if len(frenet_s) < 2:
                            new_seg_G.add_edge(seg_node, lane_name, edge_type='successor')
                            new_G.add_edge(new_seg_G.nodes[seg_node]['point_names'][-1], \
                                           new_seg_G.nodes[lane_name]['point_names'][0], edge_type='successor')
                        else:
                            nodes_pixel = frenet_to_cartesian_approx(bc_line, frenet_s)
                            new_lane_name = seg_node + ' ' + lane_name
                            new_seg_G.add_node(new_lane_name, pos=nodes_pixel.mean(0), on_same_direc=False)
                            new_seg_G.nodes[new_lane_name]['point_line'] = nodes_pixel
                            # new_seg_G.nodes[new_lane_name]['frenet'] = \
                            # frenet_by_cl if on_same_direc else frenet_by_cl[::-1]
                            new_seg_G.nodes[new_lane_name]['point_names'] = ['{}_{}'.format(new_lane_name, k) \
                                                                             for k in range(len(nodes_pixel))]
                            new_seg_G.add_edge(seg_node, new_lane_name, edge_type='successor')
                            new_seg_G.add_edge(new_lane_name, lane_name, edge_type='successor')

                            for k, k_pos in enumerate(new_seg_G.nodes[new_lane_name]['point_line']):
                                new_G.add_node('{}_{}'.format(new_lane_name, k), pos=k_pos, seg=new_lane_name)  # , \
                                #              frenet=new_seg_G.nodes[lane_name]['frenet'][n])
                            for k in range(len(nodes_pixel) - 1):
                                new_G.add_edge('{}_{}'.format(new_lane_name, k), '{}_{}'.format(new_lane_name, k + 1), \
                                               edge_type='sequential')
                            new_G.add_edge(new_seg_G.nodes[seg_node]['point_names'][-1], \
                                           new_seg_G.nodes[new_lane_name]['point_names'][0], edge_type='successor')
                            new_G.add_edge(new_seg_G.nodes[new_lane_name]['point_names'][-1], \
                                           new_seg_G.nodes[lane_name]['point_names'][0], edge_type='successor')

                if seg_node.startswith('DA-RO') and seg_node.endswith(str(lane_num)):
                    if new_seg_G.has_node(lane_name) and new_seg_G.nodes[lane_name]['on_same_direc']:
                        start_node = new_seg_G.nodes[lane_name]['point_names'][-1]
                        start_node_before = new_seg_G.nodes[lane_name]['point_names'][-2]
                        end_node = new_seg_G.nodes[seg_node]['point_names'][0]
                        end_node_after = new_seg_G.nodes[seg_node]['point_names'][1]

                        pos1 = new_G.nodes[start_node_before]['pos']
                        pos2 = new_G.nodes[start_node]['pos']
                        pos3 = new_G.nodes[end_node]['pos']
                        pos4 = new_G.nodes[end_node_after]['pos']
                        intersect_point = get_intersection_point(pos1, pos2, pos3, pos4)
                        if intersect_point[0] < 0 or intersect_point[0] > resolution \
                                or intersect_point[1] < 0 or intersect_point[1] > 0:
                            intersect_point = [resolution / 2, resolution / 2]

                        points = np.stack([pos2, intersect_point, pos3])
                        bc_line = bezier_curve(points)[::-1]
                        min_max = cartesian_to_frenet_approx(bc_line, bc_line[-1:])
                        frenet_s = np.arange(0, min_max[0, 0], 3 * 8)[1:]
                        frenet_s = np.stack([frenet_s, np.zeros_like(frenet_s)]).T

                        if len(frenet_s) < 2:
                            new_seg_G.add_edge(lane_name, seg_node, edge_type='successor')
                            new_G.add_edge(new_seg_G.nodes[lane_name]['point_names'][-1], \
                                           new_seg_G.nodes[seg_node]['point_names'][0], edge_type='successor')
                        else:
                            nodes_pixel = frenet_to_cartesian_approx(bc_line, frenet_s)
                            new_lane_name = lane_name + ' ' + seg_node
                            new_seg_G.add_node(new_lane_name, pos=nodes_pixel.mean(0), on_same_direc=True)
                            new_seg_G.nodes[new_lane_name]['point_line'] = nodes_pixel
                            # new_seg_G.nodes[new_lane_name]['frenet'] = \
                            # frenet_by_cl if on_same_direc else frenet_by_cl[::-1]
                            new_seg_G.nodes[new_lane_name]['point_names'] = ['{}_{}'.format(new_lane_name, k) \
                                                                             for k in range(len(nodes_pixel))]
                            new_seg_G.add_edge(lane_name, new_lane_name, edge_type='successor')
                            new_seg_G.add_edge(new_lane_name, seg_node, edge_type='successor')

                            for k, k_pos in enumerate(new_seg_G.nodes[new_lane_name]['point_line']):
                                new_G.add_node('{}_{}'.format(new_lane_name, k), pos=k_pos, seg=new_lane_name)  # , \
                                #              frenet=new_seg_G.nodes[lane_name]['frenet'][n])
                            for k in range(len(nodes_pixel) - 1):
                                new_G.add_edge('{}_{}'.format(new_lane_name, k), '{}_{}'.format(new_lane_name, k + 1), \
                                               edge_type='sequential')
                            new_G.add_edge(new_seg_G.nodes[lane_name]['point_names'][-1], \
                                           new_seg_G.nodes[new_lane_name]['point_names'][0], edge_type='successor')
                            new_G.add_edge(new_seg_G.nodes[new_lane_name]['point_names'][-1], \
                                           new_seg_G.nodes[seg_node]['point_names'][0], edge_type='successor')

    new_G_nodes_pixel = np.array([node['pos'] for node in new_G.nodes().values()])
    new_G_nodes_pixel_id = np.array([key for key in new_G.nodes().keys()])
    if len(new_G_nodes_pixel) == 0:
        return new_seg_G, new_G

    condition1 = np.logical_and(new_G_nodes_pixel[:, 0] > 0, \
                                new_G_nodes_pixel[:, 0] < resolution)
    condition2 = np.logical_and(new_G_nodes_pixel[:, 1] > 0, \
                                new_G_nodes_pixel[:, 1] < resolution)
    condition = np.logical_and(condition1, condition2)
    nodes_ids = new_G_nodes_pixel_id[condition]
    new_G = new_G.subgraph(nodes_ids)

    if world_coord:
        new_G = graph_pixel_to_world_coord(new_G, v_pos, theta, \
                                           resolution=resolution, x_max=x_max, y_max=y_max)
        new_seg_G = graph_pixel_to_world_coord(new_seg_G, v_pos, theta, \
                                               resolution=resolution, x_max=x_max, y_max=y_max)
    else:
        for edge in new_G.edges():
            edge_direc = new_G.nodes[edge[1]]['pos'] - new_G.nodes[edge[0]]['pos']
            new_G.edges[edge]['direc'] = edge_direc
            new_G.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))
        for edge in new_seg_G.edges():
            edge_direc = new_seg_G.nodes[edge[1]]['pos'] - new_seg_G.nodes[edge[0]]['pos']
            new_seg_G.edges[edge]['direc'] = edge_direc
            new_seg_G.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    return new_seg_G, new_G


def get_graph_from_sem_img_wo_int(lidar, sems, coord_converters, resolution=256, x_max=24, y_max=48, s=18000, \
                           min_pixel_num=20, division_method='seg', \
                           plot_method='plot_pixel', color_type=None, padding='same', d=4, \
                           world_coord=True, v_pos=None, theta=None):
    img = get_bev_sem_img(lidar, sems, coord_converters, plot_method=plot_method, color_type=color_type,
                          padding=padding, d=d)
    img_argmax = img.argmax(2)

    new_seg_G = nx.DiGraph()
    new_G = nx.DiGraph()

    # center_line
    lm_y_cluster, lm_y_line = get_dbscan(img, color_code['LM-Y'][1], s=s, min_pixel_num=min_pixel_num)
    if len(lm_y_cluster) > 0:
        lm_y_pos = [np.sum(np.abs(lmc.mean(0))) for lmc in lm_y_cluster]
        center_line = lm_y_line[np.argmin(lm_y_pos)][::-1]
    else:
        center_line = np.arange(0, y_max, 1)
        center_line = np.stack([x_max * np.ones_like(center_line), center_line], axis=1)

    # division line
    color_index_list = [color_code['DA-LL'][1], color_code['DA-L'][1], color_code['DA-C'][1], \
                        color_code['DA-R'][1], color_code['DA-RR'][1]]

    if division_method == 'line':
        lm_w_cluster, lm_w_line = get_dbscan(img, [color_code['LM-Y'][1], color_code['LM-W'][1]], s=s)
        if len(lm_w_cluster) > 0:
            lm_w_pos_cond = [lmc[:, 1].min() < 10 for lmc in lm_w_cluster]
            div_lines = [lml[::-1] for lml, lmpc in zip(lm_w_line, lm_w_pos_cond) if lmpc]
            div_lines = sorted(div_lines, key=lambda l: -l[:, 0].min())
            div_cluster, div_interp = get_division_interp(img, color_index_list, div_lines)
        else:
            div_cluster, div_interp = None, None
    else:
        div_cluster, div_interp = [], []
        for color in color_index_list:
            if np.sum(img.argmax(2) == color) > min_pixel_num:
                a, xy_new = get_dbscan_max_interpolate(img, color, s=s)
                if len(a) > min_pixel_num * 10:
                    div_cluster.append(a)
                    div_interp.append(xy_new)

    # get lane segment
    if len(div_cluster) == 0:  # pseudo cluster
        frenet_cl_start_list = [np.zeros([1, 2])]
        lane_num_list = np.array([1])
        lane_name_list = ['DA_1']
        num_left_lane = 0

        nodes_pixel = np.array([[resolution / 2, resolution], [resolution / 2, resolution - 0.5]])
        lane_name = 'DA_1'
        new_seg_G.add_node(lane_name, pos=nodes_pixel.mean(0), on_same_direc=True)
        new_seg_G.nodes[lane_name]['point_line'] = nodes_pixel
        new_seg_G.nodes[lane_name]['frenet'] = np.array([[0, 0], [0.5, 0]])
        new_seg_G.nodes[lane_name]['point_names'] = ['{}_{}'.format(lane_name, k) \
                                                     for k in range(len(nodes_pixel))]

        for k, k_pos in enumerate(new_seg_G.nodes[lane_name]['point_line']):
            new_G.add_node('{}_{}'.format(lane_name, k), pos=k_pos, seg=lane_name, \
                           frenet=new_seg_G.nodes[lane_name]['frenet'][k])
        for k in range(len(nodes_pixel) - 1):
            new_G.add_edge('{}_{}'.format(lane_name, k), '{}_{}'.format(lane_name, k + 1), \
                           edge_type='sequential')
    else:
        cl_left, cl_right = 0, 0
        frenet_cl_start_list = []
        for l_cluster, l_line in zip(div_cluster, div_interp):
            frenet_by_cl = cartesian_to_frenet_approx(center_line, l_cluster.mean(0).reshape([-1, 2]))
            frenet_cl_start_list.append(frenet_by_cl)
            on_same_direc = frenet_by_cl[:, 1].mean() > 0
            if on_same_direc:
                cl_right += 1
            else:
                cl_left += 1

        lane_num_list = np.concatenate([np.arange(-cl_left, 0), np.arange(1, cl_right + 1)])
        lane_name_list = ['DA_' + str(n) for n in lane_num_list]
        num_left_lane = len([l for l in lane_num_list if l < 0])

    for n, (l_cluster, l_line) in enumerate(zip(div_cluster, div_interp)):
        if len(l_cluster) > min_pixel_num:
            l_max = l_cluster[:, 1].max()
            frenet_s = np.concatenate([np.arange(0, l_max, 3), [l_max]])
            frenet_s = np.stack([frenet_s, np.zeros_like(frenet_s)]).T

            nodes = frenet_to_cartesian_approx(l_line, frenet_s)
            nodes_pixel = np.array([-resolution / (2 * x_max), -resolution / y_max]) * nodes \
                          + np.array([resolution / 2, resolution])
            on_same_direc = lane_num_list[n] > 0
            lane_name = lane_name_list[n]
            frenet_by_cl = cartesian_to_frenet_approx(center_line, nodes)

            new_seg_G.add_node(lane_name, pos=nodes_pixel.mean(0), on_same_direc=on_same_direc)
            new_seg_G.nodes[lane_name]['point_line'] = nodes_pixel if on_same_direc else nodes_pixel[::-1]
            new_seg_G.nodes[lane_name]['frenet'] = frenet_by_cl if on_same_direc else frenet_by_cl[::-1]
            new_seg_G.nodes[lane_name]['point_names'] = ['{}_{}'.format(lane_name, k) \
                                                         for k in range(len(nodes_pixel))]

            for k, k_pos in enumerate(new_seg_G.nodes[lane_name]['point_line']):
                new_G.add_node('{}_{}'.format(lane_name, k), pos=k_pos, seg=lane_name, \
                               frenet=new_seg_G.nodes[lane_name]['frenet'][k])
            for k in range(len(nodes_pixel) - 1):
                new_G.add_edge('{}_{}'.format(lane_name, k), '{}_{}'.format(lane_name, k + 1), \
                               edge_type='sequential')

    # connect lane segment
    for n in range(len(div_interp) - 1):
        direc1 = lane_num_list[n] > 0
        direc2 = lane_num_list[n + 1] > 0
        if direc1 == direc2:
            edge_type1 = 'right' if new_seg_G.nodes[lane_name_list[n]]['on_same_direc'] else 'left'
            edge_type2 = 'left' if new_seg_G.nodes[lane_name_list[n]]['on_same_direc'] else 'right'
            new_seg_G.add_edge(lane_name_list[n], lane_name_list[n + 1], edge_type=edge_type1)
            new_seg_G.add_edge(lane_name_list[n + 1], lane_name_list[n], edge_type=edge_type2)

    lane_change_progress_min_dist = 0.0
    lane_change_progress_max_dist = 6.0
    for seg_edge in new_seg_G.edges():
        if new_seg_G.edges[seg_edge]['edge_type'] in ['left', 'right']:
            lane_id = seg_edge[0]
            changed_lane_id = seg_edge[1]

            lane_points = new_seg_G.nodes[lane_id]['point_names']
            changed_lane_points = new_seg_G.nodes[changed_lane_id]['point_names']
            is_inverted = not new_seg_G.nodes[lane_id]['on_same_direc']

            changed_lane_idx = 0
            for p_id in lane_points:
                while (changed_lane_idx < len(changed_lane_points)):
                    changed_p_id = changed_lane_points[changed_lane_idx]

                    if is_inverted:
                        lane_s = -new_G.nodes[p_id]['frenet'][0]
                        changed_lane_s = -new_G.nodes[changed_p_id]['frenet'][0]
                    else:
                        lane_s = new_G.nodes[p_id]['frenet'][0]
                        changed_lane_s = new_G.nodes[changed_p_id]['frenet'][0]

                    if changed_lane_s < lane_s + lane_change_progress_min_dist:
                        changed_lane_idx += 1
                    else:
                        if changed_lane_s < lane_s + lane_change_progress_max_dist:
                            new_G.add_edge(p_id, changed_p_id, edge_type=new_seg_G.edges[seg_edge]['edge_type'])
                        break

    new_G_nodes_pixel = np.array([node['pos'] for node in new_G.nodes().values()])
    new_G_nodes_pixel_id = np.array([key for key in new_G.nodes().keys()])
    if len(new_G_nodes_pixel) == 0:
        return new_seg_G, new_G

    condition1 = np.logical_and(new_G_nodes_pixel[:, 0] > 0, \
                                new_G_nodes_pixel[:, 0] < resolution)
    condition2 = np.logical_and(new_G_nodes_pixel[:, 1] > 0, \
                                new_G_nodes_pixel[:, 1] < resolution)
    condition = np.logical_and(condition1, condition2)
    nodes_ids = new_G_nodes_pixel_id[condition]
    new_G = new_G.subgraph(nodes_ids)

    if world_coord:
        new_G = graph_pixel_to_world_coord(new_G, v_pos, theta, \
                                           resolution=resolution, x_max=x_max, y_max=y_max)
        new_seg_G = graph_pixel_to_world_coord(new_seg_G, v_pos, theta, \
                                               resolution=resolution, x_max=x_max, y_max=y_max)
    else:
        for edge in new_G.edges():
            edge_direc = new_G.nodes[edge[1]]['pos'] - new_G.nodes[edge[0]]['pos']
            new_G.edges[edge]['direc'] = edge_direc
            new_G.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))
        for edge in new_seg_G.edges():
            edge_direc = new_seg_G.nodes[edge[1]]['pos'] - new_seg_G.nodes[edge[0]]['pos']
            new_seg_G.edges[edge]['direc'] = edge_direc
            new_seg_G.edges[edge]['weight'] = float(np.linalg.norm(edge_direc))

    return new_seg_G, new_G

def rainbow_text(x, y, ls, lc, **kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    t = plt.gca().transData
    fig = plt.gcf()

    # horizontal version
    for s, c in zip(ls, lc):
        text = plt.text(x, y, s + " ", color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


def show_pallete_info(seg_img, fontsize=10):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    w, h = seg_img.shape[1], seg_img.shape[0]
    color_idxs = np.arange(1, NUM_COLOR, 1)
    color_code_list = list(color_code)
    pos_w, pos_h = w - 8 * fontsize, 2 * fontsize

    plt.text(pos_w - fontsize / 2, 2 * fontsize * (len(color_idxs)), \
             (" " * 17 + "\n") * (len(color_idxs) + 1), \
             fontsize=fontsize, bbox=dict(boxstyle='round', facecolor='silver', alpha=0.5))

    for c_idx in color_idxs:
        if c_idx != 0:
            color_name = color_code_list[c_idx]
            rainbow_text(pos_w, pos_h, "■ {}".format(color_name).split(), \
                         [np.array(color_code[color_name][0]) / 255, 'white'], fontsize=fontsize)
            pos_h += 2 * fontsize

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T

    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


def lidar_to_histogram_features_new(lidar, x_max=24, y_max=48, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 256 / y_max
        hist_max_per_pixel = 5
        x_meters_max = x_max
        y_meters_max = y_max
        xbins = np.linspace(-2 * x_meters_max, 2 * x_meters_max + 1, \
                            int(2 * x_meters_max * pixels_per_meter + 1))
        ybins = np.linspace(-y_meters_max, 0, int(y_meters_max * pixels_per_meter + 1))
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def bce_loss_with_logits(pred, target):
    pred = torch.sigmoid(pred)
    loss = -(target * torch.log(pred + 1e-6) + (1 - target) * torch.log(1 - pred + 1e-6))
    return loss

def graph_loss_with_logits(pred, target):
    field_idx = 1 + 3 * torch.arange(6)
    vertex_gt = target[:,0]
    edge_gt = target[:,field_idx]

    vertex_ce_loss = bce_loss_with_logits(pred[:,0], vertex_gt)
    edge_ce_loss = bce_loss_with_logits(pred[:,field_idx], edge_gt)
    l2_loss = torch.sqrt((pred[:,field_idx+1] - target[:,field_idx+1])**2 + \
              (pred[:,field_idx+2] - target[:,field_idx+2])**2)

    loss = vertex_ce_loss + vertex_gt * torch.sum(edge_gt * (edge_ce_loss + l2_loss), dim=1)
    return loss

def logits_to_feature(logits):
    result = logits.clone()
    result[:,0] = torch.sigmoid(result[:,0])
    return result.cpu().numpy()

def lidar_colorized_plot(lidar, lidar_painted):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    plt.figure()

    plt.scatter(lidar[:,])
    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def TOPO(gt_H, pred_H, d=4):
    if len(pred_H.nodes()) == 0:
        return 0, 0, 0

    # calculate cost
    gt_len, pred_len = gt_H.number_of_nodes(), pred_H.number_of_nodes()
    if gt_len == 0 or pred_len == 0:
        return 0, 0, 0

    gt_node_pos = np.stack(list(nx.get_node_attributes(gt_H, 'pos').values()))
    pred_node_pos = np.stack(list(nx.get_node_attributes(pred_H, 'pos').values()))

    a = np.expand_dims(gt_node_pos, 1)
    b = np.expand_dims(pred_node_pos, 0)
    dist = np.sqrt(np.sum((a - b) ** 2, axis=2))

    if gt_len > pred_len:
        dist = np.concatenate([dist, np.inf * np.ones([gt_len, gt_len - pred_len])], axis=1)
    elif gt_len < pred_len:
        dist = np.concatenate([dist, np.inf * np.ones([pred_len - gt_len, pred_len])], axis=0)

    cost_mat = (dist > d).astype(int)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()

    matched_pairs = max(gt_len, pred_len) - cost

    assert matched_pairs >= 0

    precision = matched_pairs / gt_len
    recall = matched_pairs / pred_len

    F_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, F_score

def APLS(gt_H, pred_H):
    if len(pred_H.nodes()) == 0:
        return 0
    path_len_gt = dict(nx.all_pairs_dijkstra_path_length(gt_H))
    path_len_pred = dict(nx.all_pairs_dijkstra_path_length(pred_H))

    gt_pos = np.stack(list(dict(gt_H.nodes('pos')).values()))
    gt_pos_keys = np.stack(list(dict(gt_H.nodes('pos')).keys()))
    pred_pos_kdtree = KDTree(np.stack(list(dict(pred_H.nodes('pos')).values())))
    pred_pos_kdtree_keys = np.stack(list(dict(pred_H.nodes('pos')).keys()))
    dist, idxs = pred_pos_kdtree.query(gt_pos, k=1)
    nn_node_list = pred_pos_kdtree_keys[idxs[:, 0]]
    nn_dict = {gt_node: nn_node for gt_node, nn_node in zip(gt_pos_keys, nn_node_list)}

    values_list = []
    for gt_start_node, pairs in path_len_gt.items():
        for gt_end_node, gt_len in pairs.items():
            if gt_len != 0:
                pred_start_node = nn_dict[gt_start_node]
                pred_end_node = nn_dict[gt_end_node]
                try:
                    pred_len = path_len_pred[pred_start_node][pred_end_node]
                except:
                    pred_len = np.inf
                value = min(1, np.abs(gt_len - pred_len) / gt_len)
                values_list.append(value)

    return 1 - np.mean(values_list)

def get_seg_accuracy(gt_image, pred_image, num_class=12):
    assert gt_image.shape[-1] != 3 and gt_image.shape == pred_image.shape
    if len(gt_image.shape) == 3:  # one-hot encoding
        gt_image = gt_image.argmax(-1)
        pred_image = pred_image.argmax(-1)

    mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[mask].astype('int') + pred_image[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)

    np.seterr(invalid='ignore')
    MPA = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    MPA = np.nanmean(MPA)

    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)

    return MPA, MIoU

def get_precision_recall(result, a):
    # a=0.63 for mPA, a=0.46 for mIoU
    tp = np.sum(result[:, 0] >= a)
    fp = np.sum(result[:, 1] >= a)
    fn = np.sum(result[:, 0] < a)
    tn = np.sum(result[:, 1] < a)

    precision = tp / (tp + fp) if tp + fp > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp+fn+fp+tn)

    return precision, recall, f1_score, accuracy


def get_ap(result):
    truths = np.concatenate([np.ones_like(result[:, 0]), np.zeros_like(result[:, 1])])
    scores = np.concatenate([result[:, 0], result[:, 1]])
    ap = average_precision_score(truths, scores)

    return ap

'''
err=np.load('error_fmtc.npy',allow_pickle=True)
result_long_DA = np.load('result_resnet_34_fmtc.npy')

not_on_traffic_signal = np.array([not e['on_traffic_signal'] for e in err])
not_err_in_data = np.array([not e['err_in_data'] for e in err])
filt = np.logical_and(not_on_traffic_signal, not_err_in_data)

r1 = result_long_DA[filt]
print(get_ap(r1[:,[0,1]]))
print('-'*10)

rand_ = (np.array([e['rand_'] for e in err])==4)
r1 = result_long_DA[np.logical_and(filt, rand_)]
print(get_ap(r1[:,[0,1]]))

rand_ = (np.array([e['rand_'] for e in err])==3)
r1 = result_long_DA[np.logical_and(filt, rand_)]
print(get_ap(r1[:,[0,1]]))

rand_ = (np.array([e['rand_'] for e in err])==1)
r1 = result_long_DA[np.logical_and(filt, rand_)]
print(get_ap(r1[:,[0,1]]))

rand_ = (np.array([e['rand_'] for e in err])==2)
r1 = result_long_DA[np.logical_and(filt, rand_)]
print(get_ap(r1[:,[0,1]]))

rand_ = (np.array([e['rand_'] for e in err])==0)
r1 = result_long_DA[np.logical_and(filt, rand_)]
print(get_ap(r1[:,[0,1]]))

rand_ = (np.array([e['rand_'] for e in err])>=5)
r1 = result_long_DA[np.logical_and(filt, rand_)]
print(get_ap(r1[:,[0,1]]))

'''