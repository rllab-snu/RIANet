import os
import json
from PIL import Image

import glob
import numpy as np
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0,'..')
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.other_utils import *
from utils.localization import *

import xml.etree.ElementTree as ET
from collections import deque
from tqdm import tqdm

class CARLA_DATA(Dataset):
    def __init__(self, root, args):

        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = 4
        self.input_resolution = 256
        self.scale_front = 2   # 800, 600 -> 400, 300
        self.scale = 1

        self.ignore_sides = True
        self.ignore_rear = True

        self.data_history_files = glob.glob(os.path.join(root, 'data_history_*.npy'))
        self.data_history_files.sort()

        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.lidar_raw = []
        self.pos = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.velocity = []
        self.target_speed = []
        self.command = []
        self.graph_feature = []

        self.data_history_files = self.data_history_files if len(self.data_history_files) < 40 else self.data_history_files[:20] + self.data_history_files[-20:]
        for file_name in tqdm(self.data_history_files):
            processed_file_name = file_name.replace('data_history', 'data_processed')
            image_folder_name = file_name.replace('data_history', 'Snaps').replace('.npy', '')

            graph_folder_name = image_folder_name.replace('Snaps', 'Processed_Graph_Data')
            if not os.path.isdir(graph_folder_name):
                os.makedirs(graph_folder_name)

            # dump to npy if no preload
            if not os.path.exists(processed_file_name):
                preload_front = []
                preload_left = []
                preload_right = []
                preload_rear = []
                preload_lidar_raw = []
                preload_pos = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_velocity = []
                preload_target_speed = []
                preload_command = []
                preload_graph_feature = []

                # num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-self.pred_len-2)//self.seq_len
                num_seq = len(glob.glob(image_folder_name + '/*_Center.png'))
                data = np.load(file_name, allow_pickle=True)
                data = data.tolist()

                pos_data = get_localization(data, method='filter')[:, :2]

                # get graph info
                root = ET.fromstring(data['HD-map'])
                geo_ref_dict = get_georeference(root)

                initial_gps = data['GPS'][0]
                initial_pos = gnss_to_xy(initial_gps, geo_ref_dict)[:2].reshape([1, 2])

                G, seg_G, processed_roads, i = None, None, [], 0
                signal_object_dict, signal_reference_dict = None, None
                while len(processed_roads) < len(root.findall('road')):
                    i += 1
                    G, seg_G, pos_kdtree_keys, pos_kdtree, processed_roads, signal_object_dict, signal_reference_dict = \
                        extract_graph_info_partial(root, cutoff_dist=200 * i, vehicle_pos=initial_pos.squeeze(),
                                                   prev_seg_G=seg_G, prev_G=G, processed_roads=processed_roads,
                                                   signal_object_dict=signal_object_dict,
                                                   signal_reference_dict=signal_reference_dict)

                G2, pos_kdtree_keys2, pos_kdtree2 = contract_graph(G, pos_kdtree_keys, pos_kdtree)

                global_plan_pos = data['global_plan_pos']
                traj, traj_node_id, progress, (plan_start_idx, plan_end_idx) = get_global_traj(global_plan_pos, G,
                                                                                               seg_G, pos_kdtree_keys,
                                                                                               pos_kdtree)
                target_points = get_waypoint_command(global_plan_pos, data['pos'][:,:2], traj, progress, plan_start_idx)

                # get image data
                for seq in range(self.seq_len + 1, num_seq - self.pred_len, 1):
                    fronts = []
                    lefts = []
                    rights = []
                    rears = []
                    lidar_raws = []

                    # read files sequentially (past and current frames)
                    for i in range(self.seq_len):
                        # images
                        fronts.append(image_folder_name + '/{:06d}_Center.png'.format(seq - self.seq_len + i + 1))
                        lefts.append(image_folder_name + '/{:06d}_left.png'.format(seq - self.seq_len + i + 1))
                        rights.append(image_folder_name + '/{:06d}_right.png'.format(seq - self.seq_len + i + 1))
                        rears.append(image_folder_name + '/{:06d}_rear.png'.format(seq - self.seq_len + i + 1))

                        # point cloud
                        lidar_raws.append(image_folder_name + '/{:06d}_lidar_raw.npy'.format(seq - self.seq_len + i + 1))

                    preload_front.append(fronts)
                    preload_left.append(lefts)
                    preload_right.append(rights)
                    preload_rear.append(rears)
                    preload_lidar_raw.append(lidar_raws)

                    poss = []
                    thetas = []
                    for i in range(self.seq_len + self.pred_len):
                        poss.append(pos_data[seq - self.seq_len + i + 1])
                        theta = data['IMU'][seq - self.seq_len + i + 1][-1]
                        theta = 0 if np.isnan(theta) else theta
                        thetas.append(theta)
                    preload_pos.append(poss)
                    preload_theta.append(thetas)

                    preload_steer.append(data['control'][seq]['steer'])
                    preload_throttle.append(data['control'][seq]['throttle'])
                    preload_brake.append(data['control'][seq]['brake'])
                    preload_velocity.append(data['speed'][seq])
                    preload_target_speed.append(data['control'][seq]['target_speed'])
                    preload_command.append(target_points[seq])

                    # get graph features
                    cur_theta = data['IMU'][seq][-1]
                    cur_theta = 0 if np.isnan(cur_theta) else cur_theta
                    graph_feature_dict = extract_graph_features(pos_data[seq], 0.5 * np.pi - cur_theta, data['speed'][seq], \
                                                    G2, pos_kdtree_keys2, pos_kdtree2, traj_node_id, use_node_filter=True)

                    graph_feature_file_name = graph_folder_name + '/{:06d}_graph_feature_dict.npy'.format(seq)
                    preload_graph_feature.append(graph_feature_file_name)

                    # save graph features
                    np.save(graph_feature_file_name, graph_feature_dict)

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['lidar_raw'] = preload_lidar_raw
                preload_dict['pos'] = preload_pos
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['velocity'] = preload_velocity
                preload_dict['target_speed'] = preload_target_speed
                preload_dict['command'] = preload_command
                preload_dict['graph_feature'] = preload_graph_feature
                np.save(processed_file_name, preload_dict)

            # load from npy if available
            preload_dict = np.load(processed_file_name, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.rear += preload_dict.item()['rear']
            self.lidar_raw += preload_dict.item()['lidar_raw']
            self.pos += preload_dict.item()['pos']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.velocity += preload_dict.item()['velocity']
            self.target_speed += preload_dict.item()['target_speed']
            self.command += preload_dict.item()['command']
            self.graph_feature += preload_dict.item()['graph_feature']

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['rears'] = []
        data['lidars'] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_rears = self.rear[index]
        seq_lidar_raws = self.lidar_raw[index]
        seq_pos = self.pos[index]
        seq_theta = self.theta[index]

        full_lidar = []
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.scale_front, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['rears'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rears[i]), scale=self.scale, crop=self.input_resolution))))

            lidar_unprocessed = np.load(seq_lidar_raws[i])[..., :3]  # lidar: XYZI
            full_lidar.append(lidar_unprocessed)

        ego_x = seq_pos[self.seq_len-1][0]
        ego_y = seq_pos[self.seq_len-1][1]
        ego_theta = seq_theta[self.seq_len-1]

        # future frames
        #for i in range(self.seq_len, self.seq_len + self.pred_len):
        #    lidar_unprocessed = np.load(seq_lidars[i])
        #    full_lidar.append(lidar_unprocessed)
        #
        # lidar and waypoint processing to local coordinates
        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1, 3)),
                                                 np.pi / 2 - seq_theta[i], -seq_pos[i][0], -seq_pos[i][1], np.pi / 2 - ego_theta,
                                                 -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0, :2]))

            if i < self.seq_len:
                # convert coordinate frame of point cloud
                full_lidar[i][:, 1] *= -1  # inverts x, y
                full_lidar[i] = transform_2d_points(full_lidar[i],
                                                    np.pi / 2 - seq_theta[i], -seq_pos[i][0], -seq_pos[i][1],
                                                    np.pi / 2 - ego_theta, -ego_x, -ego_y)
                lidar_processed = lidar_to_histogram_features(full_lidar[i], crop=self.input_resolution)
                data['lidars'].append(torch.from_numpy(lidar_processed))
        data['waypoints'] = waypoints

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])
        local_command_point = np.array([self.command[index][0] - ego_x, self.command[index][1] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = tuple(local_command_point)

        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['velocity'] = self.velocity[index]
        data['target_speed'] = self.target_speed[index]
        data['command'] = self.command[index]

        data['graph_feature'] = self.graph_feature[index]
        graph_feature_dict = np.load(self.graph_feature[index], allow_pickle=True).item()
        data['adjacency_matrix'] = torch.from_numpy(graph_feature_dict['adjacency_matrix'])
        data['node_feature_matrix'] = torch.from_numpy(graph_feature_dict['node_feature_matrix'])
        data['edge_feature_matrix'] = torch.from_numpy(graph_feature_dict['edge_feature_matrix'])
        data['node_num'] = graph_feature_dict['node_num']

        return data

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-2 * x_meters_max, 2 * x_meters_max + 1, 2 * x_meters_max * pixels_per_meter + 1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
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