import os
import json
import cv2
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import sys

sys.path.insert(0,'..')
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.other_utils import *
from utils.localization import *

import xml.etree.ElementTree as ET

class CARLA_Data(Dataset):

    def __init__(self, root, config, args):

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.ignore_sides = config.ignore_sides
        self.ignore_rear = config.ignore_rear

        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.use_2d_detection = args.use_2d_detection
        self.use_3d_detection = args.use_3d_detection
        
        self.lidar = []
        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []
        self.graph_feature = []

        self.detection_2D = []

        for sub_root in tqdm(root, file=sys.stdout):
            preload_file = os.path.join(sub_root,
                                        'rg_lidar_diag_pl_' + str(self.seq_len) + '_' + str(self.pred_len) + '.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_front = []
                preload_left = []
                preload_right = []
                preload_rear = []
                preload_lidar = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_command = []
                preload_velocity = []

                preload_graph_feature = []
                preload_detection_2D = []

                # get graph info
                town_name = 't' + sub_root.split('/')[-1].split('_')[0][1:]
                root = ET.parse('../routes/HD-map/{}.xml'.format(town_name)).getroot()

                initial_pos = np.zeros([1, 2])

                G, seg_G, processed_roads, i = None, None, [], 0
                signal_object_dict, signal_reference_dict = None, None
                while len(processed_roads) < len(root.findall('road')):
                    i += 1
                    G, seg_G, pos_kdtree_keys, pos_kdtree, processed_roads, signal_object_dict, signal_reference_dict = \
                        extract_graph_info_partial(root, cutoff_dist=400 * i, vehicle_pos=initial_pos.squeeze(),
                                                   prev_seg_G=seg_G, prev_G=G, processed_roads=processed_roads,
                                                   signal_object_dict=signal_object_dict,
                                                   signal_reference_dict=signal_reference_dict)

                G2, pos_kdtree_keys2, pos_kdtree2 = contract_graph(G, pos_kdtree_keys, pos_kdtree)

                # list sub-directories in root
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    # subtract final frames (pred_len) since there are no future waypoints
                    # first frame of sequence not used

                    num_seq = (len(os.listdir(route_dir + "/rgb_front/")) - self.pred_len - 2) // self.seq_len

                    if num_seq < 0:
                        continue

                    for seq in range(num_seq):
                        fronts = []
                        lefts = []
                        rights = []
                        rears = []
                        lidars = []
                        xs = []
                        ys = []
                        thetas = []

                        # read files sequentially (past and current frames)
                        for i in range(self.seq_len):
                            # images
                            filename = f"{str(seq * self.seq_len + 1 + i).zfill(4)}.png"
                            fronts.append(route_dir + "/rgb_front/" + filename)
                            lefts.append(route_dir + "/rgb_left/" + filename)
                            rights.append(route_dir + "/rgb_right/" + filename)
                            rears.append(route_dir + "/rgb_rear/" + filename)

                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq * self.seq_len + 1 + i).zfill(4)}.npy")

                            # position
                            with open(route_dir + f"/measurements/{str(seq * self.seq_len + 1 + i).zfill(4)}.json",
                                      "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])

                        # read files sequentially (future frames)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq * self.seq_len + 1 + i).zfill(4)}.npy")

                            # position
                            with open(route_dir + f"/measurements/{str(seq * self.seq_len + 1 + i).zfill(4)}.json",
                                      "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_front.append(fronts)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_rear.append(rears)
                        preload_lidar.append(lidars)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                    # get graph features
                    all_num_seq = len(os.listdir(route_dir + "/rgb_front/"))
                    route_x_command, route_y_command = [], []
                    route_x, route_y = [], []
                    route_theta, route_speed = [], []
                    for seq in range(all_num_seq):
                        # position
                        with open(route_dir + f"/measurements/{str(seq).zfill(4)}.json", "r") as read_file:
                            data = json.load(read_file)
                        route_x_command.append(data['x_command'])
                        route_y_command.append(data['y_command'])
                        route_x.append(data['x'])
                        route_y.append(data['y'])
                        route_theta.append(data['theta'])
                        route_speed.append(data['speed'])

                    # x and y are inverted
                    route_pos = np.stack([np.array(route_y), np.array(route_x)], axis=1)
                    global_plan_pos = np.stack([np.array(route_y_command), np.array(route_x_command)], axis=1)
                    global_plan_pos = np.unique(global_plan_pos, axis=0)
                    global_plan_pos_with_starting = np.concatenate([route_pos[0].reshape(1, 2), global_plan_pos])

                    traj, traj_node_id, progress, _ = get_global_traj(global_plan_pos_with_starting, G, seg_G,
                                                                      pos_kdtree_keys, pos_kdtree)

                    for seq in range(all_num_seq):
                        # save graph features
                        if not os.path.exists(route_dir + "/graph_data"):
                            os.mkdir(route_dir + "/graph_data")

                        cur_pos, cur_theta, cur_speed = route_pos[seq], route_theta[seq], route_speed[seq]  # x, y are inverted
                        cur_theta = 0 if np.isnan(cur_theta) else cur_theta
                        graph_feature_dict, _ = extract_graph_features(cur_pos, 0.5 * np.pi - cur_theta,
                                                                    cur_speed, \
                                                                    G2, pos_kdtree_keys2, pos_kdtree2, traj_node_id,
                                                                    nearest_node_num=96, use_node_filter=True)

                        np.save(route_dir + "/graph_data/{:04d}.npy".format(seq), graph_feature_dict)
                        if seq < num_seq:
                            preload_graph_feature.append(route_dir + "/graph_data/{:04d}.npy".format(seq * self.seq_len + 1))

                    for seq in range(num_seq):
                        preload_detection_2D.append(route_dir + "/pre_detected_data/{:04d}.npy".format(seq * self.seq_len + 1))

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['lidar'] = preload_lidar
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                preload_dict['graph_feature'] = preload_graph_feature
                preload_dict['detection_2D'] = preload_detection_2D
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.rear += preload_dict.item()['rear']
            self.lidar += preload_dict.item()['lidar']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.command += preload_dict.item()['command']
            self.velocity += preload_dict.item()['velocity']
            self.graph_feature += preload_dict.item()['graph_feature']
            if self.use_2d_detection > 0:
                self.detection_2D += preload_dict.item()['detection_2D']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

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
        seq_lidars = self.lidar[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        full_lidar = []
        pos = []
        neg = []
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['rears'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rears[i]), scale=self.scale, crop=self.input_resolution))))

            lidar_unprocessed = np.load(seq_lidars[i])[..., :3]  # lidar: XYZI
            full_lidar.append(lidar_unprocessed)

            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # future frames
        for i in range(self.seq_len, self.seq_len + self.pred_len):
            lidar_unprocessed = np.load(seq_lidars[i])
            full_lidar.append(lidar_unprocessed)

            # lidar and waypoint processing to local coordinates
        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1, 3)),
                                                 np.pi / 2 - seq_theta[i], -seq_x[i], -seq_y[i], np.pi / 2 - ego_theta,
                                                 -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0, :2]))

            # process only past lidar point clouds
            if i < self.seq_len:
                # convert coordinate frame of point cloud
                full_lidar[i][:, 1] *= -1  # inverts x, y
                full_lidar[i] = transform_2d_points(full_lidar[i],
                                                    np.pi / 2 - seq_theta[i], -seq_x[i], -seq_y[i],
                                                    np.pi / 2 - ego_theta, -ego_x, -ego_y)
                lidar_processed = lidar_to_histogram_features(full_lidar[i], crop=self.input_resolution)
                data['lidars'].append(lidar_processed)

        data['waypoints'] = waypoints

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])
        local_command_point = np.array([self.x_command[index] - ego_x, self.y_command[index] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = tuple(local_command_point)

        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['command'] = self.command[index]
        data['velocity'] = self.velocity[index]

        waypoint_stack = np.stack(waypoints)
        data['target_speed'] = np.linalg.norm(waypoint_stack[0] - waypoint_stack[1])

        # load graph features
        graph_feature_dict = np.load(self.graph_feature[index], allow_pickle=True).item()
        data['adjacency_matrix'] = torch.from_numpy(graph_feature_dict['adjacency_matrix'])
        data['node_feature_matrix'] = torch.from_numpy(graph_feature_dict['node_feature_matrix'])
        data['edge_feature_matrix'] = torch.from_numpy(graph_feature_dict['edge_feature_matrix'])
        data['node_num'] = graph_feature_dict['node_num']

        # load detection features
        if self.use_2d_detection > 0:
            detection_2D = np.load(self.detection_2D[index], allow_pickle=True)
            detection_result = get_detection_patch(detection_2D)
            for d_key in detection_result.keys():
                data[d_key] = detection_result[d_key]

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

def scale_and_crop_np_image(image, scale=1, crop=256):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, dsize=(width, height))
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = im_resized[start_x:start_x + crop, start_y:start_y + crop]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def get_detection_patch(detection_2D, height=300, width=400, max_obj_num=12):
    td_img_class = np.zeros([max_obj_num, 4], dtype=float)  # red, yellow, green, off
    td_img_mask = np.zeros([height, width, max_obj_num], dtype=int)
    td_img_mask_info = np.zeros([max_obj_num, 4], dtype=int)
    od_img_class = np.zeros([max_obj_num, 4], dtype=float)  # car, person, motorcycle, bicycle
    od_img_mask = np.zeros([height, width, max_obj_num], dtype=int)
    od_img_mask_info = np.zeros([max_obj_num, 4], dtype=int)

    td_idx, od_idx = 0, 0
    for obj in detection_2D:
        if 'state' in obj.keys() and td_idx < max_obj_num:  # traffic light
            x_min, x_max, y_min, y_max = obj['x_min'], obj['x_max'], obj['y_min'], obj['y_max']
            td_img_mask_info[td_idx] = np.array([(x_max+x_min)//2-128, (y_max+y_min)//2-128, (x_max-x_min), (y_max-y_min)])
            if obj['state'] == 'RED':
                td_img_class[td_idx, 0] = 1.0
                td_img_mask[y_min:y_max,x_min:x_max,td_idx] = 1
                td_idx += 1
            elif obj['state'] == 'YELLOW':
                td_img_class[td_idx, 1] = 1.0
                td_img_mask[y_min:y_max,x_min:x_max,td_idx] = 1
                td_idx += 1
            elif obj['state'] == 'GREEN':
                td_img_class[td_idx, 2] = 1.0
                td_img_mask[y_min:y_max,x_min:x_max,td_idx] = 1
                td_idx += 1
            else:
                td_img_class[td_idx, 3] = 1.0
                td_img_mask[y_min:y_max,x_min:x_max,td_idx] = 1
                td_idx += 1
        elif 'label' in obj.keys() and od_idx < max_obj_num:  # obstacle light
            x_min, x_max, y_min, y_max = obj['x_min'], obj['x_max'], obj['y_min'], obj['y_max']
            od_img_mask_info[od_idx] = np.array([(x_max+x_min)//2-128, (y_max+y_min)//2-128, (x_max-x_min), (y_max-y_min)])
            if obj['label'] == 'car':
                od_img_class[od_idx, 0] = 1.0
                od_img_mask[y_min:y_max, x_min:x_max, od_idx] = 1
                od_idx += 1
            elif obj['label'] == 'person':
                od_img_class[od_idx, 1] = 1.0
                od_img_mask[y_min:y_max, x_min:x_max, od_idx] = 1
                od_idx += 1
            elif obj['label'] == 'motorcycle':
                od_img_class[od_idx, 2] = 1.0
                od_img_mask[y_min:y_max, x_min:x_max, od_idx] = 1
                od_idx += 1
            elif obj['label'] == 'bicycle':
                od_img_class[od_idx, 3] =1.0
                od_img_mask[y_min:y_max, x_min:x_max, od_idx] = 1
                od_idx += 1

    td_img_class = torch.from_numpy(td_img_class).to(dtype=torch.float)  # obj x 4
    td_img_mask = torch.from_numpy(scale_and_crop_np_image(td_img_mask))  # obj x 256 x 256
    td_img_mask_info = torch.from_numpy(td_img_mask_info).to(dtype=torch.float)  # obj x 4
    od_img_class = torch.from_numpy(od_img_class).to(dtype=torch.float)  # obj x 4
    od_img_mask = torch.from_numpy(scale_and_crop_np_image(od_img_mask))  # obj x 256 x 256
    od_img_mask_info = torch.from_numpy(od_img_mask_info).to(dtype=torch.float)  # obj x 4

    result_dict = {}
    result_dict['td_img_class'] = td_img_class
    result_dict['td_img_mask'] = td_img_mask
    result_dict['td_img_mask_info'] = td_img_mask_info
    result_dict['td_num'] = td_idx
    result_dict['od_img_class'] = od_img_class
    result_dict['od_img_mask'] = od_img_mask
    result_dict['od_img_mask_info'] = od_img_mask_info
    result_dict['od_num'] = od_idx

    return result_dict
