import os
import json
import cv2
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import sys

sys.path.insert(0, '..')
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.other_utils import *
from utils.localization import *

import xml.etree.ElementTree as ET


class CARLA_Data(Dataset):

    def __init__(self, root, config):

        self.perception_type = config.perception_type
        self.use_sem_image = config.use_sem_image
        self.use_DA_sem = config.use_DA_sem
        self.x_max = config.x_max
        self.y_max = config.y_max
        self.ignore_sides = config.ignore_sides
        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.bev = []
        self.front = []
        self.left = []
        self.right = []
        self.lidar = []
        self.graph_feature_gt = []
        self.graph_feature_err = []
        self.sem_image_gt_front = []
        self.sem_image_gt_left = []
        self.sem_image_gt_right = []
        self.sem_image_err_front = []
        self.sem_image_err_left = []
        self.sem_image_err_right = []
        self.sem_image_pred_front = []
        self.sem_image_pred_left = []
        self.sem_image_pred_right = []
        self.sem_image_DA_gt_front = []
        self.sem_image_DA_gt_left = []
        self.sem_image_DA_gt_right = []
        self.sem_image_DA_err_front = []
        self.sem_image_DA_err_left = []
        self.sem_image_DA_err_right = []
        self.sem_image_DA_pred_front = []
        self.sem_image_DA_pred_left = []
        self.sem_image_DA_pred_right = []


        for sub_root in tqdm(root, file=sys.stdout):
            preload_file = os.path.join(sub_root,
                                        'error_detection_preload.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_bev = []
                preload_front = []
                preload_left = []
                preload_right = []
                preload_lidar = []
                preload_graph_feature_gt = []
                preload_graph_feature_err = []
                preload_sem_image_gt_front = []
                preload_sem_image_gt_left = []
                preload_sem_image_gt_right = []
                preload_sem_image_err_front = []
                preload_sem_image_err_left = []
                preload_sem_image_err_right = []
                preload_sem_image_pred_front = []
                preload_sem_image_pred_left = []
                preload_sem_image_pred_right = []
                preload_sem_image_DA_gt_front = []
                preload_sem_image_DA_gt_left = []
                preload_sem_image_DA_gt_right = []
                preload_sem_image_DA_err_front = []
                preload_sem_image_DA_err_left = []
                preload_sem_image_DA_err_right = []
                preload_sem_image_DA_pred_front = []
                preload_sem_image_DA_pred_left = []
                preload_sem_image_DA_pred_right = []
                
                # list sub-directories in root
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    # subtract final frames (pred_len) since there are no future waypoints
                    # first frame of sequence not used

                    num_seq = (len(os.listdir(route_dir + "/rgb_front/")) - 2)

                    for seq in range(num_seq):
                        bevs = []
                        fronts = []
                        lefts = []
                        rights = []
                        lidars = []
                        graph_features_gts = []
                        graph_features_errs = []
                        sem_image_gt_fronts = []
                        sem_image_gt_lefts = []
                        sem_image_gt_rights = []
                        sem_image_err_fronts = []
                        sem_image_err_lefts = []
                        sem_image_err_rights = []
                        sem_image_pred_fronts = []
                        sem_image_pred_lefts = []
                        sem_image_pred_rights = []
                        sem_image_DA_gt_fronts = []
                        sem_image_DA_gt_lefts = []
                        sem_image_DA_gt_rights = []
                        sem_image_DA_err_fronts = []
                        sem_image_DA_err_lefts = []
                        sem_image_DA_err_rights = []
                        sem_image_DA_pred_fronts = []
                        sem_image_DA_pred_lefts = []
                        sem_image_DA_pred_rights = []

                        # read files sequentially (past and current frames)
                        filename = f"{str(seq + 1).zfill(4)}.png"
                        
                        # check error
                        #check_dict = np.load(route_dir + "/graph_pixel/" + f"{str(seq + 1).zfill(4)}_graph_err_type.npy", allow_pickle=True)
                        #if all(['error' in c for c in check_dict]):
                        #    continue
                        
                        bevs.append(route_dir + "/bev_plot_3/" + f"{str(seq + 1).zfill(4)}_bev_plot.png")
                        fronts.append(route_dir + "/rgb_front/" + filename)
                        lefts.append(route_dir + "/rgb_left/" + filename)
                        rights.append(route_dir + "/rgb_right/" + filename)

                        # point cloud
                        lidars.append(route_dir + f"/lidar/{str(seq + 1).zfill(4)}.npy")

                        # graph feature
                        graph_features_gts.append(route_dir + f"/graph_pixel/{str(seq + 1).zfill(4)}_graph_gt.npy")
                        graph_features_errs.append(route_dir + f"/graph_pixel/{str(seq + 1).zfill(4)}_graph_err.npy")

                        sem_image_gt_fronts.append(route_dir + f"/sem_front/{str(seq + 1).zfill(4)}_gt_labelIds.png")
                        sem_image_gt_lefts.append(route_dir + f"/sem_left/{str(seq + 1).zfill(4)}_gt_labelIds.png")
                        sem_image_gt_rights.append(route_dir + f"/sem_right/{str(seq + 1).zfill(4)}_gt_labelIds.png")
                        sem_image_err_fronts.append(route_dir + f"/sem_front/{str(seq + 1).zfill(4)}_err_labelIds.png")
                        sem_image_err_lefts.append(route_dir + f"/sem_left/{str(seq + 1).zfill(4)}_err_labelIds.png")
                        sem_image_err_rights.append(route_dir + f"/sem_right/{str(seq + 1).zfill(4)}_err_labelIds.png")                        
                        sem_image_pred_fronts.append(route_dir + f"/pred_sem_front/{str(seq + 1).zfill(4)}_pred_labelIds.png")
                        sem_image_pred_lefts.append(route_dir + f"/pred_sem_left/{str(seq + 1).zfill(4)}_pred_labelIds.png")
                        sem_image_pred_rights.append(route_dir + f"/pred_sem_right/{str(seq + 1).zfill(4)}_pred_labelIds.png")

                        sem_image_DA_gt_fronts.append(route_dir + f"/sem_front/{str(seq + 1).zfill(4)}_gt_labelIds_DA.png")
                        sem_image_DA_gt_lefts.append(route_dir + f"/sem_left/{str(seq + 1).zfill(4)}_gt_labelIds_DA.png")
                        sem_image_DA_gt_rights.append(route_dir + f"/sem_right/{str(seq + 1).zfill(4)}_gt_labelIds_DA.png")
                        sem_image_DA_err_fronts.append(route_dir + f"/sem_front/{str(seq + 1).zfill(4)}_err_labelIds_DA.png")
                        sem_image_DA_err_lefts.append(route_dir + f"/sem_left/{str(seq + 1).zfill(4)}_err_labelIds_DA.png")
                        sem_image_DA_err_rights.append(route_dir + f"/sem_right/{str(seq + 1).zfill(4)}_err_labelIds_DA.png")                        
                        sem_image_DA_pred_fronts.append(route_dir + f"/pred_sem_front/{str(seq + 1).zfill(4)}_pred_labelIds_DA.png")
                        sem_image_DA_pred_lefts.append(route_dir + f"/pred_sem_left/{str(seq + 1).zfill(4)}_pred_labelIds_DA.png")
                        sem_image_DA_pred_rights.append(route_dir + f"/pred_sem_right/{str(seq + 1).zfill(4)}_pred_labelIds_DA.png")
                        
                        preload_bev.append(bevs)
                        preload_front.append(fronts)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_lidar.append(lidars)
                        preload_graph_feature_gt.append(graph_features_gts)
                        preload_graph_feature_err.append(graph_features_errs)
                        preload_sem_image_gt_front.append(sem_image_gt_fronts)
                        preload_sem_image_gt_left.append(sem_image_gt_lefts)
                        preload_sem_image_gt_right.append(sem_image_gt_rights)
                        preload_sem_image_err_front.append(sem_image_err_fronts)
                        preload_sem_image_err_left.append(sem_image_err_lefts)
                        preload_sem_image_err_right.append(sem_image_err_rights)
                        preload_sem_image_pred_front.append(sem_image_pred_fronts)
                        preload_sem_image_pred_left.append(sem_image_pred_lefts)
                        preload_sem_image_pred_right.append(sem_image_pred_rights)
                        preload_sem_image_DA_gt_front.append(sem_image_DA_gt_fronts)
                        preload_sem_image_DA_gt_left.append(sem_image_DA_gt_lefts)
                        preload_sem_image_DA_gt_right.append(sem_image_DA_gt_rights)
                        preload_sem_image_DA_err_front.append(sem_image_DA_err_fronts)
                        preload_sem_image_DA_err_left.append(sem_image_DA_err_lefts)
                        preload_sem_image_DA_err_right.append(sem_image_DA_err_rights)
                        preload_sem_image_DA_pred_front.append(sem_image_DA_pred_fronts)
                        preload_sem_image_DA_pred_left.append(sem_image_DA_pred_lefts)
                        preload_sem_image_DA_pred_right.append(sem_image_DA_pred_rights)

                # dump to npy
                preload_dict = {}
                preload_dict['bev'] = preload_bev
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['lidar'] = preload_lidar
                preload_dict['graph_feature_gt'] = preload_graph_feature_gt
                preload_dict['graph_feature_err'] = preload_graph_feature_err
                preload_dict['sem_image_gt_front'] = preload_sem_image_gt_front
                preload_dict['sem_image_gt_left'] = preload_sem_image_gt_left
                preload_dict['sem_image_gt_right'] = preload_sem_image_gt_right
                preload_dict['sem_image_err_front'] = preload_sem_image_err_front
                preload_dict['sem_image_err_left'] = preload_sem_image_err_left
                preload_dict['sem_image_err_right'] = preload_sem_image_err_right
                preload_dict['sem_image_pred_front'] = preload_sem_image_pred_front
                preload_dict['sem_image_pred_left'] = preload_sem_image_pred_left
                preload_dict['sem_image_pred_right'] = preload_sem_image_pred_right
                preload_dict['sem_image_DA_gt_front'] = preload_sem_image_DA_gt_front
                preload_dict['sem_image_DA_gt_left'] = preload_sem_image_DA_gt_left
                preload_dict['sem_image_DA_gt_right'] = preload_sem_image_DA_gt_right
                preload_dict['sem_image_DA_err_front'] = preload_sem_image_DA_err_front
                preload_dict['sem_image_DA_err_left'] = preload_sem_image_DA_err_left
                preload_dict['sem_image_DA_err_right'] = preload_sem_image_DA_err_right
                preload_dict['sem_image_DA_pred_front'] = preload_sem_image_DA_pred_front
                preload_dict['sem_image_DA_pred_left'] = preload_sem_image_DA_pred_left
                preload_dict['sem_image_DA_pred_right'] = preload_sem_image_DA_pred_right

                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.bev += preload_dict.item()['bev']
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.lidar += preload_dict.item()['lidar']
            self.graph_feature_gt += preload_dict.item()['graph_feature_gt']
            self.graph_feature_err += preload_dict.item()['graph_feature_err']
            self.sem_image_gt_front += preload_dict.item()['sem_image_gt_front']
            self.sem_image_gt_left += preload_dict.item()['sem_image_gt_left']
            self.sem_image_gt_right += preload_dict.item()['sem_image_gt_right']
            self.sem_image_err_front += preload_dict.item()['sem_image_err_front']
            self.sem_image_err_left += preload_dict.item()['sem_image_err_left']
            self.sem_image_err_right += preload_dict.item()['sem_image_err_right']
            self.sem_image_pred_front += preload_dict.item()['sem_image_pred_front']
            self.sem_image_pred_left += preload_dict.item()['sem_image_pred_left']
            self.sem_image_pred_right += preload_dict.item()['sem_image_pred_right']
            self.sem_image_DA_gt_front += preload_dict.item()['sem_image_DA_gt_front']
            self.sem_image_DA_gt_left += preload_dict.item()['sem_image_DA_gt_left']
            self.sem_image_DA_gt_right += preload_dict.item()['sem_image_DA_gt_right']
            self.sem_image_DA_err_front += preload_dict.item()['sem_image_DA_err_front']
            self.sem_image_DA_err_left += preload_dict.item()['sem_image_DA_err_left']
            self.sem_image_DA_err_right += preload_dict.item()['sem_image_DA_err_right']
            self.sem_image_DA_pred_front += preload_dict.item()['sem_image_DA_pred_front']
            self.sem_image_DA_pred_left += preload_dict.item()['sem_image_DA_pred_left']
            self.sem_image_DA_pred_right += preload_dict.item()['sem_image_DA_pred_right']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['bevs'] = []
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['lidars'] = []
        data['graph_feature_gts'] = []
        data['graph_feature_errs'] = []
        data['sem_image_gt_fronts'] = []
        data['sem_image_gt_lefts'] = []
        data['sem_image_gt_rights'] = []
        data['sem_image_err_fronts'] = []
        data['sem_image_err_lefts'] = []
        data['sem_image_err_rights'] = []
        data['sem_image_pred_fronts'] = []
        data['sem_image_pred_lefts'] = []
        data['sem_image_pred_rights'] = []

        seq_bevs = self.bev[index]
        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_lidars = self.lidar[index]

        seq_graph_feature_gts = self.graph_feature_gt[index]
        seq_graph_feature_errs = self.graph_feature_err[index]

        if self.use_DA_sem:
            seq_sem_image_gt_fronts = self.sem_image_DA_gt_front[index]
            seq_sem_image_gt_lefts = self.sem_image_DA_gt_left[index]
            seq_sem_image_gt_rights = self.sem_image_DA_gt_right[index]
            seq_sem_image_err_fronts = self.sem_image_DA_err_front[index]
            seq_sem_image_err_lefts = self.sem_image_DA_err_left[index]
            seq_sem_image_err_rights = self.sem_image_DA_err_right[index]
            seq_sem_image_pred_fronts = self.sem_image_DA_pred_front[index]
            seq_sem_image_pred_lefts = self.sem_image_DA_pred_left[index]
            seq_sem_image_pred_rights = self.sem_image_DA_pred_right[index]
            onehot = 4
        else:
            seq_sem_image_gt_fronts = self.sem_image_gt_front[index]
            seq_sem_image_gt_lefts = self.sem_image_gt_left[index]
            seq_sem_image_gt_rights = self.sem_image_gt_right[index]
            seq_sem_image_err_fronts = self.sem_image_err_front[index]
            seq_sem_image_err_lefts = self.sem_image_err_left[index]
            seq_sem_image_err_rights = self.sem_image_err_right[index]
            seq_sem_image_pred_fronts = self.sem_image_pred_front[index]
            seq_sem_image_pred_lefts = self.sem_image_pred_left[index]
            seq_sem_image_pred_rights = self.sem_image_pred_right[index]
            onehot = 12
        
        if self.perception_type == 'bev':
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_bevs[0]), scale=self.scale, crop=self.input_resolution))))
        elif self.perception_type == 'pred-sem':            
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_sem_image_pred_fronts[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_pred_lefts[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_pred_rights[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
        elif self.perception_type == 'gt-sem':
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_sem_image_gt_fronts[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_gt_lefts[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_gt_rights[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
        else:
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[0]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[0]), scale=self.scale, crop=self.input_resolution))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[0]), scale=self.scale, crop=self.input_resolution))))

        lidar_unprocessed = np.load(seq_lidars[0])[..., :3]  # lidar: XYZI

        # convert coordinate frame of point cloud
        lidar_unprocessed[:, 1] *= -1  # inverts x, y
        lidar_unprocessed = transform_2d_points(lidar_unprocessed,
                                            np.pi / 2, 0, 0,
                                            np.pi / 2, 0, 0)
        lidar_processed = lidar_to_histogram_features_new(lidar_unprocessed, x_max=self.x_max, y_max=self.y_max, crop=self.input_resolution)
        lidar_processed = np.rot90(lidar_processed, k=-1, axes=(1,2)).copy()
        data['lidars'].append(lidar_processed)

        # load graph features
        if self.use_sem_image:
            data['sem_image_gt_fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_sem_image_gt_fronts[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
            data['sem_image_err_fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_sem_image_err_fronts[0]), scale=self.scale, crop=self.input_resolution, onehot=onehot))))
            if not self.ignore_sides:
                data['sem_image_gt_lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_gt_lefts[0]), scale=self.scale,
                                         crop=self.input_resolution, onehot=onehot))))
                data['sem_image_gt_rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_gt_rights[0]), scale=self.scale,
                                         crop=self.input_resolution, onehot=onehot))))
                data['sem_image_err_lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_err_lefts[0]), scale=self.scale,
                                         crop=self.input_resolution, onehot=onehot))))
                data['sem_image_err_rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_sem_image_err_rights[0]), scale=self.scale,
                                         crop=self.input_resolution, onehot=onehot))))
        else:
            graph_feature_gt = np.load(seq_graph_feature_gts[0], allow_pickle=True)
            b_list_gt, b_feature_gt, b_shape_gt = graph_feature_gt[0], graph_feature_gt[1], graph_feature_gt[2]
            bb_gt = np.zeros(b_shape_gt)
            bb_gt[b_list_gt[0], b_list_gt[1]] = b_feature_gt
            data['graph_feature_gts'].append(np.transpose(bb_gt, [2,0,1]))

            graph_feature_err = np.load(seq_graph_feature_errs[0], allow_pickle=True)
            b_list_err, b_feature_err, b_shape_err = graph_feature_err[0], graph_feature_err[1], graph_feature_err[2]
            bb_err = np.zeros(b_shape_err)
            bb_err[b_list_err[0], b_list_err[1]] = b_feature_err
            data['graph_feature_errs'].append(np.transpose(bb_err, [2,0,1]))

        return data


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

def scale_and_crop_image(image, scale=1, crop=256, onehot=12):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    if len(image.shape) == 2:
        image = np.eye(onehot)[image]
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

