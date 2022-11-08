import os
import json
import numpy as np
import time
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import networkx as nx

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, Voronoi
from scipy import interpolate

from utils.graph_utils import *
from utils.gnss_utils import *
from utils.bev_utils import *

import glob
from tqdm import tqdm

color_code = {}
color_code['Others'] = mcolors.hex2color(mcolors.CSS4_COLORS['black'])   # Other
color_code['LM-W'] = mcolors.hex2color(mcolors.CSS4_COLORS['white'])   # lane marker white
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
    color_code[c] = (list((255*np.array(color_code[c])).astype('uint8')), n)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-path', type=str, default='/data/timothyha/carla/transfuser/clear_weather_data')
    parser.add_argument('--dest-path', type=str, default='/data/timothyha/carla/transfuser/sem_data')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=300)
    args = parser.parse_args()
    
    a = glob.glob(args.origin_path + '/*')
    a = [p for p in a if os.path.isdir(p)]
    a = sorted(a)

    b = []
    for p in a:
        town_data_list = sorted(glob.glob(p+'/*'))
        b += [f for f in town_data_list if os.path.isdir(f)]
            
    coord_converters = [PC_CoordConverter(cam_yaw=yaw) for yaw in [-60,0,60]]

    for kk, data_dir in enumerate(tqdm(b)):
        if kk < args.start_idx or kk > args.end_idx:
            continue
        traj_dataset_route_dir = data_dir
        map_name = data_dir.split('/')[-2].split('_')[0].lower()
        
        dir_name = './routes/Road_graph/' + map_name
        
        G = nx.read_gpickle(dir_name+"/map-graph.gpickle")
        G2 = nx.read_gpickle(dir_name+"/map-graph2.gpickle")
        seg_G = nx.read_gpickle(dir_name+"/map-seg-graph.gpickle")
        
        pos_kdtree_keys = np.stack(list(dict(G.nodes('pos')).keys()))
        pos_kdtree = KDTree(np.stack(list(dict(G.nodes('pos')).values())))
        
        pos_kdtree_keys2 = np.stack(list(dict(G2.nodes('pos')).keys()))
        pos_kdtree2 = KDTree(np.stack(list(dict(G2.nodes('pos')).values())))

        town_data_path = args.dest_path + '/' + data_dir.split('/')[-2]
        if not os.path.isdir(town_data_path):
            os.mkdir(town_data_path)
        
        sem_data_path = town_data_path + '/' + data_dir.split('/')[-1]
        if not os.path.isdir(sem_data_path):
            os.mkdir(sem_data_path)
            os.mkdir(sem_data_path + '/sem_left')
            os.mkdir(sem_data_path + '/sem_front')
            os.mkdir(sem_data_path + '/sem_right')
            
        # load trajectory data
        all_num_seq = len(os.listdir(traj_dataset_route_dir + "/rgb_front/"))
        route_x_command, route_y_command = [], []
        route_x, route_y = [], []
        route_theta, route_speed = [], []
        for seq in range(all_num_seq):
            # position
            with open(traj_dataset_route_dir + f"/measurements/{str(seq).zfill(4)}.json", "r") as read_file:
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
        _, idx = np.unique(global_plan_pos, axis=0, return_index=True)
        global_plan_pos = global_plan_pos[np.sort(idx)]

        global_plan_pos_with_starting = np.concatenate([route_pos[0].reshape(1, 2), global_plan_pos])

        # get Dijkstra shortest path
        traj, traj_node_id, progress, _ = get_global_traj(global_plan_pos_with_starting, G, seg_G,
                                                          pos_kdtree_keys, pos_kdtree)

        # extract road graph feature
        road_graph_feature = []
        H_list = []
        for seq in range(all_num_seq):
            cur_pos, cur_theta, cur_speed = route_pos[seq], route_theta[seq], route_speed[seq]  # x, y are inverted
            cur_theta = 0 if np.isnan(cur_theta) else cur_theta
            graph_feature_dict, H = extract_graph_features(cur_pos, 0.5 * np.pi - cur_theta,
                                                        cur_speed, \
                                                        G2, pos_kdtree_keys2, pos_kdtree2, traj_node_id,
                                                        nearest_node_num=96*4, cutoff_dist=30, use_node_filter=True)
            road_graph_feature.append(graph_feature_dict)
            H_list.append(H)
            

            i = seq
            rgbs = [np.asarray(Image.open(traj_dataset_route_dir+'/rgb_left/{:04d}.png'.format(i))),
                   np.asarray(Image.open(traj_dataset_route_dir+'/rgb_front/{:04d}.png'.format(i))),
                   np.asarray(Image.open(traj_dataset_route_dir+'/rgb_right/{:04d}.png'.format(i)))]
            pc = np.load(traj_dataset_route_dir+'/lidar/{:04d}.npy'.format(i), allow_pickle=True)
            w, h = 256, 288
            rgbs_ = [im[150-h//2:150+h//2,200-w//2:200+w//2] for im in rgbs]
            all_rgb = np.concatenate(rgbs_, axis=1)
            
            theta = 0.5*math.pi - route_theta[i]

            # Ego-centric
            gt_sem_img0 = coord_converters[0].visualize_road_graph(rgbs[0], seg_G, H_list[i], road_graph_feature[i], \
                                                            route_pos[i], theta, show_road=True)
            gt_sem_img1 = coord_converters[1].visualize_road_graph(rgbs[1], seg_G, H_list[i], road_graph_feature[i], \
                                                            route_pos[i], theta, show_road=True)
            gt_sem_img2 = coord_converters[2].visualize_road_graph(rgbs[2], seg_G, H_list[i], road_graph_feature[i], \
                                                            route_pos[i], theta, show_road=True)
            
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_labelIds.png'.format(i), gt_sem_img0.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(gt_sem_img0), cv2.COLOR_BGR2RGB))        
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_labelIds.png'.format(i), gt_sem_img1.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(gt_sem_img1), cv2.COLOR_BGR2RGB))  
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_labelIds.png'.format(i), gt_sem_img2.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(gt_sem_img2), cv2.COLOR_BGR2RGB))  
