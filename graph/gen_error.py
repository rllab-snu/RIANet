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

from utils.graph_augment_utils import *
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.bev_utils import *

import glob
from tqdm import tqdm

def sem_to_rgb_DA(sem):
    # sem : w x h
    color_map = np.array([color for color, _ in color_code_DA.values()])
    pred_imgs = [color_map[p] for p in sem]
    return np.array(pred_imgs).reshape([sem.shape[0], sem.shape[1], 3])

color_code_DA = {}
color_code_DA['Others'] = mcolors.hex2color(mcolors.CSS4_COLORS['black'])  # Other
color_code_DA['LM-W'] = mcolors.hex2color(mcolors.CSS4_COLORS['white'])  # lane marker white
color_code_DA['LM-Y'] = mcolors.hex2color(mcolors.CSS4_COLORS['yellow'])  # lane marker yellow (center)
color_code_DA['DA'] = mcolors.hex2color(mcolors.CSS4_COLORS['grey'])  # driving area
for n, c in enumerate(color_code_DA.keys()):
    color_code_DA[c] = (list((255 * np.array(color_code_DA[c])).astype('uint8')), n)
    
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
            os.mkdir(sem_data_path + '/graph')
            os.mkdir(sem_data_path + '/graph_pixel')
            
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
                                                          
        traj_seg_name_set = set(['_'.join(node_id.split('_')[:2]) for node_id in traj_node_id])
                                                          
        rgbs = [np.asarray(Image.open(traj_dataset_route_dir+'/rgb_left/{:04d}.png'.format(0)), dtype=object),
               np.asarray(Image.open(traj_dataset_route_dir+'/rgb_front/{:04d}.png'.format(0)), dtype=object),
               np.asarray(Image.open(traj_dataset_route_dir+'/rgb_right/{:04d}.png'.format(0)), dtype=object)]
               
        # extract road graph feature
        road_graph_feature = []
        H_list = []
        for seq in range(all_num_seq):    
            G = nx.read_gpickle(dir_name+"/map-graph.gpickle")
            G2 = nx.read_gpickle(dir_name+"/map-graph2.gpickle")
            seg_G = nx.read_gpickle(dir_name+"/map-seg-graph.gpickle")
        
            cur_pos, cur_theta, cur_speed = route_pos[seq], route_theta[seq], route_speed[seq]  # x, y are inverted
            cur_theta = 0 if np.isnan(cur_theta) else cur_theta
            graph_feature_dict, H = extract_graph_features(cur_pos, 0.5 * np.pi - cur_theta,
                                                        cur_speed, \
                                                        G2, pos_kdtree_keys2, pos_kdtree2, traj_node_id,
                                                        nearest_node_num=96*4, cutoff_dist=30, use_node_filter=True)

            i = seq
            theta = 0.5*math.pi - route_theta[i]

            # Ego-centric
            gt_sem_img0 = coord_converters[0].visualize_sem_image(rgbs[0], seg_G, H, graph_feature_dict, \
                                                            route_pos[i], theta)
            gt_sem_img1 = coord_converters[1].visualize_sem_image(rgbs[1], seg_G, H, graph_feature_dict, \
                                                            route_pos[i], theta)
            gt_sem_img2 = coord_converters[2].visualize_sem_image(rgbs[2], seg_G, H, graph_feature_dict, \
                                                            route_pos[i], theta)
            
            _, bev = graph_to_bev_encoder(H, route_pos[seq], 0.5*math.pi - route_theta[seq], resolution=256)
            b_list = np.stack(np.where(bev[:,:,0]==1))
            b_feature = bev[b_list[0], b_list[1]]
            
            np.save(sem_data_path + '/graph_pixel/{:04d}_graph_gt.npy'.format(seq), np.array([b_list, b_feature, bev.shape], dtype=object), allow_pickle=True)
            
            #nx.write_gpickle(H, sem_data_path + '/graph/{:04d}_map-graph-gt.gpickle'.format(i))
            #nx.write_gpickle(seg_G, sem_data_path + '/graph/{:04d}_map-seg-graph-gt.gpickle'.format(i))            
            
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_gt_labelIds.png'.format(i), gt_sem_img0.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_gt_labelIds_DA.png'.format(i), np.clip(gt_sem_img0.argmax(-1), 0, 3))            
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_gt_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(gt_sem_img0), cv2.COLOR_BGR2RGB))              
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_gt_color_DA.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb_DA(np.clip(gt_sem_img0.argmax(-1), 0, 3)), cv2.COLOR_BGR2RGB))        
                        
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_gt_labelIds.png'.format(i), gt_sem_img1.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_gt_labelIds_DA.png'.format(i), np.clip(gt_sem_img1.argmax(-1), 0, 3))   
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_gt_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(gt_sem_img1), cv2.COLOR_BGR2RGB))  
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_gt_color_DA.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb_DA(np.clip(gt_sem_img1.argmax(-1), 0, 3)), cv2.COLOR_BGR2RGB))    
                        
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_gt_labelIds.png'.format(i), gt_sem_img2.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_gt_labelIds_DA.png'.format(i), np.clip(gt_sem_img2.argmax(-1), 0, 3))   
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_gt_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(gt_sem_img2), cv2.COLOR_BGR2RGB))  
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_gt_color_DA.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb_DA(np.clip(gt_sem_img2.argmax(-1), 0, 3)), cv2.COLOR_BGR2RGB))    
                                
            # graph with error
            seg_name_set = set(['_'.join(node_id.split('_')[:-1]) for node_id in list(H.nodes.keys())])
            seg_name_list = list(traj_seg_name_set & seg_name_set)
            
            G_arrange = G2.copy()
            seg_G2 = seg_G.copy()
            
            hole_candids = get_only_frontal_subgraph(route_pos[seq], theta, H.copy(), margin=0)  # get filtered subgraph
            G_arrange, seg_G2, result = route_augment(G_arrange, seg_G2, seg_name_list, hole_candids)
            
            pos_kdtree_keys_arrange = np.stack(list(dict(G_arrange.nodes('pos')).keys()))
            pos_kdtree_arrange = KDTree(np.stack(list(dict(G_arrange.nodes('pos')).values())))

            cur_pos, cur_theta, cur_speed = route_pos[seq], route_theta[seq], route_speed[seq]  # x, y are inverted
            cur_theta = 0 if np.isnan(cur_theta) else cur_theta
            graph_feature_dict_err, H_err = extract_graph_features(cur_pos, 0.5 * np.pi - cur_theta, cur_speed,
                                                        G_arrange, pos_kdtree_keys_arrange, pos_kdtree_arrange,
                                                        traj_node_id, nearest_node_num=96*4, cutoff_dist=40, 
                                                        use_node_filter=True)            
                                                        
            err_sem_img0 = coord_converters[0].visualize_sem_image(rgbs[0], seg_G2, H_err, graph_feature_dict_err, \
                                                            route_pos[i], theta)
            err_sem_img1 = coord_converters[1].visualize_sem_image(rgbs[1], seg_G2, H_err, graph_feature_dict_err, \
                                                            route_pos[i], theta)
            err_sem_img2 = coord_converters[2].visualize_sem_image(rgbs[2], seg_G2, H_err, graph_feature_dict_err, \
                                                            route_pos[i], theta)
                
            _, bev = graph_to_bev_encoder(H_err, route_pos[seq], 0.5*math.pi - route_theta[seq], resolution=256)
            b_list = np.stack(np.where(bev[:,:,0]==1))
            b_feature = bev[b_list[0], b_list[1]]
            
            np.save(sem_data_path + '/graph_pixel/{:04d}_graph_err_type.npy'.format(seq), np.array(result, dtype=object), allow_pickle=True)
            np.save(sem_data_path + '/graph_pixel/{:04d}_graph_err.npy'.format(seq), np.array([b_list, b_feature, bev.shape], dtype=object), allow_pickle=True)
                                                            
            #nx.write_gpickle(H_err, sem_data_path + '/graph/{:04d}_map-graph-err.gpickle'.format(i))
            #nx.write_gpickle(seg_G2, sem_data_path + '/graph/{:04d}_map-seg-graph-err.gpickle'.format(i))
            
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_err_labelIds.png'.format(i), err_sem_img0.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_err_labelIds_DA.png'.format(i), np.clip(err_sem_img0.argmax(-1), 0, 3))    
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_err_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(err_sem_img0), cv2.COLOR_BGR2RGB))        
            cv2.imwrite(sem_data_path + '/sem_left/' + '{:04d}_err_color_DA.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb_DA(np.clip(err_sem_img0.argmax(-1), 0, 3)), cv2.COLOR_BGR2RGB))      
                        
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_err_labelIds.png'.format(i), err_sem_img1.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_err_labelIds_DA.png'.format(i), np.clip(err_sem_img1.argmax(-1), 0, 3))    
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_err_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(err_sem_img1), cv2.COLOR_BGR2RGB))  
            cv2.imwrite(sem_data_path + '/sem_front/' + '{:04d}_err_color_DA.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb_DA(np.clip(err_sem_img1.argmax(-1), 0, 3)), cv2.COLOR_BGR2RGB))     
                        
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_err_labelIds.png'.format(i), err_sem_img2.argmax(-1))
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_err_labelIds_DA.png'.format(i), np.clip(err_sem_img2.argmax(-1), 0, 3))    
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_err_color.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb(err_sem_img2), cv2.COLOR_BGR2RGB))  
            cv2.imwrite(sem_data_path + '/sem_right/' + '{:04d}_err_color_DA.png'.format(i), \
                        cv2.cvtColor(sem_to_rgb_DA(np.clip(err_sem_img2.argmax(-1), 0, 3)), cv2.COLOR_BGR2RGB))  
            
