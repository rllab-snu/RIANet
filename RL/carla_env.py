from __future__ import print_function

import sys
import random
import math
import numpy as np
import time
import pygame
import threading
import cv2
import gym
import networkx as nx
import copy

from collections import deque

from sklearn.neighbors import KDTree
from scipy.interpolate import interp1d

def extract_criterion(scenario_manager):
    """
    Creates the output message
    """
    criterion_list = scenario_manager.scenario.get_criteria()

    statistic = {}
    for criterion in criterion_list:

        actual_value = criterion.actual_value
        expected_value = criterion.expected_value_success
        name = criterion.name
        result = criterion.test_status

        #if result == "SUCCESS":
        #    result = '\033[92m' + 'SUCCESS' + '\033[0m'
        #elif result == "FAILURE":
        #    result = '\033[91m' + 'FAILURE' + '\033[0m'

        if name in ["InRouteTest", "AgentBlockedTest"]:
            statistic[name] = (result != "FAIL")
        else:
            statistic[name] = actual_value

        '''
        if name == "RouteCompletionTest":
            actual_value = str(actual_value) + " %"
        elif name == "OutsideRouteLanesTest":
            actual_value = str(actual_value) + " %"
        elif name == "CollisionTest":
            actual_value = str(actual_value) + " times"
        elif name == "RunningRedLightTest":
            actual_value = str(actual_value) + " times"
        elif name == "RunningStopTest":
            actual_value = str(actual_value) + " times"
        elif name == "InRouteTest":
            actual_value = ""
        elif name == "AgentBlockedTest":
            actual_value = ""
        '''

    # Timeout
    name = "OnTime"
    on_time = scenario_manager.scenario_duration_game < scenario_manager.scenario.timeout
    statistic[name] = on_time

    return statistic

class CarlaVirtualEnv(object):

    def __init__(self, args, observation_type):
    
        self.args = args
        self.observation_type = observation_type
        
        self.epi = 0
        self.env_timestep = 0
        self.env_time_limit = 10000

        self.prev_criterion_dict = None
        self.scenario_end = False

        self.set_observation_parameters()        

    def set_observation_parameters(self):
        self.observation_length = 4
        self.time_period = 5
        self.history_length = self.observation_length * self.time_period
        self.observation_time_step = self.history_length - self.time_period * np.arange(self.observation_length - 1, -1, -1) - 1

        self.data_history = deque(maxlen=self.history_length)

        t = len(self.observation_time_step)
        w, h = 800, 600
        self.observation_image_shape = [t * 3, w // 4, h // 4]

        if self.observation_type in ['dict', 'graph']:
            # graph related parameters
            self.use_node_filter = False
            self.use_edge_connection_feature = True
            self.node_feature_num = 5
            self.edge_feature_num = 2 + 5 if self.use_edge_connection_feature else 2
            self.nearest_node_num = 64
            self.num_obstacle = 16

            # other observation parameters
            n = self.nearest_node_num
            self.space_dict = {'traj': gym.spaces.Box(low=-1000, high=1000, shape=(t, 2), dtype=np.float32),
                          'traj_st': gym.spaces.Box(low=-1000, high=1000, shape=(t, 2), dtype=np.float32),
                          'traj_compass': gym.spaces.Box(low=-1000, high=1000, shape=(t, 1), dtype=np.float32),
                          'adjacency_matrix': gym.spaces.Box(low=-1, high=1, shape=(n, n), dtype=np.float32),
                          'node_feature_matrix': gym.spaces.Box(low=-1000, high=1000, shape=(2, n, self.node_feature_num),dtype=np.float32),
                          'edge_feature_matrix': gym.spaces.Box(low=-1000, high=1000, shape=(n, n, self.edge_feature_num),dtype=np.float32)}

            if self.observation_type == 'dict':
                self.space_dict['image'] = gym.spaces.Box(low=0, high=1.0, shape=self.observation_image_shape, dtype=np.float32)

            # encode into one numpy array
            self.space_shape = {}
            self.space_len = {}
            self.space_len_sum = 0
            for k in self.space_dict:
                box_shape = self.space_dict[k].shape
                self.space_shape[k] = box_shape
                npprod = np.prod(box_shape)
                self.space_len[k] = npprod
                self.space_len_sum += npprod
            self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(self.space_len_sum,), dtype=np.float32)
        elif self.observation_type == 'vec':
            self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(5,), dtype=np.float32)
        elif self.observation_type == 'image':
            self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=self.observation_image_shape, dtype=np.float32)

        if self.args.use_continuous_action_space:
            self.action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(5 * 3)

        self.reward_range = (-1., 1.)
        self.metadata = {'render.modes': []}
        self.spec = None

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def translate_action(self, action):
        steer_adjustment = 0.1 * (int(action) % 3 - 1)
        target_speed = 3.0 * (int(action) % 5)

        return steer_adjustment, target_speed

    def update_state(self, data):
        self.env_data = data
        return
    
    def reset(self):        
        self.epi += 1
        self.env_timestep = 0

        if self.scenario_end:
            self.data_history = deque(maxlen=self.history_length)
            self.prev_criterion_dict = None
        self.scenario_end = False

        return self.get_state()

    def reset_scenario_and_state(self):
        self.scenario_end = True

    def step(self, control):
        next_state = self.get_state()
        next_reward, is_terminate = self.get_reward_and_terminal()
        next_info = self.get_info()

        return next_state, next_reward, is_terminate, next_info
               
    def get_subgraph(self, center_pos_xy):
        _, idxs = self.pos_kdtree.query(center_pos_xy.reshape(1, -1), k=self.nearest_node_num)
        nearest_node_num = self.pos_kdtree_keys[idxs].reshape(-1)
        H = self.G2.subgraph(nearest_node_num)
        return H

    def get_filtered_subgraph(self, center_pos_xy, theta):
        xy = np.array([np.cos(theta), np.sin(theta)])
        xy = np.expand_dims(xy, 0)
        center_pos_xy = np.expand_dims(center_pos_xy, 0)

        candid_node_num = self.nearest_node_num * 4
        for i in range(5):
            _, idxs = self.pos_kdtree.query(center_pos_xy.reshape(1,-1), k=candid_node_num)

            nearest_nodes = self.pos_kdtree_keys[idxs].reshape(-1)
            sub_graph = self.G2.subgraph(nearest_nodes)

            # filtering
            sub_pos_kdtree_values = np.array(list(dict(sub_graph.nodes('pos')).values()))
            sub_pos_kdtree_keys = np.array(list(dict(sub_graph.nodes('pos')).keys()))

            direc = center_pos_xy - xy
            check = direc[0, 0] * (sub_pos_kdtree_values[:, 0] - xy[0, 0]) + direc[0, 1] * (
                        sub_pos_kdtree_values[:, 1] - xy[0, 1])
            margin = -5
            filter_idx = np.where(check > margin)[0]

            if len(filter_idx) > self.nearest_node_num:
                break
            else:
                candid_node_num *= 2

        filtered_pos = sub_pos_kdtree_values[filter_idx]
        filtered_pos_keys = sub_pos_kdtree_keys[filter_idx]
        filtered_pos_kdtree = KDTree(filtered_pos)

        _, idxs = filtered_pos_kdtree.query(center_pos_xy, k=self.nearest_node_num)

        filtered_nearest_nodes = filtered_pos_keys[idxs].reshape(-1)
        filtered_sub_graph = sub_graph.subgraph(filtered_nearest_nodes)

        return filtered_sub_graph

    def get_state(self):
        if self.scenario_end:  # replicate last state
            history_state = np.stack([self.data_history[t] for t in self.observation_time_step], axis=0)

            if self.observation_type == 'image':
                history_state = np.transpose(history_state, [0, 3, 2, 1])  # t x h x w x 3 -> t x 3 x w x h
                history_state = np.concatenate(history_state, axis=0)  # -> (t * 3) x w x h
            return history_state

        if self.observation_type in ['dict', 'graph']:
            center_pos_xy = self.env_data['vehicle_pos']
            theta = self.env_data['vehicle_compass']
            cos_theta = np.cos(theta).reshape([-1, 1])
            sin_theta = np.sin(theta).reshape([-1, 1])
            R = np.concatenate([cos_theta, -sin_theta, sin_theta, cos_theta], axis=1).reshape(
                [2, 2])

            # construct graph
            if self.use_node_filter:
                H = self.get_filtered_subgraph(center_pos_xy, theta)   # get filtered subgraph
            else:
                H = self.get_subgraph(center_pos_xy)   # get subgraph

            adjacency_matrix = nx.to_numpy_array(H)   # n x n

            # construct edge feature matrix
            edge_feature_matrix = np.zeros([self.nearest_node_num, self.nearest_node_num, self.edge_feature_num])
            nodes = list(H.nodes())
            for edge in H.edges():
                key1, key2 = edge
                i, j = nodes.index(key1), nodes.index(key2)
                diff = H.edges[edge]['direc']
                diff = np.matmul(diff, R)  # apply diff_rot
                edge_feature_matrix[i, j] = diff

            # construct node feature matrix
            node_feature_matrix = np.zeros([2, self.nearest_node_num, self.node_feature_num])
            for i, key in enumerate(H.nodes()):
                pos = H.nodes[key]['pos'] - center_pos_xy
                pos_rot = np.matmul(pos, R)  # apply diff_rot
                node_feature_matrix[:, i, :2] = pos_rot

            obstacle_data = np.concatenate([self.env_data['vehicle_obs'], self.env_data['walker_obs']], axis=0)

            v_nearest_nodes = np.zeros(3)
            for v_id, nearest_node_id in enumerate(v_nearest_nodes):
                for i, key in enumerate(H.nodes()):
                    if key == nearest_node_id:
                        node_feature_matrix[v_id, i, -1] = 1
                        break
                        traj_array_pos = obs_features[:, :, 4:6]   # t x v x 2
                traj_array_vel = obs_features[:, :, 6:8]   # t x v x 2
                traj_array_th = obs_features[:,:,8:9]  # t x v x 1
                traj_mask = obs_features[:, :, -1]  # t x v

                traj_array_pos = np.matmul((traj_array_pos - center_pos_xy), rot)
                traj_array_vel = np.matmul(traj_array_vel, rot)
                traj_array_th = traj_array_th - theta
                traj_array_direc = np.concatenate([np.cos(traj_array_th), np.sin(traj_array_th)], axis=-1)

                traj_input_processed = np.concatenate([traj_array_pos, traj_array_vel, traj_array_direc], axis=-1)  # t x v x 6

                next_state = {'traj_input':obs_features, 'traj_input_processed':traj_input_processed, 'adjacency_matrix':adjacency_matrix, \
                              'node_feature_matrix':node_feature_matrix, 'edge_feature_matrix':edge_feature_matrix, \
                              'obs_traj_mask':traj_mask}
        elif self.observation_type == 'vec':
            next_state = np.zeros(2)
        elif self.observation_type == 'image':
            image_data = self.env_data['image_data'] / 255.0
            h, w = image_data.shape[0] // 4, image_data.shape[1] // 4
            next_state = image_data.reshape((h, 4, w, 4, 3)).max(3).max(1).copy()

        if len(self.data_history) == 0:
            for i in range(self.data_history.maxlen):
                self.data_history.append(next_state)
        else:
            self.data_history.append(next_state)

        history_state = np.stack([self.data_history[t] for t in self.observation_time_step], axis=0)

        if self.observation_type == 'image':
            history_state = np.transpose(history_state, [0,3,2,1])   # t x h x w x 3 -> t x 3 x w x h
            history_state = np.concatenate(history_state, axis=0)   # -> (t * 3) x w x h

        return history_state
        
    def get_reward_and_terminal(self):
        scenario_manager = self.env_data['scenario_manager']
        criterion_dict = extract_criterion(scenario_manager)
        
        if self.prev_criterion_dict == None:
            self.prev_criterion_dict = copy.copy(criterion_dict)
            self.prev_criterion_dict['is_success'] = False
            return 0, False

        # not in progress
        r_no_progress = float(self.env_data['speed_data'] < 0.5)

        # deviation
        r_deviation = abs(self.env_data['vehicle_gt_dev'])

        # offroad
        r_offroad = float(criterion_dict['OutsideRouteLanesTest'] > self.prev_criterion_dict['OutsideRouteLanesTest'] + 1e-6)

        # collision
        r_collision = float(criterion_dict['CollisionTest'] > self.prev_criterion_dict['CollisionTest'] + 1e-6)

        # redlight
        r_redlight = float(criterion_dict['RunningRedLightTest'] > self.prev_criterion_dict['RunningRedLightTest'] + 1e-6)

        # stop sign
        r_stopsign = float(criterion_dict['RunningStopTest'] > self.prev_criterion_dict['RunningStopTest'] + 1e-6)

        # in route
        r_in_routes = 1. - float(criterion_dict['InRouteTest'])

        # agent blocked
        r_agent_blocked = 1. - float(criterion_dict['AgentBlockedTest'])

        # timeout
        r_timeout = 1. - float(criterion_dict['OnTime'])

        reward = -0.001 * r_no_progress + \
                -0.01 * r_deviation + \
                -0.1 * r_offroad + \
                -1.0 * r_collision + \
                -0.1 * r_redlight + \
                -0.1 * r_stopsign + \
                -1.0 * r_in_routes + \
                -1.0 * r_agent_blocked + \
                -1.0 * r_timeout

        #is_terminal = not(criterion_dict['InRouteTest'] and criterion_dict['AgentBlockedTest'] and criterion_dict['OnTime']) \
        #            or (int(criterion_dict['RouteCompletionTest']) == 100)
        is_terminal = self.scenario_end
        is_success = is_terminal and int(criterion_dict["RouteCompletionTest"]) == 100

        self.prev_criterion_dict = copy.copy(criterion_dict)
        self.prev_criterion_dict['is_success'] = is_success

        return reward, is_terminal

    def get_info(self):
        return self.prev_criterion_dict
        

