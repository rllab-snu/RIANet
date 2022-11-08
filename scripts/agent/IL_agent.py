#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import time
import json
from threading import Thread
import cv2
import os
import importlib
import numpy as np

import networkx as nx
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from sklearn.neighbors import KDTree
from matplotlib import cm

from itertools import product
from collections import deque, namedtuple

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

# from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.autoagents.autonomous_agent import Track
from autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from planner import RoutePlanner


def get_entry_point():
    return 'ILAgent'


import sys

sys.path.append('../..')
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.pid_controller import *
from utils.localization import *
from utils.lidar_utils import *
from RL.carla_env import extract_criterion

DEBUG = False


class HumanInterface(object):
    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, args):
        self._width = 800
        self._height = 600
        self._surface = None
        self.args = args

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run_interface(self, image_data):
        """
        Run the GUI
        """
        # display image
        self._surface = pygame.surfarray.make_surface(image_data.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x ** 2 +
            trigger.extent.y ** 2 +
            trigger.extent.z ** 2)
        b = np.sqrt(
            vehicle.bounding_box.extent.x ** 2 +
            vehicle.bounding_box.extent.y ** 2 +
            vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        result.append(light)

    return result


class ILAgent(AutonomousAgent):
    """
    Human agent to control the ego vehicle via keyboard
    """

    data_save_timer = 0
    current_control = None
    agent_engaged = False

    nearest_nodes_num = 64

    episode_num = 0

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP

        self.agent_engaged = False

        self._prev_timestamp = 0
        self.step = -1

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def sensors(self):
        if self.args.show_camera:
            self._hic = HumanInterface(self.args)
        if self.args.record_video:
            try:
                os.mkdir(os.path.join(self.args.log_dir, 'Snaps'))
            except:
                pass

        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}


            sensors_to_icons = {
                'sensor.camera.rgb':        'carla_camera',
                'sensor.lidar.ray_cast':    'carla_lidar',
                'sensor.other.radar':       'carla_radar',
                'sensor.other.gnss':        'carla_gnss',
                'sensor.other.imu':         'carla_imu',
                'sensor.opendrive_map':     'carla_opendrive_map',
                'sensor.speedometer':       'carla_speedometer'
            }
        ]
        """
        '''
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear'
            },
            # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'width': 640, 'height': 480, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 400, 'height': 300, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.lidar.ray_cast', 'x': 1.3, 'y': 0.0, 'z': 2.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
            'id': 'lidar'},
            # {'type': 'sensor.other.radar', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 100,
            # 'id': 'RADAR'},
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'IMU'},
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
            {'type': 'sensor.opendrive_map', 'reading_frequency': 20, 'id': 'HD-map'},
        ]

        '''
        sensors = [
        {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'width': 400, 'height': 300, 'fov': 100, 'id': 'Center'},
        {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
         'width': 960, 'height': 540, 'fov': 100, 'id': 'rgb_left'},
        {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
         'width': 960, 'height': 540, 'fov': 100, 'id': 'rgb_right'},
        {'type': 'sensor.camera.rgb', 'x': -1.3, 'y': 0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180,
         'width': 960, 'height': 540, 'fov': 100, 'id': 'rgb_rear'},
        {'type': 'sensor.lidar.ray_cast', 'x': 1.3, 'y': 0.0, 'z': 2.5, 'yaw': -90.0, 'pitch': 0.0, 'roll': 0.0,
         'id': 'LIDAR'},
        # {'type': 'sensor.other.radar', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 100,
        # 'id': 'RADAR'},
        {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 1.60, 'id': 'GPS'},
        {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
         'id': 'IMU'},
        {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
        {'type': 'sensor.opendrive_map', 'reading_frequency': 20, 'id': 'HD-map'},]
        return sensors

    def graph_updater_thread(self):
        map_name = self.args.routes.split('/')[-1].split('_')[1]  # town00
        dir_name = './routes/Road_graph/' + str(map_name)
        if os.path.exists(dir_name):
            return

        i = 0
        while (len(self.processed_roads) < self.num_of_roads):
            if hasattr(self, 'scenario_end'):
                return
            i += 1
            G, seg_G, pos_kdtree_keys, pos_kdtree, processed_roads, signal_object_dict, signal_reference_dict = \
                extract_graph_info_partial(self.xml_root, cutoff_dist=(self.args.cutoff_dist + 100 * i),
                                           vehicle_pos=self.initial_pos,
                                           prev_seg_G=self.seg_G, prev_G=self.G, processed_roads=self.processed_roads,
                                           signal_object_dict=self.signal_object_dict,
                                           signal_reference_dict=self.signal_reference_dict)
            if hasattr(self, 'scenario_end'):
                return
            torch_G = convert_graph_as_torch(G, self.device)
            torch_seg_G = convert_graph_as_torch(seg_G, self.device)

            global_traj, global_traj_node_id, path_progress, (plan_start_idx, plan_end_idx) = \
                get_global_traj(self.global_plan_pos_with_starting, G, seg_G, pos_kdtree_keys, pos_kdtree)

            G2, pos_kdtree_keys2, pos_kdtree2 = contract_graph(G, pos_kdtree_keys, pos_kdtree)
            torch_G2 = convert_graph_as_torch(G2, self.device)

            initial_pos_st_new = np.array([path_progress[plan_start_idx], 0])
            initial_pos_st_diff = (initial_pos_st_new[0] - self.initial_pos_st[0]).copy()

            if hasattr(self, 'scenario_end'):
                return
            self.initial_pos_st = initial_pos_st_new.copy()
            self.prev_pos_st[0] += initial_pos_st_diff
            self.final_st = np.array([path_progress[plan_end_idx], 0])

            self.G, self.seg_G, self.pos_kdtree_keys, self.pos_kdtree = G, seg_G, pos_kdtree_keys, pos_kdtree
            self.processed_roads = processed_roads
            self.signal_object_dict, self.signal_reference_dict = signal_object_dict, signal_reference_dict
            self.torch_G, self.torch_seg_G = torch_G, torch_seg_G
            self.global_traj, self.global_traj_node_id, self.path_progress = global_traj, global_traj_node_id, path_progress
            self.G2, self.torch_G2, self.pos_kdtree_keys2, self.pos_kdtree2 = G2, torch_G2, pos_kdtree_keys2, pos_kdtree2

            if not hasattr(self, 'scenario_end'):
                process_time = round(time.time() - self.start_time, 3)
                print('update done with : {}'.format(process_time))

        if not hasattr(self, 'scenario_end'):
            process_time = round(time.time() - self.start_time, 3)
            print('final update done with : {}'.format(process_time))

            os.makedirs(dir_name)
            nx.write_gpickle(self.G, dir_name + "/map-graph.gpickle")
            nx.write_gpickle(self.G2, dir_name + "/map-graph2.gpickle")
            nx.write_gpickle(self.seg_G, dir_name + "/map-seg-graph.gpickle")
            np.save(dir_name + '/processed_roads.npy', self.processed_roads)
            np.save(dir_name + '/signal_object_dict.npy', self.signal_object_dict)
            np.save(dir_name + '/signal_reference_dict.npy', self.signal_reference_dict)
            print('save graph')

    def input_data_process(self, input_data, timestamp):

        # convert to top-view and remove ground
        # lidar_data_ground_filter = lidar_data[:,2] > 0  # remove ground
        # lidar_data_processed = lidar_data[lidar_data_ground_filter,:2]  # ignore intensity

        reinitialize = False
        # HD-map
        if not hasattr(self, 'prev_hdmap'):  # need to be regenerated for new map
            self._vehicle = CarlaDataProvider.get_hero_actor()

            self.device = self.args.gpu_device if not self.args.no_cuda else "cpu"

            self._waypoint_planner = RoutePlanner(4.0, 50)
            self._waypoint_planner.set_route(self._plan_gps_HACK, True)

            print('pre-process start')
            self.start_time = time.time()
            self._prev_time = self.start_time
            self.prev_hdmap = input_data['HD-map'][1]['opendrive']
            self.xml_root = ET.fromstring(self.prev_hdmap)

            self.geo_ref_dict = get_georeference(self.xml_root)

            # gnss
            self.gnss_data = input_data['GPS'][1]
            loc = self._vehicle.get_location()
            self.vehicle_gt_pos = np.array([loc.x, -loc.y, loc.z])  # y is inverted
            self.vehicle_pos = gnss_to_xy(self.gnss_data, self.geo_ref_dict)  # initial vehicle pos start with raw

            self.initial_pos = self.vehicle_pos[:2].copy()
            self.num_of_roads = len(self.xml_root.findall('road'))

            # check previous road graph
            map_name = self.args.routes.split('/')[-1].split('_')[1]  #town00
            dir_name = './routes/Road_graph/' + str(map_name)
            if os.path.exists(dir_name):
                self.G = nx.read_gpickle(dir_name + "/map-graph.gpickle")
                self.G2 = nx.read_gpickle(dir_name + "/map-graph2.gpickle")
                self.seg_G = nx.read_gpickle(dir_name + "/map-seg-graph.gpickle")

                self.pos_kdtree_keys = np.stack(list(dict(self.G.nodes('pos')).keys()))
                self.pos_kdtree = KDTree(np.stack(list(dict(self.G.nodes('pos')).values())))

                self.pos_kdtree_keys2 = np.stack(list(dict(self.G2.nodes('pos')).keys()))
                self.pos_kdtree2 = KDTree(np.stack(list(dict(self.G2.nodes('pos')).values())))

                self.processed_roads = list(np.load(dir_name + '/processed_roads.npy'))
                self.signal_object_dict = np.load(dir_name + '/signal_object_dict.npy', allow_pickle=True).item()
                self.signal_reference_dict = np.load(dir_name + '/signal_reference_dict.npy', allow_pickle=True).item()
            else:
                # generate graph near to the start point
                self.G, self.seg_G, self.pos_kdtree_keys, self.pos_kdtree, self.processed_roads, self.signal_object_dict, self.signal_reference_dict = \
                    extract_graph_info_partial(self.xml_root, cutoff_dist=self.args.cutoff_dist,
                                               vehicle_pos=self.initial_pos)

                self.G2, self.pos_kdtree_keys2, self.pos_kdtree2 = contract_graph(self.G, self.pos_kdtree_keys,
                                                                                  self.pos_kdtree)

            self.torch_G = convert_graph_as_torch(self.G, self.device)
            self.torch_G2 = convert_graph_as_torch(self.G2, self.device)
            self.torch_seg_G = convert_graph_as_torch(self.seg_G, self.device)

            self.global_plan_pos = get_global_plan_pos(self._global_plan, geo_ref_dict=self.geo_ref_dict)
            self.global_plan_pos_with_starting = np.concatenate([self.initial_pos.reshape([1, 2]), self.global_plan_pos])

            self.global_traj, self.global_traj_node_id, self.path_progress, (plan_start_idx, plan_end_idx) = \
                get_global_traj(self.global_plan_pos_with_starting, self.G, self.seg_G, self.pos_kdtree_keys,
                                self.pos_kdtree)

            self.initial_pos_st = np.array([self.path_progress[plan_start_idx], 0])
            self.initial_timestamp = timestamp
            self.initial_time = time.time()

            self.prev_pos_st = self.initial_pos_st.squeeze().copy()
            self.final_st = np.array([self.path_progress[plan_end_idx], 0])

            # update graph using thread
            if not os.path.exists(dir_name):
                t1 = Thread(target=self.graph_updater_thread)
                t1.daemon = True
                t1.start()

            if self.args.localization == 'filter':
                self.localization = LocalizationOperator(0.001, 0.001, 0.000005)
            elif self.args.localization == 'window':
                self.pos_deque = deque(maxlen=5)
            pid_param = [0.8, 0.3, 0.2, 20]
            self.pid_controller = PID_controller(direc_p=pid_param[0], direc_i=pid_param[1], direc_d=pid_param[2],
                                                 win_size=pid_param[3])
            self.pid_controller.adjust_coeff = 0.6 if self.args.localization == 'gt' else 0.1
            self.front_target_dist = 3.0
            self.front_target_lane_dev = -0.1
            self.data_history = {'GPS': [], 'IMU': [], 'speed': [], 'pos': [], 'pos_gt': [], 'control': [],
                                 'timestamp': [],
                                 'pos_st_partial': [], 'pos_gt_st_partial': [], 'criterion_dict': [],
                                 'env_info_dict': []}
            self.data_history['HD-map'] = input_data['HD-map'][1]['opendrive']
            self.data_history['adjust_coeff'] = self.pid_controller.adjust_coeff
            self.data_history['front_target_dist'] = self.front_target_dist
            self.data_history['front_target_lane_dev'] = self.front_target_lane_dev
            self.data_history['pid_param'] = pid_param

            # load torch model
            with open(os.path.join(self.args.model_dir, 'args.txt')) as f:
                d = json.load(f)
                self.model_args = namedtuple('args', d.keys())(*d.values())
            self.model = importlib.import_module('IL.models.' + self.model_args.model_name).Model(self.model_args, self.device)
            if self.args.test_last_model:
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'model_101.pth'), map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'best_model.pth'), map_location=self.device))
            self.model.eval()

            process_time = round(time.time() - self.start_time, 3)
            print('done with : {} s'.format(process_time))

        # localization
        self.gnss_data = input_data['GPS'][1]

        loc = self._vehicle.get_location()
        self.vehicle_gt_pos = np.array([loc.x, -loc.y, loc.z])  # y is inverted

        if self.args.localization == 'filter':
            self.vehicle_pos, _ = self.localization.run(self.gnss_data, input_data['IMU'][1],
                                                        timestamp, self.geo_ref_dict)
        elif self.args.localization == 'raw':
            self.vehicle_pos = gnss_to_xy(self.gnss_data, self.geo_ref_dict)
        elif self.args.localization == 'window':
            vehicle_pos = gnss_to_xy(self.gnss_data, self.geo_ref_dict)
            self.pos_deque.append(vehicle_pos)
            self.vehicle_pos = np.mean(self.pos_deque, axis=0)
        else:
            self.vehicle_pos = self.vehicle_gt_pos.copy()

        if np.isnan(input_data['IMU'][1][-1]):
            self.compass_data = 0.5 * np.pi
        else:
            self.compass_data = 0.5 * np.pi - input_data['IMU'][1][-1]  # refine to Graph coord
        # self.compass_data = 0.5 * np.pi - self.localization.get_current_compass()

        # speedometer
        self.speed_data = input_data['speed'][1]['speed']

        # Image sensor
        self.image_data = cv2.cvtColor(input_data['Center'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        self.image_data_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        self.image_data_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        self.image_data_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        # LIDAR sensor
        self.lidar_data = input_data['LIDAR'][1]

        image_data = scale_and_crop_image(Image.fromarray(self.image_data), scale=1, crop=256)

        # convert coordinate frame of point cloud
        full_lidar = self.lidar_data[..., :3]
        full_lidar[:, 1] *= -1  # inverts x, y
        full_lidar = transform_2d_points(full_lidar,
                                         self.compass_data, -self.vehicle_pos[0], -self.vehicle_pos[1],
                                         self.compass_data, -self.vehicle_pos[0], -self.vehicle_pos[1])
        lidar_image = lidar_to_histogram_features(full_lidar, crop=256)

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)

        gps_pos = (self.gnss_data[:2] - self._waypoint_planner.mean) * self._waypoint_planner.scale
        next_wp, next_cmd = self._waypoint_planner.run_step(gps_pos)
        next_command = next_cmd.value

        ego_theta = input_data['IMU'][1][-1]
        ego_theta = 0 if np.isnan(ego_theta) else 0
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])
        local_command_point = np.array([next_wp[0] - gps_pos[0], next_wp[1] - gps_pos[1]])
        local_command_point = R.T.dot(local_command_point)
        target_point = tuple(local_command_point)
        target_point = np.array(target_point)

        # graph features
        graph_feature_dict = extract_graph_features(self.vehicle_pos[:2], self.compass_data, self.speed_data, \
                                                    self.G2, self.pos_kdtree_keys2, self.pos_kdtree2, self.global_traj_node_id, \
                                                    nearest_node_num=96, use_node_filter=True)

        # torch inputs
        self.torch_data = {}
        self.torch_data['images'] = [torch.from_numpy(image_data).unsqueeze(0).to(self.device, dtype=torch.float32)]
        self.torch_data['lidars'] = [torch.from_numpy(lidar_image).unsqueeze(0).to(self.device, dtype=torch.float32)]
        self.torch_data['target_point'] = torch.from_numpy(target_point).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.torch_data['velocity'] = torch.from_numpy(np.asarray(self.speed_data)).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.torch_data['adjacency_matrix'] = torch.from_numpy(graph_feature_dict['adjacency_matrix']).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.torch_data['node_feature_matrix'] = torch.from_numpy(graph_feature_dict['node_feature_matrix']).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.torch_data['edge_feature_matrix'] = torch.from_numpy(graph_feature_dict['edge_feature_matrix']).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.torch_data['node_num'] = torch.Tensor([graph_feature_dict['node_num']]).to(self.device, dtype=torch.int64)

        return

    def run_step(self, input_data, timestamp):
        self.step += 1
        """
        Execute one step of navigation.
        """
        self.agent_engaged = True

        self.input_data_process(input_data, timestamp)

        # control
        # High-Level
        vehicle_pos_xy = self.vehicle_pos[:2]
        nearest_node_ids = get_k_nearest_nodes(vehicle_pos_xy, self.G, self.pos_kdtree, self.pos_kdtree_keys,
                                               k=128).reshape(-1)

        # Low-Level
        try:
            partial_traj, start_progress = get_current_partial_traj(self.prev_pos_st, self.global_traj,
                                                                    self.path_progress)
        except:
            partial_traj, start_progress = self.global_traj, self.path_progress[0]

        vehicle_st_partial = cartesian_to_frenet_approx(partial_traj, vehicle_pos_xy.reshape(1, 2))
        target_st_partial = vehicle_st_partial + np.array([self.front_target_dist, 0])
        target_xy = frenet_to_cartesian_approx(partial_traj, target_st_partial)
        target_xy = target_xy.squeeze()
        self.prev_pos_st = vehicle_st_partial.squeeze().copy()
        self.prev_pos_st[0] += start_progress

        vehicle_gt_st_partial = cartesian_to_frenet_approx(partial_traj, self.vehicle_gt_pos[:2].reshape(1, 2))

        # Only predict every second step because we only get a LiDAR every second frame.
        if self.model_args.model_name.startswith('transfuser_origin'):
            if self.step % 2 == 0 or self.step <= 4:
                with torch.no_grad():
                    self.pred_wp = self.model(self.torch_data)
            steer, throttle, brake, metadata = self.model.control_pid(self.pred_wp, self.torch_data['velocity'])
            target_speed, steer_adjustment = metadata['desired_speed'], 0
        else:
            if self.step % 2 == 0 or self.step <= 4:
                # model output
                with torch.no_grad():
                    self.control_output = self.model(self.torch_data).squeeze().cpu().numpy()
            target_speed = self.control_output.item() * 2.0
            brake = target_speed < 0.1 or (self.speed_data / target_speed) > 1.1

            steer_adjustment = self.front_target_lane_dev
            steer, throttle = self.pid_controller.control(vehicle_pos_xy, self.compass_data, self.speed_data,
                                                          vehicle_st_partial, target_xy,
                                                          target_speed=target_speed, steer_adjustment=steer_adjustment)
        if brake:
            steer *= 0.5
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        criterion_dict = extract_criterion(self.scenario_manager)
        test_out = ''
        for key in criterion_dict.keys():
            test_out += str(key) + ": " + str(criterion_dict[key]) + '\n'

        control_dict = {'steer': steer, 'throttle': throttle, 'brake': brake,
                        'target_speed': target_speed, 'steer_adjustment': steer_adjustment}

        self.data_history['GPS'].append(input_data['GPS'][1])
        self.data_history['IMU'].append(input_data['IMU'][1])
        self.data_history['speed'].append(self.speed_data)
        self.data_history['pos'].append(self.vehicle_pos)
        self.data_history['pos_gt'].append(self.vehicle_gt_pos)
        self.data_history['control'].append(control_dict)
        self.data_history['timestamp'].append(timestamp)
        self.data_history['pos_st_partial'].append(vehicle_st_partial.squeeze())
        self.data_history['pos_gt_st_partial'].append(vehicle_gt_st_partial.squeeze())
        self.data_history['criterion_dict'].append(criterion_dict)

        # process sensor data
        image_center = input_data['Center'][1][:, :, -2::-1]
        vehicle_progress = (self.prev_pos_st[0] - self.initial_pos_st[0], self.final_st[0] - self.initial_pos_st[0], \
                            100. * (self.prev_pos_st[0] - self.initial_pos_st[0]) / (
                                        self.final_st[0] - self.initial_pos_st[0]))

        # record video
        if self.args.plot_txt_info:
            image_with_txt = Image.fromarray(image_center)
            _draw = ImageDraw.Draw(image_with_txt)

            txt_col = 10
            _draw.text((5, txt_col), 'GAME FPS: %.2f' % (1 / (timestamp - self._prev_timestamp)));
            txt_col += 10
            _draw.text((5, txt_col), 'SYS FPS: %.2f' % (1 / (time.time() - self._prev_time)));
            txt_col += 10
            _draw.text((5, txt_col), 'game time %.2f' % (timestamp - self.initial_timestamp));
            txt_col += 10
            _draw.text((5, txt_col), 'sys time %.2f' % (time.time() - self.initial_time));
            txt_col += 10
            _draw.text((5, txt_col), 'Steer: %.3f' % steer);
            txt_col += 10
            _draw.text((5, txt_col), 'Throttle: %.3f' % throttle);
            txt_col += 10
            _draw.text((5, txt_col), 'Brake: %s' % brake);
            txt_col += 10
            _draw.text((5, txt_col),
                       'GNSS Pos: %.2f %.2f' % (self.data_history['pos'][-1][0], self.data_history['pos'][-1][1]));
            txt_col += 10
            _draw.text((5, txt_col), 'GT Pos: %.2f %.2f' % (self.vehicle_gt_pos[0], self.vehicle_gt_pos[1]));
            txt_col += 10
            _draw.text((5, txt_col), 'Speed %.2f' % self.speed_data);
            txt_col += 10
            _draw.text((5, txt_col), 'Target Speed %.2f' % target_speed);
            txt_col += 10

            compass = self.compass_data
            R = np.array([np.cos(compass), -np.sin(compass), np.sin(compass), np.cos(compass)]).reshape(2, 2)
            direc_vec = np.matmul((target_xy - vehicle_pos_xy).reshape(1, 2), R)
            direc_error = -np.arctan2(direc_vec[:, 1], direc_vec[:, 0]).item()
            _draw.text((5, txt_col), 'Angle Dev %.2f' % direc_error);
            txt_col += 10
            _draw.text((5, txt_col), 'Pos Dev %.2f' % vehicle_st_partial.squeeze()[1]);
            txt_col += 10
            _draw.text((5, txt_col), 'Progress %.1f / %.1f (%.3f' % vehicle_progress + ' %)');
            txt_col += 10
            _draw.text((5, txt_col), test_out);
            txt_col += 10
            image_center = np.asarray(image_with_txt)

        if self.args.record_video:
            file_num = self.step - 1
            cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_Center.png' % file_num), self.image_data)
            cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_left.png' % file_num), self.image_data_left)
            cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_right.png' % file_num), self.image_data_right)
            cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_rear.png' % file_num), self.image_data_rear)
            np.save(os.path.join(self.args.log_dir, 'Snaps/%06d_lidar_raw.npy' % file_num), self.lidar_data)

        if self.args.show_camera:
            self._hic.run_interface(image_center)

        if not self.args.silent:
            sys.stdout.write(
                '\r' + 'Progress %.1f / %.1f (%.3f' % vehicle_progress + ' %)' + '.' * (1 + self.step % 3));
            sys.stdout.flush()

        self._prev_timestamp = timestamp
        self._prev_time = time.time()

        return control

    def destroy(self):
        """
        Cleanup
        """
        self.scenario_end = True
        del self.model

        if hasattr(self, 'data_history'):
            self.data_history['global_plan_pos'] = self.global_plan_pos
            self.data_history['global_traj'] = self.global_traj
            self.data_history['global_traj_node_id'] = self.global_traj_node_id
            self.data_history['geo_ref_dict'] = self.geo_ref_dict
            self.data_history['global_plan'] = [gnss for gnss, _ in self._global_plan]
            for k in self.data_history.keys():
                if type(self.data_history[k]) == list:
                    try:
                        self.data_history[k] = np.stack(self.data_history[k])
                    except:
                        pass

            pos_err_history = np.linalg.norm(self.data_history['pos_gt'][:,:2] - self.data_history['pos'][:,:2], axis=1)
            mean_err, max_err = np.mean(abs(pos_err_history)), np.max(abs(pos_err_history))
            print('mean err : {}, max err : {}'.format(round(mean_err,5), round(max_err,5)))
            self.data_history['mean_err'] = mean_err
            self.data_history['max_err'] = max_err

            dev_history = self.data_history['pos_gt_st_partial'][:,1]
            mean_dev, max_dev = np.mean(abs(dev_history)), np.max(abs(dev_history))
            print('mean dev : {}, max dev : {}'.format(round(mean_dev,5), round(max_dev,5)))
            self.data_history['mean_dev'] = mean_dev
            self.data_history['max_dev'] = max_dev

            target_lane_dev_history = self.data_history['pos_gt_st_partial'][:,1] - self.front_target_lane_dev
            target_lane_mean_dev, target_lane_max_dev = np.mean(abs(target_lane_dev_history)), np.max(abs(target_lane_dev_history))
            print('mean target_lane_dev : {}, max target_lane_dev : {}'.format(round(target_lane_mean_dev,5), round(target_lane_max_dev,5)))
            self.data_history['mean_target_lane_dev'] = target_lane_mean_dev
            self.data_history['max_target_lane_dev'] = target_lane_max_dev

            save_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
            print(save_time)
            if self.args.record_video:
                os.rename(os.path.join(self.args.log_dir, 'Snaps'), os.path.join(self.args.log_dir, 'Snaps_{:04d}'.format(self.NUM_SCENARIO)))
                logfile_name = 'data_history_{:04d}.npy'.format(self.NUM_SCENARIO)
                np.save(os.path.join(self.args.log_dir, logfile_name), self.data_history)
                ET.ElementTree(self.xml_root).write(os.path.join(self.args.log_dir, 'hd-map_{:04d}.xml'.format(self.NUM_SCENARIO)))
            if self.args.show_camera:
                self._hic._quit = True

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