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
import numpy as np

import networkx as nx
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from sklearn.neighbors import KDTree
from matplotlib import cm

from itertools import product
from collections import deque

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

def get_entry_point():
    return 'TestAgent'

import sys

sys.path.append('../..')
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.pid_controller import *
from utils.localization import *
from utils.lidar_utils import *
from RL.carla_env import extract_criterion

from planner import RoutePlanner

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

class TestAgent(AutonomousAgent):
    """
    Human agent to control the ego vehicle via keyboard
    """

    data_save_timer = 0
    current_control = None
    agent_engaged = False

    nearest_nodes_num = 64

    episode_num = 0

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP

        self.agent_engaged = False

        self._prev_timestamp = 0
        self.step = 0

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

        if self.args.use_keyboard_control:
            self._controller = KeyboardControl(path_to_conf_file=None)

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
             'width': 800, 'height': 600, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.lidar.ray_cast', 'x': 1.3, 'y': 0.0, 'z': 2.5, 'yaw': -90.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'},
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
         'width': 960, 'height': 540, 'fov': 100, 'id': 'Center'},
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
        i = 0
        while (len(self.processed_roads) < self.num_of_roads):
            if hasattr(self, 'scenario_end'):
                return
            i += 1
            G, seg_G, pos_kdtree_keys, pos_kdtree, processed_roads, signal_object_dict, signal_reference_dict = \
                extract_graph_info_partial(self.xml_root, cutoff_dist=(self.args.cutoff_dist + 100 * i), vehicle_pos=self.initial_pos,
                                            prev_seg_G=self.seg_G, prev_G=self.G, processed_roads=self.processed_roads,
                                           signal_object_dict=self.signal_object_dict, signal_reference_dict=self.signal_reference_dict)
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

    def input_data_process(self, input_data, timestamp):
        # Image sensor
        self.image_data = input_data['Center'][1][:, :, :3]
        self.image_data_left = input_data['rgb_left'][1][:, :, :3]
        self.image_data_right = input_data['rgb_right'][1][:, :, :3]
        self.image_data_rear = input_data['rgb_rear'][1][:, :, :3]

        # LIDAR sensor
        self.lidar_data = input_data['LIDAR'][1]
        # convert to top-view and remove ground
        # lidar_data_ground_filter = lidar_data[:,2] > 0  # remove ground
        # lidar_data_processed = lidar_data[lidar_data_ground_filter,:2]  # ignore intensity

        reinitialize = False
        # HD-map
        if not hasattr(self, 'prev_hdmap'):  # need to be regenerated for new map
            self._vehicle = CarlaDataProvider.get_hero_actor()
            self._world = self._vehicle.get_world()

            self._command_planner = RoutePlanner(7.5, 25.0, 257)
            self._command_planner.set_route(self._global_plan, True)
            self._waypoint_planner = RoutePlanner(4.0, 50)
            self._waypoint_planner.set_route(self._plan_gps_HACK, True)

            self._target_stop_sign = None  # the stop sign affecting the ego vehicle
            self._stop_completed = False  # if the ego vehicle has completed the stop sign
            self._affected_by_stop = False  # if the ego vehicle is influenced by a stop sign

            self.device = self.args.gpu_device if not self.args.no_cuda else "cpu"

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

            # generate graph near to the start point
            self.initial_pos = self.vehicle_pos[:2].copy()
            self.num_of_roads = len(self.xml_root.findall('road'))
            self.G, self.seg_G, self.pos_kdtree_keys, self.pos_kdtree, self.processed_roads, self.signal_object_dict, self.signal_reference_dict = \
                extract_graph_info_partial(self.xml_root, cutoff_dist=self.args.cutoff_dist, vehicle_pos=self.initial_pos)

            self.torch_G = convert_graph_as_torch(self.G, self.device)
            self.torch_seg_G = convert_graph_as_torch(self.seg_G, self.device)

            self.global_plan_pos = get_global_plan_pos(self._global_plan, geo_ref_dict=self.geo_ref_dict)
            self.global_plan_pos_with_starting = np.concatenate([self.initial_pos.reshape([1,2]), self.global_plan_pos])

            self.global_traj, self.global_traj_node_id, self.path_progress, (plan_start_idx, plan_end_idx) = \
                get_global_traj(self.global_plan_pos_with_starting, self.G, self.seg_G, self.pos_kdtree_keys, self.pos_kdtree)

            self.G2, self.pos_kdtree_keys2, self.pos_kdtree2 = contract_graph(self.G, self.pos_kdtree_keys, self.pos_kdtree)
            self.torch_G2 = convert_graph_as_torch(self.G2, self.device)

            self.initial_pos_st = np.array([self.path_progress[plan_start_idx], 0])
            self.initial_timestamp = timestamp
            self.initial_time = time.time()

            self.prev_pos_st = self.initial_pos_st.squeeze().copy()
            self.final_st = np.array([self.path_progress[plan_end_idx], 0])

            # update graph using thread
            t1 = Thread(target=self.graph_updater_thread)
            t1.daemon = True
            t1.start()

            if self.args.localization == 'filter':
                self.localization = LocalizationOperator(0.001, 0.001, 0.000005)
            elif self.args.localization == 'window':
                self.pos_deque = deque(maxlen=5)
            pid_param = [0.8, 0.3, 0.2, 20]
            self.pid_controller = PID_controller(direc_p=pid_param[0], direc_i=pid_param[1], direc_d=pid_param[2], win_size=pid_param[3])
            self.pid_controller.adjust_coeff = 0.6 if self.args.localization == 'gt' else 0.1
            self.front_target_dist = 3.0
            self.front_target_lane_dev = -0.1
            self.data_history = {'GPS': [], 'IMU': [], 'speed': [], 'pos': [], 'pos_gt': [], 'control': [], 'timestamp': [],
                                 'pos_st_partial': [], 'pos_gt_st_partial': [], 'criterion_dict': [], 'env_info_dict':[],
                                 'gps_pos': [], 'near_node': [], 'far_node': [], 'command': []}
            self.data_history['HD-map'] = input_data['HD-map'][1]['opendrive']
            self.data_history['adjust_coeff'] = self.pid_controller.adjust_coeff
            self.data_history['front_target_dist'] = self.front_target_dist
            self.data_history['front_target_lane_dev'] = self.front_target_lane_dev
            self.data_history['pid_param'] = pid_param
            self._traffic_lights = list()

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

        self.compass_data = 0.5 * np.pi - input_data['IMU'][1][-1]  # refine to Graph coord
        # self.compass_data = 0.5 * np.pi - self.localization.get_current_compass()

        # speedometer
        self.speed_data = input_data['speed'][1]['speed']

        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))

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
        sub_graph = self.torch_G.subgraph(nearest_node_ids)

        # Low-Level
        try:
            partial_traj, start_progress = get_current_partial_traj(self.prev_pos_st, self.global_traj, self.path_progress)
        except:
            partial_traj, start_progress = self.global_traj, self.path_progress[0]

        vehicle_st_partial = cartesian_to_frenet_approx(partial_traj, vehicle_pos_xy.reshape(1, 2))
        target_st_partial = vehicle_st_partial + np.array([self.front_target_dist, 0])
        target_xy = frenet_to_cartesian_approx(partial_traj, target_st_partial)
        target_xy = target_xy.squeeze()
        self.prev_pos_st = vehicle_st_partial.squeeze().copy()
        self.prev_pos_st[0] += start_progress

        vehicle_gt_st_partial = cartesian_to_frenet_approx(partial_traj, self.vehicle_gt_pos[:2].reshape(1, 2))

        target_speed = 4.0 if abs(vehicle_st_partial.squeeze()[1].item()) >= 0.1 else 7.0
        steer_adjustment = self.front_target_lane_dev
        steer, throttle = self.pid_controller.control(vehicle_pos_xy, self.compass_data, self.speed_data,
                                                      vehicle_st_partial, target_xy,
                                                      target_speed=target_speed, steer_adjustment=steer_adjustment)

        brake, obstacles, redlight = self._should_brake()
        vehicle_obstacles = np.array(obstacles[0])
        walker_obstacles = np.array(obstacles[1])

        if brake:
            steer *= 0.5
            throttle = 0.0

        if self.args.use_keyboard_control:
            control = self._controller.parse_events(timestamp - self._prev_timestamp)
        else:
            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = throttle
            control.brake = float(brake)

        criterion_dict = extract_criterion(self.scenario_manager)
        test_out = ''
        for key in criterion_dict.keys():
            test_out += str(key) + ": " + str(criterion_dict[key]) + '\n'

        control_dict = {'steer':steer, 'throttle':throttle, 'brake':brake,
                        'target_speed':target_speed, 'steer_adjustment':steer_adjustment}
        env_info_dict = {'vehicle_obstacles':vehicle_obstacles, 'walker_obstacles':walker_obstacles, 'redlight':redlight}

        gps_pos = (input_data['GPS'][1][:2] - self._waypoint_planner.mean) * self._waypoint_planner.scale

        near_node, near_command = self._waypoint_planner.run_step(gps_pos)
        far_node, far_command = self._command_planner.run_step(gps_pos)
        command = near_command.value

        if (self.step - 1) % self.args.record_frame_skip == 0:
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
            self.data_history['gps_pos'].append(gps_pos)
            self.data_history['near_node'].append(near_node)
            self.data_history['far_node'].append(far_node)
            self.data_history['command'].append(command)

        # process sensor data
        image_center = input_data['Center'][1][:, :, -2::-1]
        vehicle_progress = (self.prev_pos_st[0] - self.initial_pos_st[0], self.final_st[0] - self.initial_pos_st[0], \
                            100. * (self.prev_pos_st[0] - self.initial_pos_st[0]) /
                            (self.final_st[0] - self.initial_pos_st[0]))

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
                       'GNSS Pos: %.2f %.2f' % (self.vehicle_pos[0], self.vehicle_pos[1]));
            txt_col += 10
            _draw.text((5, txt_col), 'GT Pos: %.2f %.2f' % (self.vehicle_gt_pos[0], self.vehicle_gt_pos[1]));
            txt_col += 10
            _draw.text((5, txt_col), 'Speed %.2f' % self.speed_data);
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
            file_num = (self.step - 1) // self.args.record_frame_skip
            if (self.step - 1) % self.args.record_frame_skip == 0:
                cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_Center.png' % file_num), self.image_data)
                cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_left.png' % file_num), self.image_data_left)
                cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_right.png' % file_num), self.image_data_right)
                cv2.imwrite(os.path.join(self.args.log_dir, 'Snaps/%06d_rear.png' % file_num), self.image_data_rear)
                np.save(os.path.join(self.args.log_dir, 'Snaps/%06d_lidar_raw.npy' % file_num), self.lidar_data)

        if self.args.show_camera:
            self._hic.run_interface(image_center)

        if not self.args.silent:
            sys.stdout.write('\r' + 'Progress %.1f / %.1f (%.3f' % vehicle_progress + ' %)' + '.' * (1+self.step%3));
            sys.stdout.flush()

        self._prev_timestamp = timestamp
        self._prev_time = time.time()

        return control

    def destroy(self):
        """
        Cleanup
        """
        self.scenario_end = True
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

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle, vehicle_xy_list = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        walker, walker_xy_list = self._is_walker_hazard(actors.filter('*walker*'))
        redlight = self._is_light_red(actors.filter('*traffic_light*'))
        stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))

        return any([vehicle, walker, redlight, stop_sign]), (vehicle_xy_list, walker_xy_list), redlight

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._world.get_map().get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _is_stop_sign_hazard(self, stop_sign_list):
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return None
                else:
                    return self._target_stop_sign
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return None

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._world.get_map().get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    return self._target_stop_sign

        return None

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting is not None

        return False

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        total_collides = False
        walker_xy_list = []
        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            carla_vector = walker.get_location()
            walker_xy_list.append(np.float32([carla_vector.x, carla_vector.y]))

            total_collides = total_collides or collides

        return total_collides, walker_xy_list

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat

        total_collides = False
        vehicle_xy_list = []
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            carla_vector = target_vehicle.get_location()
            vehicle_xy_list.append(np.float32([carla_vector.x, carla_vector.y]))

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue
            else:
                total_collides = True

        return total_collides, vehicle_xy_list


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self, path_to_conf_file):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # Get the mode
        if path_to_conf_file:

            with (open(path_to_conf_file, "r")) as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {'records': []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except json.JSONDecodeError:
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):

        # transform strs into VehicleControl commands
        for entry in self._records['records']:
            control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                           steer=entry['control']['steer'],
                                           brake=entry['control']['brake'],
                                           hand_brake=entry['control']['hand_brake'],
                                           reverse=entry['control']['reverse'],
                                           manual_gear_shift=entry['control']['manual_gear_shift'],
                                           gear=entry['control']['gear'])
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp*1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.6
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        new_record = {
            'control': {
                'throttle': self._control.throttle,
                'steer': self._control.steer,
                'brake': self._control.brake,
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            }
        }

        self._log_data['records'].append(new_record)

    def __del__(self):
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)