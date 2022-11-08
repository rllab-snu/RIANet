import os
import json
import datetime
import pathlib
import time
import cv2
import carla
import pygame
import json
from collections import deque

from threading import Thread

import torch
import carla
import numpy as np
from PIL import Image, ImageDraw

from leaderboard.autoagents import autonomous_agent

import sys
sys.path.append('../..')
from IL.config import GlobalConfig
from IL.data_baseline import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points, get_detection_patch
from planner import RoutePlanner

import math
import importlib
from matplotlib import cm
from collections import namedtuple

SAVE_PATH = os.environ.get('SAVE_PATH', None)

def get_entry_point():
    return 'ILAgent'

from autonomous_agent import AutonomousAgent

import sys

sys.path.append('../..')
from utils.graph_utils import *
from utils.gnss_utils import *
from utils.pid_controller import *
from utils.localization import *
from utils.lidar_utils import *

from utils.traffic_light_detector import *
from utils.object_detector import *

#from RL.carla_env import extract_criterion

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

class ILAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.MAP
        self.args = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(),
                             'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

        self.device = self.args.gpu_device if not self.args.no_cuda else "cpu"
        self.config = GlobalConfig()

        # load torch model
        with open(os.path.join(self.args.model_dir, 'args.txt')) as f:
            d = json.load(f)
            self.model_args = namedtuple('args', d.keys())(*d.values())
        self.net = importlib.import_module('IL.models.' + self.model_args.model_name).Model(self.model_args,
                                                                                              self.device)
        if self.args.test_last_model:
            self.net.load_state_dict(
                torch.load(os.path.join(self.args.model_dir, 'model_101.pth'), map_location=self.device))
        else:
            self.net.load_state_dict(
                torch.load(os.path.join(self.args.model_dir, 'best_model.pth'), map_location=self.device))
        self.net.to(self.device)
        self.net.eval()

        assert self.args.use_2d_detection == 0 or self.args.use_3d_detection == 0
        if self.args.use_2d_detection > 0:
            gpu_idx = int(self.args.gpu_device[-1])
            self.td = TrafficLightDetector(400, 300, "./utils/models/traffic_light_detection/faster-rcnn/", gpu_idx=gpu_idx)
            self.od = ObjectDetector(400, 300, './utils/models/obstacle_detection/' + self.args.object_detection_model,
                                './utils/models/pylot.names', gpu_idx=gpu_idx)

        self.save_path = None
        if self.args.record_video:
            now = datetime.datetime.now()
            string = pathlib.Path('').stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(self.args.log_dir) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_rear').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_right').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'lidar_0').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'lidar_1').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'meta').mkdir(parents=True, exist_ok=False)
            (self.save_path / 'detect').mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        if self.args.show_camera:
            self._hic = HumanInterface(self.args)

        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb'
            },
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
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 100,
                'id': 'rgb_right'
            } if self.args.show_camera else {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -4, 'y': 0.0, 'z': 4,
                'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 90,
                'id': 'rgb_rear'
            } if self.args.show_camera else {
                'type': 'sensor.camera.rgb',
                'x': -1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 1.3, 'y': 0.0, 'z': 2.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'id': 'lidar'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            },
            {'type': 'sensor.opendrive_map', 'reading_frequency': 20, 'id': 'HD-map'}
        ]

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

            global_traj, global_traj_node_id, path_progress, (plan_start_idx, plan_end_idx) = \
                get_global_traj(self.global_plan_pos_with_starting, G, seg_G, pos_kdtree_keys, pos_kdtree)

            G2, pos_kdtree_keys2, pos_kdtree2 = contract_graph(G, pos_kdtree_keys, pos_kdtree)

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
            self.global_traj, self.global_traj_node_id, self.path_progress = global_traj, global_traj_node_id, path_progress
            self.G2, self.pos_kdtree_keys2, self.pos_kdtree2 = G2, pos_kdtree_keys2, pos_kdtree2

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

    def tick(self, input_data, timestamp):
        self.step += 1

        # HD-map
        if not hasattr(self, 'prev_hdmap'):  # need to be regenerated for new map
            print('pre-process start')
            self.start_time = time.time()
            self._prev_time = self.start_time
            self.prev_hdmap = input_data['HD-map'][1]['opendrive']
            self.xml_root = ET.fromstring(self.prev_hdmap)
            self.geo_ref_dict = get_georeference(self.xml_root)

            # gnss
            self.vehicle_pos = gnss_to_xy(input_data['gps'][1], self.geo_ref_dict)  # initial vehicle pos start with raw

            # generate graph near to the start point
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
                self.G, self.seg_G, self.pos_kdtree_keys, self.pos_kdtree, self.processed_roads, self.signal_object_dict, self.signal_reference_dict = \
                    extract_graph_info_partial(self.xml_root, cutoff_dist=self.args.cutoff_dist,
                                               vehicle_pos=self.initial_pos)
                self.G2, self.pos_kdtree_keys2, self.pos_kdtree2 = contract_graph(self.G, self.pos_kdtree_keys,
                                                                                  self.pos_kdtree)

            self.global_plan_pos = get_global_plan_pos(self._global_plan, geo_ref_dict=self.geo_ref_dict)
            self.global_plan_pos_with_starting = np.concatenate(
                [self.initial_pos.reshape([1, 2]), self.global_plan_pos])

            self.global_traj, self.global_traj_node_id, self.path_progress, (plan_start_idx, plan_end_idx) = \
                get_global_traj(self.global_plan_pos_with_starting, self.G, self.seg_G, self.pos_kdtree_keys,
                                self.pos_kdtree)

            self.initial_pos_st = np.array([self.path_progress[plan_start_idx], 0])

            self.prev_pos_st = self.initial_pos_st.squeeze().copy()
            self.final_st = np.array([self.path_progress[plan_end_idx], 0])

            # update graph using thread
            if not os.path.exists(dir_name):
                t1 = Thread(target=self.graph_updater_thread)
                t1.daemon = True
                t1.start()

            if self.args.localization == 'filter':
                self.localization = LocalizationOperator(0.001, 0.001, 0.000005)

            pid_param = [0.8, 0.3, 0.2, 20]
            self.pid_controller = PID_controller(direc_p=pid_param[0], direc_i=pid_param[1], direc_d=pid_param[2],
                                                 win_size=pid_param[3])
            self.pid_controller.adjust_coeff = 0.1
            self.front_target_dist = 3.0
            self.front_target_lane_dev = -0.1

            process_time = round(time.time() - self.start_time, 3)
            print('done with : {} s'.format(process_time))

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0
        lidar = input_data['lidar'][1][:, :3]

        if self.args.localization == 'filter':
            self.vehicle_pos, _ = self.localization.run(input_data['gps'][1], input_data['imu'][1],
                                                        timestamp, self.geo_ref_dict)
        elif self.args.localization == 'raw':
            self.vehicle_pos = gnss_to_xy(input_data['gps'][1], self.geo_ref_dict)

        graph_feature_dict, _ = extract_graph_features(self.vehicle_pos[:2], 0.5 * np.pi - compass, speed, \
                                                    self.G2, self.pos_kdtree_keys2, self.pos_kdtree2, self.global_traj_node_id, \
                                                    nearest_node_num=96, use_node_filter=True)

        result = {
            'rgb': rgb,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'rgb_rear': rgb_rear,
            'lidar': lidar,
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'graph_feature_dict': graph_feature_dict
        }

        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        if self.args.use_2d_detection:
            #img = Image.fromarray(rgb)
            #img_array = np.asarray(img)
            img_array = rgb

            result_td = self.td.detect(img_array)
            result_od = self.od.detect(img_array)

            traffic_light_objects = []
            detected_objects = []

            for obj in result_td:
                bbox = obj.bounding_box_2D
                td_result = {'id': obj.id, 'state': obj.state.name, 'confidence': float(obj.confidence.numpy()),
                             'x_min': bbox.x_min, 'x_max': bbox.x_max, 'y_min': bbox.y_min, 'y_max': bbox.y_max}

                traffic_light_objects.append(td_result)

            for obj in result_od:
                bbox = obj.bounding_box_2D
                od_result = {'id': obj.id, 'label': obj.label, 'confidence': float(obj.confidence.numpy()),
                             'x_min': bbox.x_min, 'x_max': bbox.x_max, 'y_min': bbox.y_min, 'y_max': bbox.y_max}

                detected_objects.append(od_result)

            detection_result = get_detection_patch(traffic_light_objects + detected_objects)
            for d_key in detection_result.keys():
                result[d_key] = detection_result[d_key]

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data, timestamp)

        # Low-Level
        try:
            partial_traj, start_progress = get_current_partial_traj(self.prev_pos_st, self.global_traj,
                                                                    self.path_progress)
        except:
            partial_traj, start_progress = self.global_traj, self.path_progress[0]

        vehicle_pos_xy = self.vehicle_pos[:2]
        vehicle_st_partial = cartesian_to_frenet_approx(partial_traj, vehicle_pos_xy.reshape(1, 2))
        target_st_partial = vehicle_st_partial + np.array([self.front_target_dist, 0])
        target_xy = frenet_to_cartesian_approx(partial_traj, target_st_partial)
        target_xy = target_xy.squeeze()
        self.prev_pos_st = vehicle_st_partial.squeeze().copy()
        self.prev_pos_st[0] += start_progress

        if self.step < self.config.seq_len:
            rgb = torch.from_numpy(
                scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb'].append(rgb.to(self.device, dtype=torch.float32))

            if not self.config.ignore_sides:
                rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']),
                                                                 crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_left'].append(rgb_left.to(self.device, dtype=torch.float32))

                rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']),
                                                                  crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_right'].append(rgb_right.to(self.device, dtype=torch.float32))

            if not self.config.ignore_rear:
                rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']),
                                                                 crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_rear'].append(rgb_rear.to(self.device, dtype=torch.float32))

            self.input_buffer['lidar'].append(tick_data['lidar'])
            self.input_buffer['gps'].append(tick_data['gps'])
            self.input_buffer['thetas'].append(tick_data['compass'])

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

            vehicle_progress = (self.prev_pos_st[0] - self.initial_pos_st[0], self.final_st[0] - self.initial_pos_st[0], \
                                100. * (self.prev_pos_st[0] - self.initial_pos_st[0]) / (
                                        self.final_st[0] - self.initial_pos_st[0]))

            sys.stdout.write(
                '\r' + 'Progress %.1f / %.1f (%.3f' % vehicle_progress + ' %)' + '.' * (1 + self.step % 3));
            sys.stdout.flush()

            return control

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to(self.device, dtype=torch.float32)
        command = torch.FloatTensor([tick_data['next_command']]).to(self.device, dtype=torch.float32)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                     torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to(self.device, dtype=torch.float32)

        encoding = []
        rgb = torch.from_numpy(
            scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
        self.input_buffer['rgb'].popleft()
        self.input_buffer['rgb'].append(rgb.to(self.device, dtype=torch.float32))

        if not self.config.ignore_sides:
            rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']),
                                                             crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_left'].popleft()
            self.input_buffer['rgb_left'].append(rgb_left.to(self.device, dtype=torch.float32))

            rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']),
                                                              crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_right'].popleft()
            self.input_buffer['rgb_right'].append(rgb_right.to(self.device, dtype=torch.float32))

        if not self.config.ignore_rear:
            rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']),
                                                             crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_rear'].popleft()
            self.input_buffer['rgb_rear'].append(rgb_rear.to(self.device, dtype=torch.float32))

        self.input_buffer['lidar'].popleft()
        self.input_buffer['lidar'].append(tick_data['lidar'])
        self.input_buffer['gps'].popleft()
        self.input_buffer['gps'].append(tick_data['gps'])
        self.input_buffer['thetas'].popleft()
        self.input_buffer['thetas'].append(tick_data['compass'])

        # transform the lidar point clouds to local coordinate frame
        ego_theta = self.input_buffer['thetas'][-1]
        ego_x, ego_y = self.input_buffer['gps'][-1]

        # Only predict every second step because we only get a LiDAR every second frame.
        if (self.step % 2 == 0 or self.step <= 4):
            for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
                curr_theta = self.input_buffer['thetas'][i]
                curr_x, curr_y = self.input_buffer['gps'][i]
                lidar_point_cloud[:, 1] *= -1  # inverts x, y
                lidar_transformed = transform_2d_points(lidar_point_cloud,
                                                        np.pi / 2 - curr_theta, -curr_x, -curr_y, np.pi / 2 - ego_theta,
                                                        -ego_x, -ego_y)
                lidar_transformed = torch.from_numpy(
                    lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
                self.lidar_processed = list()
                self.lidar_processed.append(lidar_transformed.to(self.device, dtype=torch.float32))

            graph_feature_dict = tick_data['graph_feature_dict']

            torch_data = {}
            torch_data['images'] = self.input_buffer['rgb'] + self.input_buffer['rgb_left'] + \
                                        self.input_buffer['rgb_right'] + self.input_buffer['rgb_rear']
            torch_data['lidars'] = self.lidar_processed
            torch_data['target_point'] = target_point
            torch_data['command'] = command
            torch_data['velocity'] = gt_velocity
            torch_data['adjacency_matrix'] = torch.from_numpy(graph_feature_dict['adjacency_matrix']).unsqueeze(
                0).to(self.device, dtype=torch.float32)
            torch_data['node_feature_matrix'] = torch.from_numpy(
                graph_feature_dict['node_feature_matrix']).unsqueeze(0).to(self.device, dtype=torch.float32)
            torch_data['edge_feature_matrix'] = torch.from_numpy(
                graph_feature_dict['edge_feature_matrix']).unsqueeze(0).to(self.device, dtype=torch.float32)
            torch_data['node_num'] = torch.Tensor([graph_feature_dict['node_num']]).to(self.device,
                                                                                            dtype=torch.int64)
            if self.args.use_2d_detection > 0:
                torch_data['td_img_class'] = tick_data['td_img_class'].unsqueeze(0).to(self.device, dtype=torch.float32)
                torch_data['td_img_mask'] = tick_data['td_img_mask'].unsqueeze(0).to(self.device, dtype=torch.float32)
                torch_data['td_num'] = torch.Tensor([tick_data['td_num']]).to(self.device, dtype=torch.int64)
                torch_data['td_img_mask_info'] = tick_data['td_img_mask_info'].unsqueeze(0).to(self.device, dtype=torch.float32)
                torch_data['od_img_class'] = tick_data['od_img_class'].unsqueeze(0).to(self.device, dtype=torch.float32)
                torch_data['od_img_mask'] = tick_data['od_img_mask'].unsqueeze(0).to(self.device, dtype=torch.float32)
                torch_data['od_num'] = torch.Tensor([tick_data['od_num']]).to(self.device, dtype=torch.int64)
                torch_data['od_img_mask_info'] = tick_data['od_img_mask_info'].unsqueeze(0).to(self.device, dtype=torch.float32)

            if hasattr(self.net.config, "pred_target_point") and self.net.config.pred_target_point:
                self.pred_wp = self.net(torch_data)
            elif self.model_args.model_name.startswith('cilrs'):
                self.steer, self.throttle, self.brake, self.velocity = self.net(torch_data)
            else:
                self.control_output = self.net(torch_data).squeeze().cpu().numpy()

        if hasattr(self.net.config, "pred_target_point") and self.net.config.pred_target_point:
            if self.args.speed_combination:
                _, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
                self.pid_metadata = metadata

                waypoints = self.pred_wp[0].data.cpu().numpy()
                # flip y is (forward is negative in our waypoints)
                waypoints[:, 1] *= -1
                target_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
                steer_adjustment = self.front_target_lane_dev
                steer, _ = self.pid_controller.control(vehicle_pos_xy, 0.5 * np.pi - tick_data['compass'], tick_data['speed'],
                                                              vehicle_st_partial, target_xy,
                                                              target_speed=target_speed, steer_adjustment=steer_adjustment)
            else:
                steer, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
                self.pid_metadata = metadata
        elif self.model_args.model_name.startswith('cilrs'):
            steer, throttle, brake, velocity = self.steer, self.throttle, self.brake, self.velocity
        else:
            target_speed = self.control_output.item() * 2.0
            brake = target_speed < 0.1 or (tick_data['speed'] / target_speed) > 1.1
            steer_adjustment = self.front_target_lane_dev
            steer, throttle = self.pid_controller.control(vehicle_pos_xy, 0.5 * np.pi - tick_data['compass'], tick_data['speed'],
                                                          vehicle_st_partial, target_xy,
                                                          target_speed=target_speed, steer_adjustment=steer_adjustment)
        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        vehicle_progress = (self.prev_pos_st[0] - self.initial_pos_st[0], self.final_st[0] - self.initial_pos_st[0], \
                            100. * (self.prev_pos_st[0] - self.initial_pos_st[0]) / (
                                        self.final_st[0] - self.initial_pos_st[0]))

        sys.stdout.write(
            '\r' + 'Progress %.1f / %.1f (%.3f' % vehicle_progress + ' %)' + '.' * (1 + self.step % 3));
        sys.stdout.flush()

        if self.args.show_camera:
            image_center = input_data['rgb_rear'][1][:, :, -2::-1]

            # record video
            if self.args.plot_txt_info:
                image_cam = input_data['rgb'][1][:, :, -2::-1]
                image_center[:150,-200:] = image_cam[::2, ::2]

                with open(self.args.checkpoint) as json_file:
                    json_data = json.load(json_file)
                    record = json_data['_checkpoint']['records']
                    num_record = len(record)
                    num_success = len([r['status'] == 'Completed' for r in record])
                    num_infraction = sum([any([len(infraction) != 0 for infraction in r['infractions'].values()]) for r in record])

                    image_with_txt = Image.fromarray(image_center)
                    _draw = ImageDraw.Draw(image_with_txt)

                    txt_col = 5
                    _draw.text((650, txt_col), 'Input Camera Image');
                    _draw.text((10, txt_col), 'Progress: %.1f m / %.1f m (%.3f' % vehicle_progress + ' %)'); txt_col += 15
                    _draw.text((10, txt_col), 'Route Completion Success Rate: {} / {}'.format(num_success, num_record)); txt_col += 15
                    _draw.text((10, txt_col), 'Traffic Rule Violation Rate: {} / {}'.format(num_infraction, num_record)); txt_col += 15
                    image_center = np.asarray(image_with_txt)

            self._hic.run_interface(image_center)

        if self.args.record_video and self.step % self.args.record_frame_skip == 0:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.args.record_frame_skip

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        if self.args.show_camera:
            Image.fromarray(tick_data['rgb_rear']).save(self.save_path / 'rgb_rear' / ('%04d.png' % frame))
            Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))

        '''
        Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(
            self.save_path / 'lidar_0' / ('%04d.png' % frame))
        Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(
            self.save_path / 'lidar_1' / ('%04d.png' % frame))

        detection_data = {}
        detection_data['td_img_class'] = tick_data['td_img_class']
        detection_data['td_img_mask'] = tick_data['td_img_mask']
        detection_data['td_num'] = tick_data['td_num']
        detection_data['td_img_mask_info'] = tick_data['td_img_mask_info']
        detection_data['od_img_class'] = tick_data['od_img_class']
        detection_data['od_img_mask'] = tick_data['od_img_mask']
        detection_data['od_num'] = tick_data['od_num']
        detection_data['od_img_mask_info'] = tick_data['od_img_mask_info']

        np.save(self.save_path / 'detect' / ('%04d.npy' % frame), detection_data)

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()
        '''

    def destroy(self):
        self.scenario_end = True

        if self.args.show_camera:
            self._hic._quit = True

        del self.net
