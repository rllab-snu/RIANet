import os
import numpy as np
from tqdm import tqdm
import sys
import argparse

from PIL import Image

sys.path.insert(0,'..')
from utils.traffic_light_detector import *
from utils.object_detector import *

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Directory to dataset')
parser.add_argument('--object-detection-model', type=str, default='faster_rcnn_resnet101_coco_2018_01_28/480p', help='Directory to model')

args = parser.parse_args()

root_dir = args.data_dir
train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07']
val_towns = ['Town05']
train_data, val_data = [], []
for town in train_towns:
    train_data.append(os.path.join(root_dir, town + '_tiny'))
    train_data.append(os.path.join(root_dir, town + '_short'))
    if town != 'Town07':
        train_data.append(os.path.join(root_dir, town + '_long'))
for town in val_towns:
    val_data.append(os.path.join(root_dir, town + '_short'))

gpu_idx = 0
td = TrafficLightDetector(400, 300, "../utils/models/traffic_light_detection/faster-rcnn/", gpu_idx=gpu_idx)
od = ObjectDetector(400, 300, '../utils/models/obstacle_detection/'+args.object_detection_model, '../utils/models/pylot.names', gpu_idx=gpu_idx)

for sub_root in tqdm(train_data + val_data, file=sys.stdout):
    root_files = os.listdir(sub_root)
    routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]

    for route in routes:
        route_dir = os.path.join(sub_root, route)
        print(route_dir)

        num_seq = len(os.listdir(route_dir + "/rgb_front/"))

        if not os.path.exists(route_dir + "/pre_detected_data"):
            os.mkdir(route_dir + "/pre_detected_data")

        for seq in range(num_seq):
            img = Image.open(route_dir + '/rgb_front/{:04d}.png'.format(seq))
            img_array = np.asarray(img)

            result_td = td.detect(img_array)
            result_od = od.detect(img_array)

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

            np.save(route_dir + "/pre_detected_data/{:04d}.npy".format(seq),
                    traffic_light_objects + detected_objects)
