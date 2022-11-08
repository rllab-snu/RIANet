"""Implements an operator that detects obstacles."""
import time
import numpy as np

from utils.pylot_utils import *

import tensorflow as tf

class ObjectDetector(object):
    """Detects obstacles using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.
    """
    def __init__(self, camera_setup_width, camera_setup_height, \
                 model_path="./models/obstacle_detection/faster-rcnn/",
                coco_label_path="./models/pylot.names", gpu_idx=0):

        self.camera_setup_width = camera_setup_width
        self.camera_setup_height = camera_setup_height

        # Only sets memory growth for flagged GPU
        self.gpu_idx = gpu_idx
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_visible_devices(
                [physical_devices[self.gpu_idx]], 'GPU')
            tf.config.experimental.set_memory_growth(
                physical_devices[self.gpu_idx], True)

        self.obstacle_detection_model_paths = model_path
        self.path_coco_labels = coco_label_path

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(model_path)

        self.obstacle_detection_min_score_threshold = 0.5

        self._coco_labels = load_coco_labels(self.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0

        # Serve some junk image to load up the model.
        self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))

    def detect_multiple(self, image_np):

        num_detections, res_boxes, res_scores, res_classes = self.__run_model(image_np)

    def detect(self, image_np):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
        """
        num_detections, res_boxes, res_scores, res_classes = self.__run_model(image_np)
        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >=
                        self.obstacle_detection_min_score_threshold):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] *
                                    self.camera_setup_width),
                                int(res_boxes[i][3] *
                                    self.camera_setup_width),
                                int(res_boxes[i][0] *
                                    self.camera_setup_height),
                                int(res_boxes[i][2] *
                                    self.camera_setup_height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id))
                        self._unique_id += 1
                    else:
                        print(
                            'Ignoring non essential detection {}'.format(
                                self._coco_labels[res_classes[i]]))
            else:
                print('Filtering unknown class: {}'.format(
                    res_classes[i]))

        return obstacles

    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # [None, None, 3] -> [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return num_detections, res_boxes, res_scores, res_classes

