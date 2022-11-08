"""Implements an operator that detects traffic lights."""
import logging
import numpy as np
import tensorflow as tf
from enum import Enum
from utils.pylot_utils import *

class TrafficLightColor(Enum):
    """Enum to represent the states of a traffic light."""
    RED = 1
    YELLOW = 2
    GREEN = 3
    OFF = 4

    def get_label(self):
        """Gets the label of a traffic light color.

        Returns:
            :obj:`str`: The label string.
        """
        if self.value == 1:
            return 'red traffic light'
        elif self.value == 2:
            return 'yellow traffic light'
        elif self.value == 3:
            return 'green traffic light'
        else:
            return 'off traffic light'

    def get_color(self):
        if self.value == 1:
            return [255, 0, 0]
        elif self.value == 2:
            return [255, 165, 0]
        elif self.value == 3:
            return [0, 255, 0]
        else:
            return [0, 0, 0]
        
class TrafficLight(Obstacle):
    """Class used to store info about traffic lights.

    Args:
        confidence (:obj:`float`): The confidence of the detection.
        state (:py:class:`.TrafficLightColor`): The state of the traffic light.
        id (:obj:`int`, optional): The identifier of the traffic light.
        transform (:py:class:`~pylot.utils.Transform`, optional): Transform of
            the traffic light.
        trigger_volume_extent (:py:class:`pylot.utils.Vector3D`, optional): The
            extent of the trigger volume of the light.
        bounding_box (:py:class:`.BoundingBox2D`, optional): The bounding box
            of the traffic light in camera view.

    """
    def __init__(self,
                 confidence: float,
                 state: TrafficLightColor,
                 id: int = -1,
                 transform: Transform = None,
                 trigger_volume_extent: Vector3D = None,
                 bounding_box: BoundingBox2D = None):
        super(TrafficLight, self).__init__(bounding_box, confidence,
                                           state.get_label(), id, transform)
        self.state = state
        self.trigger_volume_extent = trigger_volume_extent

    def is_traffic_light_visible(self,
                                 camera_transform: Transform,
                                 town_name: str = None,
                                 distance_threshold: int = 70):
        """Checks if the traffic light is visible from the camera transform.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): Transform of the
                camera in the world frame of reference.
            distance_threshold (:obj:`int`): Maximum distance to the camera
                (in m).

        Returns:
            bool: True if the traffic light is visible from the camera
            transform.
        """
        # We dot product the forward vectors (i.e., orientation).
        # Note: we have to rotate the traffic light forward vector
        # so that it's pointing out from the traffic light in the
        # opposite direction in which the ligth is beamed.
        prod = np.dot([
            self.transform.forward_vector.y, -self.transform.forward_vector.x,
            self.transform.forward_vector.z
        ], [
            camera_transform.forward_vector.x,
            camera_transform.forward_vector.y,
            camera_transform.forward_vector.z
        ])
        if self.transform.location.distance(
                camera_transform.location) > distance_threshold:
            return prod > 0.4

        if town_name is None:
            return prod > -0.80
        else:
            if town_name == 'Town01' or town_name == 'Town02':
                return prod > 0.3
        return prod > -0.80

    def get_all_detected_traffic_light_boxes(self, town_name: str, depth_frame,
                                             segmented_image):
        """ Returns traffic lights for all boxes of a simulator traffic light.

        Note:
            All the traffic lights returned will have the same id and
            transform.

        Args:
            town_name (:obj:`str`): Name of the town in which the traffic light
                is.
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                 Depth frame.
            segmented_image: A segmented image np array used to refine the
                 bounding boxes.

        Returns:
            list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`):
            Detected traffic lights, one for each traffic light box.
        """
        traffic_lights = []
        bboxes = self._get_bboxes(town_name)
        # Convert the returned bounding boxes to 2D and check if the
        # light is occluded. If not, add it to the traffic lights list.
        for bbox in bboxes:
            bounding_box = [
                loc.to_camera_view(
                    depth_frame.camera_setup.get_extrinsic_matrix(),
                    depth_frame.camera_setup.get_intrinsic_matrix())
                for loc in bbox
            ]
            bbox_2d = get_bounding_box_in_camera_view(
                bounding_box, depth_frame.camera_setup.width,
                depth_frame.camera_setup.height)
            if not bbox_2d:
                continue

            # Crop the segmented and depth image to the given bounding box.
            cropped_image = segmented_image[bbox_2d.y_min:bbox_2d.y_max,
                                            bbox_2d.x_min:bbox_2d.x_max]
            cropped_depth = depth_frame.frame[bbox_2d.y_min:bbox_2d.y_max,
                                              bbox_2d.x_min:bbox_2d.x_max]

            if cropped_image.size > 0:
                masked_image = np.zeros_like(cropped_image)
                masked_image[np.where(
                    np.logical_or(cropped_image == 12,
                                  cropped_image == 18))] = 1
                if np.sum(masked_image) >= 0.20 * masked_image.size:
                    masked_depth = cropped_depth[np.where(masked_image == 1)]
                    mean_depth = np.mean(masked_depth) * 1000
                    if abs(mean_depth -
                           bounding_box[0].z) <= 2 and mean_depth < 150:
                        traffic_lights.append(
                            TrafficLight(1.0, self.state, self.id,
                                         self.transform,
                                         self.trigger_volume_extent, bbox_2d))
        return traffic_lights

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'TrafficLight(confidence: {}, state: {}, id: {}, ' \
            'transform: {}, trigger_volume_extent: {}, bbox: {})'.format(
                self.confidence, self.state, self.id, self.transform,
                self.trigger_volume_extent, self.bounding_box)

    def _relative_to_traffic_light(self, points):
        """Transforms the bounding box specified in the points relative to the
        light.

        Args:
            points: An array of length 4 representing the 4 points of the
                rectangle.
        """
        def rotate(yaw, location):
            """ Rotate a given 3D vector around the Z-axis. """
            rotation_matrix = np.identity(3)
            rotation_matrix[0, 0] = np.cos(yaw)
            rotation_matrix[0, 1] = -np.sin(yaw)
            rotation_matrix[1, 0] = np.sin(yaw)
            rotation_matrix[1, 1] = np.cos(yaw)
            location_vector = np.array([[location.x], [location.y],
                                        [location.z]])
            transformed = np.dot(rotation_matrix, location_vector)
            return Location(x=transformed[0, 0],
                                        y=transformed[1, 0],
                                        z=transformed[2, 0])

        transformed_points = [
            rotate(np.radians(self.transform.rotation.yaw), point)
            for point in points
        ]
        base_relative_points = [
            self.transform.location + point for point in transformed_points
        ]
        return base_relative_points

    def _get_bboxes(self, town_name: str):
        if town_name == 'Town01' or town_name == 'Town02':
            return self._get_bboxes_for_town1_or_2()
        elif town_name == 'Town03':
            return self._get_bboxes_for_town3()
        elif town_name == 'Town04':
            return self._get_bboxes_for_town4()
        elif town_name == 'Town05':
            return self._get_bboxes_for_town5()
        else:
            raise ValueError(
                'Could not find a town named {}'.format(town_name))

    def _get_bboxes_for_town1_or_2(self):
        points = [
            # Back Plane
            Location(x=-0.5, y=-0.1, z=2),
            Location(x=+0.1, y=-0.1, z=2),
            Location(x=+0.1, y=-0.1, z=3),
            Location(x=-0.5, y=-0.1, z=3),
            # Front Plane
            Location(x=-0.5, y=0.5, z=2),
            Location(x=+0.1, y=0.5, z=2),
            Location(x=+0.1, y=0.5, z=3),
            Location(x=-0.5, y=0.5, z=3),
        ]
        return [self._relative_to_traffic_light(points)]

    def _get_bboxes_for_town3(self):
        bboxes = []
        if (self.trigger_volume_extent.x > 2 or self.id in [
                66,
                67,
                68,
                71,
                72,
                73,
                75,
                81,
        ]):
            points = [
                # Back Plane
                Location(x=-5.2, y=-0.2, z=5.5),
                Location(x=-4.8, y=-0.2, z=5.5),
                Location(x=-4.8, y=-0.2, z=6.5),
                Location(x=-5.2, y=-0.2, z=6.5),
                # Front Plane
                Location(x=-5.2, y=0.4, z=5.5),
                Location(x=-4.8, y=0.4, z=5.5),
                Location(x=-4.8, y=0.4, z=6.5),
                Location(x=-5.2, y=0.4, z=6.5),
            ]
            bboxes.append(self._relative_to_traffic_light(points))
            right_points = [
                point + Location(x=-3.0) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(right_points))
            if self.id not in [51, 52, 53]:
                left_points = [
                    point + Location(x=-6.5) for point in points
                ]
                bboxes.append(self._relative_to_traffic_light(left_points))
        else:
            points = [
                # Back Plane
                Location(x=-0.5, y=-0.1, z=2),
                Location(x=+0.1, y=-0.1, z=2),
                Location(x=+0.1, y=-0.1, z=3),
                Location(x=-0.5, y=-0.1, z=3),
                # Front Plane
                Location(x=-0.5, y=0.5, z=2),
                Location(x=+0.1, y=0.5, z=2),
                Location(x=+0.1, y=0.5, z=3),
                Location(x=-0.5, y=0.5, z=3),
            ]
            bboxes.append(self._relative_to_traffic_light(points))

        return bboxes

    def _get_bboxes_for_town4(self):
        bboxes = []
        points = [
            # Back Plane
            Location(x=-5.2, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=6.5),
            Location(x=-5.2, y=-0.2, z=6.5),
            # Front Plane
            Location(x=-5.2, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=6.5),
            Location(x=-5.2, y=0.4, z=6.5),
        ]
        bboxes.append(self._relative_to_traffic_light(points))
        if self.trigger_volume_extent.x > 5:
            # This is a traffic light with 4 signs, we need to come up with
            # more bounding boxes.
            middle_points = [  # Light in the middle of the pole.
                # Back Plane
                Location(x=-0.5, y=-0.1, z=2.5),
                Location(x=+0.1, y=-0.1, z=2.5),
                Location(x=+0.1, y=-0.1, z=3.5),
                Location(x=-0.5, y=-0.1, z=3.5),
                # Front Plane
                Location(x=-0.5, y=0.5, z=2.5),
                Location(x=+0.1, y=0.5, z=2.5),
                Location(x=+0.1, y=0.5, z=3.5),
                Location(x=-0.5, y=0.5, z=3.5),
            ]
            right_points = [
                point + Location(x=-3.0) for point in points
            ]
            left_points = [
                point + Location(x=-5.5) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(middle_points))
            bboxes.append(self._relative_to_traffic_light(right_points))
            bboxes.append(self._relative_to_traffic_light(left_points))
        return bboxes

    def _get_bboxes_for_town5(self):
        bboxes = []
        points = [
            # Back Plane
            Location(x=-5.2, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=6.5),
            Location(x=-5.2, y=-0.2, z=6.5),
            # Front Plane
            Location(x=-5.2, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=6.5),
            Location(x=-5.2, y=0.4, z=6.5),
        ]
        # Town05 randomizes the identifiers for the traffic light at each
        # reload of the world. We cannot depend on static identifiers for
        # figuring out which lights only have a single traffic light.
        bboxes.append(self._relative_to_traffic_light(points))
        # There's a traffic light with extent.x < 2, which only has one box.
        if self.trigger_volume_extent.x >= 2:
            # This is a traffids light with 4 signs, we need to come up
            # with more bounding boxes.
            middle_points = [  # Light in the middle of the pole.
                # Back Plane
                Location(x=-0.4, y=-0.1, z=2.55),
                Location(x=+0.2, y=-0.1, z=2.55),
                Location(x=+0.2, y=-0.1, z=3.55),
                Location(x=-0.4, y=-0.1, z=3.55),
                # Front Plane
                Location(x=-0.4, y=0.5, z=2.55),
                Location(x=+0.2, y=0.5, z=2.55),
                Location(x=+0.2, y=0.5, z=3.55),
                Location(x=-0.5, y=0.5, z=3.55),
            ]
            right_points = [
                point + Location(x=-3.0) for point in points
            ]
            left_points = [
                point + Location(x=-5.5) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(middle_points))
            bboxes.append(self._relative_to_traffic_light(right_points))
            bboxes.append(self._relative_to_traffic_light(left_points))
        return bboxes
        
class TrafficLightDetector(object):
    """Detects traffic lights using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        traffic_lights_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_setup_width, camera_setup_height, \
                 model_path="./models/traffic_light_detection/faster-rcnn/", gpu_idx=0):

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

        # Load the model from the saved_model format file.
        self.traffic_light_det_model_path = model_path

        self._model = tf.saved_model.load(
            self.traffic_light_det_model_path)

        self.traffic_light_det_min_score_threshold = 0.3

        self._labels = {
            1: TrafficLightColor.GREEN,
            2: TrafficLightColor.YELLOW,
            3: TrafficLightColor.RED,
            4: TrafficLightColor.OFF
        }
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0
        # Serve some junk image to load up the model.
        self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))

    def detect(self, image_np):
        boxes, scores, labels = self.__run_model(image_np)

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, self.camera_setup_height,
            self.camera_setup_width)
        
        return traffic_lights


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
        res_labels = [
            self._labels[int(label)] for label in classes[0][:num_detections]
        ]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return res_boxes, res_scores, res_labels

    def __convert_to_detected_tl(self, boxes, scores, labels, height, width):
        traffic_lights = []
        for index in range(len(scores)):
            if scores[
                    index] > self.traffic_light_det_min_score_threshold:
                bbox = BoundingBox2D(
                    int(boxes[index][1] * width),  # x_min
                    int(boxes[index][3] * width),  # x_max
                    int(boxes[index][0] * height),  # y_min
                    int(boxes[index][2] * height)  # y_max
                )
                traffic_lights.append(
                    TrafficLight(scores[index],
                                 labels[index],
                                 id=self._unique_id,
                                 bounding_box=bbox))
                self._unique_id += 1
        return traffic_lights

