from collections import deque
from functools import partial

import numpy as np

from .gnss_utils import *

from utils.pylot_utils import Location, Pose, Quaternion, Rotation, Transform, \
        Vector3D

class LocalizationOperator(object):
    def __init__(self, var_imu_f=0.001, var_imu_w=0.001, var_gnss=0.000005):
        # Gravity vector.
        self._g = np.array([0, 0, -9.81])

        # Previous timestamp values.
        self._last_pos_estimate = None
        self._last_timestamp = None

        # NOTE: At the start of the simulation, the vehicle drops down from
        # the sky, during which the IMU values screw up the calculations.
        # This boolean flag takes care to start the prediction only when the
        # values have stabilized.

        # Constants required for the Kalman filtering.
        self.var_imu_f, self.var_imu_w, self.var_gnss = var_imu_f, var_imu_w, var_gnss
        self.__Q = np.identity(6)
        self.__Q[0:3, 0:3] = self.__Q[0:3, 0:3] * var_imu_f
        self.__Q[3:6, 3:6] = self.__Q[3:6, 3:6] * var_imu_w

        self.__F = np.identity(9)

        self.__L = np.zeros([9, 6])
        self.__L[3:9, :] = np.identity(6)

        self.__R_GNSS = np.identity(3) * var_gnss

        self._last_covariance = np.zeros((9, 9))

    def __skew_symmetric(self, v):
        """Skew symmetric form of a 3x1 vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                        dtype=np.float64)
                        
    def __update_using_gnss(self, location_estimate, velocity_estimate,
                            rotation_estimate, gnss_reading, delta_t):
        # Construct H_k = [I, 0, 0] (shape=(3, 9))
        H_k = np.zeros((3, 9))
        H_k[:, :3] = np.identity(3)

        # Propogate uncertainty.
        Q = self.__Q * delta_t * delta_t
        self._last_covariance = (self.__F.dot(self._last_covariance).dot(
            self.__F.T)) + (self.__L.dot(Q).dot(self.__L.T))

        # Compute Kalman gain. (shape=(9, 3))
        K_k = self._last_covariance.dot(
            H_k.T.dot(
                np.linalg.inv(
                    H_k.dot(self._last_covariance.dot(H_k.T)) +
                    self.__R_GNSS)))

        # Compute error state. (9x3) x ((3x1) - (3x1)) = shape(9, 1)
        delta_x_k = K_k.dot(gnss_reading - location_estimate)

        # Correct predicted state.
        corrected_location_estimate = location_estimate + delta_x_k[0:3]
        corrected_velocity_estimate = velocity_estimate + delta_x_k[3:6]
        roll, pitch, yaw = delta_x_k[6:]
        corrected_rotation_estimate = Quaternion.from_rotation(
            Rotation(roll=roll, pitch=pitch, yaw=yaw)) * rotation_estimate

        # Fix the covariance.
        self._last_covariance = (np.identity(9) - K_k.dot(H_k)).dot(
            self._last_covariance)

        return (
            corrected_location_estimate,
            corrected_velocity_estimate,
            corrected_rotation_estimate,
        )

    def get_estimation(self):
        return self._last_pos_estimate
        
    def get_current_pos(self):
        return np.array([self._last_pos_estimate.transform.location.x, self._last_pos_estimate.transform.location.y,
                         self._last_pos_estimate.transform.location.z])
        
    def get_current_compass(self):
        return self._last_pos_estimate.transform.rotation.yaw * math.pi / 180

    def run(self, gnss_data, imu_data, timestamp, geo_ref_dict=None):
        timestamp *= 1000

        if self._last_pos_estimate is None:
            location_estimate = gnss_to_xy(gnss_data, geo_ref_dict=geo_ref_dict)
            compass_data = imu_data[-1]   # refine to Graph coord
            rotation_estimate = Rotation(yaw=compass_data)
            velocity_estimate = np.zeros(3)

            self._last_timestamp = timestamp
            self._last_pos_estimate = Pose(
                transform=Transform(location=Location(*location_estimate),
                                    rotation=rotation_estimate),
                forward_speed=Vector3D(*velocity_estimate).magnitude(),
                velocity_vector=Vector3D(*velocity_estimate),
                localization_time=timestamp,
            )
            return self.get_current_pos(), self.get_current_compass()
        elif abs(imu_data[1]) > 100:
            return self.get_current_pos(), self.get_current_compass()
        
        # Initialize the delta_t
        current_ts = timestamp
        delta_t = (current_ts - self._last_timestamp) / 1000.0
        
        imu_msg_acceleration = Vector3D(*imu_data[0:3])
        imu_msg_gyro = Vector3D(*imu_data[3:6])

        # Estimate the rotation.
        last_rotation_estimate = Quaternion.from_rotation(
            self._last_pos_estimate.transform.rotation)
        rotation_estimate = (
            last_rotation_estimate *
            Quaternion.from_angular_velocity(imu_msg_gyro, delta_t))

        # Transform the IMU accelerometer data from the body frame to the
        # world frame, and retrieve location and velocity estimates.
        accelerometer_data = last_rotation_estimate.matrix.dot(
            imu_msg_acceleration.as_numpy_array()) + self._g
        last_location_estimate = \
            self._last_pos_estimate.transform.location.as_numpy_array()
        last_velocity_estimate = \
            self._last_pos_estimate.velocity_vector.as_numpy_array()

        # Estimate the location.
        location_estimate = last_location_estimate + (
            delta_t * last_velocity_estimate) + ((
                (delta_t**2) / 2.0) * accelerometer_data)

        # Estimate the velocity.
        velocity_estimate = last_velocity_estimate + (delta_t *
                                                      accelerometer_data)

        # Fuse the GNSS values using an EKF to fix drifts and noise in
        # the estimates.

        # Linearize the motion model and compute Jacobians.
        self.__F[0:3, 3:6] = np.identity(3) * delta_t
        self.__F[3:6, 6:9] = last_rotation_estimate.matrix.dot(
            -self.__skew_symmetric(accelerometer_data.reshape(
                (3, 1)))) * delta_t

        # Fix estimates using GNSS
        gnss_reading = gnss_to_xy(gnss_data, geo_ref_dict=geo_ref_dict)

        (
            location_estimate,
            velocity_estimate,
            rotation_estimate,
        ) = self.__update_using_gnss(location_estimate, velocity_estimate,
                                     rotation_estimate, gnss_reading,
                                     delta_t)

        # Create the PoseMessage and send it downstream.
        current_pose = Pose(
            transform=Transform(location=Location(*location_estimate),
                                rotation=rotation_estimate.as_rotation()),
            forward_speed=Vector3D(*velocity_estimate).magnitude(),
            velocity_vector=Vector3D(*velocity_estimate),
            localization_time=current_ts,
        )

        # Set the estimates for the next iteration.
        self._last_timestamp = current_ts
        self._last_pos_estimate = current_pose
        
        return self.get_current_pos(), self.get_current_compass()

def get_localization(data, method='filter'):
    root = ET.fromstring(data['HD-map'])
    geo_ref_dict = get_georeference(root)

    if method == 'filter':
        localization = LocalizationOperator(0.001, 0.001, 0.000005)

        pos_list = []
        for i in range(len(data['pos'])):
            gnss_data = data['GPS'][i]
            imu_data = data['IMU'][i]
            timestamp = data['timestamp'][i]
            vehicle_pos, _ = localization.run(gnss_data, imu_data, timestamp, geo_ref_dict)
            pos_list.append(vehicle_pos)
        pos_list = np.array(pos_list)
    elif method == 'window':
        window = deque(maxlen=5)
        pos_list = []
        for i in range(len(data['pos'])):
            gnss_data = data['GPS'][i]
            pos = gnss_to_xy(gnss_data, geo_ref_dict)[:2]
            window.append(pos)
            pos_list.append(np.mean(window, axis=0))
        pos_list = np.array(pos_list)
    elif method == 'raw':
        pos_list = []
        for i in range(len(data['pos'])):
            gnss_data = data['GPS'][i]
            pos = gnss_to_xy(gnss_data, geo_ref_dict)[:2]
            pos_list.append(pos)
        pos_list = np.array(pos_list)
    elif method == 'pos':
        pos_list = data['pos']
    elif method == 'pos_gt':
        pos_list = data['pos_gt']

    return pos_list