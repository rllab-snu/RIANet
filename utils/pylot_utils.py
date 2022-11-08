import math
import time
from enum import Enum

import numpy as np


OBSTACLE_LABELS = {
    'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle', 'person',
    'stop sign', 'parking meter', 'cat', 'dog', 'speed limit 30',
    'speed limit 60', 'speed limit 90'
}

PYLOT_BBOX_COLOR_MAP = {
    'person': [0, 128, 0],
    'vehicle': [128, 0, 0],
    'car': [128, 0, 0],
    'bicycle': [128, 0, 0],
    'motorcycle': [128, 0, 0],
    'bus': [128, 0, 0],
    'truck': [128, 0, 0],
    'stop marking': [128, 128, 0],
    'speed limit': [255, 255, 0],
    'red traffic light': [0, 0, 255],
    'yellow traffic light': [0, 255, 255],
    'green traffic light': [0, 255, 0],
    'off traffic light': [0, 0, 0],
    '': [255, 255, 255],
}

coco_bbox_color_list = np.array([
    1.000, 1.000, 1.000, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
    0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
    0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
    0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
    0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
    0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
    0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
    0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50, 0.5, 0
]).astype(np.float32)


class Rotation(object):
    """Used to represent the rotation of an actor or obstacle.

    Rotations are applied in the order: Roll (X), Pitch (Y), Yaw (Z).
    A 90-degree "Roll" maps the positive Z-axis to the positive Y-axis.
    A 90-degree "Pitch" maps the positive X-axis to the positive Z-axis.
    A 90-degree "Yaw" maps the positive X-axis to the positive Y-axis.

    Args:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.

    Attributes:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.
    """
    def __init__(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    @classmethod
    def from_simulator_rotation(cls, rotation):
        """Creates a pylot Rotation from a simulator rotation.

        Args:
            rotation: An instance of a simulator rotation.

        Returns:
            :py:class:`.Rotation`: A pylot rotation.
        """
        from carla import Rotation
        if not isinstance(rotation, Rotation):
            raise ValueError('rotation should be of type Rotation')
        return cls(rotation.pitch, rotation.yaw, rotation.roll)

    def as_simulator_rotation(self):
        """ Retrieves the rotation as an instance of a simulator rotation.

        Returns:
            An instance of a simulator class representing the rotation.
        """
        from carla import Rotation
        return Rotation(self.pitch, self.yaw, self.roll)

    def as_numpy_array(self):
        """Retrieves the Rotation as a numpy array."""
        return np.array([self.pitch, self.yaw, self.roll])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Rotation(pitch={}, yaw={}, roll={})'.format(
            self.pitch, self.yaw, self.roll)


class Quaternion(object):
    """ Represents the Rotation of an obstacle or vehicle in quaternion
    notation.

    Args:
        w: The real-part of the quaternion.
        x: The x-part (i) of the quaternion.
        y: The y-part (j) of the quaternion.
        z: The z-part (k) of the quaternion.

    Attributes:
        w: The real-part of the quaternion.
        x: The x-part (i) of the quaternion.
        y: The y-part (j) of the quaternion.
        z: The z-part (k) of the quaternion.
        matrix: A 3x3 numpy array that can be used to rotate 3D vectors from
            body frame to world frame.
    """
    def __init__(self, w: float, x: float, y: float, z: float):
        norm = np.linalg.norm([w, x, y, z])
        if norm < 1e-50:
            self.w, self.x, self.y, self.z = 0, 0, 0, 0
        else:
            self.w = w / norm
            self.x = x / norm
            self.y = y / norm
            self.z = z / norm
        self.matrix = Quaternion._create_matrix(self.w, self.x, self.y, self.z)

    @staticmethod
    def _create_matrix(w, x, y, z):
        """Creates a Rotation matrix that can be used to transform 3D vectors
        from body frame to world frame.

        Note that this yields the same matrix as a Transform object with the
        quaternion converted to the Euler rotation except this matrix only does
        rotation and no translation.

        Specifically, this matrix is equivalent to:
            Transform(location=Location(0, 0, 0),
                      rotation=self.as_rotation()).matrix[:3, :3]

        Returns:
            A 3x3 numpy array that can be used to rotate 3D vectors from body
            frame to world frame.
        """
        x2, y2, z2 = x * 2, y * 2, z * 2
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        m = np.array([[1.0 - (yy + zz), xy - wz, xz + wy],
                      [xy + wz, 1.0 - (xx + zz), yz - wx],
                      [xz - wy, yz + wx, 1.0 - (xx + yy)]])
        return m

    @classmethod
    def from_rotation(cls, rotation: Rotation):
        """Creates a Quaternion from a rotation including pitch, roll, yaw.

        Args:
            rotation (:py:class:`.Rotation`): A pylot rotation representing
                the rotation of the object in degrees.

        Returns:
            :py:class:`.Quaternion`: The quaternion representation of the
            rotation.
        """
        roll_by_2 = np.radians(rotation.roll) / 2.0
        pitch_by_2 = np.radians(rotation.pitch) / 2.0
        yaw_by_2 = np.radians(rotation.yaw) / 2.0

        cr, sr = np.cos(roll_by_2), np.sin(roll_by_2)
        cp, sp = np.cos(pitch_by_2), np.sin(pitch_by_2)
        cy, sy = np.cos(yaw_by_2), np.sin(yaw_by_2)

        w = cr * cp * cy + sr * sp * sy
        x = cr * sp * sy - sr * cp * cy
        y = -cr * sp * cy - sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def from_angular_velocity(cls, angular_velocity, dt: float):
        """Creates a Quaternion from an angular velocity vector and the time
        delta to apply it for.

        Args:
            angular_velocity (:py:class:`.Vector3D`): The vector representing
                the angular velocity of the object in the body-frame.
            dt (float): The time delta to apply the angular velocity for.

        Returns:
            :py:class:`.Quaternion`: The quaternion representing the rotation
                undergone by the object with the given angular velocity in the
                given delta time.
        """
        angular_velocity_np = angular_velocity.as_numpy_array() * dt
        magnitude = np.linalg.norm(angular_velocity_np)

        w = np.cos(magnitude / 2.0)
        if magnitude < 1e-50:
            # To avoid instabilities and nan.
            x, y, z = 0, 0, 0
        else:
            imaginary = angular_velocity_np / magnitude * np.sin(
                magnitude / 2.0)
            x, y, z = imaginary
        return cls(w, x, y, z)

    def as_rotation(self) -> Rotation:
        """Retrieve the Quaternion as a Rotation in degrees.

        Returns:
            :py:class:`.Rotation`: The euler-angle equivalent of the
                Quaternion in degrees.
        """
        SINGULARITY_THRESHOLD = 0.4999995
        RAD_TO_DEG = (180.0) / np.pi

        singularity_test = self.z * self.x - self.w * self.y
        yaw_y = 2.0 * (self.w * self.z + self.x * self.y)
        yaw_x = (1.0 - 2.0 * (self.y**2 + self.z**2))

        pitch, yaw, roll = None, None, None
        if singularity_test < -SINGULARITY_THRESHOLD:
            pitch = -90.0
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = -yaw - (2.0 * np.arctan2(self.x, self.w) * RAD_TO_DEG)
        elif singularity_test > SINGULARITY_THRESHOLD:
            pitch = 90.0
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = yaw - (2.0 * np.arctan2(self.x, self.w) * RAD_TO_DEG)
        else:
            pitch = np.arcsin(2.0 * singularity_test) * RAD_TO_DEG
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = np.arctan2(-2.0 * (self.w * self.x + self.y * self.z),
                              (1.0 - 2.0 *
                               (self.x**2 + self.y**2))) * RAD_TO_DEG
        return Rotation(pitch, yaw, roll)

    def __mul__(self, other):
        """Returns the product self * other.  The product is NOT commutative.

        The product is defined in Unreal as:
         [ (Q2.w * Q1.x) + (Q2.x * Q1.w) + (Q2.y * Q1.z) - (Q2.z * Q1.y),
           (Q2.w * Q1.y) - (Q2.x * Q1.z) + (Q2.y * Q1.w) + (Q2.z * Q1.x),
           (Q2.w * Q1.z) + (Q2.x * Q1.y) - (Q2.y * Q1.x) + (Q2.z * Q1.w),
           (Q2.w * Q1.w) - (Q2.x * Q1.x) - (Q2.y * Q1.y) - (Q2.z * Q1.z) ]
        Copied from DirectX's XMQuaternionMultiply function.
        """
        q1, q2 = other, self
        x = (q2.w * q1.x) + (q2.x * q1.w) + (q2.y * q1.z) - (q2.z * q1.y)
        y = (q2.w * q1.y) - (q2.x * q1.z) + (q2.y * q1.w) + (q2.z * q1.x)
        z = (q2.w * q1.z) + (q2.x * q1.y) - (q2.y * q1.x) + (q2.z * q1.w)
        w = (q2.w * q1.w) - (q2.x * q1.x) - (q2.y * q1.y) - (q2.z * q1.z)
        return Quaternion(w, x, y, z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Quaternion (w={}, x={}, y={}, z={})'.format(
            self.w, self.x, self.y, self.z)


class Vector3D(object):
    """Represents a 3D vector and provides useful helper functions.

    Args:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.

    Attributes:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def from_simulator_vector(cls, vector):
        """Creates a pylot Vector3D from a simulator 3D vector.

        Args:
            vector: An instance of a simulator 3D vector.

        Returns:
            :py:class:`.Vector3D`: A pylot 3D vector.
        """
        from carla import Vector3D
        if not isinstance(vector, Vector3D):
            raise ValueError('The vector must be a Vector3D')
        return cls(vector.x, vector.y, vector.z)

    def as_numpy_array(self):
        """Retrieves the 3D vector as a numpy array."""
        return np.array([self.x, self.y, self.z])

    def as_numpy_array_2D(self):
        """Drops the 3rd dimension."""
        return np.array([self.x, self.y])

    def as_simulator_vector(self):
        """Retrieves the 3D vector as an instance of simulator 3D vector.

        Returns:
            An instance of the simulator class representing the 3D vector.
        """
        from carla import Vector3D
        return Vector3D(self.x, self.y, self.z)

    def l1_distance(self, other):
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector3D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z -
                                                                   other.z)

    def l2_distance(self, other) -> float:
        """Calculates the L2 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector3D`): The other vector used to
                calculate the L2 distance to.

        Returns:
            :obj:`float`: The L2 distance between the two points.
        """
        vec = np.array([self.x - other.x, self.y - other.y, self.z - other.z])
        return np.linalg.norm(vec)

    def magnitude(self):
        """Returns the magnitude of the 3D vector."""
        return np.linalg.norm(self.as_numpy_array())

    def to_camera_view(self, extrinsic_matrix, intrinsic_matrix):
        """Converts the given 3D vector to the view of the camera using
        the extrinsic and the intrinsic matrix.

        Args:
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            :py:class:`.Vector3D`: An instance with the coordinates converted
            to the camera view.
        """
        position_vector = np.array([[self.x], [self.y], [self.z], [1.0]])

        # Transform the points to the camera in 3D.
        transformed_3D_pos = np.dot(np.linalg.inv(extrinsic_matrix),
                                    position_vector)

        # Transform the points to 2D.
        position_2D = np.dot(intrinsic_matrix, transformed_3D_pos[:3])

        # Normalize the 2D points.
        location_2D = type(self)(float(position_2D[0] / position_2D[2]),
                                 float(position_2D[1] / position_2D[2]),
                                 float(position_2D[2]))
        return location_2D

    def rotate(self, angle: float):
        """Rotate the vector by a given angle.

        Args:
            angle (float): The angle to rotate the Vector by (in degrees).

        Returns:
            :py:class:`.Vector3D`: An instance with the coordinates of the
            rotated vector.
        """
        x_ = math.cos(math.radians(angle)) * self.x - math.sin(
            math.radians(angle)) * self.y
        y_ = math.sin(math.radians(angle)) * self.x - math.cos(
            math.radians(angle)) * self.y
        return type(self)(x_, y_, self.z)

    def __add__(self, other):
        """Adds the two vectors together and returns the result."""
        return type(self)(x=self.x + other.x,
                          y=self.y + other.y,
                          z=self.z + other.z)

    def __sub__(self, other):
        """Subtracts the other vector from self and returns the result."""
        return type(self)(x=self.x - other.x,
                          y=self.y - other.y,
                          z=self.z - other.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector3D(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Vector2D(object):
    """Represents a 2D vector and provides helper functions."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_numpy_array(self):
        """Retrieves the 2D vector as a numpy array."""
        return np.array([self.x, self.y])

    def get_angle(self, other) -> float:
        """Computes the angle between the vector and another vector
           in radians."""
        angle = math.atan2(self.y, self.x) - math.atan2(other.y, other.x)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def l1_distance(self, other) -> float:
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def l2_distance(self, other) -> float:
        """Calculates the L2 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L2 distance to.

        Returns:
            :obj:`float`: The L2 distance between the two points.
        """
        vec = np.array([self.x - other.x, self.y - other.y])
        return np.linalg.norm(vec)

    def magnitude(self):
        """Returns the magnitude of the 2D vector."""
        return np.linalg.norm(self.as_numpy_array())

    def __add__(self, other):
        """Adds the two vectors together and returns the result. """
        return type(self)(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        """Subtracts the other vector from self and returns the result. """
        return type(self)(x=self.x - other.x, y=self.y - other.y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector2D(x={}, y={})'.format(self.x, self.y)


class Location(Vector3D):
    """Stores a 3D location, and provides useful helper methods.

    Args:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.

    Attributes:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super(Location, self).__init__(x, y, z)

    @classmethod
    def from_simulator_location(cls, location):
        """Creates a pylot Location from a simulator location.

        Args:
            location: An instance of a simulator location.

        Returns:
            :py:class:`.Location`: A pylot location.
        """
        from carla import Location, Vector3D
        if not (isinstance(location, Location)
                or isinstance(location, Vector3D)):
            raise ValueError('The location must be a Location or Vector3D')
        return cls(location.x, location.y, location.z)

    @classmethod
    def from_gps(cls, latitude: float, longitude: float, altitude: float):
        """Creates Location from GPS (latitude, longitude, altitude).

        This is the inverse of the _location_to_gps method found in
        https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
        """
        EARTH_RADIUS_EQUA = 6378137.0
        # The following reference values are applicable for towns 1 through 7,
        # and are taken from the corresponding OpenDrive map files.
        # LAT_REF = 49.0
        # LON_REF = 8.0
        # TODO: Do not hardcode. Get the references from the open drive file.
        LAT_REF = 0.0
        LON_REF = 0.0

        scale = math.cos(LAT_REF * math.pi / 180.0)
        basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
        basey = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + LAT_REF) * math.pi / 360.0))

        x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
        y = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

        # This wasn't in the original method, but seems to be necessary.
        #y *= -1

        return cls(x, y, altitude)

    def distance(self, other) -> float:
        """Calculates the Euclidean distance between the given point and the
        other point.

        Args:
            other (:py:class:`~.Location`): The other location used to
                calculate the Euclidean distance to.

        Returns:
            :obj:`float`: The Euclidean distance between the two points.
        """
        return (self - other).magnitude()

    def as_vector_2D(self) -> Vector2D:
        """Transforms the Location into a Vector2D.

        Note:
            The method just drops the z-axis.

        Returns:
            :py:class:`.Vector2D`: A 2D vector.
        """
        return Vector2D(self.x, self.y)

    def as_simulator_location(self):
        """Retrieves the location as a simulator location instance.

        Returns:
            An instance of the simulator class representing the location.
        """
        from carla import Location
        return Location(self.x, self.y, self.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Transform(object):
    """A class that stores the location and rotation of an obstacle.

    It can be created from a simulator transform, defines helper functions
    needed in Pylot, and makes the simulator transform serializable.

    A transform object is instantiated with either a location and a rotation,
    or using a matrix.

    Args:
        location (:py:class:`.Location`, optional): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`, optional): The rotation  (in degrees)
            of the object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.

    Attributes:
        location (:py:class:`.Location`): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`): The rotation (in degrees) of the
            object represented by the transform.
        forward_vector (:py:class:`.Vector3D`): The forward vector of the
            object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.
    """
    def __init__(self,
                 location: Location = None,
                 rotation: Rotation = None,
                 matrix=None):
        if matrix is not None:
            self.matrix = matrix
            self.location = Location(matrix[0, 3], matrix[1, 3], matrix[2, 3])

            # Forward vector is retrieved from the matrix.
            self.forward_vector = \
                Vector3D(self.matrix[0, 0], self.matrix[1, 0],
                         self.matrix[2, 0])
            pitch_r = math.asin(np.clip(self.forward_vector.z, -1, 1))
            yaw_r = math.acos(
                np.clip(self.forward_vector.x / math.cos(pitch_r), -1, 1))
            roll_r = math.asin(
                np.clip(matrix[2, 1] / (-1 * math.cos(pitch_r)), -1, 1))
            self.rotation = Rotation(math.degrees(pitch_r),
                                     math.degrees(yaw_r), math.degrees(roll_r))
        else:
            self.location, self.rotation = location, rotation
            self.matrix = Transform._create_matrix(self.location,
                                                   self.rotation)

            # Forward vector is retrieved from the matrix.
            self.forward_vector = \
                Vector3D(self.matrix[0, 0], self.matrix[1, 0],
                         self.matrix[2, 0])

    @classmethod
    def from_simulator_transform(cls, transform):
        """Creates a pylot transform from a simulator transform.

        Args:
            transform: A simulator transform.

        Returns:
            :py:class:`.Transform`: An instance of a pylot transform.
        """
        from carla import Transform
        if not isinstance(transform, Transform):
            raise ValueError('transform should be of type Transform')
        return cls(Location.from_simulator_location(transform.location),
                   Rotation.from_simulator_rotation(transform.rotation))

    @staticmethod
    def _create_matrix(location, rotation):
        """Creates a transformation matrix to convert points in the 3D world
        coordinate space with respect to the object.

        Use the transform_points function to transpose a given set of points
        with respect to the object.

        Args:
            location (:py:class:`.Location`): The location of the object
                represented by the transform.
            rotation (:py:class:`.Rotation`): The rotation of the object
                represented by the transform.

        Returns:
            A 4x4 numpy matrix which represents the transformation matrix.
        """
        matrix = np.identity(4)
        cy = math.cos(np.radians(rotation.yaw))
        sy = math.sin(np.radians(rotation.yaw))
        cr = math.cos(np.radians(rotation.roll))
        sr = math.sin(np.radians(rotation.roll))
        cp = math.cos(np.radians(rotation.pitch))
        sp = math.sin(np.radians(rotation.pitch))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = (cp * cy)
        matrix[0, 1] = (cy * sp * sr - sy * cr)
        matrix[0, 2] = -1 * (cy * sp * cr + sy * sr)
        matrix[1, 0] = (sy * cp)
        matrix[1, 1] = (sy * sp * sr + cy * cr)
        matrix[1, 2] = (cy * sr - sy * sp * cr)
        matrix[2, 0] = (sp)
        matrix[2, 1] = -1 * (cp * sr)
        matrix[2, 2] = (cp * cr)
        return matrix

    def __transform(self, points, matrix):
        """Internal function to transform the points according to the
        given matrix. This function either converts the points from
        coordinate space relative to the transform to the world coordinate
        space (using self.matrix), or from world coordinate space to the
        space relative to the transform (using inv(self.matrix))

        Args:
            points: An n by 3 numpy array, where each row is the
                (x, y, z) coordinates of a point.
            matrix: The matrix of the transformation to apply.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        # Needed format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]].
        # So let's transpose the point matrix.
        points = points.transpose()

        # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)

        # Point transformation (depends on the given matrix)
        points = np.dot(matrix, points)

        # Get all but the last row in array form.
        points = np.asarray(points[0:3].transpose()).astype(np.float16)

        return points

    def transform_points(self, points):
        """Transforms the given set of points (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        For example, if the transform is at location (3, 0, 0) and the
        location passed to the argument is (10, 0, 0), this function will
        return (13, 0, 0) i.e. the location of the argument in the world
        coordinate space.

        Args:
            points: A (number of points) by 3 numpy array, where each row is
                the (x, y, z) coordinates of a point.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        return self.__transform(points, self.matrix)

    def inverse_transform_points(self, points):
        """Transforms the given set of points (specified in world coordinate
        space) to be relative to the given transform.

        For example, if the transform is at location (3, 0, 0) and the location
        passed to the argument is (10, 0, 0), this function will return
        (7, 0, 0) i.e. the location of the argument relative to the given
        transform.

        Args:
            points: A (number of points) by 3 numpy array, where each row is
                the (x, y, z) coordinates of a point.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        return self.__transform(points, np.linalg.inv(self.matrix))

    def transform_locations(self, locations):
        """Transforms the given set of locations (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        This method has the same functionality as transform_points, and
        is provided for convenience; when dealing with a large number of
        points, it is advised to use transform_points to avoid the slow
        conversion between a numpy array and list of locations.

        Args:
            points (list(:py:class:`.Location`)): List of locations.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """
        points = np.array([loc.as_numpy_array() for loc in locations])
        transformed_points = self.__transform(points, self.matrix)
        return [Location(x, y, z) for x, y, z in transformed_points]

    def inverse_transform_locations(self, locations):
        """Transforms the given set of locations (specified in world coordinate
        space) to be relative to the given transform.

        This method has the same functionality as inverse_transform_points,
        and is provided for convenience; when dealing with a large number of
        points, it is advised to use inverse_transform_points to avoid the slow
        conversion between a numpy array and list of locations.

        Args:
            points (list(:py:class:`.Location`)): List of locations.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """

        points = np.array([loc.as_numpy_array() for loc in locations])
        transformed_points = self.__transform(points,
                                              np.linalg.inv(self.matrix))
        return [Location(x, y, z) for x, y, z in transformed_points]

    def as_simulator_transform(self):
        """Converts the transform to a simulator transform.

        Returns:
            An instance of the simulator class representing the Transform.
        """
        from carla import Location, Rotation, Transform
        return Transform(
            Location(self.location.x, self.location.y, self.location.z),
            Rotation(pitch=self.rotation.pitch,
                     yaw=self.rotation.yaw,
                     roll=self.rotation.roll))

    def get_angle_and_magnitude(self, target_loc):
        """Computes relative angle between the transform and a target location.

        Args:
            target_loc (:py:class:`.Location`): Location of the target.

        Returns:
            Angle in radians and vector magnitude.
        """
        target_vec = target_loc.as_vector_2D() - self.location.as_vector_2D()
        magnitude = target_vec.magnitude()
        if magnitude > 0:
            forward_vector = Vector2D(
                math.cos(math.radians(self.rotation.yaw)),
                math.sin(math.radians(self.rotation.yaw)))
            angle = target_vec.get_angle(forward_vector)
        else:
            angle = 0
        return angle, magnitude

    def is_within_distance_ahead(self, dst_loc: Location,
                                 max_distance: float) -> bool:
        """Checks if a location is within a distance.

        Args:
            dst_loc (:py:class:`.Location`): Location to compute distance to.
            max_distance (:obj:`float`): Maximum allowed distance.

        Returns:
            bool: True if other location is within max_distance.
        """
        d_angle, norm_dst = self.get_angle_and_magnitude(dst_loc)
        # Return if the vector is too small.
        if norm_dst < 0.001:
            return True
        # Return if the vector is greater than the distance.
        if norm_dst > max_distance:
            return False
        return d_angle < 90.0

    def inverse_transform(self):
        """Returns the inverse transform of this transform."""
        new_matrix = np.linalg.inv(self.matrix)
        return Transform(matrix=new_matrix)

    def __mul__(self, other):
        new_matrix = np.dot(self.matrix, other.matrix)
        return Transform(matrix=new_matrix)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.location:
            return "Transform(location: {}, rotation: {})".format(
                self.location, self.rotation)
        else:
            return "Transform({})".format(str(self.matrix))


class Pose(object):
    """Class used to wrap ego-vehicle information.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the ego
            vehicle.
        forward_speed (:obj:`int`): Forward speed in m/s.
        velocity_vector (:py:class:`~pylot.utils.Vector3D`): Velocity vector
            in world frame

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the ego
            vehicle.
        forward_speed (:obj:`int`): Forward speed in m/s.
        velocity_vector (:py:class:`~pylot.utils.Vector3D`): Velocity vector
            in world frame
    """
    def __init__(self,
                 transform: Transform,
                 forward_speed: float,
                 velocity_vector: Vector3D = None,
                 localization_time: float = None):
        if not isinstance(transform, Transform):
            raise ValueError(
                'transform should be of type pylot.utils.Transform')
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed
        self.velocity_vector = velocity_vector
        if localization_time is None:
            self.localization_time = time.time()
        else:
            self.localization_time = localization_time

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Pose(transform: {}, forward speed: {}, velocity vector: {})"\
            .format(self.transform, self.forward_speed, self.velocity_vector)


class LaneMarkingColor(Enum):
    """Enum that defines the lane marking colors according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane colors are correctly
    propogated from the simulator to Pylot.
    """
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4
    OTHER = 5



class BoundingBox2D(object):
    """Class that stores a 2D bounding box."""
    def __init__(self, x_min, x_max, y_min, y_max):
        assert x_min < x_max and y_min < y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_min_point(self) -> Vector2D:
        return Vector2D(self.x_min, self.y_min)

    def get_max_point(self) -> Vector2D:
        return Vector2D(self.x_max, self.y_max)

    def get_height(self):
        return self.y_max - self.y_min

    def get_width(self):
        return self.x_max - self.x_min

    def get_center_point(self) -> Vector2D:
        return Vector2D((self.x_min + self.x_max) // 2,
                                    (self.y_min + self.y_max) // 2)

    def as_width_height_bbox(self):
        return [self.x_min, self.y_min, self.get_width(), self.get_height()]

    def is_within(self, point) -> bool:
        """Checks if a point is within the bounding box."""
        return (point.x >= self.x_min and point.x <= self.x_max
                and point.y >= self.y_min and point.y <= self.y_max)

    def calculate_iou(self, other_bbox) -> float:
        """Calculate the IoU of a single bounding box.

        Args:
            other_bbox (:py:class:`.BoundingBox2D`): The other bounding box.

        Returns:
            :obj:`float`: The IoU of the two bounding boxes.
        """
        if (other_bbox.x_min > other_bbox.x_max
                or other_bbox.y_min > other_bbox.y_max):
            raise AssertionError(
                "Other bbox is malformed {}".format(other_bbox))

        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise AssertionError("Bounding box is malformed {}".format(self))

        if (self.x_max < other_bbox.x_min or other_bbox.x_max < self.x_min
                or self.y_max < other_bbox.y_min
                or other_bbox.y_max < self.y_min):
            return 0.0

        inter_x1 = max([self.x_min, other_bbox.x_min])
        inter_x2 = min([self.x_max, other_bbox.x_max])

        inter_y1 = max([self.y_min, other_bbox.y_min])
        inter_y2 = min([self.y_max, other_bbox.y_max])

        inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        gt_area = (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)
        pred_area = (other_bbox.x_max - other_bbox.x_min +
                     1) * (other_bbox.y_max - other_bbox.y_min + 1)
        return float(inter_area) / (gt_area + pred_area - inter_area)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'BoundingBox2D(xmin: {}, xmax: {}, ymin: {}, ymax: {})'.format(
            self.x_min, self.x_max, self.y_min, self.y_max)

class BoundingBox3D(object):
    """Class used to store a 3D bounding box.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            bounding box (rotation is (0, 0, 0)).
        extent (:py:class:`~pylot.utils.Vector3D`): The extent of the bounding
            box.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            bounding box (rotation is (0, 0, 0)).
        extent (:py:class:`~pylot.utils.Vector3D`): The extent of the bounding
            box.
    """
    def __init__(self,
                 transform: Transform = None,
                 extent: Vector3D = None,
                 corners=None):
        self.transform = transform
        self.extent = extent
        self.corners = corners

    @classmethod
    def from_dimensions(cls, bbox_dimensions, location, rotation_y):
        """Creates a 3D bounding box.

        Args:
            bbox_dimensions: The height, width and length of the bbox.
            location: The location of the box in the camera frame.
            rotation: The rotation of the bbox.

        Returns:
            :py:class:`.BoundingBox3D`: A bounding box instance.
        """
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        l, w, h = bbox_dimensions[2], bbox_dimensions[1], bbox_dimensions[0]
        x_corners = [
            l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2
        ]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [
            w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2
        ]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners).transpose(1, 0)
        corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
            1, 3)
        return cls(corners=corners_3d)

    @classmethod
    def from_simulator_bounding_box(cls, bbox):
        """Creates a pylot bounding box from a simulator bounding box.

        Args:
            bbox: The bounding box to transform.

        Returns:
            :py:class:`.BoundingBox3D`: A bounding box instance.
        """
        transform = Transform(
            Location.from_simulator_location(bbox.location),
            Rotation())
        extent = Vector3D.from_simulator_vector(bbox.extent)
        return cls(transform, extent)

    def as_simulator_bounding_box(self):
        """Retrieves the bounding box as instance of a simulator bounding box.

        Returns:
            A instance of a simulator class that represents the bounding box.
        """
        from carla import BoundingBox
        bb_loc = self.transform.location.as_simulator_location()
        bb_extent = self.extent.as_simulator_vector()
        return BoundingBox(bb_loc, bb_extent)

    def visualize(self, world, actor_transform, time_between_frames=100):
        """Visualizes the bounding box on the world.

        Args:
            world: The simulator world instance to visualize the bounding
                box on.
            actor_transform (:py:class:`~pylot.utils.Transform`): The current
                transform of the actor that the bounding box is of.
            time_between_frames (:obj:`float`): Time in ms to show the bounding
                box for.
        """
        bb = self.as_simulator_bounding_box()
        bb.location += actor_transform.location()
        world.debug.draw_box(bb,
                             actor_transform.rotation.as_simulator_rotation(),
                             life_time=time_between_frames / 1000.0)

    def to_camera_view(self, obstacle_transform: Transform,
                       extrinsic_matrix, intrinsic_matrix):
        """Converts the coordinates of the bounding box for the given obstacle
        to the coordinates in the view of the camera.

        This method retrieves the extent of the bounding box, transforms them
        to coordinates relative to the bounding box origin, then converts those
        to coordinates relative to the obstacle.

        These coordinates are then considered to be in the world coordinate
        system, which is mapped into the camera view. A negative z-value
        signifies that the bounding box is behind the camera plane.

        Note that this function does not cap the coordinates to be within the
        size of the camera image.

        Args:
            obstacle_transform (:py:class:`~pylot.utils.Transform`): The
                transform of the obstacle that the bounding box is associated
                with.
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            A list of 8 Location instances specifying the 8 corners of the
            bounding box.
        """
        # Retrieve the eight coordinates of the bounding box with respect to
        # the origin of the bounding box.
        import numpy as np
        if self.corners is not None:
            pts_2d = np.dot(intrinsic_matrix,
                            self.corners.transpose(1, 0)).transpose(1, 0)
            pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
            camera_coordinates = [
                Vector2D(pt[0], pt[1]) for pt in pts_2d
            ]
            return camera_coordinates

        extent = self.extent
        bbox = np.array([
            Location(x=+extent.x, y=+extent.y, z=-extent.z),
            Location(x=-extent.x, y=+extent.y, z=-extent.z),
            Location(x=-extent.x, y=-extent.y, z=-extent.z),
            Location(x=+extent.x, y=-extent.y, z=-extent.z),
            Location(x=+extent.x, y=+extent.y, z=+extent.z),
            Location(x=-extent.x, y=+extent.y, z=+extent.z),
            Location(x=-extent.x, y=-extent.y, z=+extent.z),
            Location(x=+extent.x, y=-extent.y, z=+extent.z)
        ])

        # Transform the vertices with respect to the bounding box transform.
        bbox = self.transform.transform_locations(bbox)

        # Convert the bounding box relative to the world.
        bbox = obstacle_transform.transform_locations(bbox)

        # Obstacle's transform is relative to the world. Thus, the bbox
        # contains the 3D bounding box vertices relative to the world.
        camera_coordinates = []
        for vertex in bbox:
            location_2D = vertex.to_camera_view(extrinsic_matrix,
                                                intrinsic_matrix)

            # Add the points to the image.
            camera_coordinates.append(location_2D)

        return camera_coordinates

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox3D(transform: {}, extent: {})".format(
            self.transform, self.extent)


def get_bounding_box_in_camera_view(bb_coordinates, image_width, image_height):
    """Creates the bounding box in the view of the camera image using the
    coordinates generated with respect to the camera transform.

    Args:
        bb_coordinates: 8 :py:class:`~pylot.utils.Location` coordinates of
            the bounding box relative to the camera transform.
        image_width (:obj:`int`): The width of the image being published by the
            camera.
        image_height (:obj:`int`): The height of the image being published by
            the camera.

    Returns:
        :py:class:`.BoundingBox2D`: a bounding box, or None if the bounding box
            does not fall into the view of the camera.
    """
    # Make sure that atleast 2 of the bounding box coordinates are in front.
    z_vals = [loc.z for loc in bb_coordinates if loc.z >= 0]
    if len(z_vals) < 2:
        return None

    # Create the thresholding line segments of the camera view.
    from shapely.geometry import LineString
    left = LineString(((0, 0), (0, image_height)))
    bottom = LineString(((0, image_height), (image_width, image_height)))
    right = LineString(((image_width, image_height), (image_width, 0)))
    top = LineString(((image_width, 0), (0, 0)))
    camera_thresholds = [left, bottom, right, top]

    def threshold(p1, p2):
        points = []
        # If the points are themselves within the image, add them to the
        # set of thresholded points.
        if (p1[0] >= 0 and p1[0] < image_width and p1[1] >= 0
                and p1[1] < image_height):
            points.append(p1)

        if (p2[0] >= 0 and p2[0] < image_width and p2[1] >= 0
                and p2[1] < image_height):
            points.append(p2)

        # Compute the intersection of the line segment formed by p1 -- p2
        # with all the thresholds of the camera image.
        p12 = LineString((p1, p2))
        for camera_threshold in camera_thresholds:
            p = p12.intersection(camera_threshold)
            if not p.is_empty:
                if p.geom_type == 'Point':
                    points.append((p.x, p.y))
                elif p.geom_type == 'LineString':
                    for coord in p.coords:
                        points.append((coord[0], coord[1]))
        return points

    # Go over each of the segments of the bounding box and threshold it to
    # be inside the image.
    thresholded_points = []
    points = [(int(loc.x), int(loc.y)) for loc in bb_coordinates]
    # Bottom plane thresholded.
    thresholded_points.extend(threshold(points[0], points[1]))
    thresholded_points.extend(threshold(points[1], points[2]))
    thresholded_points.extend(threshold(points[2], points[3]))
    thresholded_points.extend(threshold(points[3], points[0]))

    # Top plane thresholded.
    thresholded_points.extend(threshold(points[4], points[5]))
    thresholded_points.extend(threshold(points[5], points[6]))
    thresholded_points.extend(threshold(points[6], points[7]))
    thresholded_points.extend(threshold(points[7], points[4]))

    # Remaining segments thresholded.
    thresholded_points.extend(threshold(points[0], points[4]))
    thresholded_points.extend(threshold(points[1], points[5]))
    thresholded_points.extend(threshold(points[2], points[6]))
    thresholded_points.extend(threshold(points[3], points[7]))

    if len(thresholded_points) == 0:
        return None
    else:
        x = [int(x) for x, _ in thresholded_points]
        y = [int(y) for _, y in thresholded_points]
        if min(x) < max(x) and min(y) < max(y):
            return BoundingBox2D(min(x), max(x), min(y), max(y))
        else:
            return None

def load_coco_labels(labels_path):
    """Returns a map from index to label.

    Args:
        labels_path (:obj:`str`): Path to a file storing a label on each line.
    """
    labels_map = {}
    with open(labels_path) as labels_file:
        labels = labels_file.read().splitlines()
        index = 1
        for label in labels:
            labels_map[index] = label
            index += 1
    return labels_map

def load_coco_bbox_colors(coco_labels):
    """Returns a map from label to color."""
    # Transform to RGB values.
    bbox_color_list = coco_bbox_color_list.reshape((-1, 3)) * 255
    # Transform to ints
    bbox_colors = [(bbox_color_list[_]).astype(np.uint8)
                   for _ in range(len(bbox_color_list))]
    bbox_colors = np.array(bbox_colors,
                           dtype=np.uint8).reshape(len(bbox_colors), 1, 1, 3)

    colors = {}
    for category, label in coco_labels.items():
        colors[label] = bbox_colors[category - 1][0][0].tolist()
    return colors


class Obstacle(object):
    """Class used to store info about obstacles.

    This class provides helper functions to detect obstacles and provide
    bounding boxes for them.

    Args:
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~pylot.utils.Transform`, optional): Transform of
            the obstacle in the world.

    Attributes:
        bounding_box (:py:class:`~pylot.utisl.BoundingBox2D`): Bounding box of
            the obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            obstacle.
    """
    def __init__(self,
                 bounding_box,
                 confidence: float,
                 label: str,
                 id: int = -1,
                 transform: Transform = None,
                 detailed_label: str = '',
                 bounding_box_2D: BoundingBox2D = None):
        self.bounding_box = bounding_box
        if isinstance(bounding_box, BoundingBox2D):
            self.bounding_box_2D = bounding_box
        else:
            self.bounding_box_2D = bounding_box_2D
        self.confidence = confidence
        self.label = label
        self.id = id
        self.transform = transform
        self.detailed_label = detailed_label
        if label == 'vehicle':
            self.segmentation_class = 10
        elif label == 'person':
            self.segmentation_class = 4
        else:
            self.segmentation_class = None
        # Thresholds to be used for detection of the obstacle.
        self.__segmentation_threshold = 0.20
        self.__depth_threshold = 5

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates an Obstacle from a simulator actor.

        Args:
            actor: The actor to initialize the obstacle with.

        Returns:
            :py:class:`.Obstacle`: An obstacle instance.
        """
        from carla import Vehicle, Walker
        if not isinstance(actor, (Vehicle, Walker)):
            raise ValueError("The actor should be of type Vehicle or "
                             "Walker to initialize the Obstacle class.")
        # We do not use everywhere from_simulator* methods in order to reduce
        # runtime.
        # Convert the transform provided by the simulation to the Pylot class.
        transform = Transform.from_simulator_transform(
            actor.get_transform())
        # Convert the bounding box from the simulation to the Pylot one.
        bounding_box = BoundingBox3D.from_simulator_bounding_box(
            actor.bounding_box)
        if isinstance(actor, Vehicle):
            label = 'vehicle'
        else:
            label = 'person'
        # Get the simulator actor from type_id (e.g. vehicle.ford.mustang).
        detailed_label = actor.type_id
        # TODO (Sukrit): Move from vehicles and people to separate classes
        # for bicycles, motorcycles, cars and persons.
        return cls(bounding_box, 1.0, label, actor.id, transform,
                   detailed_label)

    def as_mot16_str(self, timestamp):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        log_line = "{},{},{},{},{},{},{},{},{},{}\n".format(
            timestamp, self.id, self.bounding_box_2D.x_min,
            self.bounding_box_2D.y_min, self.bounding_box_2D.get_width(),
            self.bounding_box_2D.get_height(), 1.0, -1, -1, -1)
        return log_line

    def _distance(self, other_transform: Transform):
        """Computes the distance from the obstacle to the other transform.

        The distance provides an estimate of the depth returned by the depth
        camera sensor in the simulator. As a result, the distance is defined
        as the displacement of the obstacle along either the X or the Y axis.

        Args:
            other_transform (:py:class:`~pylot.utils.Transform`): The other
                transform.

        Returns:
            :obj:`float`: The distance (in metres) of the obstacle from the
            transform.
        """
        import numpy as np
        if self.transform is None:
            raise ValueError('Obstacle {} does not have a transform'.format(
                self.id))
        # Get the location of the vehicle and the obstacle as numpy arrays.
        other_location = other_transform.location.as_numpy_array()
        obstacle_location = self.transform.location.as_numpy_array()

        # Calculate the vector from the vehicle to the obstacle.
        # Scale it by the forward vector, and calculate the norm.
        relative_vector = other_location - obstacle_location
        distance = np.linalg.norm(
            relative_vector * other_transform.forward_vector.as_numpy_array())
        return distance

    def draw_on_frame(self,
                      frame,
                      bbox_color_map,
                      ego_transform: Transform = None,
                      text: str = None):
        """Annotate the image with the bounding box of the obstacle."""
        if text is None:
            text = '{}, {:.1f}'.format(self.label, self.confidence)
            if self.id != -1:
                text += ', id:{}'.format(self.id)
            if ego_transform is not None and self.transform is not None:
                text += ', {:.1f}m'.format(
                    ego_transform.location.distance(self.transform.location))
        if self.label in bbox_color_map:
            color = bbox_color_map[self.label]
        else:
            color = [255, 255, 255]
        # Show bounding box.
        if self.bounding_box_2D:
            # Draw the 2D bounding box if available.
            frame.draw_box(self.bounding_box_2D.get_min_point(),
                           self.bounding_box_2D.get_max_point(), color)
            frame.draw_text(self.bounding_box_2D.get_min_point(), text, color)
        elif isinstance(self.bounding_box, BoundingBox3D):
            if self.bounding_box.corners is None:
                raise ValueError(
                    'Obstacle {} does not have bbox corners'.format(self.id))
            corners = self.bounding_box.to_camera_view(
                None, frame.camera_setup.get_extrinsic_matrix(),
                frame.camera_setup.get_intrinsic_matrix())
            frame.draw_3d_box(corners, color)
        else:
            raise ValueError('Obstacle {} does not have bounding box'.format(
                self.id))

    def draw_trajectory_on_frame(self,
                                 trajectory,
                                 frame,
                                 point_color,
                                 draw_label: bool = False):
        # Intrinsic and extrinsic matrix of the top down camera.
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        if isinstance(self.bounding_box, BoundingBox3D):
            # Draw bounding boxes.
            start_location = self.bounding_box.transform.location - \
                self.bounding_box.extent
            end_location = self.bounding_box.transform.location + \
                self.bounding_box.extent
            for transform in trajectory:
                [start_transform,
                 end_transform] = transform.transform_locations(
                     [start_location, end_location])
                start_point = start_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                end_point = end_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(start_point) or frame.in_frame(end_point):
                    frame.draw_box(start_point, end_point, point_color)
        else:
            # Draw points.
            for transform in trajectory:
                screen_point = transform.location.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(screen_point):
                    # Draw trajectory on frame.
                    frame.draw_point(screen_point, point_color)
        if draw_label and len(trajectory) > 0:
            text = '{}, {}'.format(self.label, self.id)
            screen_point = trajectory[-1].location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_text(screen_point, text, point_color)

    def get_bounding_box_corners(self,
                                 obstacle_transform,
                                 obstacle_radius=None):
        """Gets the corners of the obstacle's bounding box.
        Note:
            The bounding box is applied on the given obstacle transfom, and not
            on the default obstacle transform.
        """
        # Use 3d bounding boxes if available, otherwise use default
        if isinstance(self.bounding_box, BoundingBox3D):
            start_location = (self.bounding_box.transform.location -
                              self.bounding_box.extent)
            end_location = (self.bounding_box.transform.location +
                            self.bounding_box.extent)
            [start_location,
             end_location] = obstacle_transform.transform_locations(
                 [start_location, end_location])
        else:
            obstacle_radius_loc = Location(obstacle_radius,
                                                       obstacle_radius)
            start_location = obstacle_transform.location - obstacle_radius_loc
            end_location = obstacle_transform.location + obstacle_radius_loc
        return [
            min(start_location.x, end_location.x),
            min(start_location.y, end_location.y),
            max(start_location.x, end_location.x),
            max(start_location.y, end_location.y)
        ]

    def get_in_log_format(self):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        min_point = self.bounding_box_2D.get_min_point()
        max_point = self.bounding_box_2D.get_max_point()
        return (self.label, self.detailed_label, self.id,
                ((min_point.x, min_point.y), (max_point.x, max_point.y)))

    def is_animal(self):
        return self.label in [
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe'
        ]

    def is_person(self):
        return self.label == 'person'

    def is_speed_limit(self):
        return self.label in [
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        ]

    def is_stop_sign(self):
        return self.label == 'stop sign' or self.label == 'stop marking'

    def is_traffic_light(self):
        return self.label in [
            'red traffic light', 'yellow traffic light', 'green traffic light',
            'off traffic light'
        ]

    def is_vehicle(self):
        # Might want to include train.
        return self.label in VEHICLE_LABELS

    def populate_bounding_box_2D(self, depth_frame, segmented_frame):
        """Populates the 2D bounding box for the obstacle.

        Heuristically uses the depth frame and segmentation frame to figure out
        if the obstacle is in view of the camera or not.

        Args:
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                Depth frame used to compare the depth to the distance of the
                obstacle from the sensor.
            segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):  # noqa: E501
                Segmented frame used to refine the conversions.

        Returns:
            :py:class:`~pylot.utisl.BoundingBox2D`: An instance representing a
            rectangle over the obstacle if the obstacle is deemed to be
            visible, None otherwise.
        """
        if self.bounding_box_2D:
            return self.bounding_box_2D
        # Convert the bounding box of the obstacle to the camera coordinates.
        bb_coordinates = self.bounding_box.to_camera_view(
            self.transform, depth_frame.camera_setup.get_extrinsic_matrix(),
            depth_frame.camera_setup.get_intrinsic_matrix())

        # Threshold the bounding box to be within the camera view.
        bbox_2d = get_bounding_box_in_camera_view(
            bb_coordinates, depth_frame.camera_setup.width,
            depth_frame.camera_setup.height)
        if not bbox_2d:
            return None
        # Crop the segmented and depth image to the given bounding box.
        cropped_image = segmented_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]
        cropped_depth = depth_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]

        # If the size of the bounding box is greater than 0, ensure that the
        # bounding box contains more than a threshold of pixels corresponding
        # to the required segmentation class.
        if cropped_image.size > 0:
            masked_image = np.zeros_like(cropped_image)
            masked_image[np.where(
                cropped_image == self.segmentation_class)] = 1
            seg_threshold = self.__segmentation_threshold * masked_image.size
            if np.sum(masked_image) >= seg_threshold:
                # The bounding box contains the required number of pixels that
                # belong to the required class. Ensure that the depth of the
                # obstacle is the depth in the image.
                masked_depth = cropped_depth[np.where(masked_image == 1)]
                mean_depth = np.mean(masked_depth) * 1000
                depth = self._distance(
                    depth_frame.camera_setup.get_transform())
                if abs(depth - mean_depth) <= self.__depth_threshold:
                    self.bounding_box_2D = bbox_2d
                    return bbox_2d
        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        obstacle = 'Obstacle(id: {}, label: {}, confidence: {}, '\
            'bbox: {})'.format(self.id, self.label, self.confidence,
                               self.bounding_box)
        if self.transform:
            return obstacle + ' at ' + str(self.transform)
        else:
            return obstacle

class LaneMarkingType(Enum):
    """Enum that defines the lane marking types according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane markings are correctly
    propogated from the simulator to Pylot.
    """
    OTHER = 0
    BROKEN = 1
    SOLID = 2
    SOLIDSOLID = 3
    SOLIDBROKEN = 4
    BROKENSOLID = 5
    BROKENBROKEN = 6
    BOTTSDOTS = 7
    GRASS = 8
    CURB = 9
    NONE = 10


class LaneChange(Enum):
    """ Enum that defines the permission to turn either left, right, both or
    none for a given lane.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from the simulator to Pylot.
    """
    NONE = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3


class LaneType(Enum):
    """Enum that defines the type of the lane according to OpenDrive 1.4.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from the simulator to Pylot.
    """
    NONE = 1
    DRIVING = 2
    STOP = 4
    SHOULDER = 8
    BIKING = 16
    SIDEWALK = 32
    BORDER = 64
    RESTRICTED = 128
    PARKING = 256
    BIDIRECTIONAL = 512
    MEDIAN = 1024
    SPECIAL1 = 2048
    SPECIAL2 = 4096
    SPECIAL3 = 8192
    ROADWORKS = 16384
    TRAM = 32768
    RAIL = 65536
    ENTRY = 131072
    EXIT = 262144
    OFFRAMP = 524288
    ONRAMP = 1048576
    ANY = 4294967294


class RoadOption(Enum):
    """Enum that defines the possible high-level route plans.

    RoadOptions are usually attached to waypoints we receive from
    the challenge environment.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name


class LaneMarking(object):
    """Used to represent a lane marking.

    Attributes:
        marking_color (:py:class:`.LaneMarkingColor`): The color of the lane
            marking
        marking_type (:py:class:`.LaneMarkingType`): The type of the lane
            marking.
        lane_change (:py:class:`.LaneChange`): The type that defines the
            permission to either turn left, right, both or none.
    """
    def __init__(self, marking_color, marking_type, lane_change):
        self.marking_color = LaneMarkingColor(marking_color)
        self.marking_type = LaneMarkingType(marking_type)
        self.lane_change = LaneChange(lane_change)

    @classmethod
    def from_simulator_lane_marking(cls, lane_marking):
        """Creates a pylot LaneMarking from a simulator lane marking.

        Args:
            lane_marking: An instance of a simulator lane marking.

        Returns:
            :py:class:`.LaneMarking`: A pylot lane-marking.
        """
        return cls(lane_marking.color, lane_marking.type,
                   lane_marking.lane_change)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "LaneMarking(color: {}, type: {}, change: {})".format(
            self.marking_color, self.marking_type, self.lane_change)


def add_timestamp(image_np, timestamp):
    """Adds a timestamp text to an image np array.

    Args:
        image_np: A numpy array of the image.
        timestamp (:obj:`int`): The timestamp of the image.
    """
    import cv2
    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp_txt = '{}'.format(timestamp)
    # Put timestamp text.
    cv2.putText(image_np,
                timestamp_txt, (5, 15),
                txt_font,
                0.5, (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)


def get_top_down_transform(transform, top_down_camera_altitude):
    # Calculation relies on the fact that the camera's FOV is 90.
    top_down_location = (transform.location +
                         Location(0, 0, top_down_camera_altitude))
    return Transform(top_down_location, Rotation(-90, 0, 0))


def time_epoch_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


def set_tf_loglevel(level):
    """To be used to suppress TensorFlow logging."""
    import logging
    import os
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def run_visualizer_control_loop(control_display_stream):
    """Runs a pygame loop that waits for user commands.

    The user commands are send on the control_display_stream
    to control the pygame visualization window.
    """
    import erdos
    import pygame
    clock = pygame.time.Clock()
    from pygame.locals import K_n
    while True:
        clock.tick_busy_loop(60)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == K_n:
                    control_display_stream.send(
                        erdos.Message(erdos.Timestamp(coordinates=[0]),
                                      event.key))
            elif event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_c
                        and pygame.key.get_mods() & pygame.KMOD_CTRL):
                    raise KeyboardInterrupt


def verify_keys_in_dict(required_keys, arg_dict):
    assert set(required_keys).issubset(set(arg_dict.keys())), \
            "one or more of {} not found in {}".format(required_keys, arg_dict)
