import numpy as np


def _is_light_red(self, lights_list):
    if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
        affecting = self._vehicle.get_traffic_light()

        for light in self._traffic_lights:
            if light.id == affecting.id:
                return affecting

    return None

def _is_walker_hazard(self, walkers_list):
    z = self._vehicle.get_location().z
    p1 = _numpy(self._vehicle.get_location())
    v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

    self._draw_line(p1, v1, z+2.5, (0, 0, 255))

    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat

        self._draw_line(p2, v2, z+2.5)

        collides, collision_point = get_collision(p1, v1, p2, v2)

        if collides:
            return walker

    return None

def _is_vehicle_hazard(self, vehicle_list):
    z = self._vehicle.get_location().z

    o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
    p1 = _numpy(self._vehicle.get_location())
    s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
    v1_hat = o1
    v1 = s1 * v1_hat

    self._draw_line(p1, v1, z+2.5, (255, 0, 0))

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

        self._draw_line(p2, v2, z+2.5, (255, 0, 0))

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        return target_vehicle

    return None
