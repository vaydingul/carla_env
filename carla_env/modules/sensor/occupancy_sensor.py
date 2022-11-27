from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import math
import time
import weakref
logger = logging.getLogger(__name__)


def rotate_z(x, y, theta):
    """Rotate a point (x, y) by theta degrees around the z-axis."""
    theta = math.radians(theta)
    return x * math.cos(theta) - y * math.sin(theta), x * \
        math.sin(theta) + y * math.cos(theta)


class OccupancySensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for collision sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.angle_list = [
            np.rad2deg(2 * np.pi /
                       (self.config["number_of_radars"])) *
            k for k in range(
                self.config['number_of_radars'])]

        self.occupancy = np.zeros((self.config["number_of_radars"], ))

        self.reset()

    def _start(self):
        """Start the sensor module"""

        bound_x = 0.5 + self.actor.get_actor().bounding_box.extent.x
        bound_y = 0.5 + self.actor.get_actor().bounding_box.extent.y
        bound_z = 0.5 + self.actor.get_actor().bounding_box.extent.z

        self.radar_sensor_list = []
        for (i, angle) in enumerate(self.angle_list):

            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', str(5))
            radar_bp.set_attribute('vertical_fov', str(5))
            radar_bp.set_attribute('range', str(10))
            radar_bp.set_attribute('points_per_second', str(100))

            x, y = rotate_z(math.sqrt(math.pow(bound_x, 2) +
                            math.pow(bound_y, 2)), 0, angle)
            print(x, y)

            self.radar_sensor_list.append(self.world.spawn_actor(
                radar_bp,
                carla.Transform(
                    location=carla.Location(x=x, y=y, z=0.5),
                    rotation=carla.Rotation(
                        yaw=angle)),
                attach_to=self.actor.get_actor()))

        for (i, radar_sensor) in enumerate(self.radar_sensor_list):

            def callback(radar_data, i=i): return self._get_sensor_data(
                radar_data, i)

            radar_sensor.listen(callback)

        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""
        if self.is_attached:
            for sensor in self.sensor_list:
                sensor.stop()
                sensor.destroy()

        self.is_attached = False

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, radar_data, i):
        """Get the sensor data"""
        # To get a numpy [[vel, azimuth, altitude, depth],...[,,,]]:

        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))
        depth = points[:, -1]

        self.occupancy[i] = np.nanmean(depth)
        if np.any(depth < self.config["threshold"]):
            self.occupancy[i] = 1
        else:
            self.occupancy[i] = 0

        data = {'frame': radar_data.frame,
                'occupancy': self.occupancy
                }

        if self.save_to_queue:
            self._queue_operation(data)

    def step(self):
        """Step the sensor"""
        self._tick()

    def reset(self):
        """Reset the sensor"""
        self._stop()
        self._start()

    def render(self):
        """Render the sensor"""

        self.render_dict['occupancy'] = self.occupancy

        return self.render_dict

    def close(self):
        """Close the sensor"""
        self._stop()

    def seed(self):
        """Seed the sensor"""
        pass

    def get_config(self):
        """Get the config of the sensor"""
        return self.config

    def _set_default_config(self):
        """Set the default config of the sensor"""
        self.config = {"threshold": 3.0,
                       "number_of_radars": 4}

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
