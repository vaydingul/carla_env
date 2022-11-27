from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time
import weakref
logger = logging.getLogger(__name__)


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
            np.rad2deg(np.pi /
                       4) *
            k for k in range(
                self.config['number_of_radars'])]

        self.occupancy = np.zeros((self.config["number_of_radars"], ))

        self.reset()

    def _start(self):
        """Start the sensor module"""

        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')

        for (i, angle) in enumerate(self.angle_list):
			
            radar_sensor = self.world.spawn_actor(
                radar_bp,
                carla.Transform(
                    rotation=carla.Rotation(
                        yaw=angle)),
                attach_to=self.actor.get_actor())

            radar_sensor.listen(
                lambda radar_data: self._get_sensor_data(radar_data, i))

        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""
        if self.is_attached:
            self.radar_sensor.destroy()

        self.is_attached = False

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, radar_data, i):
        """Get the sensor data"""

        print(i)
        # To get a numpy [[vel, azimuth, altitude, depth],...[,,,]]:
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        depth = points[:, -1]

        self.occupancy[i] = np.mean(depth)
        # if np.any(depth < self.config["threshold"]):
        # 	self.occupancy[i] = 1
        # else:
        # 	self.occupancy[i] = 0

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
        self.config = {"threshold": 10.0,
                       "number_of_radars": 8}

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
