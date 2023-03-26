from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class RadarSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for rgb sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.radar_data = None
        self.render_dict = {}

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.reset()

    def _start(self):
        """Start the sensor module"""

        radar_bp = self.world.get_blueprint_library().find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(self.config["horizontal_fov"]))
        radar_bp.set_attribute("vertical_fov", str(self.config["vertical_fov"]))
        radar_bp.set_attribute(
            "points_per_second", str(self.config["points_per_second"])
        )
        radar_bp.set_attribute("range", str(self.config["range"]))

        self.radar_transform = carla.Transform(
            carla.Location(x=self.config["x"], y=self.config["y"], z=self.config["z"]),
            carla.Rotation(
                roll=self.config["roll"],
                pitch=self.config["pitch"],
                yaw=self.config["yaw"],
            ),
        )

        self.radar = self.world.spawn_actor(
            radar_bp,
            self.radar_transform,
            attach_to=self.actor.get_actor(),
        )

        self.radar.listen(lambda radar_data: self._get_sensor_data(radar_data))

        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""

        if self.is_attached:

            self.radar.destroy()

        self.is_attached = False

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, radar_data):
        """Get the sensor data"""

        radar_data_ = np.frombuffer(radar_data.raw_data, dtype=np.dtype("f4"))
        radar_data_ = copy.deepcopy(radar_data_)
        radar_data_ = np.reshape(radar_data_, (int(radar_data_.shape[0] / 4), 4))

        # To get a numpy [[vel, azimuth, altitude, depth],...[,,,]]:

        self.radar_data = radar_data_

        data = {
            "frame": radar_data.frame,
            "transform": radar_data.transform,
            "data": radar_data_,
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
        if self.radar_data is not None:
            if self.radar_data.shape[0] > 0:
                self.render_dict["vel_max"] = np.max(self.radar_data[:, 0])
                self.render_dict["vel_min"] = np.min(self.radar_data[:, 0])
                self.render_dict["azimuth_max"] = np.max(self.radar_data[:, 1])
                self.render_dict["azimuth_min"] = np.min(self.radar_data[:, 1])
                self.render_dict["altitude_max"] = np.max(self.radar_data[:, 2])
                self.render_dict["altitude_min"] = np.min(self.radar_data[:, 2])
                self.render_dict["depth_max"] = np.max(self.radar_data[:, -1])  
            self.render_dict["depth_min"] = np.min(self.radar_data[:, -1])

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
        self.config = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "horizontal_fov": 0.0,
            "vertical_fov": 0.0,
            "points_per_second": 0.0,
            "range": 0.0,
        }

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
