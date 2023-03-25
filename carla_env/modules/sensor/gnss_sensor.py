from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class GNSSSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for rgb sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.gnss_data = None
        self.render_dict = {}

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.reset()

    def _start(self):
        """Start the sensor module"""

        gnss_bp = self.world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("noise_alt_stddev", str(self.config["noise_alt_stddev"]))
        gnss_bp.set_attribute("noise_lat_stddev", str(self.config["noise_lat_stddev"]))
        gnss_bp.set_attribute("noise_lon_stddev", str(self.config["noise_lon_stddev"]))
        gnss_bp.set_attribute("noise_alt_bias", str(self.config["noise_alt_bias"]))
        gnss_bp.set_attribute("noise_lat_bias", str(self.config["noise_lat_bias"]))
        gnss_bp.set_attribute("noise_lon_bias", str(self.config["noise_lon_bias"]))

        self.gnss = self.world.spawn_actor(
            gnss_bp, carla.Transform(), attach_to=self.actor.get_actor()
        )

        self.gnss.listen(lambda gnss_data: self._get_sensor_data(gnss_data))

        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""

        if self.is_attached:

            self.gnss.destroy()

        self.is_attached = False

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, gnss_data):
        """Get the sensor data"""

        gnss_data_ = [gnss_data.latitude, gnss_data.longitude, gnss_data.altitude]

        self.gnss_data = gnss_data_

        data = {"frame": gnss_data.frame, "data": gnss_data_}

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
        if self.gnss_data_ is not None:
            self.render_dict["latitude"] = self.gnss_data_[0]
            self.render_dict["longitude"] = self.gnss_data_[1]
            self.render_dict["altitude"] = self.gnss_data_[2]

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
            "noise_alt_stddev": 0.000005,
            "noise_lat_stddev": 0.000005,
            "noise_lon_stddev": 0.000005,
            "noise_alt_bias": 0,
            "noise_lat_bias": 0,
            "noise_lon_bias": 0,
        }

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
