from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class IMUSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for rgb sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.imu_data = None
        self.render_dict = {}

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.reset()

    def _start(self):
        """Start the sensor module"""

        imu_bp = self.world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute(
            "noise_accel_stddev_x", str(self.config["noise_accel_stddev_x"])
        )
        imu_bp.set_attribute(
            "noise_accel_stddev_y", str(self.config["noise_accel_stddev_y"])
        )
        imu_bp.set_attribute(
            "noise_accel_stddev_z", str(self.config["noise_accel_stddev_z"])
        )
        imu_bp.set_attribute(
            "noise_gyro_stddev_x", str(self.config["noise_gyro_stddev_x"])
        )
        imu_bp.set_attribute(
            "noise_gyro_stddev_y", str(self.config["noise_gyro_stddev_y"])
        )
        imu_bp.set_attribute(
            "noise_gyro_stddev_z", str(self.config["noise_gyro_stddev_z"])
        )

        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.actor.get_actor()
        )

        self.imu.listen(lambda imu_data: self._get_sensor_data(imu_data))

        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""

        if self.is_attached:

            self.imu.destroy()

        self.is_attached = False

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, imu_data):
        """Get the sensor data"""

        imu_data_ = [
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z,
            imu_data.gyroscope.x,
            imu_data.gyroscope.y,
            imu_data.gyroscope.z,
            imu_data.compass,
        ]

        self.imu_data = imu_data_

        data = {"frame": imu_data.frame, "data": imu_data_}

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
        if self.imu_data is not None:
            self.render_dict["accelerometer_x"] = self.imu_data[0]
            self.render_dict["accelerometer_y"] = self.imu_data[1]
            self.render_dict["accelerometer_z"] = self.imu_data[2]
            self.render_dict["gyroscope_x"] = self.imu_data[3]
            self.render_dict["gyroscope_y"] = self.imu_data[4]
            self.render_dict["gyroscope_z"] = self.imu_data[5]
            self.render_dict["compass"] = self.imu_data[6]

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
            "noise_accel_stddev_x": 0.001,
            "noise_accel_stddev_y": 0.001,
            "noise_accel_stddev_z": 0.015,
            "noise_gyro_stddev_x": 0.001,
            "noise_gyro_stddev_y": 0.001,
            "noise_gyro_stddev_z": 0.001,
        }

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
