from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class CollisionSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for collision sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.impulse = None

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.reset()

    def _start(self):
        """Start the sensor module"""

        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")

        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.actor.get_actor()
        )

        self.collision_sensor.listen(
            lambda collision_data: self._get_sensor_data(collision_data)
        )

        self.is_attached = True

    def _stop(self):
        """Stop the sensor module"""
        if self.is_attached:
            self.collision_sensor.destroy()

        self.is_attached = False

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, collision_data):
        """Get the sensor data"""

        # logger.info("Received an image of frame: " + str(image.frame))

        impulse = collision_data.normal_impulse
        impulse = np.array([impulse.x, impulse.y, impulse.z])
        impulse = copy.deepcopy(impulse)

        self.impulse = impulse

        data = {
            "frame": collision_data.frame,
            "transform": collision_data.transform,
            "impulse": impulse,
        }

        if self.save_to_queue:
            self._queue_operation(data)

    def step(self, *args, **kwargs):
        """Step the sensor"""
        self._tick()

    def reset(self):
        """Reset the sensor"""
        self._stop()
        self._start()

    def render(self):
        """Render the sensor"""
        if self.impulse is not None:
            self.render_dict["impulse"] = self.impulse

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
        self.config = {}

    def _queue_operation(self, data):
        """Queue the sensor data and additional stuff"""
        self.queue.put(data)
