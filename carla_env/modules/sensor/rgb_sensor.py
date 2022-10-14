from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class RGBSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for rgb sensor management"""

    def __init__(self, config, client, actor=None, id=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.image_data = None

        if actor is not None:
            self.attach_to_actor(actor, id)

        self.reset()

    def _start(self):
        """Start the sensor module"""

        rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.camera = self.world.spawn_actor(
            rgb_bp, self.camera_transform, attach_to=self.actor.get_actor())

        self.camera.listen(lambda image: self._get_sensor_data(image))

    def _stop(self):
        """Stop the sensor module"""
        self.camera.destroy()

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, image):
        """Get the sensor data"""

        #logger.info("Received an image of frame: " + str(image.frame))

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_data = copy.deepcopy(image_data)
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        image_data = image_data[:, :, :3]
        image_data = image_data[:, :, ::-1]

        self.image_data = image_data
        self.image_transform = image.transform

        data = {'frame': image.frame,
                'transform': image.transform,
                'data': image_data
                }

        if self.save_to_queue:
            self._queue_operation(data)

    def step(self):
        """Step the sensor"""
        self._tick()

    def reset(self):
        """Reset the sensor"""
        self._start()

    def render(self):
        """Render the sensor"""
        if self.image_data is not None:
            self.render_dict["image_data"] = self.image_data
            self.render_dict["image_transform"] = self.image_transform

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
