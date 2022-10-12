from carla_env.modules.sensor import sensor
from queue import Queue, Empty
import carla
import copy
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

OTHER_MAP = [1, 2, 5, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


class SemanticSensorModule(sensor.SensorModule):
    """Concrete implementation of SensorModule abstract base class for semantic sensor management"""

    def __init__(self, config, client, actor=None) -> None:
        super().__init__(config, client)

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]
        
        self.image_data = None

        if actor is not None:
            self.attach_to_actor(actor)

        self.reset()

    def _start(self):
        """Start the sensor module"""

        semantic_bp = self.world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('fov', str(50))
        self.camera_transform = carla.Transform(
            carla.Location(
                z=50), carla.Rotation(
                pitch=-90))

        self.camera = self.world.spawn_actor(
            semantic_bp, self.camera_transform, attach_to=self.actor.get_actor())

        self.camera.listen(lambda image: self._get_sensor_data(image))

    def _stop(self):
        """Stop the sensor module"""
        self.camera.destroy()

    def _tick(self):
        """Tick the sensor"""
        pass

    def _get_sensor_data(self, image):
        """Get the sensor data"""

        image_data_transformed, image_data = self._image_to_basic_palette(
            image)

        self.image_data = image_data_transformed
        self.image_transform = image.transform

        data = {'frame': image.frame,
                'transform': image.transform,
                'data': image_data_transformed
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

    def _to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""

        if not isinstance(image, carla.Image):
            raise ValueError("Argument must be a carla.sensor.Image")
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def _to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self._to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def _labels_to_array(self, image):
        """
        Convert an image containing CARLA semantic segmentation labels to a 2D array
        containing the label of each pixel.
        """
        return self._to_bgra_array(image)[:, :, 2]

    def _image_to_basic_palette(self, image):
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes = {
            0: [0, 0, 0],         # None
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            10: [0, 0, 255],      # Vehicles
        }
        array = self._labels_to_array(image)
        result = np.zeros((array.shape[0], array.shape[1], 3))
        for key, value in classes.items():
            result[np.where(array == key)] = value
        return result, array
