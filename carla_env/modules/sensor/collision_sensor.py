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

	def __init__(self, config, client, actor=None) -> None:
		super().__init__(config, client)

		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]
		self.sensor_dict = {}
		self.client = client
		self.world = self.client.get_world()
		self.map = self.world.get_map()

		if actor is not None:
			self.attach_to_actor(actor)

		self.reset()

	def _start(self):
		"""Start the sensor module"""

		self.queue = Queue()

		collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')


		self.collision_sensor = self.world.spawn_actor(
			collision_bp, carla.Transform(), attach_to=self.actor.player)

		self.collision_sensor.listen(lambda collision_data: self._get_sensor_data(collision_data))

	def _stop(self):
		"""Stop the sensor module"""
		self.collision_sensor.destroy()

	def _tick(self):
		"""Tick the sensor"""
		pass

	def _get_sensor_data(self, collision_data):
		"""Get the sensor data"""

		#logger.info("Received an image of frame: " + str(image.frame))


		impulse = collision_data.normal_impulse
		impulse = np.array([impulse.x, impulse.y, impulse.z])
		impulse = copy.deepcopy(impulse)


		data = {'frame': collision_data.frame,
				'transform': collision_data.transform,
				'impulse': impulse
				}
		self._queue_operation(data)

	def step(self):
		"""Step the sensor"""
		self._tick()
		
	def reset(self):
		"""Reset the sensor"""
		self._start()

	def render(self):
		"""Render the sensor"""
		pass

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
