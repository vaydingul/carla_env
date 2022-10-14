from carla_env.modules import module
import carla
import logging
import time
logger = logging.getLogger(__name__)


class TrafficManagerModule(module.Module):
	"""Concrete implementation Module abstract base class for traffic manager module"""

	def __init__(self, config, client) -> None:
		super().__init__()

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		self.client = client
		self.world = self.get_world()
		self.traffic_manager = self.client.get_trafficmanager(
			self.config["port"])

		self.render_dict = {}

		self.reset()

	def step(self):
		"""Step the client"""
		self._tick()

	def reset(self):
		"""Reset the client"""
		
		self.traffic_manager.set_synchronous_mode(
			self.config["synchronous_mode"])
		
		if self.config["vehicle_list"]:
			for vehicle in self.config["vehicle_list"]:
				vehicle.set_autopilot(True, self.traffic_manager.get_port())

	def render(self):
		"""Render the client"""
		pass

	def close(self):
		"""Close the client"""
		for vehicle in self.config["vehicle_list"]:
			vehicle.close()

	def seed(self):
		"""Seed the client"""
		pass

	def get_config(self):
		"""Get the config of the client"""
		return self.config

	def get_world(self):
		"""Get the world"""
		return self.client.get_world()

	def get_client(self):
		"""Get the client"""
		return self.client

	def _set_default_config(self):
		"""Set the default config of the client"""
		self.config = {
			"port": 8000,
			"synchronous_mode": True,
			"vehicle_list": [],
			"walker_list": [],
		}

	@property
	def spawn_transforms(self):
		"""Get all the spawn point in the map"""
		spawn_transforms = self.map.get_spawn_points()
		return spawn_transforms
