from carla_env.modules import module
import carla
import logging
logger = logging.getLogger(__name__)
class ClientModule(module.Module):
	"""Concrete implementation Module abstract base class for client module"""

	def __init__(self, config) -> None:
		super().__init__()

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		self.render_dict = {}

		self.reset()

	def _tick(self):
		self.frame_id = self.world.tick()

	def _start(self):
		"""Start the client"""
		
				
		self.client = carla.Client(self.config["host"], self.config["port"])
		self.client.set_timeout(self.config["timeout"])
				
		self.world = self.client.load_world(self.config["world"])
		self.map = self.world.get_map()
		
		self.blueprint_library = self.world.get_blueprint_library()

		self.settings = self.world.get_settings()
		self.settings.synchronous_mode = self.config["synchronous_mode"]
		self.settings.fixed_delta_seconds = self.config["fixed_delta_seconds"]
		self.world.apply_settings(self.settings)
		logger.info("Client started")

		self._tick()

	def step(self):
		"""Step the client"""
		self._tick()


	def _stop(self):
		"""Stop the client"""
		pass

	def reset(self):
		"""Reset the client"""
		self._start()
		

	def render(self):
		"""Render the client"""
		pass
	
	def close(self):
		"""Close the client"""
		pass

	def seed(self):
		"""Seed the client"""
		pass

	def get_config(self):
		"""Get the config of the client"""
		return self.config
	
	def _set_default_config(self):
		"""Set the default config of the client"""
		self.config = {
			"host": "localhost",
			"port": 2000,
			"timeout": 10.0,
			"world": "Town01",
			"synchronous_mode": True,
			"fixed_delta_seconds": 0.01
		}

	@property
	def spawn_transforms(self):
		"""Get all the spawn point in the map"""
		spawn_transforms = self.map.get_spawn_points()
		return spawn_transforms
