from carla_env.modules import module
import carla

class ClientModule(module.Module):
	"""Concrete implementation Module abstract base class for client module"""

	def __init__(self, config) -> None:
		super().__init__()

		if config is None:
			self.set_default_config()
		else:
			self.config = config
		self.render_dict = {}
		
	def _tick(self):
		self.frame_id = self.world.tick()

	def _start(self):
		"""Start the client"""
		self.client = carla.Client(self.config["host"], self.config["port"])
		self.client.set_timeout(self.config["timeout"])
		
		self.world = self.client.load_world(self.config["world"])
		
		self.blueprint_library = self.world.get_blueprint_library()

		self.settings = self.world.get_settings()
		self.settings.synchronous_mode = self.config["synchronous_mode"]
		self.settings.fixed_delta_seconds = self.config["fixed_delta_seconds"]
		self.world.apply_settings(self.settings)


		self._tick()

	def step(self):
		"""Step the client"""
		self._tick()


	def _stop(self):
		"""Stop the client"""
		pass

	def reset(self):
		"""Reset the client"""
		self.client.reload_world()
		pass

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
	
	def set_default_config(self):
		"""Set the default config of the client"""
		self.config = {
			"host": "localhost",
			"port": 2000,
			"timeout": 10.0,
			"world": "Town01",
			"synchronous_mode": False,
			"fixed_delta_seconds": 0.01
		}
		