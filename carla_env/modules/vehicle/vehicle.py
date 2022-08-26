from typing_extensions import Self
from carla_env.modules import module
import carla

class VehicleModule(module.Module):
	"""Concrete implementation of Module abstract base class for vehicle management"""
	def __init__(self, config, client) -> None:
		super().__init__()
		if config is None:
			self.set_default_config()
		else:
			self.config = config

		self.client = client
		self.world = self.client.get_world()
		self.blueprint = self.config[f"vehicle.{self.config['vehicle_model']}"][0]

			
	def _start(self, spawn_transform):
		"""Start the vehicle manager"""
		pass

	
	def step(self, control = None):
		"""Step the vehicle manager"""
		pass
	
	def _stop(self):
		"""Stop the vehicle manager"""
		pass

	def reset(self):
		"""Reset the vehicle manager"""
		pass
	
	def render(self):
		"""Render the vehicle manager"""
		pass

	def close(self):
		"""Close the vehicle manager"""
		pass
	def seed(self):
		"""Seed the vehicle manager"""
		pass
	
	def get_config(self):
		"""Get the config of the vehicle manager"""
		return self.config

	def set_default_config(self):
		"""Set the default config of the vehicle"""
		self.config = {"vehicle_model" : "lincoln.mkz2017"}
	
