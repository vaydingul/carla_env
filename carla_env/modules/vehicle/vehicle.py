from carla_env.modules import module
import carla

class VehicleModule(module.Module):
	"""Concrete implementation of Module abstract base class for vehicle management"""
	def __init__(self, config, client) -> None:
		super().__init__()
		self.client = client

		
		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

			
		self.world = self.client.get_world()
		self.blueprint = self.world.get_blueprint_library().filter(f"vehicle.{self.config['vehicle_model']}")[0]

		self.render_dict = {}
			
	def _start(self, spawn_transform):
		"""Start the vehicle manager"""
		pass

	
	def step(self):
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
		self.render_dict["vehicle_model"] = self.config['vehicle_model']
		return self.render_dict
		
	def close(self):
		"""Close the vehicle manager"""
		pass
	def seed(self):
		"""Seed the vehicle manager"""
		pass
	
	def get_config(self):
		"""Get the config of the vehicle manager"""
		return self.config

	def _set_default_config(self):
		"""Set the default config of the vehicle"""
		self.config = {"vehicle_model" : "lincoln.mkz2017"}
	
