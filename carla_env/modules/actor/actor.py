from carla_env.modules import module
from carla_env.modules.vehicle import vehicle
import carla

class ActorModule(module.Module):
	"""Concrete implementation of Module abstract base class for actor management"""
	def __init__(self, config, client) -> None:
		super().__init__()
		self.client = client

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		self.actor = self.config["actor"]
		self.world = self.client.get_world()
		self.hero = self.config["hero"]
		self.render_dict = {}
	
	def _start(self, spawn_transform):
		"""Start the actor manager"""
		self.player = self.world.try_spawn_actor(self.actor.blueprint, spawn_transform)

	
	def step(self, control = None):
		"""Step the actor manager"""
		if self.hero and control is not None:
			vehicle_control = carla.VehicleControl(control)
			self.actor.apply_control(vehicle_control)

	
	def _stop(self):
		"""Stop the actor manager"""
		self.actor.destroy()
	
	def reset(self):
		"""Reset the actor manager"""
		pass
	
	def render(self):
		"""Render the actor manager"""
		self.render_dict["id"] = self.actor.id
		self.render_dict["transform"] = self.actor.get_transform()
		self.render_dict["velocity"] = self.actor.get_velocity()
		self.render_dict["location"] = Self.actor.get_location()

	def close(self):
		"""Close the actor manager"""
		pass

	def seed(self):
		"""Seed the actor manager"""
		pass
	
	def get_config(self):
		"""Get the config of the actor manager"""
		return self.config
	
	def _set_default_config(self):
		"""Set the default config of actor manager"""
		self.config = {"actor" : vehicle.VehicleModule(None, self.client), 
		"hero" : True}