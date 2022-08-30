from pickle import FALSE
from carla_env.modules import module
from carla_env.modules.vehicle import vehicle
import carla
import numpy as np

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
		self.sensor_dict = {}
		self.last_applied_control = [0, 0, 0]
		self.reset()

	def _start(self):
		"""Start the actor manager"""
		
		not_spawned = True

		while not_spawned:

			selected_spawn_point = np.random.choice(self.world.get_map().get_spawn_points())
			
			self.player = self.world.try_spawn_actor(self.actor.blueprint, selected_spawn_point)
			
			not_spawned = self.player is None
	

	def step(self, action = None):
		"""Step the actor manager"""
		# for sensor in self.sensor_dict.values():
		# 	sensor.step()

		if self.hero and action is not None:

			if type(action) == list:
				
				vehicle_control = carla.VehicleControl(throttle = action[0], steer = action[1], brake = action[2])
				self.player.apply_control(vehicle_control)
				self.last_applied_control = action

	
	def _stop(self):
		"""Stop the actor manager"""
		self.player.destroy()
	
	def reset(self):
		"""Reset the actor manager"""
		self._start()
	
	def render(self):
		"""Render the actor manager"""
		self.render_dict["id"] = self.actor.id
		self.render_dict["transform"] = self.actor.get_transform()
		self.render_dict["velocity"] = self.actor.get_velocity()
		self.render_dict["location"] = self.actor.get_location()

	def close(self):
		"""Close the actor manager"""
		self._stop()
		

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