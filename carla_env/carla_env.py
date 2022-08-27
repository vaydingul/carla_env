from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server as s
from carla_env.modules.client import client as c
from carla_env.modules.actor import actor as a
from carla_env.modules.vehicle import vehicle as v

# Import utils
import time
import numpy as np
class CarlaEnvironment(Environment):
	"""Concrete implementation of Environment abstract base class"""

	
	def __init__(self, config):
		"""Initialize the environment"""
		super().__init__()

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		# We have our server and client up and running
		server = s.ServerModule(None)
		server._start()

		time.sleep(5.0)

		client = c.ClientModule(None)
		client._start()

		self.spectator = client.world.get_spectator()

		# Let's initialize a vehicle

		vehicle = v.VehicleModule(
			{"vehicle_model": "lincoln.mkz2017"}, client.client)

		# Make this vehicle actor

		actor = a.ActorModule({"actor": vehicle, "hero": True}, client.client)

		spawn_points = client._get_spawn_points()
		selected_spawn_point = np.random.choice(spawn_points)
		actor._start(selected_spawn_point)
		self.spectator.set_transform(selected_spawn_point)
	

		self.reset()

	def reset(self):
		"""Reset the environment"""
		pass

	def step(self, action):
		"""Perform an action in the environment"""
		pass

	def render(self):
		"""Render the environment"""
		pass

	def close(self):
		"""Close the environment"""
		pass

	def seed(self, seed):
		"""Set the seed for the environment"""
		pass

	def get_state(self):
		"""Get the state of the environment"""
		pass

	def get_action_space(self):
		"""Get the action space of the environment"""
		pass

	def get_observation_space(self):
		"""Get the observation space of the environment"""
		pass

	def get_config(self):
		"""Get the config of the environment"""
		return self.config

	def _set_default_config(self):
		"""Set the default config of the environment"""
		self.config = {}