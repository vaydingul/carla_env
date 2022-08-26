from environment import Environment
from modules.sensor.sensor import SensorModule

class CarlaEnvironment(Environment):
	"""Concrete implementation of Environment abstract base class"""
	
	def __init__(self, config):
		"""Initialize the environment"""
		super().__init__(config)
		self.config = config
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

