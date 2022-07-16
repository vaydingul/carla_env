import abc

class Module(abc.ABC):
	"""Abstract base class for modules"""
	
	@abc.abstractmethod
	def reset(self):
		"""Reset the module"""
		pass
	
	@abc.abstractmethod
	def step(self, action):
		"""Perform an action in the module"""
		pass
	
	@abc.abstractmethod
	def render(self):
		"""Render the module"""
		pass
	
	@abc.abstractmethod
	def close(self):
		"""Close the module"""
		pass
	
	@abc.abstractmethod
	def seed(self, seed):
		"""Set the seed for the module"""
		pass
	
	
	
	@abc.abstractmethod
	def get_config(self):
		"""Get the config of the module"""
		pass
