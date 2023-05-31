from abc import ABC, abstractmethod


class Adapter(ABC):
    
	@abstractmethod
	def __init__(self, config):
		pass
	
	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def step(self, action):
		pass

	@abstractmethod
	def get_state(self):
		pass

	@abstractmethod
	def get_action(self, state):
		pass
	
	