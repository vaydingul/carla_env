from abc import ABC, abstractmethod


class Adapter(ABC):
    
	@abstractmethod
	def __init__(self, config):
		pass
	
	@abstractmethod
	def get_action(self, state):
		pass
	
	@abstractmethod
	def train(self, state, action, reward, next_state, done):
		pass
	
	@abstractmethod
	def save(self, path):
		pass
	
	@abstractmethod
	def load(self, path):
		pass