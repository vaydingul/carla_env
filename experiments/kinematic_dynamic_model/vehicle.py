import numpy as np

class Vehicle:
	"""
	Define a kinematic bicycle model
	"""
	def __init__(self, x, y, yaw, v, L=2.5):
		
		self.x = x
		self.y = y
		self.yaw = yaw
		self.v = v
		self.L = L
		
		self.state = np.array([x, y, yaw, v])
	
		self.dt = 0.1
	
		

	def step(self, a, delta):
		self.state = self.state + np.array([self.v*np.cos(self.yaw), self.v*np.sin(self.yaw), self.v*np.tan(delta), a])*self.dt
		self.x = self.state[0]
		self.y = self.state[1]
		self.yaw = self.state[2]
		self.v = self.state[3]
		self.state = np.array([self.x, self.y, self.yaw, self.v])
		return self.state