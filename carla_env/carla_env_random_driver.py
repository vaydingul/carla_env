from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server as s
from carla_env.modules.client import client as c
from carla_env.modules.actor import actor as a
from carla_env.modules.vehicle import vehicle as v
from carla_env.modules.sensor import vehicle_sensor as vs
from carla_env.modules.sensor import rgb_sensor as rgbs
from carla_env.modules.sensor import collision_sensor as cs
# Import utils
import carla
import time
import numpy as np
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)
maps = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06"]

class RandomActionDesigner(object):
	
	def __init__(self, brake_probability = 0.03, max_throttle = 1.0, max_steering_angle = 1.0):
		self.brake_probability = brake_probability
		self.max_throttle = max_throttle
		self.max_steering_angle = max_steering_angle

	def step(self, t = None):
		
		
		# Randomize control
		if np.random.random() < self.brake_probability:
			throttle = 0.
			steer = 0.
			brake = 1.
		else:
			throttle = np.random.uniform(0, self.max_throttle)
			steer = np.random.uniform(-self.max_steering_angle, self.max_steering_angle)
			brake = 0.

		action = [throttle, steer, brake]
		logger.debug(f"Action: {action}")

		return action
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
		self.server = s.ServerModule(None)
		#Select a random map
		self.client = c.ClientModule(config = {"world":maps[np.random.randint(0, len(maps))]})


		self.first_time_step = True
		self.is_done = False
		self.counter = 0
		self.data = Queue()

		self.action_designer = RandomActionDesigner()

		self.reset()

	def reset(self):
		"""Reset the environment"""
		self.server.reset()
		self.client.reset()

		self.spectator = self.client.world.get_spectator()

		# Let's initialize a vehicle
		self.vehicle = v.VehicleModule(
			{"vehicle_model": "lincoln.mkz2017"}, self.client.client)

		# Make this vehicle actor
		self.actor = a.ActorModule(
			{"actor": self.vehicle, "hero": True}, self.client.client)

		self.vehicle_sensor = vs.VehicleSensorModule(
			None, self.client.client, self.actor)

		self.collision_sensor = cs.CollisionSensorModule(None, self.client.client, self.actor)



		self.rgb_sensor = rgbs.RGBSensorModule(
			None, self.client.client, self.actor)

		time.sleep(1.0)
		logger.info("Everything is set!")
		# self.actor.reset()
		# self.vehicle.reset()
		# self.vehicle_sensor.reset()
		# self.rgb_sensor.reset()

	def step(self, action=None):
		"""Perform an action in the environment"""

		

		
		self.server.step()
		self.client.step()

		snapshot = self.client.world.get_snapshot()
		t = snapshot.timestamp.elapsed_seconds
		action = self.action_designer.step(t)

		self.is_done = action is None or (self.counter > 5000)
		
		if self.is_done:
			return True

		self.actor.step(action)
		self.vehicle.step()
		self.vehicle_sensor.step()

		data_dict = {}

		for (k, v) in self.actor.sensor_dict.items():

			if v.queue.qsize() > 0:

				try:

					equivalent_frame_fetched = False

					while not equivalent_frame_fetched:

						data_ = v.queue.get(True, 10)

						equivalent_frame_fetched =  data_["frame"] == snapshot.frame #, f"Frame number mismatch: {data_['frame']} != {snapshot.frame} \n Current Sensor: {k} \n Current Data Queue Size {self.data.qsize()}"

				except Empty:
					print("Empty")

				data_dict[k] = data_

				if k == "VehicleSensorModule":

					ego_transform = data_dict[k]["transform"]
					transform = ego_transform
					transform.location.z += 2.0

					if self.first_time_step:
						self.initial_vehicle_transform = ego_transform
						self.first_time_step = False
				
				elif k == "CollisionSensorModule":

					impulse = data_dict[k]["impulse"]
					impulse_amplitude = np.linalg.norm(impulse)
					logger.debug(f"Collision impulse: {impulse_amplitude}")
					if impulse_amplitude > 1:
						self.is_done = True

		data_dict["snapshot"] = snapshot

		self.data.put(data_dict)

		self.spectator.set_transform(transform)

		self.counter += 1

		return False

	def render(self):
		"""Render the environment"""
		pass

	def close(self):
		"""Close the environment"""
		self.vehicle.close()
		self.actor.close()
		self.client.close()
		self.server.close()

	def seed(self, seed):
		"""Set the seed for the environment"""
		pass

	def get_config(self):
		"""Get the config of the environment"""
		return self.config

	def _set_default_config(self):
		"""Set the default config of the environment"""
		self.config = {}
