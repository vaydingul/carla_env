from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server as s
from carla_env.modules.client import client as c
from carla_env.modules.actor import actor as a
from carla_env.modules.vehicle import vehicle as v
from carla_env.modules.sensor import vehicle_sensor as vs
from carla_env.modules.sensor import rgb_sensor as rgbs
from carla_env.modules.sensor import collision_sensor as cs
from carla_env.modules.route import route as r
# Import utils
import carla
import time
import numpy as np
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)


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
		self.client = c.ClientModule(None)

		self.render_dict = {}

		self.first_time_step = True
		self.is_done = False
		self.counter = 0
		self.data = Queue()
		
		self.reset()

	def reset(self):
		"""Reset the environment"""
		self.server.reset()
		self.client.reset()	

		self.world = self.client.world
		self.map = self.world.get_map()

		self.spectator = self.client.world.get_spectator()

		# Fetch all spawn points
		spawn_points = self.client.world.get_map().get_spawn_points()
		# Select two random spawn points
		start_end_spawn_point = np.random.choice(spawn_points, 2)
		start = start_end_spawn_point[0]
		end = start_end_spawn_point[1]

		self.route = r.RouteModule({"start": start, "end": end, "sampling_resolution" : 0.5, "debug": True}, self.client.client)
		# Let's initialize a vehicle
		self.vehicle = v.VehicleModule(
			{"vehicle_model": "lincoln.mkz2017"}, self.client.client)
		# Make this vehicle actor
		self.actor = a.ActorModule(
			{"actor": self.vehicle, "hero": True, "selected_spawn_point": start}, self.client.client)
		

		# Sensor suite
		self.vehicle_sensor = vs.VehicleSensorModule(
			None, self.client.client, self.actor)
		self.collision_sensor = cs.CollisionSensorModule(None, self.client.client, self.actor)
		self.rgb_sensor = rgbs.RGBSensorModule(
			None, self.client.client, self.actor)






		time.sleep(1.0)
		logger.info("Everything is set!")

		for _ in range(int(1/self.client.config["fixed_delta_seconds"]) * 2):
			self.client.step()
	

	def step(self, action=None):
		"""Perform an action in the environment"""

	
		snapshot = self.client.world.get_snapshot()
	

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

					current_transform = data_dict[k]["transform"]
					current_velocity = data_dict[k]["velocity"]

					transform = current_transform
					transform.location.z += 2.0

					if self.first_time_step:
						self.initial_vehicle_transform = current_transform
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

		self.is_done = False
		
		target_waypoint, _ = self.route.step(self.map.get_waypoint(current_transform.location))
		# target_waypoint, _ = self.route.step(current_transform)
		self.server.step()
		self.client.step()

		self.counter += 1

		return current_transform, current_velocity, target_waypoint

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
