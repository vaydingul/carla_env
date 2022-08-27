from carla_env.modules import module


import carla


class SensorModule(module.Module):
	"""Concrete implementation of Module abstract base class for sensor management"""
	def __init__(self, config, client) -> None:
		super().__init__()

		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]
		self.sensor_dict = {}
		self.client = client
		self.world = self.client.get_world()
		self.map = self.world.get_map()
		
	

	def _start(self):
		"""Start the sensor module"""
		self.client = self.config["client"]
		self.world = self.client.get_world()
		self.map = self.world.get_map()
		self.sensor_dict = self.config["sensor_dict"]
		self.sensor_dict["client"] = self.client
		self.sensor_dict["world"] = self.world
		self.sensor_dict["map"] = self.map
		self.sensor_dict["config"] = self.config
		self.sensor_dict["sensor_module"] = self
		for k in self.sensor_dict.keys():
			self.sensor_dict[k]._start()
		self.started = True

	def _stop(self):
		"""Stop the sensor module"""
		for k in self.sensor_dict.keys():
			self.sensor_dict[k]._stop()
		self.started = False

	def _get_sensor_data(self):
		"""Get the sensor data"""
		for k in self.sensor_dict.keys():
			self.sensor_dict[k]._get_sensor_data()

	def step(self):
		"""Step the sensor"""
		self._tick()



	def reset(self):
		"""Reset the sensor"""
		pass

	def render(self):
		"""Render the sensor"""
		pass
	
	def close(self):
		"""Close the sensor"""
		pass

	def seed(self):
		"""Seed the sensor"""
		pass

	def get_config(self):
		"""Get the config of the sensor"""
		return self.config
	
	def _set_default_config(self):
		"""Set the default config of the sensor"""
		self.config = {}


