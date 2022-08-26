from carla_env.modules import module


import carla


class SensorModule(module.Module):
	"""Concrete implementation of Module abstract base class for sensor management"""
	def __init__(self, config) -> None:
		super().__init__()
		for k in config.keys():
    self.config[k] = config[k]
		self.sensor_dict = {}
		self.client = None
		self.world = None
		self.map = None
	





if __name__ == "__main__":

	pass