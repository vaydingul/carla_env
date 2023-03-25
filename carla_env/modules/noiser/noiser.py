

class Noiser(object):
	"""
	This class defines the noiser callback that will be fed to CARLA World .on_tick function
	"""

	def __init__(self, config, client):
		super().__init__()

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		self.client = client
		self.world = self.client.get_world()

		self.render_dict = {}

		self.reset()
	

	def reset(self):
		"""Reset the noiser"""
		pass
		
		def callback(self, timestamp):
			"""Callback function to be fed to CARLA World .on_tick function"""
			pass

		return callback
	

	def _set_default_config(self):
		"""Set the default config of the noiser"""
		self.config = {}
	