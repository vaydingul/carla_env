import subprocess
import os
import signal
import time
import logging
from carla_env.modules import module

CARLA_ROOT = os.getenv("CARLA_ROOT")
CARLA_EXECUTABLE = os.path.join(CARLA_ROOT, "CarlaUE4.sh") # Path to the carla executable
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

logger = logging.getLogger(__name__)
class ServerModule(module.Module):
	"""Concrete implementation of Module abstract base class for server module"""

	def __init__(self, config) -> None:
		super().__init__()

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		self._is_running = False # Boolean to check if the server is running
		self.render_dict = {} # Dictionary to store the render information

		self.reset()

	def _generate_command(self):
		"""Generate the command to start the server based on the config file"""

		self.command = [CARLA_EXECUTABLE, "-carla-server"]

		if self.config["quality"]  is not None:
			self.command += ["-quality-level", config["quality"]]
		else:
			self.command += ["-quality-level", "epic"]

		if self.config["port"]  is not None:
			self.command += ["-carla-rpc-port", config["port"]]
		else:
			self.command += ["-carla-rpc-port", "2000"]

		self.command += ["-vulkan"]

		if self.config["no_screen"] :
			self.command = "".join(self.command)

	@property
	def is_running(self):
		"""Check if the server is running"""

		if hasattr(self, 'process'):
			self._is_running = self.process.poll() is None
		else:
			self._is_running = False

		return self._is_running
		

	def _start(self):
		"""Start the server"""
		
		self._generate_command()
		self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
		logger.info("Server started")
		time.sleep(5.0)

	def _stop(self):
		"""Kill the server"""

		# self.process.terminate()
		os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
		logger.info("Server stopped")
		time.sleep(5.0)


	def reset(self):
		"""Reset the module"""
		if self.is_running:
			self._stop()
		
		self._start()
		logger.info("Server reset")
	
	def step(self):
		"""Perform an action in the module"""
		pass
	
	def render(self):
		"""Render the module"""
		self.render_dict["is_running"] = self.is_running

		return self.render_dict
	
	def close(self):
		"""Close the module"""
		self._stop()
		time.sleep(5.0)
		
	def seed(self, seed):
		"""Set the seed for the module"""
		raise(NotImplementedError)
	
	
	def get_config(self):
		"""Get the config of the module"""
		return self.config

	def _set_default_config(self):
		"""Set the default config for the module"""
		self.config = {"quality":None,
		"port": None,
		"no_screen": False}





