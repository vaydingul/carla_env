import subprocess
import os
import signal
import time

from carla_env.modules import module

CARLA_ROOT = os.getenv("CARLA_ROOT")

class ServerModule(module.Module):
	"""Concrete implementation of Module abstract base class for server module"""

	def __init__(self, config) -> None:
		super().__init__()

		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]

		self.carla_exec = os.path.join(CARLA_ROOT, "CarlaUE4.sh") # Path to the carla executable
		self.is_running = None # Boolean to check if the server is running
		self.render_dict = {} # Dictionary to store the render information

	def _generate_command(self):
		"""Generate the command to start the server based on the config file"""

		self.command = [self.carla_exec, "-carla-server"]

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


	def _is_running(self):
			"""Check if the server is running"""
			return self.process.poll() is None


	def _start(self):
		"""Start the server"""
		
		self._generate_command()
		self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
		

	def _stop(self):
		"""Kill the server"""

		# self.process.terminate()
		os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
		
	def reset(self):
		"""Reset the module"""
		self.stop()
		time.sleep(3.0)
		self.start()

	
	def step(self, action):
		"""Perform an action in the module"""
		self.is_running = self._is_running()
	
	def render(self):
		"""Render the module"""
		self.render_dict["is_running"] = self.is_running
	
	def close(self):
		"""Close the module"""
		self.stop()
		
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

if __name__ == "__main__":

	config = {"quality": "epic", "port": "2000", "no_screen": False}
	server = ServerModule(config)
	server._start()
	time.sleep(10.0)
	print("Stopping....")
	server._stop()
	pass



