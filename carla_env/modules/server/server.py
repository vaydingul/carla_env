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
		self.config = config
		self.carla_exec = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
		self.is_running = None
		self.render_dict = {}

	def _generate_command(self):
		"""Generate the command to start the server based on the config file"""

		self.command = [self.carla_exec, "-carla-server"]

		if config["quality"] is not None:
			self.command += ["-quality-level", config["quality"]]
		else:
			self.command += ["-quality-level", "epic"]

		if config["port"] is not None:
			self.command += ["-carla-rpc-port", config["port"]]
		else:
			self.command += ["-carla-rpc-port", "2000"]

		self.command += ["-opengl"]

		if config["no_screen"]:
			self.command = "".join(self.command)


	def _is_running(self):
			"""Check if the server is running"""
			return self.process.poll() is None


	def start(self):
		"""Start the server"""
		
		self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE)
		

	def stop(self):
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
		

	def seed(self, seed):
		"""Set the seed for the module"""
		raise(NotImplementedError)
	
	
	def get_config(self):
		"""Get the config of the module"""
		return self.config


if __name__ == "__main__":

	config = {"quality": "epic", "port": "2000", "no_screen": False}
	server = ServerModule(config)
	server._start()
	time.sleep(10.0)
	print("Stopping....")
	server._stop()
	pass



