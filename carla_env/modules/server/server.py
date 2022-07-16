import subprocess
import os

from carla_env.modules import module

CARLA_ROOT = os.getenv("CARLA_ROOT")


class ServerModule(module.Module):
	"""Concrete implementation of Module abstract base class for server module"""

	def __init__(self, config) -> None:
		super().__init__()
		self.config = config
		self.carla_exec = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
		self.command = [self.carla_exec, "-carla-server"]

		if config["quality"] is not None:
			self.command += ["-quality-level", config["quality"]]
		else:
			self.command += ["-quality-level", "epic"]

		if config["port"] is not None:
			self.command += ["-carla-rpc-port", config["port"]]
		else:
			self.command += ["-carla-rpc-port", "2000"]

		if config["no_screen"]:
			self.command = "".join(self.command)
		

		
	def _start(self):
		"""Start the server"""
		
		self.process = subprocess.Popen(self.command, shell = True)
		

	def _stop(self):
		"""Kill the server"""

		self.process.kill()
		self.process.wait()
		self.process = None

	def _is_running(self):
		"""Check if the server is running"""
		return self.process.poll() is None

	def _get_status(self):
		"""Get the status of the server"""
		return self.process.poll()


	def reset(self):
		"""Reset the module"""
		pass
	
	def step(self, action):
		"""Perform an action in the module"""
		pass
	
	def render(self):
		"""Render the module"""
		pass
	
	def close(self):
		"""Close the module"""
		pass
	
	def seed(self, seed):
		"""Set the seed for the module"""
		pass
	
	
	def get_config(self):
		"""Get the config of the module"""
		pass


if __name__ == "__main__":

	config = {"quality": "epic", "port": "2000", "no_screen": False}
	server = ServerModule(config)
	server._start()
	pass



