from carla_env.modules.noiser.noiser import Noiser
import numpy as np
import carla
from utils.log_utils import get_logger


logger = get_logger(__name__)


class GaussianNoiser(Noiser):
	def __init__(self, config, client, actor):
		super().__init__(config, client)
		self._set_default_config()
		if config is not None:
			for k in config.keys():
				self.config[k] = config[k]
		self.client = client
		self.actor = actor
		self.world = self.client.get_world()
		self.render_dict = {}
		self.reset()

	def reset(self):
		"""Reset the noiser"""
		self.probability = self.config["probability"]
		self.mean_acceleration = self.config["mean_acceleration"]
		self.std_acceleration = self.config["std_acceleration"]
		self.mean_steer = self.config["mean_steer"]
		self.std_steer = self.config["std_steer"]

		self.world.on_tick(lambda snapshot: self.callback(self.actor, snapshot))

	def callback(self, actor, snapshot):

		if np.random.rand() < self.probability:

			# Get control from hero actor
			actor.set_autopilot(False)

			logger.debug(actor.attributes)

			logger.debug(f"Frame: {snapshot.frame}")

			action = actor.get_control()

			logger.debug(f"Throttle: {action.throttle}")
			logger.debug(f"Steer: {action.steer}")
			logger.debug(f"Brake: {action.brake}")

			# Sample additive actions from Gaussian distribution
			acceleration = np.random.normal(
				self.mean_acceleration, self.std_acceleration
			)
			steer = np.random.normal(self.mean_steer, self.std_steer)

			logger.debug(f"Acceleration Noise: {acceleration}")
			logger.debug(f"Steer Noise: {steer}")

			action.steer += steer

			if (action.throttle * acceleration) > 0:
				action.throttle += acceleration

			if (action.brake * acceleration) > 0:
				action.brake += -acceleration

			actor.apply_control(action)

			logger.debug(f"Noisy Throttle: {action.throttle}")
			logger.debug(f"Noisy Steer: {action.steer}")
			logger.debug(f"Noisy Brake: {action.brake}")

			# actor.add_impulse(carla.Vector3D(0, 0, 10000))

		else:

			actor.set_autopilot(True)

	def _set_default_config(self):
		"""Set the default config of the noiser"""
		self.config = {
			"probability": 0.05,
			"mean_acceleration": 0,
			"std_acceleration": 0.1,
			"mean_steer": 0,
			"std_steer": 0.1,
			"seed": 0,
		}

	def get_config(self):
		"""Get the config of the noiser"""
		return self.config
