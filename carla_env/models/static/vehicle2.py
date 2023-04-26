from carla_env.models.dynamic import KinematicBicycleModel as kbm
from carla_env.models.dynamic import DynamicBicycleModel as dbm
from torch import nn


class KinematicBicycleModel(nn.Module):
    def __init__(self, config) -> None:
        super(KinematicBicycleModel).__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.model = kbm(config=None)
        # TODO: Set model parameters to config
        pass

    def forward(self, ego_state, action):
        return self.model(ego_state, action)

    def build_from_config(self):
        self.dt = self.config["dt"]
        self.front_wheelbase = self.config["front_wheelbase"]
        self.rear_wheelbase = self.config["rear_wheelbase"]
        self.steer_gain = self.config["steer_gain"]
        self.acceleration_gain = self.config["acceleration_gain"]

    def set_default_config(self):
        self.config = {
            "dt": 0.1,
            "front_wheelbase": 1.5,
            "rear_wheelbase": 1.5,
            "steer_gain": 1.0,
            "acceleration_gain": 1.0,
        }

    def append_config(self, config):
        self.config.update(config)


class DynamicBicycleModel(nn.Module):
    def __init__(self, config) -> None:
        super(DynamicBicycleModel).__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.model = dbm(config=None)

    def build_from_config(self):
        self.dt = self.config["dt"]
        self.front_wheelbase = self.config["front_wheelbase"]
        self.rear_wheelbase = self.config["rear_wheelbase"]
        # TODO: Set model parameters to config
