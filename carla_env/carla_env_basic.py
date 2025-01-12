from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server as s
from carla_env.modules.client import client as c
from carla_env.modules.actor import actor as a
from carla_env.modules.vehicle import vehicle as v
from carla_env.modules.sensor import vehicle_sensor as vs
from carla_env.modules.sensor import rgb_sensor as rgbs

# Import utils
import carla
import time
import numpy as np
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)


class ActionDesigner(object):
    @classmethod
    def step(cls, t=None):

        if t < 5:

            action = [0.0, 0.0, 0.0]

        elif t > 5 and t < 10:

            action = [1.0, 0.0, 0.0]

        elif t > 10 and t < 10.5:

            action = [0.0, -0.1, 0.0]

        elif t > 10.5 and t < 13.5:

            action = [0.1, 0.0, 0.0]

        elif t > 13.5 and t < 14:

            action = [0.0, 0.1, 0.0]

        elif t > 14 and t < 16:

            action = [0.0, 0.0, 1.0]

        else:

            action = None

        return action


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
        self.client = c.ClientModule(None)

        self.first_time_step = True
        self.is_done = False
        self.data = Queue()

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.server.reset()
        self.client.reset()

        self.spectator = self.client.world.get_spectator()

        # Let's initialize a vehicle
        self.vehicle_module = v.VehicleModule(
            config={"vehicle_model": "lincoln.mkz_2017"},
            client=self.client.get_client(),
        )
        # Make this vehicle actor
        self.actor_module = a.ActorModule(
            config={
                "actor": self.vehicle_module,
                "hero": True,
            },
            client=self.client.get_client(),
        )

        self.vehicle_sensor = vs.VehicleSensorModule(
            None, self.client.get_client(), self.actor_module, id="ego"
        )

        self.rgb_sensor = rgbs.RGBSensorModule(
            None, self.client.get_client(), self.actor_module, id="rgb"
        )

        time.sleep(1.0)
        logger.info("Everything is set!")
        # self.actor.reset()
        # self.vehicle.reset()
        # self.vehicle_sensor.reset()
        # self.rgb_sensor.reset()

    def step(self, action=None):
        """Perform an action in the environment"""

        self.server.step()
        self.client.step()

        snapshot = self.client.world.get_snapshot()
        t = snapshot.timestamp.elapsed_seconds
        action = ActionDesigner.step(t)

        self.is_done = action is None

        if self.is_done:
            return True

        self.actor_module.step(action)
        self.vehicle_module.step()
        self.vehicle_sensor.step()

        data_dict = {}

        for (k, v) in self.actor_module.sensor_dict.items():

            if v.get_queue().qsize() > 0:

                try:

                    equivalent_frame_fetched = False

                    while not equivalent_frame_fetched:

                        data_ = v.get_queue().get(True, 10)

                        # , f"Frame number mismatch: {data_['frame']} != {snapshot.frame} \n Current Sensor: {k} \n Current Data Queue Size {self.data.qsize()}"
                        equivalent_frame_fetched = data_["frame"] == snapshot.frame

                except Empty:
                    print("Empty")

                data_dict[k] = data_

                if k == "ego":

                    ego_transform = data_dict[k]["transform"]
                    transform = ego_transform
                    transform.location.z += 2.0

                    if self.first_time_step:
                        self.initial_vehicle_transform = ego_transform
                        self.first_time_step = False

        data_dict["snapshot"] = snapshot

        self.data.put(data_dict)

        self.spectator.set_transform(transform)

        return False

    def render(self):
        """Render the environment"""
        pass

    def close(self):
        """Close the environment"""
        self.vehicle_module.close()
        self.actor_module.close()
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
