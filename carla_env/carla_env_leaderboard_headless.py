from carla_env.environment import Environment
from agents.navigation.local_planner import RoadOption

# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.module import Module
from carla_env.bev import BirdViewProducer

# Import utils
import time
import numpy as np
import cv2
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)


class CarlaEnvironment(Environment):
    """Concrete implementation of Environment abstract base class"""

    def __init__(self, config):
        """Initialize the environment"""
        super().__init__()

        self._set_default_config()
        self.config.update(config)
        self.build_from_config()

        # We have our server and client up and running
        self.server_module = server.ServerModule(config=None)

        self.counter = 0

        self.data = Queue()
        self.data_dict = {}
        self.reset()

    def build_from_config(self):

        pass

    def reset(self):
        """Reset the environment"""
        # self.server_module.reset()

        # Select a random task
        self.client_module = client.ClientModule(config=None)

        self.world = self.client_module.get_world()
        self.map = self.client_module.get_map()
        self.client = self.client_module.get_client()

        self.vehicle_module = vehicle.VehicleModule(
            config={"vehicle_model": "lincoln.mkz_2017"}, client=self.client
        )
        # Make this vehicle actor
        self.hero_actor_module = actor.ActorModule(
            config={
                "actor": self.vehicle_module,
                "hero": True,  # self.random,
            },
            client=self.client,
        )

        time.sleep(1.0)
        logger.info("Everything is set!")

        # for _ in range(int((1 / self.fixed_delta_seconds))):
        #     self.client_module.step()

    def step(self, action=None):
        """Perform an action in the environment"""

        pass

    def render(self, bev, control, occupancy):
        """Render the environment"""
        self.generate_sensor_dict()

        self.canvas = np.zeros((bev.shape[0], 800, 3), dtype=np.uint8)
        self.canvas[:, :, :] = 255

        bev = cv2.cvtColor(
            BirdViewProducer.as_rgb(
                bev,
            ),
            cv2.COLOR_BGR2RGB,
        )

        self.canvas[0 : bev.shape[0], 0 : bev.shape[1], :] = bev

        cv2.imshow("Canvas", self.canvas)
        cv2.waitKey(1)

    def close(self):
        """Close the environment"""

        self.client_module.close()
        self.server_module.close()
        logger.info("Environment is closed")

    def seed(self, seed):
        """Set the seed for the environment"""
        pass

    def get_config(self):
        """Get the config of the environment"""
        return self.config

    def get_world(self):
        """Get the world of the environment"""
        return self.world

    def get_map(self):
        """Get the map of the environment"""
        return self.map

    def get_client(self):
        """Get the client of the environment"""
        return self.client

    def get_hero_actor(self):
        """Get the hero actor of the environment"""
        return self.hero_actor_module.get_actor()

    def get_data(self):
        """Get the data of the environment"""
        return self.data.get()

    def get_counter(self):
        """Get the counter of the environment"""
        return self.counter

    def set_hero_actor(self, hero_actor):
        """Set the hero actor of the environment"""
        self.hero_actor_module.set_actor(hero_actor)

    def generate_sensor_dict(self):
        """Generate a sensor dict from the sensor config"""
        self.render_dict = {}
        for (k, v) in self.__dict__.items():
            if isinstance(v, Module):
                self.render_dict[k] = v.render()

    def _set_default_config(self):
        """Set the default config of the environment"""
        self.config = {}
