from asyncio.log import logger
from carla_env.modules import module
from carla_env.modules.vehicle import vehicle
import carla
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActorModule(module.Module):
    """Concrete implementation of Module abstract base class for actor management"""

    def __init__(self, config, client) -> None:
        super().__init__()
        self.client = client

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.vehicle = self.config["vehicle"]
        self.world = self.client.get_world()
        self.hero = self.config["hero"]
        self.spawned = False
        self.render_dict = {}
        self.sensor_dict = {}

        self.reset()

    def _start(self):
        """Start the actor manager"""

        while not self.spawned:

            if "selected_spawn_point" in self.config.keys():

                selected_spawn_point = self.config["selected_spawn_point"]

            else:

                selected_spawn_point = np.random.choice(
                    self.world.get_map().get_spawn_points())

            self.actor = self.world.try_spawn_actor(
                self.vehicle.blueprint, selected_spawn_point)

            self.spawned = self.actor is not None

    def step(self, action=None):
        """Step the actor manager"""
        # for sensor in self.sensor_dict.values():
        # 	sensor.step()

        if self.hero and action is not None:

            if isinstance(action, list):

                vehicle_control = carla.VehicleControl(
                    throttle=float(
                        action[0]), steer=float(
                        action[1]), brake=float(
                        action[2]))
                self.actor.apply_control(vehicle_control)

    def _stop(self):
        """Stop the actor manager"""
        if self.spawned:
            
            self.actor.destroy()

            for sensor in self.sensor_dict.values():
                sensor.close()

    def reset(self):
        """Reset the actor manager"""
        self._stop()
        self._start()

    def render(self):
        """Render the actor manager"""
        if self.spawned:
            self.render_dict["velocity"] = self.actor.get_velocity()
            self.render_dict["location"] = self.actor.get_location()
            self.render_dict["rotation"] = self.actor.get_transform().rotation
            self.render_dict["acceleration"] = self.actor.get_acceleration()
            self.render_dict["control"] = self.actor.get_control()
            self.render_dict["x_extent_meters"] = self.actor.bounding_box.extent.x * 2
            self.render_dict["y_extent_meters"] = self.actor.bounding_box.extent.y * 2
        return self.render_dict

    def close(self):
        """Close the actor manager"""
        self._stop()
        logger.info("Actor manager closed")

    def seed(self):
        """Seed the actor manager"""
        pass

    def get_config(self):
        """Get the config of the actor manager"""
        return self.config

    def get_actor(self):
        """Get the actor"""
        return self.actor

    def get_sensor_dict(self):
        """Get the sensor dictionary"""
        return self.sensor_dict

    def set_autopilot(self, autopilot, port=8000):
        """Set the actor to autopilot"""
        self.actor.set_autopilot(autopilot, port)

    def _set_default_config(self):
        """Set the default config of actor manager"""
        self.config = {"vehicle": vehicle.VehicleModule(None, self.client),
                       "hero": True}
