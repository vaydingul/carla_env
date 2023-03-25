import random
from carla_env.modules import module
import carla
import logging
import time

logger = logging.getLogger(__name__)


class TrafficManagerModule(module.Module):
    """Concrete implementation Module abstract base class for traffic manager module"""

    def __init__(self, config, client) -> None:
        super().__init__()

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]

        self.client = client
        self.world = self.get_world()
        self.traffic_manager = self.client.get_trafficmanager(self.config["port"])

        self.render_dict = {}

        self.reset()

    def step(self):
        """Step the client"""
        self._tick()

    def reset(self):
        """Reset the client"""

        self.traffic_manager.set_synchronous_mode(self.config["synchronous_mode"])
        # self.traffic_manager.set_hybrid_physics_mode(True)
        # self.traffic_manager.set_hybrid_physics_radius(200)

        if self.config["vehicle_list"]:
            for vehicle in self.config["vehicle_list"]:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())

        if self.config["walker_list"]:

            self.world.set_pedestrians_cross_factor(
                self.config["pedestrians_cross_factor"]
            )

            walker_ai_controller_blueprint = self.world.get_blueprint_library().find(
                "controller.ai.walker"
            )

            for walker in self.config["walker_list"]:
                controller_actor = self.world.try_spawn_actor(
                    walker_ai_controller_blueprint,
                    carla.Transform(),
                    attach_to=walker.get_actor(),
                )


                if controller_actor is not None:

                    controller_actor.start()
                    controller_actor.go_to_location(
                        self.world.get_random_location_from_navigation()
                    )
                    controller_actor.set_max_speed(1 + random.random())

    def render(self):
        """Render the client"""
        self.render_dict["port"] = self.traffic_manager.get_port()
        self.render_dict["num_vehicles"] = len(self.config["vehicle_list"])
        self.render_dict["num_walkers"] = len(self.config["walker_list"])
        return self.render_dict

    def close(self):
        """Close the client"""
        for vehicle in self.config["vehicle_list"]:
            vehicle.close()

    def seed(self):
        """Seed the client"""
        pass

    def get_config(self):
        """Get the config of the client"""
        return self.config

    def get_world(self):
        """Get the world"""
        return self.client.get_world()

    def get_client(self):
        """Get the client"""
        return self.client

    def get_traffic_manager(self):
        """Get the traffic manager"""
        return self.traffic_manager

    def get_next_action(self, actor):
        """Get the next action"""
        return self.get_traffic_manager().get_next_action(actor)

    def _set_default_config(self):
        """Set the default config of the client"""
        self.config = {
            "port": 8000,
            "synchronous_mode": True,
            "vehicle_list": [],
            "walker_list": [],
            "pedestrians_cross_factor": 0.5,
            "running_walker_ratio": 0.5,
        }

    @property
    def spawn_transforms(self):
        """Get all the spawn point in the map"""
        spawn_transforms = self.map.get_spawn_points()
        return spawn_transforms
