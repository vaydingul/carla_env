from carla_env.environment import Environment
from agents.navigation.local_planner import RoadOption

# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.traffic_manager import traffic_manager

from carla_env.modules.module import Module
from carla_env.renderer.renderer import Renderer, COLORS
from carla_env.bev import BirdViewProducer, BirdViewCropType, BIRDVIEW_CROP_TYPE
from carla_env.bev.mask import PixelDimensions
from utils.carla_utils import (
    create_multiple_vehicle_actors_for_traffic_manager,
    create_multiple_walker_actors_for_traffic_manager,
)

# Import utils
import time
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)


class RandomActionDesigner(object):
    def __init__(
        self,
        brake_probability=0.03,
        max_throttle=1.0,
        max_steering_angle=1.0,
        action_repeat=1,
    ):
        self.brake_probability = brake_probability
        self.max_throttle = max_throttle
        self.max_steering_angle = max_steering_angle

        self.previous_action = None
        self.previous_count = 0
        self.action_repeat = action_repeat

    def step(self):

        if (self.previous_count < self.action_repeat) and self.previous_action:

            self.previous_count += 1

            return self.previous_action

        # Randomize control
        if np.random.random() < self.brake_probability:

            acceleration = np.random.uniform(-self.max_throttle, 0)
            steer = np.random.uniform(-self.max_steering_angle, self.max_steering_angle)

            action = [0, steer, -acceleration]

        else:

            acceleration = np.random.uniform(0, self.max_throttle)
            steer = np.random.uniform(-self.max_steering_angle, self.max_steering_angle)

            action = [acceleration, steer, 0]

        self.previous_action = action
        self.previous_count = 1

        return action


class CarlaEnvironment(Environment):
    """Concrete implementation of Environment abstract base class"""

    def __init__(self, config):
        """Initialize the environment"""
        super().__init__()

        self._set_default_config()
        self.config.update(config)
        self.build_from_config()

        # We have our server and client up and running
        self.server_module = server.ServerModule(config={"port": self.port})

        self.is_first_reset = True
        self.is_done = False
        self.counter = 0

        if self.random:
            self.action_designer = RandomActionDesigner(
                action_repeat=self.action_repeat
            )

        if self.renderer is not None:
            self.renderer_module = Renderer(config=self.renderer)

        self.data = Queue()

        self.reset()

    def build_from_config(self):

        self.random = self.config["random"]
        self.action_repeat = self.config["action_repeat"]
        self.fixed_delta_seconds = self.config["fixed_delta_seconds"]
        self.port = self.config["port"]
        self.max_steps = self.config["max_steps"]
        self.tm_port = self.config["tm_port"]
        self.tasks = self.config["tasks"]
        self.sensors = self.config["sensors"]
        self.noiser = self.config["noiser"]
        self.bevs = self.config["bevs"]
        self.renderer = self.config["renderer"]

    def reset(self):
        """Reset the environment"""
        # self.server_module.reset()
        if self.is_first_reset:
            self.is_first_reset = False
        else:
            if not self.config["random"]:
                self.traffic_manager_module.close()
            else:
                self.hero_actor_module.close()

        # Select a random task
        selected_task = np.random.choice(self.tasks)
        self.client_module = client.ClientModule(
            config={
                "world": selected_task["world"],
                "fixed_delta_seconds": self.fixed_delta_seconds,
                "port": self.port,
            }
        )

        self.world = self.client_module.get_world()
        self.map = self.client_module.get_map()
        self.client = self.client_module.get_client()

        self.spectator = self.world.get_spectator()

        number_of_vehicle_actors = (
            np.random.randint(*selected_task["num_vehicles"])
            if isinstance(selected_task["num_vehicles"], list)
            else selected_task["num_vehicles"]
        )
        logger.info(f"Number of actors: {number_of_vehicle_actors}")

        number_of_walker_actors = (
            np.random.randint(*selected_task["num_walkers"])
            if isinstance(selected_task["num_walkers"], list)
            else selected_task["num_walkers"]
        )
        logger.info(f"Number of walkers: {number_of_walker_actors}")

        # Fetch all spawn points
        spawn_points = self.map.get_spawn_points()
        # Select two random spawn points
        start_end_spawn_point = np.random.choice(spawn_points, 2)
        start = start_end_spawn_point[0]

        # Let's initialize a vehicle
        self.vehicle_module = vehicle.VehicleModule(
            config={"vehicle_model": "lincoln.mkz_2017"}, client=self.client
        )
        # Make this vehicle actor
        self.hero_actor_module = actor.ActorModule(
            config={
                "actor": self.vehicle_module,
                "hero": self.random,
                "selected_spawn_point": start,
            },
            client=self.client,
        )

        vehicle_actor_list = create_multiple_vehicle_actors_for_traffic_manager(
            self.client, n=number_of_vehicle_actors
        )

        self.client_module.step()

        walker_actor_list = create_multiple_walker_actors_for_traffic_manager(
            self.client, n=number_of_walker_actors
        )

        self.client_module.step()

        vehicle_actor_list.append(self.hero_actor_module)

        if not self.random:
            self.traffic_manager_module = traffic_manager.TrafficManagerModule(
                config={
                    "vehicle_list": vehicle_actor_list,
                    "walker_list": walker_actor_list,
                    "port": self.tm_port,
                },
                client=self.client,
            )

        # Sensor suite

        self.sensor_modules = []
        for sensor in self.sensors:

            self.sensor_modules.append(
                {
                    "id": sensor["id"],
                    "module": sensor["class"](
                        config=sensor["config"],
                        client=self.client,
                        actor=self.hero_actor_module,
                        id=sensor["id"],
                    ),
                }
            )

        # Noiser

        self.noiser_module = self.noiser["class"](
            config=self.noiser["config"],
            client=self.client,
            actor=self.hero_actor_module.get_actor(),
        )

        # Bird's eye view
        self.bev_modules = []
        for bev in self.bevs:

            self.bev_modules.append(
                {
                    "id": bev["id"],
                    "module": BirdViewProducer(
                        client=self.client,
                        target_size=PixelDimensions(
                            bev["config"]["width"], bev["config"]["height"]
                        ),
                        render_lanes_on_junctions=bev["config"][
                            "render_lanes_on_junctions"
                        ],
                        pixels_per_meter=bev["config"]["pixels_per_meter"],
                        crop_type=BIRDVIEW_CROP_TYPE[bev["config"]["crop_type"]],
                        road_on_off=bev["config"]["road_on_off"],
                        road_light=bev["config"]["road_light"],
                        light_circle=bev["config"]["light_circle"],
                        lane_marking_thickness=bev["config"]["lane_marking_thickness"],
                    ),
                }
            )

        time.sleep(1.0)
        logger.info("Everything is set!")

        for _ in range(int((1 / self.fixed_delta_seconds))):
            self.client_module.step()

        self.is_done = False
        self.counter = 0
        self.data = Queue()

    def step(self, action=None):
        """Perform an action in the environment"""

        snapshot = self.client_module.world.get_snapshot()

        if self.config["random"]:
            action = self.action_designer.step()
            self.hero_actor_module.step(action=action)

        for sensor in self.sensor_modules:
            sensor["module"].step()

        self.data_dict = {}

        for (k, v) in self.hero_actor_module.get_sensor_dict().items():

            if v.get_queue().qsize() > 0:

                try:

                    equivalent_frame_fetched = False

                    while not equivalent_frame_fetched:

                        data_ = v.get_queue().get(True, 10)

                        equivalent_frame_fetched = data_["frame"] == snapshot.frame

                except Empty:

                    print("Empty")

                self.data_dict[k] = data_

                if k == "ego":

                    current_transform = self.data_dict[k]["transform"]

                    transform = current_transform
                    transform.location.z += 2.0

                if k == "col":

                    impulse = self.data_dict[k]["impulse"]
                    impulse_amplitude = np.linalg.norm(impulse)
                    logger.debug(f"Collision impulse: {impulse_amplitude}")
                    if impulse_amplitude > 100:
                        self.is_done = True

        self.data_dict["snapshot"] = snapshot

        self.spectator.set_transform(transform)

        for bev_module in self.bev_modules:
            bev_output = bev_module["module"].step(
                agent_vehicle=self.hero_actor_module.get_actor()
            )
            self.data_dict[bev_module["id"]] = bev_output

        if not self.config["random"]:
            self._next_agent_command = RoadOption.VOID.value
            self._next_agent_waypoint = [-1, -1, -1]
            try:
                _next_agent_navigational_action = (
                    self.traffic_manager_module.get_next_action(
                        self.hero_actor_module.get_actor()
                    )
                )
                self._next_agent_command = RoadOption[
                    _next_agent_navigational_action[0].upper()
                ].value
                self._next_agent_waypoint = [
                    _next_agent_navigational_action[1].transform.location.x,
                    _next_agent_navigational_action[1].transform.location.y,
                    _next_agent_navigational_action[1].transform.location.z,
                ]

                self.data_dict["navigation"] = {
                    "command": self._next_agent_command,
                    "waypoint": self._next_agent_waypoint,
                }
            except BaseException:
                self.data_dict["navigation"] = {
                    "command": self._next_agent_command,
                    "waypoint": self._next_agent_waypoint,
                }

        self.data.put(self.data_dict)

        self.server_module.step()
        self.client_module.step()

        self.counter += 1

        self.is_done = self.is_done or (self.counter >= self.max_steps)

    def render(self):
        """Render the environment"""

        self.renderer_module.reset()

        self.render_dict = {}
        for (k, v) in self.__dict__.items():
            if isinstance(v, Module):
                self.render_dict[k] = v.render()
            elif isinstance(v, list):
                if len(v) > 0:
                    if isinstance(v[0], Module):
                        for item in v:
                            self.render_dict[k] = item.render()

        # Put all of the rgb cameras as a 2x3 grid
        if "rgb_front" in self.data_dict.keys():
            rgb_image_front = self.data_dict["rgb_front"]["data"]
            rgb_image_front = cv2.cvtColor(rgb_image_front, cv2.COLOR_BGR2RGB)
            (h_image, w_image, c_image) = rgb_image_front.shape

            self.renderer_module.render_image(rgb_image_front, move_cursor="down")

            if "bev_world" in self.data_dict.keys():
                bev = self.data_dict["bev_world"]
                bev = self.bev_modules[0]["module"].as_rgb(bev)
                bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
                (h_bev, w_bev, c_bev) = bev.shape
                # Put image into canvas
                self.renderer_module.render_image(bev, move_cursor="down")

                self.renderer_module.move_cursor(
                    direction="right-up", amount=(h_image + h_bev, w_image + 20)
                )

        self.renderer_module.render_text("", move_cursor="down", font_color=COLORS.RED)

        for (module, render_dict) in self.render_dict.items():

            if "rgb" not in module:

                if bool(render_dict):

                    self.renderer_module.render_text(
                        f"{module.capitalize()}",
                        move_cursor="down",
                        font_color=COLORS.RED,
                    )

                    for (k, v) in render_dict.items():

                        self.renderer_module.render_text(
                            f"{k}: {v}",
                            move_cursor="down",
                            font_color=COLORS.YELLOW,
                        )

        self.renderer_module.show()

        self.renderer_module.save(info=self.counter)

    def close(self):
        """Close the environment"""
        if not self.config["random"]:
            self.traffic_manager_module.close()
        else:
            self.hero_actor_module.close()
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

    def _set_default_config(self):
        """Set the default config of the environment"""
        self.config = {
            "random": False,
            "action_repeat": 1,
            "fixed_delta_seconds": 0.05,
            "port": 2000,
            "tm_port": 8000,
            "max_steps": 1000,
            "tasks": [{"world": "Town02", "num_vehicles": 80}],
            "sensors": [],
            "bevs": [],
            "renderer": None,
        }
