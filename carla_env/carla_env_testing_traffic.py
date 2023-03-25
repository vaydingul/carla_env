from carla_env.environment import Environment
from agents.navigation.local_planner import RoadOption

# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.traffic_manager import traffic_manager
from carla_env.modules.route import route
from carla_env.modules.module import Module
from carla_env.renderer.renderer import Renderer, COLORS
from carla_env.bev import BirdViewProducer, BIRDVIEW_CROP_TYPE
from carla_env.bev.mask import PixelDimensions
from utils.carla_utils import create_multiple_vehicle_actors_for_traffic_manager
from utils.render_utils import *

# Import utils
import time
import numpy as np
import cv2
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)


class RandomActionDesigner(object):
    def __init__(
        self,
        brake_probability=0.1,
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
        self.is_collided = False

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
        self.bevs = self.config["bevs"]
        self.route = self.config["route"]
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

        # Fetch all spawn points
        spawn_points = self.map.get_spawn_points()
        # Select two random spawn points
        start_end_spawn_point = np.random.choice(spawn_points, 2)
        start = start_end_spawn_point[0]
        end = start_end_spawn_point[1]

        self.route = route.RouteModule(
            config={"start": start, "end": end, **self.route["config"]},
            client=self.client,
        )

        self.route_bev = route.RouteModule(
            config={
                "start": start,
                "end": end,
                "sampling_resolution": 5,
                "distance_threshold": 5,
            },
            client=self.client,
        )

        # Let's initialize a vehicle
        self.vehicle_module = vehicle.VehicleModule(
            config={"vehicle_model": "lincoln.mkz_2017"}, client=self.client
        )
        # Make this vehicle actor
        self.hero_actor_module = actor.ActorModule(
            config={
                "actor": self.vehicle_module,
                "hero": True,  # self.random,
                "selected_spawn_point": start,
            },
            client=self.client,
        )

        # Let's initialize a traffic manager
        number_of_actors = (
            np.random.randint(*selected_task["num_vehicles"])
            if isinstance(selected_task["num_vehicles"], list)
            else selected_task["num_vehicles"]
        )
        logger.info(f"Number of actors: {number_of_actors}")
        actor_list = create_multiple_vehicle_actors_for_traffic_manager(
            self.client, n=number_of_actors
        )

        # Traffic manager
        self.traffic_manager_module = traffic_manager.TrafficManagerModule(
            config={"vehicle_list": actor_list, "port": self.tm_port},
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
                    if impulse_amplitude > 1:
                        self.is_collided = True

        self.data_dict["snapshot"] = snapshot

        self.spectator.set_transform(transform)

        route_step = self.route.step(self.map.get_waypoint(current_transform.location))
        route_bev_step = self.route_bev.step(
            self.map.get_waypoint(current_transform.location)
        )

        for bev_module in self.bev_modules:
            bev_output = bev_module["module"].step(
                agent_vehicle=self.hero_actor_module.get_actor(),
                waypoint=route_bev_step[0],
            )
            self.data_dict[bev_module["id"]] = bev_output

        self.data_dict["navigation"] = {}
        self.data_dict["navigation"]["waypoint"] = route_step[0]
        self.data_dict["navigation"]["command"] = route_step[1]

        self.data.put(self.data_dict)

        self.server_module.step()
        self.client_module.step()

        self.counter += 1

        self.is_done = self.is_collided or (self.counter >= self.max_steps)

        return self.is_done

    def render(self, *args, **kwargs):
        """Render the environment"""

        self.renderer_module.reset()

        self.generate_sensor_dict()

        world_2_camera_transformation = self.render_dict["rgb_front"][
            "image_transform"
        ].get_inverse_matrix()
        fov = self.render_dict["rgb_front"]["image_fov"]
        ego_current_location = self.render_dict["hero_actor_module"]["location"]
        ego_current_location_ = postprocess_location(ego_current_location)
        ego_yaw = self.render_dict["hero_actor_module"]["rotation"].yaw
        pixels_per_meter = self.render_dict["bev_world"]["pixels_per_meter"]

        # Put all of the rgb cameras as a 2x3 grid
        if "rgb_front" in self.data_dict.keys():
            rgb_image_front = self.data_dict["rgb_front"]["data"]
            rgb_image_front = cv2.cvtColor(rgb_image_front, cv2.COLOR_BGR2RGB)
            (h_image, w_image, c_image) = rgb_image_front.shape

            point_rgb_front_left_up = tuple(reversed(self.renderer_module.get_cursor()))

            self.renderer_module.render_image(rgb_image_front, move_cursor="down")

            if "bev_world" in self.data_dict.keys():
                bev = self.data_dict["bev_world"]
                bev = self.bev_modules[0]["module"].as_rgb(bev)
                bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
                (h_bev, w_bev, c_bev) = bev.shape
                # Put image into canvas

                point_bev_world_left_up = tuple(
                    reversed(self.renderer_module.get_cursor())
                )

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
                        font_thickness=1,
                    )

                    for (k, v) in render_dict.items():

                        self.renderer_module.render_text(
                            f"{k}: {v}",
                            move_cursor="down",
                            font_color=COLORS.YELLOW,
                        )

                    self.renderer_module.move_cursor(direction="down", amount=(10, 0))

        if bool(kwargs):

            self.renderer_module.render_text(
                f"ADDITIONAL ARGUMENTS",
                move_cursor="down",
                font_color=COLORS.PURPLE,
                font_thickness=1,
            )

            for (k, v) in kwargs.items():

                if not isinstance(v, dict):

                    self.renderer_module.render_text(
                        f"{k}: {v}",
                        move_cursor="down",
                        font_color=COLORS.BLUE,
                    )

        if "cost_viz" in kwargs.keys():
            world_future_bev_predicted = kwargs["cost_viz"][
                "world_future_bev_predicted"
            ]
            mask_dict = kwargs["cost_viz"]["mask_dict"]
            bev_selected_channels = kwargs["cost_viz"]["bev_selected_channels"]

            _, S, _, H, W = world_future_bev_predicted.shape

            cursor_master = self.renderer_module.get_cursor()

            for (mask_key, mask_value) in mask_dict.items():

                for s in range(S - 1):

                    bev = postprocess_bev(
                        world_future_bev_predicted[0, s + 1],
                        bev_selected_channels=bev_selected_channels,
                    )

                    mask = postprocess_mask(mask_value[0, s])

                    self.renderer_module.render_overlay_image(
                        bev, mask, 0.5, 0.5, move_cursor="right"
                    )

                    self.renderer_module.move_cursor("right", amount=(0, 10))

                self.renderer_module.render_text(
                    f"{mask_key}",
                    move_cursor="down",
                    font_color=COLORS.WHITE,
                )

                self.renderer_module.move_cursor("point", amount=cursor_master)
                self.renderer_module.move_cursor("down", amount=(H + 10, 0))

        if "ego_viz" in kwargs.keys():
            ego_viz = kwargs["ego_viz"]
            ego_future_location_predicted = ego_viz["ego_future_location_predicted"]
            control_selected = ego_viz["control_selected"]

            _, S, _ = ego_future_location_predicted.shape

            self.renderer_module.move_cursor("point", amount=(0, 0))

            if ("rgb_front" in self.data_dict.keys()) and (
                "bev_world" in self.data_dict.keys()
            ):

                for k in range(S):

                    self.renderer_module.move_cursor("point", amount=(0, 0))

                    ego_future_location = postprocess_location(
                        ego_future_location_predicted[0, k],
                        ego_current_location=ego_current_location,
                    )

                    ego_future_location_pixel = world_2_pixel(
                        ego_future_location,
                        world_2_camera_transformation,
                        h_image,
                        w_image,
                        fov,
                    )

                    ego_future_location_bev = world_2_bev(
                        ego_future_location,
                        ego_current_location_,
                        ego_yaw,
                        h_bev,
                        w_bev,
                        pixels_per_meter,
                    )

                    if ego_future_location_pixel is not None:
                        render_position = (
                            ego_future_location_pixel[0] + point_rgb_front_left_up[0],
                            ego_future_location_pixel[1] + point_rgb_front_left_up[1],
                        )
                        self.renderer_module.render_point(
                            pos=render_position, color=COLORS.YELLOW
                        )

                    if ego_future_location_bev is not None:
                        render_position = (
                            ego_future_location_bev[0] + point_bev_world_left_up[0],
                            ego_future_location_bev[1] + point_bev_world_left_up[1],
                        )
                        self.renderer_module.render_point(
                            pos=render_position, color=COLORS.YELLOW
                        )

        for k in range(
            self.render_dict["route"]["route_index"],
            np.minimum(
                self.render_dict["route"]["route_index"] + 10,
                self.render_dict["route"]["route_length"],
            ),
        ):

            self.renderer_module.move_cursor("point", amount=(0, 0))

            route_current_location = self.render_dict["route"]["current_waypoint"]

            route_current_location = postprocess_location(
                route_current_location, ego_current_location
            )

            route_current_location_pixel = world_2_pixel(
                route_current_location,
                world_2_camera_transformation,
                h_image,
                w_image,
                fov,
            )

            route_current_location_bev = world_2_bev(
                route_current_location,
                ego_current_location_,
                ego_yaw,
                h_bev,
                w_bev,
                pixels_per_meter,
            )

            if route_current_location_pixel is not None:
                render_position = (
                    route_current_location_pixel[0] + point_rgb_front_left_up[0],
                    route_current_location_pixel[1] + point_rgb_front_left_up[1],
                )
                self.renderer_module.render_point(
                    pos=render_position, color=COLORS.BLUE
                )

            if route_current_location_bev is not None:
                render_position = (
                    route_current_location_bev[0] + point_bev_world_left_up[0],
                    route_current_location_bev[1] + point_bev_world_left_up[1],
                )
                self.renderer_module.render_point(
                    pos=render_position, color=COLORS.BLUE
                )

        self.renderer_module.show()

        saved_image_path = self.renderer_module.save(info=f"step_{self.counter}")

        return saved_image_path

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

    def generate_sensor_dict(self):
        """Generate a sensor dict from the sensor config"""
        self.render_dict = {}
        for (k, v) in self.__dict__.items():
            if isinstance(v, Module):
                self.render_dict[k] = v.render()
        for item in self.sensor_modules:
            self.render_dict[item["id"]] = item["module"].render()

        for item in self.bevs:
            self.render_dict[item["id"]] = item["config"]

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
            "route": None,
            "renderer": None,
        }
