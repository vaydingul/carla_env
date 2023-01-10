from carla_env.environment import Environment
from agents.navigation.local_planner import RoadOption
# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.traffic_manager import traffic_manager
from carla_env.modules.sensor import vehicle_sensor
from carla_env.modules.sensor import rgb_sensor
from carla_env.modules.sensor import collision_sensor
from carla_env.modules.sensor import occupancy_sensor
from carla_env.modules.module import Module
from carla_env.bev import (
    BirdViewProducer,
    BirdViewCropType,
)
from carla_env.bev.mask import PixelDimensions
from utils.carla_utils import (fetch_all_vehicles)

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
            brake_probability=0.1,
            max_throttle=1.0,
            max_steering_angle=1.0,
            action_repeat=1):
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
            steer = np.random.uniform(-self.max_steering_angle,
                                      self.max_steering_angle)

            action = [0, steer, -acceleration]

        else:

            acceleration = np.random.uniform(0, self.max_throttle)
            steer = np.random.uniform(-self.max_steering_angle,
                                      self.max_steering_angle)

            action = [acceleration, steer, 0]

        self.previous_action = action
        self.previous_count = 0

        return action


def create_multiple_actors_for_traffic_manager(client, n=20):
    """Create multiple vehicles in the world"""
    vehicles = fetch_all_vehicles(client)
    vehicles = vehicles * (n // len(vehicles) + 1)
    # Shuffle the list and take first n vehicles
    np.random.shuffle(vehicles)
    actors = [actor.ActorModule(
        config={
            "actor": vehicle.VehicleModule(
                config=None,
                client=client),
            "hero": True},
        client=client)]

    for k in range(n):

        actors.append(actor.ActorModule(
            config={
                "vehicle": vehicle.VehicleModule(
                    config={"vehicle_model": vehicles[k], },
                    client=client),
                "hero": False},
            client=client))

    return actors


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
        self.server_module = server.ServerModule(None)

        self.render_dict = {}

        self.is_first_reset = True
        self.is_done = False
        self.counter = 0

        self.action_designer = RandomActionDesigner(action_repeat=2)

        self.data = Queue()

        self.reset()

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
        selected_task = np.random.choice(self.config["tasks"])
        self.client_module = client.ClientModule(
            config={
                "world": selected_task["world"],
                "fixed_delta_seconds": self.config["fixed_delta_seconds"]})

        self.world = self.client_module.get_world()
        self.map = self.client_module.get_map()
        self.client = self.client_module.get_client()

        self.spectator = self.world.get_spectator()

        # Make this vehicle actor
        number_of_actors = np.random.randint(
            *
            selected_task["num_vehicles"]) if isinstance(
            selected_task["num_vehicles"],
            list) else selected_task["num_vehicles"]
        logger.info(f"Number of actors: {number_of_actors}")

        # Fetch all spawn points
        spawn_points = self.map.get_spawn_points()
        # Select two random spawn points
        start_end_spawn_point = np.random.choice(spawn_points, 2)
        start = start_end_spawn_point[0]
        # Let's initialize a vehicle
        self.vehicle_module = vehicle.VehicleModule(
            config={
                "vehicle_model": "lincoln.mkz_2017"},
            client=self.client)
        # Make this vehicle actor
        self.hero_actor_module = actor.ActorModule(config={
            "actor": self.vehicle_module,
            "hero": True,
            "selected_spawn_point": start},
            client=self.client)

        actor_list = create_multiple_actors_for_traffic_manager(
            self.client,
            n=number_of_actors + 1)

        actor_list.append(self.hero_actor_module)

        if not self.config["random"]:
            self.traffic_manager_module = traffic_manager.TrafficManagerModule(
                config={"vehicle_list": actor_list}, client=self.client)

        # Sensor suite
        self.vehicle_sensor = vehicle_sensor.VehicleSensorModule(
            config=None, client=self.client,
            actor=self.hero_actor_module, id="ego")
        self.collision_sensor = collision_sensor.CollisionSensorModule(
            config=None, client=self.client,
            actor=self.hero_actor_module, id="col")
        self.rgb_sensor_1 = rgb_sensor.RGBSensorModule(
            config={"yaw": -60, "width": 900, "height": 256}, client=self.client,
            actor=self.hero_actor_module, id="rgb_right")
        self.rgb_sensor_2 = rgb_sensor.RGBSensorModule(
            config={"yaw": 0, "width": 900, "height": 256}, client=self.client,
            actor=self.hero_actor_module, id="rgb_front")
        self.rgb_sensor_3 = rgb_sensor.RGBSensorModule(
            config={"yaw": 60, "width": 900, "height": 256}, client=self.client,
            actor=self.hero_actor_module, id="rgb_left")
        self.occupancy_sensor = occupancy_sensor.OccupancySensorModule(
            config=None, client=self.client, actor=self.hero_actor_module, id="occ")

        self.bev_module_world = BirdViewProducer(
            client=self.client,
            target_size=PixelDimensions(
                192,
                192),
            render_lanes_on_junctions=False,
            pixels_per_meter=5,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY,
            road_on_off=True,
            road_light=True,
            light_circle=True,
            lane_marking_thickness=2)

        self.bev_module_ego = BirdViewProducer(
            client=self.client,
            target_size=PixelDimensions(
                192,
                192),
            render_lanes_on_junctions=False,
            pixels_per_meter=20,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            light_circle=True)

        time.sleep(1.0)
        logger.info("Everything is set!")

        for _ in range(
                int(1 / self.client_module.config["fixed_delta_seconds"]) * 2):
            self.client_module.step()

        self.is_done = False
        self.counter = 0
        self.data = Queue()

        if self.config["save"]:

            self._create_save_folder()

        if self.config["render"]:

            self._create_render_window()

    def step(self, action=None):
        """Perform an action in the environment"""

        snapshot = self.client_module.world.get_snapshot()

        if self.config["random"]:
            action = self.action_designer.step()
            self.hero_actor_module.step(action=action)

        self.vehicle_sensor.step()

        data_dict = {}

        for (k, v) in self.hero_actor_module.get_sensor_dict().items():

            if v.get_queue().qsize() > 0:

                try:

                    equivalent_frame_fetched = False

                    while not equivalent_frame_fetched:

                        data_ = v.get_queue().get(True, 10)

                        equivalent_frame_fetched = data_[
                            "frame"] == snapshot.frame

                except Empty:

                    print("Empty")

                data_dict[k] = data_

                if k == "ego":

                    current_transform = data_dict[k]["transform"]

                    transform = current_transform
                    transform.location.z += 2.0

                if k == "col":

                    impulse = data_dict[k]["impulse"]
                    impulse_amplitude = np.linalg.norm(impulse)
                    logger.debug(f"Collision impulse: {impulse_amplitude}")
                    if impulse_amplitude > 100:
                        self.is_done = True

        data_dict["snapshot"] = snapshot

        self.spectator.set_transform(transform)

        bev_world = self.bev_module_world.step(
            agent_vehicle=self.hero_actor_module.get_actor())
        bev_ego = self.bev_module_ego.step(
            agent_vehicle=self.hero_actor_module.get_actor())

        data_dict["bev_world"] = bev_world
        data_dict["bev_ego"] = bev_ego

        if not self.config["random"]:
            self._next_agent_command = RoadOption.VOID.value
            self._next_agent_waypoint = [-1, -1, -1]
            try:
                _next_agent_navigational_action = self.traffic_manager_module.get_next_action(
                    self.hero_actor_module.get_actor())
                self._next_agent_command = RoadOption[_next_agent_navigational_action[0].upper(
                )].value
                self._next_agent_waypoint = [
                    _next_agent_navigational_action[1].transform.location.x,
                    _next_agent_navigational_action[1].transform.location.y,
                    _next_agent_navigational_action[1].transform.location.z]

                data_dict["navigation"] = {
                    "command": self._next_agent_command,
                    "waypoint": self._next_agent_waypoint}
            except BaseException:
                data_dict["navigation"] = {
                    "command": self._next_agent_command,
                    "waypoint": self._next_agent_waypoint}

        self.data.put(data_dict)

        self.server_module.step()
        self.client_module.step()

        self.counter += 1

        self.is_done = self.is_done or (
            self.counter >= self.config["max_steps"])

    def render(self, bev_list):
        """Render the environment"""
        if not self.config["render"]:
            return
        for (k, v) in self.__dict__.items():
            if isinstance(v, Module):
                self.render_dict[k] = v.render()

        self.canvas = np.zeros_like(self.canvas)

        # Put all of the rgb cameras as a 2x3 grid
        rgb_image_1 = self.render_dict["rgb_sensor_1"]["image_data"]
        rgb_image_1 = cv2.cvtColor(rgb_image_1, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:rgb_image_1.shape[0], :rgb_image_1.shape[1]] = rgb_image_1

        rgb_image_2 = self.render_dict["rgb_sensor_2"]["image_data"]
        rgb_image_2 = cv2.cvtColor(rgb_image_2, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:rgb_image_2.shape[0], rgb_image_1.shape[1]:rgb_image_1.shape[1] + rgb_image_2.shape[1]] = rgb_image_2

        rgb_image_3 = self.render_dict["rgb_sensor_3"]["image_data"]
        rgb_image_3 = cv2.cvtColor(rgb_image_3, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:rgb_image_3.shape[0], rgb_image_1.shape[1] +
                    rgb_image_2.shape[1]:rgb_image_1.shape[1] +
                    rgb_image_2.shape[1] +
                    rgb_image_3.shape[1]] = rgb_image_3

        offset = 0
        # Draw bev as
        for bev in bev_list:

            bev = self.bev_module_world.as_rgb(bev)
            bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
            # Put image into canvas
            self.canvas[rgb_image_1.shape[0] + 10:rgb_image_1.shape[0] +
                        10 + bev.shape[0], rgb_image_1.shape[1] +
                        rgb_image_2.shape[1] + offset:rgb_image_1.shape[1] +
                        rgb_image_2.shape[1] +
                        bev.shape[1] + offset] = bev

            offset += bev.shape[1] + 10

        # Put text for other modules
        position_x = rgb_image_1.shape[1] * 3 + 10
        position_y = 20
        for (module, render_dict) in self.render_dict.items():
            if "rgb" not in module:

                if bool(render_dict):

                    cv2.putText(
                        self.canvas,
                        f"{module.capitalize()}",
                        (position_x,
                         position_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,
                         255,
                         255),
                        1,
                        cv2.LINE_AA)
                    position_y += 40

                    for (k, v) in render_dict.items():

                        cv2.putText(
                            self.canvas,
                            f"{k}: {v}",
                            (position_x,
                             position_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,
                             255,
                             0),
                            1)
                        position_y += 20

        canvas_display = cv2.resize(
            src=self.canvas, dsize=(
                0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Environment", canvas_display)

        if self.config["save"]:
            canvas_save = self.canvas
            cv2.imwrite(str(self.debug_path /
                            Path(f"{self.counter}.png")), canvas_save)

        if self.config["save_video"]:
            self.video_writer.write(canvas_save)

        cv2.waitKey(1)

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
        self.config = {}

    def _create_render_window(self):
        cv2.destroyAllWindows()
        self.canvas = np.zeros((256 + 192 + 50, 900 * 3 + 1200, 3), np.uint8)
        cv2.imshow("Environment", self.canvas)

        if self.config["save_video"]:
            fourcc = VideoWriter_fourcc(*'mp4v')
            self.video_writer = VideoWriter(
                str(
                    self.debug_path /
                    Path("video.mp4")),
                fourcc,
                int(20),
                (self.canvas.shape[1] //
                 2,
                 self.canvas.shape[0] //
                 2))

    def _create_save_folder(self):

        debug_path = Path("figures/env_debug")

        date_ = Path(datetime.today().strftime('%Y-%m-%d'))
        time_ = Path(datetime.today().strftime('%H-%M-%S'))

        self.debug_path = debug_path / date_ / time_
        self.debug_path.mkdir(parents=True, exist_ok=True)
