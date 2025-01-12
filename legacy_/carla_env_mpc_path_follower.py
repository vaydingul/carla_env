from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.sensor import vehicle_sensor
from carla_env.modules.sensor import rgb_sensor
from carla_env.modules.sensor import semantic_sensor
from carla_env.modules.sensor import collision_sensor
from carla_env.modules.route import route
from carla_env.modules.module import Module

from utils.camera_utils import world_2_pixel
from utils.bev_utils import world_2_bev
# Import utils
import carla
import time
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import logging

logger = logging.getLogger(__name__)


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
        self.server = server.ServerModule(None)
        # Select a random map
        self.client = client.ClientModule(None)

        self.render_dict = {}

        self.first_time_step = True
        self.is_done = False
        self.counter = 0
        self.data = Queue()

        if self.config["save"]:

            self._create_save_folder()

        if self.config["render"]:

            self._create_render_window()

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.server.reset()
        self.client.reset()

        self.world = self.client.world
        self.map = self.world.get_map()

        self.spectator = self.client.world.get_spectator()

        # Fetch all spawn points
        spawn_points = self.client.world.get_map().get_spawn_points()
        # Select two random spawn points
        start_end_spawn_point = np.random.choice(spawn_points, 2)
        start = start_end_spawn_point[0]
        end = start_end_spawn_point[1]

        self.route = route.RouteModule(
            {"start": start, "end": end, "sampling_resolution": 2, "debug": True}, self.client.client)
        # Let's initialize a vehicle
        self.vehicle = vehicle.VehicleModule(
            {"vehicle_model": "lincoln.mkz_2017"}, self.client.client)
        # Make this vehicle actor
        self.actor = actor.ActorModule(
            {"actor": self.vehicle, "hero": True, "selected_spawn_point": start}, self.client.client)

        # Sensor suite
        self.vehicle_sensor = vehicle_sensor.VehicleSensorModule(
            None, self.client.client, self.actor)
        self.collision_sensor = collision_sensor.CollisionSensorModule(
            None, self.client.client, self.actor)
        self.rgb_sensor = rgb_sensor.RGBSensorModule(
            None, self.client.client, self.actor)
        self.semantic_sensor = semantic_sensor.SemanticSensorModule(
            None, self.client.client, self.actor)

        for (k, v) in self.actor.sensor_dict.items():
            if k not in self.config["allowed_sensors"]:
                v.save_to_queue = False

        time.sleep(1.0)
        logger.info("Everything is set!")

        for _ in range(int(1 / self.client.config["fixed_delta_seconds"]) * 2):
            self.client.step()

    def step(self, action=None):
        """Perform an action in the environment"""

        snapshot = self.client.world.get_snapshot()

        self.actor.step(action)
        self.vehicle.step()
        self.vehicle_sensor.step()

        data_dict = {}

        for (k, v) in self.actor.sensor_dict.items():

            if k in self.config["allowed_sensors"]:

                if v.get_queue().qsize() > 0:

                    try:

                        equivalent_frame_fetched = False

                        while not equivalent_frame_fetched:

                            data_ = v.get_queue().get(True, 10)

                            # , f"Frame number mismatch: {data_['frame']} != {snapshot.frame} \n Current Sensor: {k} \n Current Data Queue Size {self.data.qsize()}"
                            equivalent_frame_fetched = data_[
                                "frame"] == snapshot.frame

                    except Empty:

                        print("Empty")

                    data_dict[k] = data_

                    if k == "VehicleSensorModule":

                        current_transform = data_dict[k]["transform"]
                        current_velocity = data_dict[k]["velocity"]

                        transform = current_transform
                        transform.location.z += 2.0

                        if self.first_time_step:
                            self.initial_vehicle_transform = current_transform
                            self.first_time_step = False

                    elif k == "CollisionSensorModule":

                        impulse = data_dict[k]["impulse"]
                        impulse_amplitude = np.linalg.norm(impulse)
                        logger.debug(f"Collision impulse: {impulse_amplitude}")
                        if impulse_amplitude > 1:
                            self.is_done = True

        data_dict["snapshot"] = snapshot

        self.data.put(data_dict)

        self.spectator.set_transform(transform)

        self.is_done = False

        route_step = self.route.step(
            self.map.get_waypoint(current_transform.location))
        # target_waypoint, _ = self.route.step(current_transform)
        self.server.step()
        self.client.step()

        self.counter += 1

        if (route_step is not None) and (self.counter <= 500):
            return current_transform, current_velocity, route_step[0]

        else:
            self.is_done = True
            self.video_writer.release()
            return current_transform, current_velocity, None

    def render(self, predicted_location, **kwargs):
        """Render the environment"""
        for (k, v) in self.__dict__.items():
            if isinstance(v, Module):
                self.render_dict[k] = v.render()

        self.canvas = np.zeros_like(self.canvas)

        rgb_image = self.render_dict["rgb_sensor"]["image_data"]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:rgb_image.shape[0], :rgb_image.shape[1]] = rgb_image

        semantic_image = self.render_dict["semantic_sensor"]["image_data"]
        # semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[rgb_image.shape[0]:rgb_image.shape[0] +
                    semantic_image.shape[0], :semantic_image.shape[1]] = semantic_image

        # Put text for other modules
        position_x = rgb_image.shape[1] + 10
        position_y = 20
        for (module, render_dict) in self.render_dict.items():
            if module not in ["rgb_sensor", "semantic_sensor"]:

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

        if bool(kwargs):

            cv2.putText(
                self.canvas,
                f"Additional Info",
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

            for (k, v) in kwargs.items():
                cv2.putText(self.canvas, f"{k}: {v}", (position_x, position_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                position_y += 20

        for k in range(predicted_location.shape[0]):
            loc_ = predicted_location[k]
            loc_ = np.array(
                [loc_[0], loc_[1], self.render_dict["actor"]["location"].z])
            pixel_loc_ = world_2_pixel(
                loc_,
                self.render_dict["rgb_sensor"]["image_transform"].get_inverse_matrix(),
                rgb_image.shape[0],
                rgb_image.shape[1])

            ego_loc = np.array([self.render_dict["actor"]["location"].x,
                                self.render_dict["actor"]["location"].y,
                                self.render_dict["actor"]["location"].z])
            bev_loc_ = world_2_bev(
                loc_,
                ego_loc,
                self.render_dict["actor"]["rotation"].yaw,
                rgb_image.shape[0],
                rgb_image.shape[1])

            if pixel_loc_.shape[0] > 0:
                cv2.circle(self.canvas, (int(pixel_loc_[0][0]), int(
                    pixel_loc_[0][1])), 5, (255, 0, 0), -1)
            cv2.circle(self.canvas, (int(bev_loc_[0]), int(
                bev_loc_[1] + rgb_image.shape[0])), 5, (0, 255, 0), -1)

        for k in range(
            self.render_dict["route"]["route_index"],
            np.minimum(
                self.render_dict["route"]["route_index"] + 5,
                self.render_dict["route"]["route_length"])):
            loc_ = self.route.route[k][0].transform.location
            loc_ = np.array([loc_.x, loc_.y, loc_.z])
            pixel_loc_ = world_2_pixel(
                loc_,
                self.render_dict["rgb_sensor"]["image_transform"].get_inverse_matrix(),
                rgb_image.shape[0],
                rgb_image.shape[1])

            ego_loc = np.array([self.render_dict["actor"]["location"].x,
                                self.render_dict["actor"]["location"].y,
                                self.render_dict["actor"]["location"].z])
            bev_loc_ = world_2_bev(
                loc_,
                ego_loc,
                self.render_dict["actor"]["rotation"].yaw,
                rgb_image.shape[0],
                rgb_image.shape[1])

            if pixel_loc_.shape[0] > 0:
                cv2.circle(self.canvas, (int(pixel_loc_[0][0]), int(
                    pixel_loc_[0][1])), 5, (0, 255, 0), -1)
            cv2.circle(self.canvas, (int(bev_loc_[0]), int(
                bev_loc_[1] + rgb_image.shape[0])), 5, (255, 0, 0), -1)

        canvas_display = cv2.resize(self.canvas, (0, 0), fx=0.8, fy=0.8)
        cv2.imshow("Environment", canvas_display)

        canvas_save = cv2.resize(
            self.canvas,
            (self.canvas.shape[1] // 2,
             self.canvas.shape[0] // 2))
        cv2.imwrite(str(self.debug_path /
                        Path(f"{self.counter}.png")), canvas_save)

        if self.config["save_video"]:
            self.video_writer.write(canvas_save)

        cv2.waitKey(1)

    def close(self):
        """Close the environment"""
        self.vehicle.close()
        self.actor.close()
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

    def _create_render_window(self):

        self.canvas = np.zeros((1200, 1600, 3), np.uint8)
        cv2.imshow("Environment", self.canvas)

        if cv2.waitKey(1) == ord('q'):

            # press q to terminate the loop
            cv2.destroyAllWindows()

        if self.config["save_video"]:
            fourcc = VideoWriter_fourcc(*'mp4v')
            self.video_writer = VideoWriter(
                str(self.debug_path / Path("video.mp4")), fourcc, int(20), (800, 600))

    def _create_save_folder(self):

        debug_path = Path("figures/env_debug")

        date_ = Path(datetime.today().strftime('%Y-%m-%d'))
        time_ = Path(datetime.today().strftime('%H-%M-%S'))

        self.debug_path = debug_path / date_ / time_
        self.debug_path.mkdir(parents=True, exist_ok=True)
