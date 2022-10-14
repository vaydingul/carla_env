from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.traffic_manager import traffic_manager
from carla_env.modules.sensor import vehicle_sensor
from carla_env.modules.sensor import rgb_sensor
from carla_env.modules.sensor import collision_sensor
from carla_env.modules.module import Module
from carla_env.bev import (
    BirdViewProducer,
    BirdViewCropType,
)
from carla_env.bev.mask import PixelDimensions


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


def create_multiple_actors_for_traffic_manager(client, n=20):
    """Create multiple vehicles in the world"""

    return [
        actor.ActorModule(
            config={
                "actor": vehicle.VehicleModule(
                    config=None,
                    client=client),
                "hero": False},
            client=client) for _ in range(n)]


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
        # Select a random town
        world_ = np.random.choice(self.config["worlds"])
        self.client_module = client.ClientModule(config={"world": world_})

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
        self.server_module.reset()
        self.client_module.reset()

        self.world = self.client_module.get_world()
        self.map = self.client_module.get_map()
        self.client = self.client_module.get_client()

        self.spectator = self.world.get_spectator()

        # Make this vehicle actor
        actor_list = create_multiple_actors_for_traffic_manager(
            self.client, 50)
        self.hero_actor_module = actor_list[0]

        self.traffic_manager_module = traffic_manager.TrafficManagerModule(
            config={"vehicle_list": actor_list}, client=self.client)

        # Sensor suite
        self.vehicle_sensor = vehicle_sensor.VehicleSensorModule(
            config=None, client=self.client,
            actor=self.hero_actor_module)
        self.collision_sensor = collision_sensor.CollisionSensorModule(
            config=None, client=self.client,
            actor=self.hero_actor_module)
        self.rgb_sensor_1 = rgb_sensor.RGBSensorModule(
            config={"yaw": -60}, client=self.client,
            actor=self.hero_actor_module)
        # self.rgb_sensor_2 = rgb_sensor.RGBSensorModule(
        #     config={"yaw": 0}, client=self.client,
        #     actor=self.hero_actor_module)
        # self.rgb_sensor_3 = rgb_sensor.RGBSensorModule(
        #     config={"yaw": 60}, client=self.client,
        #     actor=self.hero_actor_module)
        # self.rgb_sensor_4 = rgb_sensor.RGBSensorModule(
        #     config={"yaw": 120}, client=self.client,
        #     actor=self.hero_actor_module)
        # self.rgb_sensor_5 = rgb_sensor.RGBSensorModule(
        #     config={"yaw": 180}, client=self.client,
        #     actor=self.hero_actor_module)
        # self.rgb_sensor_6 = rgb_sensor.RGBSensorModule(
        #     config={"yaw": 240}, client=self.client,
        #     actor=self.hero_actor_module)

        self.bev_module = BirdViewProducer(
            client=self.client,
            target_size=PixelDimensions(
                1600,
                1200),
            render_lanes_on_junctions=False,
            pixels_per_meter=15,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA)

        time.sleep(1.0)
        logger.info("Everything is set!")

        for _ in range(
                int(1 / self.client_module.config["fixed_delta_seconds"]) * 2):
            self.client_module.step()

    def step(self, action=None):
        """Perform an action in the environment"""

        snapshot = self.client_module.world.get_snapshot()

        # self.hero_actor_module.step(action=action)
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

                if k == "VehicleSensorModule_0":

                    current_transform = data_dict[k]["transform"]
                    current_velocity = data_dict[k]["velocity"]

                    transform = current_transform
                    transform.location.z += 2.0

                    if self.first_time_step:
                        self.initial_vehicle_transform = current_transform
                        self.first_time_step = False

                elif k == "CollisionSensorModule_0":

                    impulse = data_dict[k]["impulse"]
                    impulse_amplitude = np.linalg.norm(impulse)
                    logger.debug(f"Collision impulse: {impulse_amplitude}")
                    if impulse_amplitude > 1:
                        self.is_done = True

        data_dict["snapshot"] = snapshot

        self.spectator.set_transform(transform)

        bev = self.bev_module.step(
            agent_vehicle=self.hero_actor_module.get_actor())

        data_dict["bev"] = bev

        self.data.put(data_dict)

        self.server_module.step()
        self.client_module.step()

        self.counter += 1

    def render(self, bev):
        """Render the environment"""
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
        self.canvas[:rgb_image_2.shape[0], rgb_image_1.shape[1]
            :rgb_image_1.shape[1] + rgb_image_2.shape[1]] = rgb_image_2

        rgb_image_3 = self.render_dict["rgb_sensor_3"]["image_data"]
        rgb_image_3 = cv2.cvtColor(rgb_image_3, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:rgb_image_3.shape[0], rgb_image_1.shape[1] +
                    rgb_image_2.shape[1]:rgb_image_1.shape[1] +
                    rgb_image_2.shape[1] +
                    rgb_image_3.shape[1]] = rgb_image_3

        rgb_image_4 = self.render_dict["rgb_sensor_4"]["image_data"]
        rgb_image_4 = cv2.cvtColor(rgb_image_4, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[rgb_image_1.shape[0]:rgb_image_1.shape[0] +
                    rgb_image_4.shape[0], rgb_image_1.shape[1] +
                    rgb_image_2.shape[1]:rgb_image_1.shape[1] +
                    rgb_image_2.shape[1] +
                    rgb_image_4.shape[1]] = rgb_image_4

        rgb_image_5 = self.render_dict["rgb_sensor_5"]["image_data"]
        rgb_image_5 = cv2.cvtColor(rgb_image_5, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[rgb_image_1.shape[0]:rgb_image_1.shape[0] +
                    rgb_image_5.shape[0], rgb_image_1.shape[1]:rgb_image_1.shape[1] +
                    rgb_image_5.shape[1]] = rgb_image_5

        rgb_image_6 = self.render_dict["rgb_sensor_6"]["image_data"]
        rgb_image_6 = cv2.cvtColor(rgb_image_6, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[rgb_image_1.shape[0]:rgb_image_1.shape[0] +
                    rgb_image_6.shape[0], :rgb_image_6.shape[1]] = rgb_image_6

        bev_1 = self.bev_module.as_rgb(bev)
        bev_1 = cv2.cvtColor(bev_1, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:bev_1.shape[0], rgb_image_1.shape[1] +
                    rgb_image_2.shape[1] +
                    rgb_image_3.shape[1]:rgb_image_1.shape[1] +
                    rgb_image_2.shape[1] +
                    rgb_image_3.shape[1] +
                    bev_1.shape[1]] = bev_1

        bev_2 = bev.copy()
        bev_2[..., 2] = 0
        bev_2[..., 3] = 0
        bev_2 = self.bev_module.as_rgb(bev_2)
        bev_2 = cv2.cvtColor(bev_2, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[bev_1.shape[0]:bev_1.shape[0] +
                    bev_2.shape[0], :bev_2.shape[1]] = bev_2

        bev_3 = bev.copy()
        bev_3[..., 0] = 0
        bev_3[..., 1] = 0
        bev_3[..., 4] = 0
        bev_3[..., 5] = 0
        bev_3[..., 6] = 0
        bev_3[..., 7] = 0
        bev_3 = self.bev_module.as_rgb(bev_3)
        bev_3 = cv2.cvtColor(bev_3, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[bev_1.shape[0]:bev_1.shape[0] +
                    bev_3.shape[0], bev_2.shape[1] +
                    100:bev_2.shape[1] +
                    100 +
                    bev_3.shape[1]] = bev_3

        canvas_display = cv2.resize(
            src=self.canvas, dsize=(
                0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Environment", canvas_display)

        canvas_save = self.canvas
        cv2.imwrite(str(self.debug_path /
                        Path(f"{self.counter}.png")), canvas_save)

        if self.config["save_video"]:
            self.video_writer.write(canvas_save)

        cv2.waitKey(1)

    def close(self):
        """Close the environment"""
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
        return self.data.get_data()

    def get_counter(self):
        """Get the counter of the environment"""
        return self.counter

    def _set_default_config(self):
        """Set the default config of the environment"""
        self.config = {}

    def _create_render_window(self):

        self.canvas = np.zeros((2400, 4000, 3), np.uint8)
        cv2.imshow("Environment", self.canvas)

        if cv2.waitKey(1) == ord('q'):

            # press q to terminate the loop
            cv2.destroyAllWindows()

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
