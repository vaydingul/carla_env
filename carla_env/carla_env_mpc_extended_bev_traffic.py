from carla_env.environment import Environment

# Import modules
from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
from carla_env.modules.traffic_manager import traffic_manager
from carla_env.modules.sensor import vehicle_sensor
from carla_env.modules.sensor import rgb_sensor
from carla_env.modules.sensor import semantic_sensor
from carla_env.modules.sensor import collision_sensor
from carla_env.modules.sensor import occupancy_sensor
from carla_env.modules.route import route
from carla_env.modules.module import Module
from carla_env.bev import (
    BirdViewProducer,
    BirdViewCropType,
)
from carla_env.bev.mask import PixelDimensions
from agents.navigation.local_planner import RoadOption

from utils.camera_utils import world_2_pixel
from utils.bev_utils import world_2_bev
from utils.carla_utils import (fetch_all_vehicles)


# Import utils
import time
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
import carla
import logging

logger = logging.getLogger(__name__)


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
            "hero": False},
        client=client)]

    for k in range(n):

        actors.append(actor.ActorModule(
            config={
                "vehicle": vehicle.VehicleModule(
                    config=None, #{"vehicle_model": vehicles[k], },
                    client=client),
                "hero": False},
            client=client))

    return actors

# def create_multiple_actors_for_traffic_manager(client, n=20):
#     """Create multiple vehicles in the world"""

#     return [
#         actor.ActorModule(
#             config={
#                 "actor": vehicle.VehicleModule(
#                     config=None,
#                     client=client),
#                 "hero": False},
#             client=client) for _ in range(n)]


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

        self.data = Queue()

        self.reset()

    def reset(self):
        """Reset the environment"""

        # self.server_module.reset()
        if self.is_first_reset:
            self.is_first_reset = False
        else:
            self.traffic_manager_module.close()

        self.client_module = client.ClientModule(
            config={
                "world": "Town02",
                "fixed_delta_seconds": self.config["fixed_delta_seconds"]})

        self.world = self.client_module.get_world()
        self.map = self.client_module.get_map()
        self.client = self.client_module.get_client()

        self.spectator = self.world.get_spectator()

        # while True:
        #     # Fetch all spawn points
        #     spawn_points = self.map.get_spawn_points()
        #     # Select two random spawn points
        #     start_end_spawn_point = np.random.choice(spawn_points, 2)
        #     start = start_end_spawn_point[0]
        #     end = start_end_spawn_point[1]
        #     self.route = route.RouteModule(config={"start": start,
        #                                            "end": end,
        #                                            "sampling_resolution": 10,
        #                                            "debug": False},
        #                                    client=self.client)

        #     if ((RoadOption.LEFT not in [x[1] for x in self.route.get_route()]) and (RoadOption.RIGHT in [
        #             x[1] for x in self.route.get_route()[:10]]) and (len(self.route.get_route()) < 50) and
        #             (len(self.route.get_route()) > 0)):
        #         break

        # Fetch all spawn points
        spawn_points = self.map.get_spawn_points()
        # Select two random spawn points
        start_end_spawn_point = np.random.choice(spawn_points, 2)
        start = start_end_spawn_point[0]
        end = start_end_spawn_point[1]
        self.route = route.RouteModule(config={"start": start,
                                               "end": end,
                                               "sampling_resolution": 10,
                                               "debug": True},
                                       client=self.client)

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
            self.client, 80)
        self.traffic_manager_module = traffic_manager.TrafficManagerModule(
            config={"vehicle_list": actor_list}, client=self.client)
        # Sensor suite
        self.vehicle_sensor = vehicle_sensor.VehicleSensorModule(
            config=None, client=self.client,
            actor=self.hero_actor_module, id="ego")
        self.collision_sensor = collision_sensor.CollisionSensorModule(
            config=None, client=self.client,
            actor=self.hero_actor_module, id="col")
        self.occupancy_sensor = occupancy_sensor.OccupancySensorModule(
            config=None, actor=self.hero_actor_module, client=self.client, id="occ")
        self.rgb_sensor = rgb_sensor.RGBSensorModule(
            config={"yaw": 0, "width": 900, "height": 256}, client=self.client,
            actor=self.hero_actor_module, id="rgb_front")
        self.bev_module = BirdViewProducer(
            client=self.client,
            target_size=PixelDimensions(
                192,
                192),
            render_lanes_on_junctions=False,
            pixels_per_meter=5,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY,
            road_on_off=True,
            road_light=True,
            light_circle=False,
            lane_marking_thickness=2)

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
                    current_velocity = data_dict[k]["velocity"]

                    spectator_transform = carla.Transform(
                        location=carla.Location(
                            x=current_transform.location.x,
                            y=current_transform.location.y,
                            z=20.0),
                        rotation=carla.Rotation(
                            roll=0.0,
                            yaw=0.0,
                            pitch=-90.0))

                elif k == "col":

                    impulse = data_dict[k]["impulse"]
                    impulse_amplitude = np.linalg.norm(impulse)
                    logger.debug(f"Collision impulse: {impulse_amplitude}")
                    if impulse_amplitude > 1:
                        self.is_done = True
                        if self.config["save_video"]:
                            self.video_writer.release()

        data_dict["snapshot"] = snapshot

        self.spectator.set_transform(spectator_transform)

        route_step = self.route.step(
            self.map.get_waypoint(current_transform.location))
        # route_step = self.route.step(current_transform)
        # target_waypoint, _ = self.route.step(current_transform)

        bev = self.bev_module.step(
            agent_vehicle=self.hero_actor_module.get_actor())

        data_dict["bev"] = bev

        self.data.put(data_dict)

        self.server_module.step()
        self.client_module.step()

        self.counter += 1

        if (route_step is not None) and (self.counter <= 50000):
            return (
                current_transform,
                current_velocity,
                route_step[0],
                route_step[1])

        else:
            self.is_done = True
            if self.config["save_video"]:
                self.video_writer.release()
            return (current_transform, current_velocity, None, None)

    def render(self, predicted_location, bev, cost_canvas=None, **kwargs):
        """Render the environment"""
        for (k, v) in self.__dict__.items():
            if isinstance(v, Module):
                self.render_dict[k] = v.render()

        self.canvas = np.zeros_like(self.canvas)

        rgb_image = self.render_dict["rgb_sensor"]["image_data"]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # Put image into canvas
        self.canvas[:rgb_image.shape[0], :rgb_image.shape[1]] = rgb_image

        # semantic_image = self.render_dict["semantic_sensor"]["image_data"]
        # # semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2RGB)
        # # Put image into canvas
        # self.canvas[rgb_image.shape[0]:rgb_image.shape[0] +
        # semantic_image.shape[0], :semantic_image.shape[1]] = semantic_image

        bev = cv2.cvtColor(
            self.bev_module.as_rgb(
                bev,
            ),
            cv2.COLOR_BGR2RGB)
        # Put image into canvas
        bev = cv2.resize(bev, (0, 0), fx=4, fy=4)
        self.canvas[rgb_image.shape[0]:rgb_image.shape[0] +
                    bev.shape[0], :bev.shape[1]] = bev

        # Draw control as arrowed line to left corner of bev
        if "control" in kwargs.keys():
            action = kwargs["control"]
            action *= 100
            action = action.astype(np.int32)
            cv2.arrowedLine(
                self.canvas,
                (100, rgb_image.shape[0] + 100),
                (100 + action[1], rgb_image.shape[0] + 100 - action[0]),
                (0,
                 255,
                 0),
                1,
                tipLength=0.1)
        # Put text for other modules
        position_x = rgb_image.shape[1] + 10
        position_y = 20
        for (module, render_dict) in self.render_dict.items():
            if module not in ["rgb_sensor", "semantic_sensor"]:

                if bool(render_dict):

                    if "collision" not in module:

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
                    else:
                        # Draw a big GAME OVER text in red color
                        cv2.putText(
                            self.canvas,
                            "GAME OVER",
                            (position_x,
                             position_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            8,
                            (0,
                             0,
                             255),
                            5,
                            cv2.LINE_AA)

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
                [loc_[0], loc_[1], self.render_dict["hero_actor_module"]["location"].z])
            pixel_loc_ = world_2_pixel(
                loc_,
                self.render_dict["rgb_sensor"]["image_transform"].get_inverse_matrix(),
                rgb_image.shape[0],
                rgb_image.shape[1])

            ego_loc = np.array([self.render_dict["hero_actor_module"]["location"].x,
                                self.render_dict["hero_actor_module"]["location"].y,
                                self.render_dict["hero_actor_module"]["location"].z])
            bev_loc_ = world_2_bev(
                loc_,
                ego_loc,
                self.render_dict["hero_actor_module"]["rotation"].yaw,
                bev.shape[0],
                bev.shape[1],
                pixels_per_meter=20)

            if pixel_loc_.shape[0] > 0:
                cv2.circle(self.canvas, (int(pixel_loc_[0][0]), int(
                    pixel_loc_[0][1])), 5, (255, 0, 0), -1)
            cv2.circle(self.canvas, (int(bev_loc_[0]), int(
                bev_loc_[1] + rgb_image.shape[0])), 5, (255, 0, 0), -1)

        for k in range(
            self.render_dict["route"]["route_index"],
            np.minimum(
                self.render_dict["route"]["route_index"] + 10,
                self.render_dict["route"]["route_length"])):
            loc_ = self.route.route[k][0].transform.location
            loc_ = np.array([loc_.x, loc_.y, loc_.z])
            pixel_loc_ = world_2_pixel(
                loc_,
                self.render_dict["rgb_sensor"]["image_transform"].get_inverse_matrix(),
                rgb_image.shape[0],
                rgb_image.shape[1])

            ego_loc = np.array([self.render_dict["hero_actor_module"]["location"].x,
                                self.render_dict["hero_actor_module"]["location"].y,
                                self.render_dict["hero_actor_module"]["location"].z])

            bev_loc_ = world_2_bev(
                loc_,
                ego_loc,
                self.render_dict["hero_actor_module"]["rotation"].yaw,
                bev.shape[0],
                bev.shape[1],
                pixels_per_meter=20)
                

            if pixel_loc_.shape[0] > 0:
                cv2.circle(self.canvas, (int(pixel_loc_[0][0]), int(
                    pixel_loc_[0][1])), 5, (0, 255, 0), -1)
            if ((bev_loc_[0] < bev.shape[1]) and (
                    bev_loc_[1] + rgb_image.shape[0] > rgb_image.shape[0])):
                cv2.circle(self.canvas, (int(bev_loc_[0]), int(
                    bev_loc_[1] + rgb_image.shape[0])), 5, (0, 0, 0), -1)

        if cost_canvas is not None:

            scale_factor = np.minimum(self.canvas[position_y:,
                                                  bev.shape[1]:].shape[0] / cost_canvas.shape[0],
                                      self.canvas[position_y:,
                                                  bev.shape[1]:].shape[1] / cost_canvas.shape[1])
            scaled_cost_canvas = cv2.resize(
                cost_canvas, (0, 0), fx=scale_factor, fy=scale_factor)
            self.canvas[position_y:position_y +
                        scaled_cost_canvas.shape[0], bev.shape[1]:bev.shape[1] +
                        scaled_cost_canvas.shape[1]] = scaled_cost_canvas

        canvas_display = cv2.resize(
            src=self.canvas, dsize=(
                0, 0), fx=0.8, fy=0.8)

        cv2.imshow("Environment", canvas_display)

        canvas_save = cv2.resize(
            self.canvas,
            (self.canvas.shape[1] // 1,
             self.canvas.shape[0] // 1))
        cv2.imwrite(str(self.debug_path /
                        Path(f"{self.counter}.png")), canvas_save)

        if self.config["save_video"]:
            self.video_writer.write(canvas_save)

        cv2.waitKey(1)

    def close(self):
        """Close the environment"""
        self.traffic_manager_module.close()
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
            "save_video": False,
            "save": False,
            "render": True,
            "fixed_delta_seconds": 0.05,
        }

    def _create_render_window(self):

        self.canvas = np.zeros((1200, 2400, 3), np.uint8)
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

        debug_path = Path("figures/env_debug/test_mpc")

        date_ = Path(datetime.today().strftime('%Y-%m-%d'))
        time_ = Path(datetime.today().strftime('%H-%M-%S'))

        self.debug_path = debug_path / date_ / time_
        self.debug_path.mkdir(parents=True, exist_ok=True)
