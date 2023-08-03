import numpy as np
import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py
from carla_env.bev import BirdViewProducer, BIRDVIEW_CROP_TYPE, PixelDimensions

from externals.roach.carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from externals.roach.carla_gym.utils.traffic_light import TrafficLightHandler


class ObsManager(ObsManagerBase):
    def __init__(self, obs_configs):
        self._width = obs_configs["width"]
        self._height = obs_configs["height"]
        self._render_lanes_on_junctions = obs_configs["render_lanes_on_junctions"]
        self._pixels_per_meter = obs_configs["pixels_per_meter"]
        self._crop_type = obs_configs["crop_type"]
        self._road_on_off = obs_configs["road_on_off"]
        self._road_light = obs_configs["road_light"]
        self._light_circle = obs_configs["light_circle"]
        self._lane_marking_thickness = obs_configs["lane_marking_thickness"]
        self._bev_agent_channel = obs_configs["bev_agent_channel"]
        self._bev_vehicle_channel = obs_configs["bev_vehicle_channel"]
        self._bev_selected_channels = obs_configs["bev_selected_channels"]
        self._bev_calculate_offroad = obs_configs["bev_calculate_offroad"]

        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {
                "rendered": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._height, self._width, 3),
                    dtype=np.uint8,
                ),
                "masks": spaces.Box(
                    low=0,
                    high=255,
                    shape=(len(self._bev_selected_channels), self._height, self._width),
                    dtype=np.uint8,
                ),
            }
        )

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

        self._world = parent_actor.get_world()
        self._map = self._world.get_map()

        self._birdview_producer = BirdViewProducer(
            world=self._world,
            target_size=PixelDimensions(width=self._width, height=self._height),
            render_lanes_on_junctions=self._render_lanes_on_junctions,
            pixels_per_meter=self._pixels_per_meter,
            crop_type=BIRDVIEW_CROP_TYPE[self._crop_type],
            road_on_off=self._road_on_off,
            road_light=self._road_light,
            light_circle=self._light_circle,
            lane_marking_thickness=self._lane_marking_thickness,
        )

    def get_observation(self):
        actor = self._parent_actor.get_vehicle()

        # Get birdview masks
        masks = self._birdview_producer.step(
            agent_vehicle=actor,
            waypoint=None,
        )

        masks[..., self._bev_vehicle_channel] = np.logical_and(
            masks[..., self._bev_vehicle_channel],
            np.logical_not(masks[..., self._bev_agent_channel]),
        )

        bev = masks[..., self._bev_selected_channels]

        # Get birdview image
        rendered = BirdViewProducer.as_rgb_with_indices(
            birdview=bev, indices=self._bev_selected_channels
        )

        bev = np.transpose(bev, (2, 0, 1))

        obs_dict = {
            "rendered": rendered,
            "masks": bev,
        }

        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._world = None
