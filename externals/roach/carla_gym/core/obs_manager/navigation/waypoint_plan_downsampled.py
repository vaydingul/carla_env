import numpy as np
import carla
from gym import spaces

from externals.roach.carla_gym.core.obs_manager.obs_manager import ObsManagerBase

import externals.roach.carla_gym.utils.transforms as trans_utils


class ObsManager(ObsManagerBase):
    """
    Template config
    "obs_configs" = {
        "module": "navigation.waypoint_plan_downsampled",
        "steps": 10
    }
    [command, loc_x, loc_y]
    """

    def __init__(self, obs_configs):
        # self._steps = obs_configs["steps"]
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {
                "location": spaces.Box(
                    low=-100, high=1000, shape=(2,), dtype=np.float32
                ),
                "rotation": spaces.Box(
                    low=-180, high=180, shape=(3,), dtype=np.float32
                ),
                "command": spaces.Box(low=-1, high=6, shape=(1,), dtype=np.uint8),
                "road_id": spaces.Box(low=0, high=6000, shape=(1,), dtype=np.uint8),
                "lane_id": spaces.Box(low=-20, high=20, shape=(1,), dtype=np.int8),
                "is_junction": spaces.MultiBinary(1),
            }
        )

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = self._parent_actor.get_world()

    def get_observation(self):
        ev_transform = self._parent_actor.get_transform()

        route_plan = self._parent_actor.global_plan_world_coord

        route_length = len(route_plan)

        if 0 < route_length:
            waypoint, road_option = route_plan[0]
        else:
            waypoint, road_option = route_plan[-1]

        if isinstance(waypoint, carla.Transform):
            waypoint = self._world.get_map().get_waypoint(waypoint.location)
        if isinstance(waypoint, carla.Location):
            waypoint = self._world.get_map().get_waypoint(waypoint)

        wp_location_world_coord = waypoint.transform.location
        wp_rotation_world_coord = waypoint.transform.rotation

        location_list = [wp_location_world_coord.x, wp_location_world_coord.y]
        rotation_list = [
            wp_rotation_world_coord.roll,
            wp_rotation_world_coord.pitch,
            wp_rotation_world_coord.yaw,
        ]

        command_list = road_option.value
        road_id = waypoint.road_id
        lane_id = waypoint.lane_id
        is_junction = waypoint.is_junction

        obs_dict = {
            "location": np.array(location_list, dtype=np.float32),
            "rotation": np.array(rotation_list, dtype=np.float32),
            "command": np.array(command_list, dtype=np.int8),
            "road_id": np.array(road_id, dtype=np.int8),
            "lane_id": np.array(lane_id, dtype=np.int8),
            "is_junction": np.array(is_junction, dtype=np.int8),
        }
        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._world = None


# VOID = 0
# LEFT = 1
# RIGHT = 2
# STRAIGHT = 3
# LANEFOLLOW = 4
# CHANGELANELEFT = 5
# CHANGELANERIGHT = 6
