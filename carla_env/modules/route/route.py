from carla_env.modules import module
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

RED = carla.Color(r=255, g=0, b=0)
GREEN = carla.Color(r=0, g=255, b=0)
BLUE = carla.Color(r=0, g=0, b=255)
MAGENTA = carla.Color(r=255, g=0, b=255)
CYAN = carla.Color(r=0, g=255, b=255)
YELLOW = carla.Color(r=255, g=255, b=0)
BLACK = carla.Color(r=0, g=0, b=0)
WHITE = carla.Color(r=255, g=255, b=255)
GRAY = carla.Color(r=128, g=128, b=128)
PINK = carla.Color(r=255, g=0, b=255)


class RouteModule(module.Module):
    """Concrete implementation of Module abstract base class for route management"""

    def __init__(self, config, client) -> None:
        super().__init__()
        self.client = client

        self._set_default_config()
        if config is not None:
            for k in config.keys():
                self.config[k] = config[k]
        self.world = self.client.get_world()
        self.grp = GlobalRoutePlanner(
            self.world.get_map(),
            self.config["sampling_resolution"])

        self.route = self.grp.trace_route(
            self.config["start"].location,
            self.config["end"].location)
        self.route_length = len(self.route)

        self.render_dict = {}

        if self.config["debug"]:
            self._visualize_route()

        self.route_index = 0

    def _start(self, spawn_transform):
        """Start the vehicle manager"""
        pass

    def step(self, current_location):
        """Step the vehicle manager"""
        if self.route_index < self.route_length - 1:
            # self.config["sampling_resolution"]:
            if _get_distance_between_waypoints(
                    self.route[self.route_index][0], current_location) < 0.5:
                self.route_index += 1
            return self.route[self.route_index]
        else:
            return None

    def _stop(self):
        """Stop the vehicle manager"""
        pass

    def reset(self):
        """Reset the vehicle manager"""
        pass

    def render(self):
        """Render the vehicle manager"""
        self.render_dict["current_waypoint"] = self.route[self.route_index][
            0].transform.location if self.route_index < self.route_length else None
        self.render_dict["current_command"] = self.route[self.route_index][1] if self.route_index < self.route_length else None
        self.render_dict["route_index"] = self.route_index
        self.render_dict["route_length"] = self.route_length
        return self.render_dict

    def close(self):
        """Close the vehicle manager"""
        pass

    def seed(self):
        """Seed the vehicle manager"""
        pass

    def get_config(self):
        """Get the config of the vehicle manager"""
        return self.config

    def _set_default_config(self):
        """Set the default config of the vehicle"""
        self.config = {"sampling_resolution": 1}

    def _visualize_route(self):
        """Visualize the route"""
        for ix, (waypoint, road_option) in enumerate(self.route):

            # print(f"Coordinate {ix}: {waypoint.transform.location}")
            # print(f"Road option {ix}: {road_option}")

            if ix <= 10:

                self.world.debug.draw_string(
                    waypoint.transform.location, 'o', color=RED, life_time=10000)

            elif ix >= self.route_length - 10:

                self.world.debug.draw_string(
                    waypoint.transform.location, 'o', color=GREEN, life_time=10000)

            else:

                self.world.debug.draw_string(
                    waypoint.transform.location, 'o', color=BLUE, life_time=10000)


def _get_distance_between_waypoints(waypoint1, waypoint2):
    """Get the distance between two waypoints"""
    if isinstance(waypoint2, carla.Transform):
        return waypoint1.transform.location.distance(waypoint2.location)
    else:
        return waypoint1.transform.location.distance(
            waypoint2.transform.location)
