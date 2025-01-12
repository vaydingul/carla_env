import carla
from agents.navigation.local_planner_behavior import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import numpy as np

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


ROAD_OPTION_COLOR_MAP = {
    "LEFT": CYAN,
    "RIGHT": MAGENTA,
    "STRAIGHT": RED,
    "LANEFOLLOW": GREEN,
    "CHANGELANELEFT": YELLOW,
    "CHANGELANERIGHT": BLUE,
}


def calculate_route(world, start=None, end=None):
    """Calculate route between two points."""

    dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=0.1)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    route = grp.trace_route(start, end)
    return route


def initialize_client(world):
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.load_world(world)
    return world


def visualize_route(world, start=None, end=None):
    """Visualize route waypoints in world."""

    map = world.get_map()
    points = np.random.choice(map.get_spawn_points(), 2)

    if start is None:
        start = points[0].location
    if end is None:
        end = points[1].location

    route = calculate_route(world, start, end)

    for ix, (waypoint, road_option) in enumerate(route):
        print(f"Coordinate {ix}: {waypoint.transform.location}")
        print(f"Road option {ix}: {road_option}")

        if ix <= 10:

            world.debug.draw_string(
                waypoint.transform.location, "o", color=WHITE, life_time=10000
            )

        elif ix >= len(route) - 10:

            world.debug.draw_string(
                waypoint.transform.location, "o", color=BLACK, life_time=10000
            )

        else:

            world.debug.draw_string(
                waypoint.transform.location, "o", color=GRAY, life_time=10000
            )

        # offset_loc = waypoint.transform.location + carla.Location(z=0.5)
        # world.debug.draw_string(offset_loc, str(waypoint.transform.location), color=RED, life_time=10000)
    return start, end


def main(world="Town06"):
    world = initialize_client(world)

    start, _ = visualize_route(world)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(start, carla.Rotation()))


if __name__ == "__main__":
    main()
