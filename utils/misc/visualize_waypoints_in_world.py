import carla

RED = carla.Color(r=255, g=0, b=0)
GREEN = carla.Color(r=0, g=255, b=0)
BLUE = carla.Color(r=0, g=0, b=255)
MAGENTA = carla.Color(r=255, g=0, b=255)
CYAN = carla.Color(r=0, g=255, b=255)


def visualize_waypoints_in_world(world="Town04"):
    """Visualize waypoints in world."""
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    world = client.load_world(world)
    map = world.get_map()
    points = map.get_spawn_points()

    for point in points:
        waypoint = map.get_waypoint(point.location)
        world.debug.draw_point(
            waypoint.transform.location,
            size=0.5,
            color=RED,
            life_time=10000)
        offset_loc = waypoint.transform.location + carla.Location(z=0.5)
        world.debug.draw_string(
            offset_loc,
            f"{str(waypoint.transform.location)} -- {str(waypoint.transform.rotation)}",
            color=RED,
            life_time=10000)


if __name__ == "__main__":
    visualize_waypoints_in_world()
