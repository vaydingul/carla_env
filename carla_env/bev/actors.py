import carla
from typing import NamedTuple, List


def is_vehicle(actor): return "vehicle" in actor.type_id
def is_pedestrian(actor): return ("walker" in actor.type_id) and ("controller.ai" not in actor.type_id)
def is_traffic_light(actor): return "traffic_light" in actor.type_id


class SegregatedActors(NamedTuple):
    vehicles: List[carla.Actor]
    pedestrians: List[carla.Actor]
    traffic_lights: List[carla.Actor]


def segregate_by_type(actors: List[carla.Actor], agent_vehicle: carla.Actor) -> SegregatedActors:
    vehicles = []
    pedestrians = []
    traffic_lights = []
    for actor in actors:
        if is_vehicle(actor) and actor.id != agent_vehicle.id:
            vehicles.append(actor)
        elif is_pedestrian(actor):
            pedestrians.append(actor)
        elif is_traffic_light(actor):
            traffic_lights.append(actor)
    return SegregatedActors(vehicles, pedestrians, traffic_lights)


def query_all(world: carla.World) -> List[carla.Actor]:
    snapshot: carla.WorldSnapshot = world.get_snapshot()
    all_actors = []
    for actor_snapshot in snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if actor is not None:
            all_actors.append(actor)
    return all_actors
