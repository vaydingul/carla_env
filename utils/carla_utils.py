import carla
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle
import numpy as np


BIG_VEHICLES = ["carlacola", "firetruck", "ambulance"]


def fetch_all_vehicles(client, spawn_big_vehicle=False):
    world = client.get_world()
    vehicles = world.get_blueprint_library().filter("*vehicle*")

    vehicles_ = []
    if spawn_big_vehicle:
        vehicles_ = [".".join(vehicle.id.split(".")[1:]) for vehicle in vehicles]
    else:
        for vehicle in vehicles:
            if vehicle.id.split(".")[-1] not in BIG_VEHICLES:
                vehicles_.append(".".join(vehicle.id.split(".")[1:]))
    return vehicles_


def create_multiple_actors_for_traffic_manager(client, n=20):
    """Create multiple vehicles in the world"""

    vehicles = fetch_all_vehicles(client)
    vehicles = vehicles * (n // len(vehicles) + 1)
    # Shuffle the list and take first n vehicles
    np.random.shuffle(vehicles)
    actors = []

    for k in range(n):

        actors.append(
            actor.ActorModule(
                config={
                    "vehicle": vehicle.VehicleModule(
                        config={
                            "vehicle_model": vehicles[k],
                        },
                        client=client,
                    ),
                    "hero": False,
                },
                client=client,
            )
        )

    return actors
