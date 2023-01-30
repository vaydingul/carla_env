import carla

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
