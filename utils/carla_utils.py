import carla

def fetch_all_vehicles(client):
	world = client.get_world()
	vehicles = world.get_blueprint_library().filter("*vehicle*")
	vehicles = [".".join(vehicle.id.split(".")[1:]) for vehicle in vehicles]
	return vehicles