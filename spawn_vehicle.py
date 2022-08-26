from carla_env.modules.server import server
from carla_env.modules.client import client
from carla_env.modules.actor import actor
from carla_env.modules.vehicle import vehicle

import time
import numpy as np

if __name__ == "__main__":

	# We have our server and client up and running
	server = server.ServerModule(None)
	server._start()

	time.sleep(5.0)

	client = client.ClientModule(None)
	client._start()

	spectator = client.world.get_spectator()

	# Let's initialize a vehicle

	vehicle = vehicle.VehicleModule({"vehicle_model" : "lincoln.mkz2017"}, client.client)

	# Make this vehicle actor

	actor = actor.ActorModule({"actor" : vehicle, "hero" : True}, client.client)

	spawn_points = client._get_spawn_points()
	selected_spawn_point = np.random.choice(spawn_points)
	actor._start(selected_spawn_point)
	spectator.set_transform(selected_spawn_point)
	
