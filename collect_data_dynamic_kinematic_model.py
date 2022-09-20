from ast import arg
from carla_env import carla_env_basic, carla_env_random_driver

import time
import logging
import cv2
import os
import numpy as np
import argparse
import sys
# Save the log to a file name with the current date
#logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.DEBUG)

def main(config):

	for k in range(config.num_episodes):

		c = carla_env_random_driver.CarlaEnvironment(None)

		t_init = time.time()

		while not c.is_done:

			c.step()

		c.close()

		# os.makedirs("images", exist_ok=True)
		# for ix, rgb_image in enumerate(rgbs):

		#     img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

		#     cv2.imwrite("images/{}.png".format(ix), img)

		vehicle_location_list = []
		vehicle_rotation_list = []
		vehicle_velocity_list = []
		vehicle_acceleration_list = []
		vehicle_control_list = []
		snapshot_list = []

		initial_vehicle_transform = np.array(c.initial_vehicle_transform.get_matrix())

		for _ in range(c.data.qsize()):

			data_point = c.data.get()
			snapshot_list.append(data_point["snapshot"])

			if data_point != {} and "VehicleSensorModule" in data_point.keys():

				vehicle_control =  data_point["VehicleSensorModule"]["control"]
				vehicle_location = data_point["VehicleSensorModule"]["location"]
				vehicle_rotation = data_point["VehicleSensorModule"]["rotation"]
				vehicle_velocity = data_point["VehicleSensorModule"]["velocity"]
				vehicle_acceleration = data_point["VehicleSensorModule"]["acceleration"]

				vehicle_location_list.append(np.array(
					[vehicle_location.x, vehicle_location.y, vehicle_location.z]))
				vehicle_velocity_list.append(np.array(
					[vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z]))
				vehicle_acceleration_list.append(np.array(
					[vehicle_acceleration.x, vehicle_acceleration.y, vehicle_acceleration.z]))
				vehicle_rotation_list.append(np.deg2rad(np.array(
					[vehicle_rotation.pitch, vehicle_rotation.yaw, vehicle_rotation.roll])))
				vehicle_control_list.append(np.array(vehicle_control))



		vehicle_location = np.array(vehicle_location_list)
		vehicle_velocity = np.array(vehicle_velocity_list)
		vehicle_acceleration = np.array(vehicle_acceleration_list)
		vehicle_rotation = np.array(vehicle_rotation_list)
		vehicle_control = np.array(vehicle_control_list)
		elapsed_time = np.array(
			[snapshot.timestamp.elapsed_seconds for snapshot in snapshot_list])

		np.savez(f"{config.data_save_path}/dynamic_kinematic_model_data_{k+2}", vehicle_location=vehicle_location, vehicle_rotation = vehicle_rotation, vehicle_velocity=vehicle_velocity,
				vehicle_acceleration=vehicle_acceleration, vehicle_control=vehicle_control, elapsed_time=elapsed_time)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Collect data from the CARLA simulator")
	parser.add_argument("--data_save_path", type=str, default="./data/kinematic_model_data/", help="Path to save the data")
	parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect data from")
	config = parser.parse_args()

	main(config)
