from carla_env import carla_env_basic, carla_env_random_driver, carla_env_mpc
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
import torch
import time
import logging
import math
import numpy as np
import argparse

torch.autograd.set_detect_anomaly(True)
# Save the log to a file name with the current date
#logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.DEBUG)

def main(config):
	
	

	ego_forward_model = KinematicBicycleModel()
	ego_forward_model.load_state_dict(torch.load(config.ego_forward_model_path))
	ego_forward_model.to(config.device)
	mpc_module = mpc.MPC(config.device, 10, 40)
	mpc_module.to(config.device)

	c = carla_env_mpc.CarlaEnvironment(None)
	
	current_state = torch.zeros((1, 4), device=config.device).unsqueeze(0)
	
	c.step()
	
	while True:

		data_point = c.data.get()["VehicleSensorModule"]

		location_ = data_point["location"]
		rotation_ = data_point["rotation"]
		velocity_ = data_point["velocity"]	

		current_state[..., 0] = location_.x
		current_state[..., 1] = location_.y
		current_state[..., 2] = rotation_.yaw
		current_state[..., 3] = math.sqrt(velocity_.x**2 + velocity_.y**2)
		
		logging.debug(f"Current state: {current_state}")

		# Get the control from the MPC module
		temp_ = torch.zeros((1,2), device=config.device).unsqueeze(0)
		temp_[..., 0] = 1
		temp_ += current_state[..., 0:2]
		control, _, _, _ = mpc_module.optimize_action(current_state, temp_, ego_forward_model)
		
		#control[0] = -float(control[0])
		#control[1] = float(control[1])
		#control[2] = 0#float(control[2])

		c.step(control)

	c.close()


	# vehicle_location_list = []
	# vehicle_rotation_list = []
	# vehicle_velocity_list = []
	# vehicle_acceleration_list = []
	# vehicle_control_list = []
	# snapshot_list = []

	# initial_vehicle_transform = np.array(c.initial_vehicle_transform.get_matrix())

	# for _ in range(c.data.qsize()):

	# 	data_point = c.data.get()
	# 	snapshot_list.append(data_point["snapshot"])

	# 	if data_point != {} and "VehicleSensorModule" in data_point.keys():

	# 		vehicle_control =  data_point["VehicleSensorModule"]["control"]
	# 		vehicle_location = data_point["VehicleSensorModule"]["location"]
	# 		vehicle_rotation = data_point["VehicleSensorModule"]["rotation"]
	# 		vehicle_velocity = data_point["VehicleSensorModule"]["velocity"]
	# 		vehicle_acceleration = data_point["VehicleSensorModule"]["acceleration"]

	# 		vehicle_location_list.append(np.array(
	# 			[vehicle_location.x, vehicle_location.y, vehicle_location.z]))
	# 		vehicle_velocity_list.append(np.array(
	# 			[vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z]))
	# 		vehicle_acceleration_list.append(np.array(
	# 			[vehicle_acceleration.x, vehicle_acceleration.y, vehicle_acceleration.z]))
	# 		vehicle_rotation_list.append(np.deg2rad(np.array(
	# 			[vehicle_rotation.pitch, vehicle_rotation.yaw, vehicle_rotation.roll])))
	# 		vehicle_control_list.append(np.array(vehicle_control))



	# vehicle_location = np.array(vehicle_location_list)
	# vehicle_velocity = np.array(vehicle_velocity_list)
	# vehicle_acceleration = np.array(vehicle_acceleration_list)
	# vehicle_rotation = np.array(vehicle_rotation_list)
	# vehicle_control = np.array(vehicle_control_list)
	# elapsed_time = np.array(
	# 	[snapshot.timestamp.elapsed_seconds for snapshot in snapshot_list])

	# np.savez(f"{config.data_save_path}/dynamic_kinematic_model_data_{k}", vehicle_location=vehicle_location, vehicle_rotation = vehicle_rotation, vehicle_velocity=vehicle_velocity,
	# 		vehicle_acceleration=vehicle_acceleration, vehicle_control=vehicle_control, elapsed_time=elapsed_time)

	# del vehicle_location_list, vehicle_rotation_list, vehicle_velocity_list, vehicle_acceleration_list, vehicle_control_list, snapshot_list, vehicle_location, vehicle_velocity, vehicle_acceleration, vehicle_rotation, vehicle_control, elapsed_time


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Collect data from the CARLA simulator")
	parser.add_argument("--ego_forward_model_path", type=str, default="pretrained_models/2022-09-22/16-24-58/ego_model_new.pt", help="Path to the forward model of the ego vehicle")
	parser.add_argument("--device", type=str, default="cuda", help="Device to use for the forward model")
	config = parser.parse_args()

	main(config)
