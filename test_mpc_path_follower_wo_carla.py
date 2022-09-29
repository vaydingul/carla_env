from carla_env import carla_env_basic, carla_env_random_driver, carla_env_mpc
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
import torch
import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils.plot_utils import plot_result_mpc_path_follow

torch.autograd.set_detect_anomaly(True)
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def go_to_waypoint(current_state, target_state, mpc_module, ego_forward_model):

	counter = 0

	state_list = []
	action_list = []

	while torch.norm(current_state[..., 0:2] - target_state[..., 0:2]) > 0.2:

		if counter % 1 == 0:

			logging.info(f"Target State: {target_state}")
			logging.info(f"Current state: {current_state}")

			action = mpc_module.optimize_action(
				current_state, target_state)

			logging.info(f"Action: {action}")

			action = torch.Tensor(action).unsqueeze(0).unsqueeze(0)

		location = current_state[:, :, 0:2]
		yaw = current_state[:, :, 2]
		speed = current_state[:, :, 3]

		location_, yaw_, speed_ = ego_forward_model(
			location, yaw, speed, action)

		current_state = torch.cat(
			(location_, yaw_, speed_), dim=-1)

		state_list.append(current_state.detach().cpu().numpy())
		action_list.append(action)

		mpc_module.reset()

		counter += 1


	state = np.concatenate(state_list, axis=0)
	action = np.concatenate(action_list, axis=0)

	return state, action

def main(config):

	# Initialize the environment
	ego_forward_model = KinematicBicycleModel(dt=1/20)
	ego_forward_model.load_state_dict(
		torch.load(config.ego_forward_model_path))
	ego_forward_model.to(config.device)
	mpc_module = mpc.MPC(config.device, 10, 30, ego_forward_model)
	mpc_module.to(config.device)

	data = np.load(config.validation_data_path)
	vehicle_location = torch.Tensor(data["vehicle_location"])
	vehicle_rotation = torch.Tensor(data["vehicle_rotation"])
	vehicle_velocity = torch.Tensor(data["vehicle_velocity"])
	vehicle_control = torch.Tensor(data["vehicle_control"])
	elapsed_time = torch.Tensor(data["elapsed_time"])

	state_list = []
	action_list = []
	target_state_list = []

	stride = 10
	for k in range(50, vehicle_location.shape[0] - 1 - stride, stride):

		current_state = torch.cat(
			(vehicle_location[k, 0:2], vehicle_rotation[k, 1:2], torch.norm(vehicle_velocity[k], dim = -1, keepdim=True)), dim=-1).unsqueeze(0).unsqueeze(0)
		current_state.requires_grad = True

		target_state = torch.cat(
			(vehicle_location[k+stride, 0:2], vehicle_rotation[k+stride, 1:2], torch.norm(vehicle_velocity[k+stride], dim = -1, keepdim=True)), dim=-1).unsqueeze(0).unsqueeze(0)
		target_state_list.append(target_state.detach().cpu().numpy())

		state, action = go_to_waypoint(current_state, target_state, mpc_module, ego_forward_model)
		
		state_list.append(state)
		action_list.append(action)

	state = np.concatenate(state_list, axis=0)
	action = np.concatenate(action_list, axis=0)
	target_state = np.concatenate(target_state_list, axis=0)

	vehicle_location = vehicle_location.detach().cpu().numpy()
	vehicle_rotation = vehicle_rotation.detach().cpu().numpy()
	vehicle_velocity = vehicle_velocity.detach().cpu().numpy()
	vehicle_control = vehicle_control.detach().cpu().numpy()


	savedir =  f"figures/mpc_toy_examples/path_follow_1/"
	os.makedirs(os.path.dirname(savedir), exist_ok=True)

	plot_result_mpc_path_follow(state, action, vehicle_location, vehicle_rotation, vehicle_velocity, vehicle_control, target_state, 50, k+stride, Path(savedir))
	
	



	



if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="Collect data from the CARLA simulator")
	# parser.add_argument("--ego_forward_model_path", type=str, default="pretrained_models/2022-09-22/16-24-58/ego_model_new.pt",
	# 					help="Path to the forward model of the ego vehicle")
	parser.add_argument("--ego_forward_model_path", type=str, default="pretrained_models/2022-09-28/03-24-39/ego_model_new.pt",
						help="Path to the forward model of the ego vehicle")
	parser.add_argument("--validation_data_path", type=str,
						default="data/kinematic_model_data_val_2/dynamic_kinematic_model_data_5.npz")
	parser.add_argument("--device", type=str, default="cpu",
						help="Device to use for the forward model")
	parser.add_argument("--wandb", type=bool, default=False)

	config = parser.parse_args()

	main(config)
