from carla_env import carla_env_basic, carla_env_random_driver, carla_env_mpc
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
import torch
import time
import logging
import math
import numpy as np
import argparse
import wandb

torch.autograd.set_detect_anomaly(True)
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def main(config):

	# Initialize the environment
	ego_forward_model = KinematicBicycleModel(dt = 1/5)
	ego_forward_model.load_state_dict(
		torch.load(config.ego_forward_model_path))
	ego_forward_model.to(config.device)
	mpc_module = mpc.MPC(config.device, 30, 30, ego_forward_model)
	mpc_module.to(config.device)

	current_state = torch.zeros(
		(1, 4), device=config.device).unsqueeze(0)
	current_state[0,0,-1] = torch.pi/2
	current_state.requires_grad_(True)
	current_state.retain_grad()

	target_state = torch.zeros((1, 2), device=config.device).unsqueeze(0)
	target_state[0, 0, 0] = 1
	

    
	if config.wandb:
		run = wandb.init(project="mbl", group="mpc_module", tags = ["Debugging"],
						name="grad_investigation", config=config)
	
		run.watch(ego_forward_model, log = "all")

	counter = 0

	while torch.norm(target_state - current_state[:, :, 0:2]) > 0.1:

		if counter % 5 == 0:

			logging.info(f"Current state: {current_state}")

			action, _, _, _ = mpc_module.optimize_action(
				current_state, target_state)
			
			logging.info(f"Action: {action}")

			#action[0] = 1
			#action[1] = 0
			#action[2] = 0.0

			action = torch.Tensor(action).unsqueeze(0).unsqueeze(0)

			
			location = current_state[:, :, 0:2]
			yaw = current_state[:, :, 2]
			speed = current_state[:, :, 3]

			location_, yaw_, speed_ = ego_forward_model(
				location, yaw, speed, action)

			current_state = torch.cat(
				(location_, yaw_, speed_), dim=-1)
			
			mpc_module.reset()

		counter += 1


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="Collect data from the CARLA simulator")
	parser.add_argument("--ego_forward_model_path", type=str, default="pretrained_models/2022-09-22/16-24-58/ego_model_new.pt",
						help="Path to the forward model of the ego vehicle")
	parser.add_argument("--device", type=str, default="cpu",
						help="Device to use for the forward model")
	parser.add_argument("--wandb", type=bool, default=False)

	config = parser.parse_args()

	main(config)
