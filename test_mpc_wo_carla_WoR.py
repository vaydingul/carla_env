from carla_env import carla_env_basic, carla_env_random_driver, carla_env_mpc
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle_WoR import EgoModel
import torch
import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
# Save the log to a file name with the current date
# logging.basicConfig(filename=f"logs/sim_log_debug",level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)


def main(config):

	# Initialize the environment
	ego_forward_model = EgoModel(dt = 1/20)
	ego_forward_model.load_state_dict(
		torch.load(config.ego_forward_model_path))
	ego_forward_model.to(config.device)
	mpc_module = mpc.MPC(config.device, 10, 30, ego_forward_model)
	mpc_module.to(config.device)

	current_state = torch.zeros(
		(1, 4), device=config.device).unsqueeze(0)
	current_state[0, 0, 3] = 1
	current_state.requires_grad_(True)
	current_state.retain_grad()

	target_state = torch.zeros((1, 4), device=config.device).unsqueeze(0)
	target_state[0, 0, 0] = 5
	target_state[0, 0, 1] = 20
	target_state[0, 0, 2] = torch.pi / 4
	target_state[0, 0, 3] = 5
	
	counter = 0

	state_list = []
	action_list = []

	while torch.norm(current_state[..., 0:2] - target_state[..., 0:2]) > 0.5:

		if counter % 5 == 0:

			logging.info(f"Target State: {target_state}")
			logging.info(f"Current state: {current_state}")

			action = mpc_module.optimize_action(
				current_state, target_state)
			
			logging.info(f"Action: {action}")

			action = torch.Tensor(action).unsqueeze(0).unsqueeze(0)


			# action[0] = 1
			# action[1] = 0
			# action[2] = 0.0

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

	plt.figure()
	plt.plot(state[:, 0, 1], state[:, 0, 0])
	plt.plot(target_state[0, 0, 1], target_state[0, 0, 0], 'ro')
	plt.show()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="Collect data from the CARLA simulator")
	parser.add_argument("--ego_forward_model_path", type=str, default="pretrained_models/WoR/ego_model.th",
						help="Path to the forward model of the ego vehicle")
	parser.add_argument("--device", type=str, default="cpu",
						help="Device to use for the forward model")
	parser.add_argument("--wandb", type=bool, default=False)

	config = parser.parse_args()

	main(config)
