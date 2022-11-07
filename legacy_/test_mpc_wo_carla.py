from carla_env import carla_env_basic, carla_env_random_driver, carla_env_mpc
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle import KinematicBicycleModel, KinematicBicycleModelV2, KinematicBicycleModelWoR
from utils.plot_utils import plot_result_mpc
import torch
import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)


def main(config):

    # Initialize the environment
    if config.kinematic_model == "v1":

        ego_forward_model = KinematicBicycleModel(dt=1 / 20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.ModelPredictiveControl(config.device, 3, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    elif config.kinematic_model == "v2":

        ego_forward_model = KinematicBicycleModelV2(dt=1 / 20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.ModelPredictiveControl(config.device, 2, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    elif config.kinematic_model == "WoR":

        ego_forward_model = KinematicBicycleModelWoR(dt=1 / 20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.ModelPredictiveControl(config.device, 3, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    else:
        raise ValueError("Invalid kinematic model")

    current_state = torch.zeros(
        (1, 4), device=config.device).unsqueeze(0)
    current_state[0, 0, 3] = 0.01
    current_state.requires_grad_(True)
    current_state.retain_grad()

    target_state = torch.zeros((1, 4), device=config.device).unsqueeze(0)
    target_state[0, 0, 0] = -10
    target_state[0, 0, 1] = -10
    target_state[0, 0, 2] = 0
    target_state[0, 0, 3] = 3

    counter = 0

    state_list = []
    action_list = []

    while (torch.norm(current_state[..., 0:2] - target_state[..., 0:2]) > 0.2):

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

    savedir = f"figures/mpc_toy_examples/go_backward_direction/"
    os.makedirs(os.path.dirname(savedir), exist_ok=True)

    plot_result_mpc(state, action, target_state, savedir=Path(savedir))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")
    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle")
    parser.add_argument("--kinematic_model", type=str, default="v2")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for the forward model")
    parser.add_argument("--wandb", type=bool, default=False)

    config = parser.parse_args()

    main(config)
