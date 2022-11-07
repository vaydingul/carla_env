from carla_env import carla_env_basic, carla_env_random_driver, carla_env_mpc
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle import KinematicBicycleModel, KinematicBicycleModelV2, KinematicBicycleModelWoR
import torch
import time
import logging
import math
import numpy as np
import argparse
from utils.kinematic_utils import acceleration_to_throttle_brake


logging.basicConfig(level=logging.DEBUG)


def main(config):

    # Initialize the environment
    if config.kinematic_model == "v1":

        ego_forward_model = KinematicBicycleModel(dt=1 / 20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.MPC(config.device, 3, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    elif config.kinematic_model == "v2":

        ego_forward_model = KinematicBicycleModelV2(dt=1 / 20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.MPC(config.device, 2, 20, 40, ego_forward_model)
        mpc_module.to(config.device)

    elif config.kinematic_model == "WoR":

        ego_forward_model = KinematicBicycleModelWoR(dt=1 / 20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.MPC(config.device, 3, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    else:
        raise ValueError("Invalid kinematic model")

    c = carla_env_mpc.CarlaEnvironment(None)

    c.step()
    counter = 0

    first_iteration = False
    while not c.is_done:

        if counter % 1 == 0:

            data_point = c.data.get()["VehicleSensorModule"]

            location_ = data_point["location"]
            rotation_ = data_point["rotation"]
            velocity_ = data_point["velocity"]

            current_state = torch.zeros(
                (1, 4), device=config.device).unsqueeze(0)

            current_state[..., 0] = location_.x
            current_state[..., 1] = location_.y
            current_state[..., 2] = rotation_.yaw * torch.pi / 180.0
            current_state[..., 3] = math.sqrt(
                velocity_.x**2 + velocity_.y**2)
            current_state.requires_grad_(True)

            logging.debug(f"Current state: {current_state}")

            if not first_iteration:

                target_location = np.array([20, 0, location_.z, 0])
                transformation_matrix = np.array(
                    c.initial_vehicle_transform.get_matrix())
                target_location = np.matmul(
                    transformation_matrix, target_location)

                target_state = current_state.clone()
                target_state[..., 0] += target_location[0]
                target_state[..., 1] += target_location[1]
                target_state[..., 3] = 2
                target_state.to(config.device)
                first_iteration = True

            logging.debug(f"Target state: {target_state}")
            # Get the control from the MPC module
            control = mpc_module.optimize_action(
                current_state, target_state)

        throttle, brake = acceleration_to_throttle_brake(control[0])
        control = [throttle, control[1], brake]
        c.step(control)
        mpc_module.reset()

        counter += 1

    c.close()


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
    config = parser.parse_args()

    main(config)
