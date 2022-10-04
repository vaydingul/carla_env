from carla_env import carla_env_mpc_path_follower
from carla_env.mpc import mpc
from carla_env.models.dynamic.vehicle import KinematicBicycleModel, KinematicBicycleModelV2
from carla_env.models.dynamic.vehicle_WoR import EgoModel
import torch
import time
import logging
import math
import numpy as np
import argparse
from utils.kinematic_utils import acceleration_to_throttle_brake


logging.basicConfig(level=logging.INFO)


def main(config):

    # Initialize the environment
    if config.kinematic_model == "v1":

        ego_forward_model = KinematicBicycleModel(dt=1/20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.MPC(config.device, 3, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    elif config.kinematic_model == "v2":

        ego_forward_model = KinematicBicycleModelV2(dt=1/20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.MPC(config.device, 2, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    elif config.kinematic_model == "WoR":

        ego_forward_model = EgoModel(dt=1/20)
        ego_forward_model.load_state_dict(
            torch.load(config.ego_forward_model_path))
        ego_forward_model.to(config.device)

        mpc_module = mpc.MPC(config.device, 3, 10, 30, ego_forward_model)
        mpc_module.to(config.device)

    else:
        raise ValueError("Invalid kinematic model")

    c = carla_env_mpc_path_follower.CarlaEnvironment(config = {"render" : True, "save": True, "allowed_sensors" : ["VehicleSensorModule", "CollisionSensorModule"]}) 

    current_transform, current_velocity, target_waypoint = c.step()

    counter = 0

    while not c.is_done:

        if counter % 1 == 0:

            # Set the current state of the ego vehicle for the kinematic model
            current_state = torch.zeros(
                (1, 4), device=config.device).unsqueeze(0)

            current_state[..., 0] = current_transform.location.x
            current_state[..., 1] = current_transform.location.y
            current_state[..., 2] = current_transform.rotation.yaw * \
                torch.pi / 180.0
            current_state[..., 3] = math.sqrt(
                current_velocity.x**2 + current_velocity.y**2) + 0.01
            current_state.requires_grad_(True)

            logging.debug(f"Current state: {current_state}")

            target_state = torch.zeros(
                (1, 4), device=config.device).unsqueeze(0)

            target_state[..., 0] = target_waypoint.transform.location.x
            target_state[..., 1] = target_waypoint.transform.location.y
            target_state[..., 2] = target_waypoint.transform.rotation.yaw * \
                torch.pi / 180.0
            target_state[..., 3] = 5

            logging.debug(f"Target state: {target_state}")
            # Get the control from the MPC module
            control, location_predicted, cost = mpc_module.optimize_action(
                current_state, target_state)

        throttle, brake = acceleration_to_throttle_brake(control[0])
        control = [throttle, control[1], brake]



        current_transform, current_velocity, target_waypoint = c.step(control)
        c.render(location_predicted, cost = cost, control = control, current_state = current_state, target_state = target_state)


        mpc_module.reset()

        counter += 1

    c.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")
    parser.add_argument("--ego_forward_model_path", type=str, default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
                        help="Path to the forward model of the ego vehicle")
    parser.add_argument("--kinematic_model", type=str, default="v2")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for the forward model")
    config = parser.parse_args()

    main(config)
