from carla_env.carla_env_mpc_bev_traffic import CarlaEnvironment
from carla_env.models.dfm_km_cp import DecoupledForwardModelKinematicsCoupledPolicy
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy import Policy
from carla_env.cost.masked_cost_batched_policy_extended_bev import Cost
from carla_env.bev import BirdViewProducer
import torch
import time
import logging
import wandb
import numpy as np
import math
import argparse
import cv2
from collections import deque
from utils.kinematic_utils import acceleration_to_throttle_brake
from utils.model_utils import (
    load_world_model_from_wandb_run,
    load_ego_model_from_checkpoint,
    load_policy_model_from_wandb_run,
    fetch_checkpoint_from_wandb_link,
    convert_standard_bev_to_model_bev,
)
from utils.train_utils import seed_everything, get_device

logging.basicConfig(level=logging.INFO)


def main(config):

    # config.seed = int(np.random.rand(1) * 1000)
    seed_everything(config.seed)
    device = get_device()

    # ---------------------------------------------------------------------------- #
    #                              Import what you need                            #
    # --------------------  ------------------------------------------------------ #

    # model = SomeFancyTransformerModel()

    # ---------------------------------------------------------------------------- #
    #                               CARLA Environment                              #
    # ---------------------------------------------------------------------------- #
    c = CarlaEnvironment(
        config={
            "render": True,
            "save": True,
            "save_video": True,
            "fixed_delta_seconds": config.dt,
        }
    )

    NUM_TIME_STEP_PREVIOUS = 5  # Number of conditioning frames
    bev_tensor_deque = deque(maxlen=NUM_TIME_STEP_PREVIOUS)

    (
        current_transform,
        current_velocity,
        target_waypoint,
        navigational_command,
    ) = c.step()  # Initial step to fill the deque

    # Fetch all sensor data from environment
    data = c.get_data()
    # Take BEV from data
    bev = data["bev"]

    for i in range(NUM_TIME_STEP_PREVIOUS):

        # Some preprocessing on BEV to make it
        # compatible with the PyTorch and model

        # You will understand what function does
        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev,
                agent_channel=7,
                vehicle_channel=6,
                selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
                calculate_offroad=False,
                device=device,
            )
        )

    counter = 0

    with torch.no_grad():
        while not c.is_done:

            t0 = time.time()

            # Play with the return of the environment below

            # location = torch.zeros(
            #     size=(1, 2), device=device)
            # location[..., 0] = current_transform.location.x
            # location[..., 1] = current_transform.location.y
            # yaw = torch.zeros(
            #     size=(1, 1), device=device)
            # yaw[..., 0] = current_transform.rotation.yaw * math.pi / 180
            # speed = torch.zeros(
            #     size=(1, 1), device=device)
            # speed[..., 0] = math.sqrt(
            #     current_velocity.x**2 + current_velocity.y**2)
            # location.requires_grad_(True).to(device=device)
            # yaw.requires_grad_(True).to(device=device)
            # speed.requires_grad_(True).to(device=device)

            # ego_state = {"location": location,
            #              "yaw": yaw,
            #              "speed": speed}

            # target_location = torch.zeros(
            #     size=(1, 2), device=device)
            # target_location[..., 0] = target_waypoint.transform.location.x
            # target_location[..., 1] = target_waypoint.transform.location.y
            # target_location = target_location.to(device=device)

            # Convert bev tensor deque to torch tensor
            bev_tensor = torch.cat(list(bev_tensor_deque), dim=0).unsqueeze(
                0
            )  # Batch dimension is added

            # Convert navigational command to one-hot torch tensor
            navigational_command = torch.tensor(
                navigational_command.value - 1, device=device
            ).unsqueeze(0)
            navigational_command = (
                torch.nn.functional.one_hot(navigational_command, num_classes=6)
                .float()
                .to(device=device)
            )

            # Predict control from your model
            # control = model(bev_tensor, navigational_command)

            # Some helper function to convert acceleration to throttle and
            # brake
            throttle, brake = acceleration_to_throttle_brake(acceleration=control[0])

            # Final control vector that will be passed to the environment
            control = [throttle, control[1], brake]

            (
                current_transform,
                current_velocity,
                target_waypoint,
                navigational_command,
            ) = c.step(action=control)

            data = c.get_data()
            bev = data["bev"]
            bev_ = bev[..., [0, 1, 2, 3, 4, 5, 6, 11]]
            bev_tensor_deque.append(
                convert_standard_bev_to_model_bev(
                    bev,
                    agent_channel=7,
                    vehicle_channel=6,
                    selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
                    calculate_offroad=False,
                    device=device,
                )
            )

            t1 = time.time()

            # The messiest part of my whole codebase :(
            # I am sorry for this
            # Just try to survive
            c.render(
                predicted_location=None,
                bev=bev_,
                cost_canvas=None,  # After here everything is optional, it will be rendered as text
                control=control,
                counter=counter,
                sim_fps=1 / (t1 - t0),
                seed=config.seed,
            )

            counter += 1

    c.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test pretrained model on Carla environment"
    )

    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--dt", type=float, default=0.1)

    config = parser.parse_args()

    main(config)
