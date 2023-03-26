from carla_env import carla_env_mpc_bev_traffic
from carla_env.mpc.mpc_bev import ModelPredictiveControl
from carla_env.models.dynamic.vehicle import KinematicBicycleModelV2

from carla_env.cost.masked_cost_batched_simple_bev_with_pedestrian import Cost
from agents.navigation.local_planner import RoadOption
import torch
import time
import logging
import math
import argparse
from utils.kinematic_utils import acceleration_to_throttle_brake
from utils.model_utils import convert_standard_bev_to_model_bev

logging.basicConfig(level=logging.INFO)


def main(config):

    cost = Cost(image_width=192, image_height=192, device=config.device)

    ego_forward_model = KinematicBicycleModelV2(dt=1 / 20)

    ego_forward_model.load_state_dict(
        state_dict=torch.load(f=config.ego_forward_model_path)
    )

    ego_forward_model.to(device=config.device)

    mpc_module = ModelPredictiveControl(
        device=config.device,
        action_size=2,
        rollout_length=config.rollout_length,
        number_of_optimization_iterations=40,
        cost=cost,
        ego_model=ego_forward_model,
        render_cost=True,
    )

    mpc_module.to(device=config.device)

    c = carla_env_mpc_bev_traffic.CarlaEnvironment(
        config={
            "render": True,
            "save": True,
            "allowed_sensors": ["VehicleSensorModule", "CollisionSensorModule"],
            "save_video": True,
        }
    )

    (
        current_transform,
        current_velocity,
        target_waypoint,
        navigational_command,
    ) = c.step()

    data = c.get_data()
    bev_ = data["bev"]

    bev = convert_standard_bev_to_model_bev(bev=bev_, device=config.device)

    counter = 0

    while True:

        t0 = time.time()

        if counter % 1 == 0:

            # Set the current state of the ego vehicle for the kinematic model
            current_state = torch.zeros(size=(1, 4), device=config.device).unsqueeze(
                dim=0
            )

            current_state[..., 0] = current_transform.location.x
            current_state[..., 1] = current_transform.location.y
            current_state[..., 2] = current_transform.rotation.yaw * torch.pi / 180.0
            current_state[..., 3] = (
                math.sqrt(current_velocity.x**2 + current_velocity.y**2) + 0.01
            )
            current_state.requires_grad_(True)

            logging.debug(f"Current state: {current_state}")

            target_state = torch.zeros(size=(1, 4), device=config.device).unsqueeze(0)

            target_state[..., 0] = target_waypoint.transform.location.x
            target_state[..., 1] = target_waypoint.transform.location.y
            target_state[..., 2] = (
                target_waypoint.transform.rotation.yaw * torch.pi / 180.0
            )
            if (navigational_command != RoadOption.LANEFOLLOW) and (
                navigational_command != RoadOption.STRAIGHT
            ):

                target_state[..., 3] = 3

            else:

                target_state[..., 3] = 5

            logging.debug(f"Target state: {target_state}")
            # Get the control from the ModelPredictiveControl module
            control, location_predicted, cost, cost_canvas = mpc_module.step(
                current_state, target_state, bev
            )

        throttle, brake = acceleration_to_throttle_brake(acceleration=control[0])

        control = [throttle, control[1], brake]

        (
            current_transform,
            current_velocity,
            target_waypoint,
            navigational_command,
        ) = c.step(action=control)

        if c.is_done:
            break

        data = c.get_data()
        bev_ = data["bev"]

        bev = convert_standard_bev_to_model_bev(bev=bev_, device=config.device)

        t1 = time.time()

        c.render(
            predicted_location=location_predicted,
            bev=bev_,
            cost_canvas=cost_canvas,
            cost=cost,
            control=control,
            current_state=current_state,
            target_state=target_state,
            counter=counter,
            sim_fps=1 / (t1 - t0),
        )

        mpc_module.reset()

        counter += 1

    c.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle",
    )
    parser.add_argument("--rollout_length", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for the forward model"
    )
    config = parser.parse_args()

    main(config)
