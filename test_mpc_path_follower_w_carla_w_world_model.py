from carla_env import carla_env_mpc_path_follower_bev_traffic
from carla_env.mpc.mpc_bev import ModelPredictiveControl
from carla_env.models.dynamic.vehicle import KinematicBicycleModelV2
from carla_env.models.world.world import WorldBEVModel
from carla_env.cost.masked_cost_batched import Cost
import torch
import time
import logging
import wandb
import math
import argparse
from collections import deque
from utils.kinematic_utils import acceleration_to_throttle_brake
from utils.model_utils import (
    load_world_model_from_wandb_run,
    load_ego_model_from_checkpoint,
    fetch_checkpoint_from_wandb_link,
    convert_standard_bev_to_model_bev)

logging.basicConfig(level=logging.INFO)


def main(config):

    cost = Cost(image_width=192, image_height=192, device=config.world_device)

    ego_forward_model = load_ego_model_from_checkpoint(
        checkpoint=config.ego_forward_model_path,
        cls=KinematicBicycleModelV2,
        dt=1 / 20)
    ego_forward_model.to(device=config.ego_device)

    run = wandb.Api().run(config.wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.wandb_link, config.checkpoint_number)

    world_forward_model, _ = load_world_model_from_wandb_run(
        run=run,
        checkpoint=checkpoint,
        cls=WorldBEVModel,
        world_model_device=config.world_device)
    world_forward_model.to(device=config.world_device)

    mpc_module = ModelPredictiveControl(
        device=config.mpc_device,
        action_size=2,
        rollout_length=config.rollout_length,
        number_of_optimization_iterations=20,
        cost=cost,
        ego_model=ego_forward_model,
        world_model=world_forward_model,
        render_cost=True)

    c = carla_env_mpc_path_follower_bev_traffic.CarlaEnvironment(
        config={
            "render": True,
            "save": True,
            "save_video": True})

    bev_tensor_deque = deque(maxlen=world_forward_model.num_time_step_previous)

    (current_transform, current_velocity,
     target_waypoint, navigational_command) = c.step()

    data = c.get_data()
    bev = data["bev"]
    agent_mask = bev[:, :, 3]

    for i in range(world_forward_model.num_time_step_previous):

        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev, device=config.world_device))

    counter = 0

    while not c.is_done:

        t0 = time.time()

        # Set the current state of the ego vehicle for the kinematic model
        current_state = torch.zeros(
            size=(1, 4), device=config.ego_device).unsqueeze(dim=0)

        current_state[..., 0] = current_transform.location.x
        current_state[..., 1] = current_transform.location.y
        current_state[..., 2] = current_transform.rotation.yaw * \
            torch.pi / 180.0
        current_state[..., 3] = math.sqrt(
            current_velocity.x**2 + current_velocity.y**2) + 0.01
        current_state.requires_grad_(True)

        logging.debug(f"Current state: {current_state}")

        target_state = torch.zeros(
            size=(1, 4), device=config.ego_device).unsqueeze(0)

        target_state[..., 0] = target_waypoint.transform.location.x
        target_state[..., 1] = target_waypoint.transform.location.y
        target_state[..., 2] = target_waypoint.transform.rotation.yaw * \
            torch.pi / 180.0

        logging.debug(f"Target state: {target_state}")

        # Get the control from the ModelPredictiveControl module

        # Convert bev tensor deque to torch tensor
        bev_tensor = torch.stack(list(bev_tensor_deque), dim=0).unsqueeze(0)
        (control,
         location_predicted,
         cost,
         cost_canvas) = mpc_module.step(initial_state=current_state,
                                        target_state=target_state,
                                        bev=bev_tensor,
                                        agent_mask=agent_mask)

        throttle, brake = acceleration_to_throttle_brake(
            acceleration=control[0])

        control = [throttle, control[1], brake]

        (current_transform, current_velocity, target_waypoint,
         navigational_command) = c.step(action=control)

        data = c.get_data()
        bev = data["bev"]
        agent_mask = bev[:, :, 3]
        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev, device=config.world_device))

        t1 = time.time()

        c.render(
            predicted_location=location_predicted,
            bev=bev,
            cost_canvas=cost_canvas,
            cost=cost,
            control=control,
            current_state=current_state,
            target_state=target_state,
            counter=counter,
            sim_fps=1 / (t1 - t0))

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

    parser.add_argument("--rollout_length", type=int, default=10)

    parser.add_argument(
        "--wandb_link",
        type=str,
        default="vaydingul/mbl/phys7134")

    parser.add_argument("--checkpoint_number", type=int, default=24)

    parser.add_argument("--ego_device", type=str, default="cuda:0",
                        help="Device to use for the forward model")

    parser.add_argument("--world_device", type=str, default="cuda:0",
                        help="Device to use for the world model")

    parser.add_argument("--mpc_device", type=str, default="cuda:0",
                        help="Device to use for the MPC module")

    config = parser.parse_args()

    main(config)
