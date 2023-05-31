from carla_env.carla_env_mpc_extended_bev_traffic import CarlaEnvironment
from carla_env.mpc.mpc_extended_bev import ModelPredictiveControl
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.models.world.world import WorldBEVModel
from carla_env.cost.masked_cost_batched_mpc_extended_bev import Cost
import torch
import time
import logging
import wandb
import math
import argparse
from collections import deque
from utilities.kinematic_utils import acceleration_to_throttle_brake
from utilities.model_utils import (
    load_world_model_from_wandb_run,
    load_ego_model_from_checkpoint,
    fetch_checkpoint_from_wandb_link,
    convert_standard_bev_to_model_bev,
)
from utilities.train_utils import seed_everything

logging.basicConfig(level=logging.INFO)


def main(config):

    # seed_everything(seed=config.seed)

    # ---------------------------------------------------------------------------- #
    #                              Cost initialization                             #
    # ---------------------------------------------------------------------------- #
    cost = Cost(
        image_width=192,
        image_height=192,
        device=config.world_device,
    )

    # ---------------------------------------------------------------------------- #
    #                         Pretrained ego forward model                         #
    # ---------------------------------------------------------------------------- #
    ego_model_run = wandb.Api().run(config.ego_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        wandb_link=config.ego_forward_model_wandb_link,
        checkpoint_number=config.ego_forward_model_checkpoint_number,
    )
    ego_forward_model = KinematicBicycleModel.load_model_from_wandb_run(
        run=ego_model_run, checkpoint=checkpoint, device=config.ego_device
    )
    ego_forward_model = ego_forward_model.to(device=config.ego_device).eval()

    # ---------------------------------------------------------------------------- #
    #                        Pretrained world forward model                        #
    # ---------------------------------------------------------------------------- #
    world_model_run = wandb.Api().run(config.world_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.world_forward_model_wandb_link,
        config.world_forward_model_checkpoint_number,
    )
    world_forward_model = WorldBEVModel.load_model_from_wandb_run(
        run=world_model_run, checkpoint=checkpoint, device=config.world_device
    )
    world_forward_model = world_forward_model.to(device=config.world_device).eval()

    mpc_module = ModelPredictiveControl(
        device=config.mpc_device,
        batch_size=config.batch_size,
        rollout_length=config.rollout_length,
        action_size=config.action_size,
        number_of_optimization_iterations=40,
        cost=cost,
        ego_model=ego_forward_model,
        init_action="zeros",
        world_model=world_forward_model,
        render_cost=True,
    )

    c = CarlaEnvironment(
        config={
            "render": True,
            "save": True,
            "save_video": True,
            "fixed_delta_seconds": 0.1,
        }
    )

    bev_tensor_deque = deque(maxlen=world_forward_model.num_time_step_previous)

    (
        current_transform,
        current_velocity,
        target_waypoint,
        navigational_command,
    ) = c.step()

    data = c.get_data()
    bev = data["bev"]

    for i in range(world_forward_model.num_time_step_previous):

        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev,
                device=config.world_device,
                agent_channel=7,
                vehicle_channel=6,
                selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
                calculate_offroad=False,
            )
        )

    frame_counter = 0
    skip_counter = 0
    repeat_counter = 0

    while not c.is_done:

        t0 = time.time()

        # Set the current state of the ego vehicle for the kinematic model
        current_state = torch.zeros(
            size=(config.batch_size, 1, 4), device=config.ego_device
        )

        current_state[..., 0] = current_transform.location.x
        current_state[..., 1] = current_transform.location.y
        current_state[..., 2] = current_transform.rotation.yaw * torch.pi / 180.0
        current_state[..., 3] = (
            math.sqrt(current_velocity.x**2 + current_velocity.y**2) + 1e-2
        )
        current_state.requires_grad_(True)

        logging.debug(f"Current state: {current_state}")

        target_state = torch.zeros(
            size=(config.batch_size, 1, 4), device=config.ego_device
        )

        target_state[..., 0] = target_waypoint.transform.location.x
        target_state[..., 1] = target_waypoint.transform.location.y
        target_state[..., 2] = target_waypoint.transform.rotation.yaw * torch.pi / 180.0
        target_state[..., 3] = 0.0
        target_state.requires_grad_(True)

        logging.debug(f"Target state: {target_state}")

        # Get the control from the ModelPredictiveControl module

        # Convert bev tensor deque to torch tensor
        bev_tensor = torch.cat(list(bev_tensor_deque), dim=0).unsqueeze(0)

        if (skip_counter == 0) and (repeat_counter == 0):

            (control, location_predicted, cost, cost_canvas) = mpc_module.step(
                initial_state=current_state,
                target_state=target_state,
                bev=bev_tensor.detach(),
            )

        control_selected = control[0][skip_counter].copy()

        throttle, brake = acceleration_to_throttle_brake(
            acceleration=control_selected[0]
        )

        control_ = [throttle, control_selected[1], brake]

        (
            current_transform,
            current_velocity,
            target_waypoint,
            navigational_command,
        ) = c.step(action=control_)

        data = c.get_data()
        bev = data["bev"]
        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev,
                device=config.world_device,
                agent_channel=7,
                vehicle_channel=6,
                selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
                calculate_offroad=False,
            )
        )

        t1 = time.time()

        c.render(
            predicted_location=location_predicted,
            bev=bev,
            cost_canvas=cost_canvas,
            cost=cost,
            control=control_selected,
            current_state=current_state,
            target_state=target_state,
            frame_counter=frame_counter,
            skip_counter=skip_counter,
            repeat_counter=repeat_counter,
            sim_fps=1 / (t1 - t0),
            ego_forward_model_wandb_link=config.ego_forward_model_wandb_link,
            ego_forward_model_checkpoint_number=config.ego_forward_model_checkpoint_number,
            world_forward_model_wandb_link=config.world_forward_model_wandb_link,
            world_forward_model_checkpoint_number=config.world_forward_model_checkpoint_number,
        )

        mpc_module.reset()

        frame_counter += 1
        skip_counter = (
            skip_counter + (repeat_counter + 1 == (config.repeat_frames))
        ) % config.skip_frames
        repeat_counter = (repeat_counter + 1) % config.repeat_frames

    c.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )
    parser.add_argument("--seed", type=int, default=333)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rollout_length", type=int, default=5)
    parser.add_argument("--action_size", type=int, default=2)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--repeat_frames", type=int, default=2)

    parser.add_argument(
        "--ego_forward_model_wandb_link", type=str, default="vaydingul/mbl/ssifa1go"
    )
    parser.add_argument("--ego_forward_model_checkpoint_number", type=int, default=459)

    parser.add_argument(
        "--world_forward_model_wandb_link", type=str, default="vaydingul/mbl/3bzai68t"
    )

    parser.add_argument("--world_forward_model_checkpoint_number", type=int, default=47)

    parser.add_argument(
        "--ego_device",
        type=str,
        default="cuda:0",
        help="Device to use for the forward model",
    )

    parser.add_argument(
        "--world_device",
        type=str,
        default="cuda:0",
        help="Device to use for the world model",
    )

    parser.add_argument(
        "--mpc_device",
        type=str,
        default="cuda:0",
        help="Device to use for the MPC module",
    )

    config = parser.parse_args()

    main(config)
