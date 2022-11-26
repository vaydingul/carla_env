from carla_env import carla_env_mpc_path_follower_bev_traffic
from carla_env.models.dfm_km_cp import DecoupledForwardModelKinematicsCoupledPolicy
from carla_env.models.dynamic.vehicle import KinematicBicycleModelV2
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.dfm_km_cp import Policy
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
    load_policy_model_from_wandb_run,
    fetch_checkpoint_from_wandb_link,
    convert_standard_bev_to_model_bev)
from utils.train_utils import (seed_everything, get_device)

logging.basicConfig(level=logging.INFO)


def main(config):

    device = get_device()

    # ---------------------------------------------------------------------------- #
    #                                 Cost Function                                #
    # ---------------------------------------------------------------------------- #
    cost = Cost(image_width=192, image_height=192, device=device)

    # ---------------------------------------------------------------------------- #
    #                         Pretrained ego forward model                         #
    # ---------------------------------------------------------------------------- #
    ego_forward_model = load_ego_model_from_checkpoint(
        checkpoint=config.ego_forward_model_path,
        cls=KinematicBicycleModelV2,
        dt=1 / 20)
    ego_forward_model = ego_forward_model.to(device=device)

    # ---------------------------------------------------------------------------- #
    #                        Pretrained world forward model                        #
    # ---------------------------------------------------------------------------- #
    world_model_run = wandb.Api().run(
        config.world_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.world_forward_model_wandb_link,
        config.world_forward_model_checkpoint_number)
    world_forward_model, _ = load_world_model_from_wandb_run(
        run=world_model_run,
        checkpoint=checkpoint,
        cls=WorldBEVModel,
        world_model_device=device)
    world_forward_model = world_forward_model.to(device=device)

    # ---------------------------------------------------------------------------- #
    #                           Pretrained policy model                                  #
    # ---------------------------------------------------------------------------- #
    policy_model_run = wandb.Api().run(
        config.policy_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.policy_model_wandb_link,
        config.policy_model_checkpoint_number)
    policy_model, _ = load_policy_model_from_wandb_run(
        run=policy_model_run,
        checkpoint=checkpoint,
        cls=Policy,
        policy_model_device=device)
    policy_model = policy_model.to(device=device)
    # ---------------------------------------------------------------------------- #
    #                              DFM_KM with Policy                              #
    # ---------------------------------------------------------------------------- #
    model = DecoupledForwardModelKinematicsCoupledPolicy(
        ego_model=ego_forward_model,
        world_model=world_forward_model,
        policy_model=policy_model)
    model = model.to(device=device).eval()

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

    for i in range(world_forward_model.num_time_step_previous):

        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev, device=device))

    counter = 0

    while not c.is_done:

        t0 = time.time()

        # Set the current state of the ego vehicle for the kinematic model
        location = torch.zeros(
            size=(1, 2), device=device)
        location[..., 0] = current_transform.location.x
        location[..., 1] = current_transform.location.y
        yaw = torch.zeros(
            size=(1, 1), device=device)
        yaw[..., 0] = current_transform.rotation.yaw * math.pi / 180
        speed = torch.zeros(
            size=(1, 1), device=device)
        speed[..., 0] = math.sqrt(
            current_velocity.x**2 + current_velocity.y**2)
        location.requires_grad_(True).to(device=device)
        yaw.requires_grad_(True).to(device=device)
        speed.requires_grad_(True).to(device=device)
        ego_state = {"location": location,
                     "yaw": yaw,
                     "speed": speed}

        target_location = torch.zeros(
            size=(1, 2), device=device)
        target_location[..., 0] = target_waypoint.transform.location.x
        target_location[..., 1] = target_waypoint.transform.location.y
        target_location = target_location.to(device=device)

        logging.debug(f"Ego State: {ego_state}")
        logging.debug(f"Target Location: {target_location}")

        # Get the control from the ModelPredictiveControl module

        # Convert bev tensor deque to torch tensor
        bev_tensor = torch.cat(list(bev_tensor_deque), dim=0).unsqueeze(0)

        navigational_command = torch.tensor(
            navigational_command.value - 1, device=device).unsqueeze(0)
        navigational_command = torch.nn.functional.one_hot(
            navigational_command, num_classes=6).float().to(device=device)

        output = model(
            ego_state=ego_state,
            world_state=bev_tensor,
            command=navigational_command,
            target_location=target_location)

        control = output["action"][0]
        throttle, brake = acceleration_to_throttle_brake(
            acceleration=control[0])

        control = [throttle, control[1], brake]

        (current_transform, current_velocity, target_waypoint,
         navigational_command) = c.step(action=control)

        data = c.get_data()
        bev = data["bev"]
        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev, device=device))

        t1 = time.time()

        c.render(
            predicted_location=output["ego_state_next"]["location"].detach().cpu().numpy(),
            bev=bev,
            control=output["action"][0],
            current_state=ego_state["location"],
            target_state=target_location,
            counter=counter,
            sim_fps=1 / (t1 - t0))

        counter += 1

    c.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle")

    parser.add_argument("--rollout_length", type=int, default=10)

    parser.add_argument(
        "--world_forward_model_wandb_link",
        type=str,
        default="vaydingul/mbl/1gftiw9w")

    parser.add_argument(
        "--world_forward_model_checkpoint_number",
        type=int,
        default=49)

    parser.add_argument(
        "--policy_model_wandb_link",
        type=str,
        default="vaydingul/mbl/1aispzqm")

    parser.add_argument(
        "--policy_model_checkpoint_number",
        type=int,
        default=19)


    config = parser.parse_args()

    main(config)
