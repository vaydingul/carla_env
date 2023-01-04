from carla_env import carla_env_mpc_path_follower_bev_traffic
from carla_env.models.dfm_km_cp import DecoupledForwardModelKinematicsCoupledPolicy
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy import Policy
from carla_env.cost.masked_cost_batched import Cost
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
    convert_standard_bev_to_model_bev)
from utils.train_utils import (seed_everything, get_device)

logging.basicConfig(level=logging.INFO)


def main(config):

    # config.seed = int(np.random.rand(1) * 1000)
    seed_everything(config.seed)
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
        cls=KinematicBicycleModel,
        dt=1 / 20)
    ego_forward_model = ego_forward_model.to(device=device).eval()

    # ---------------------------------------------------------------------------- #
    #                        Pretrained world forward model                        #
    # ---------------------------------------------------------------------------- #
    world_model_run = wandb.Api().run(
        config.world_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.world_forward_model_wandb_link,
        config.world_forward_model_checkpoint_number)
    world_forward_model = WorldBEVModel.load_model_from_wandb_run(
        run=world_model_run,
        checkpoint=checkpoint,
        device=device)
    world_forward_model = world_forward_model.to(device=device).eval()

    # ---------------------------------------------------------------------------- #
    #                           Pretrained policy model                                  #
    # ---------------------------------------------------------------------------- #
    policy_model_run = wandb.Api().run(
        config.policy_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.policy_model_wandb_link,
        config.policy_model_checkpoint_number)
    policy_model = Policy.load_model_from_wandb_run(
        run=policy_model_run,
        checkpoint=checkpoint,
        device=device)
    policy_model = policy_model.to(device=device).eval()
    # ---------------------------------------------------------------------------- #
    #                              DFM_KM with Policy                              #
    # --------------------  -------------------------------------------------------- #
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
    occupancy = data["occ"]["occupancy"]
    for i in range(world_forward_model.num_time_step_previous):

        bev_tensor_deque.append(
            convert_standard_bev_to_model_bev(
                bev, device=device))

    occupancy = torch.tensor(
        occupancy,
        dtype=torch.float32).unsqueeze(0).to(
        device=device)
    occupancy[occupancy <= 5] = 1
    occupancy[occupancy > 5] = 0

    counter = 0

    with torch.no_grad():
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

            # output = model(
            #     ego_state=ego_state,
            #     world_state=bev_tensor,
            #     command=navigational_command,
            #     target_location=target_location,
            #     occupancy=occupancy,
            #     benchmark=True
            # )

            world_future_bev_predicted_list = []
            ego_future_location_predicted_list = []
            ego_future_yaw_predicted_list = []
            ego_future_speed_predicted_list = []
            ego_future_action_predicted_list = []

            ego_state_previous = ego_state
            world_previous_bev = bev_tensor
            command = navigational_command

            for k in range(config.rollout_length):

                # Predict the future bev
                output = model(
                    ego_state_previous,
                    world_previous_bev,
                    command,
                    target_location,
                    occupancy)

                ego_state_next = output["ego_state_next"]
                world_state_next = output["world_state_next"]
                action = output["action"]

                world_future_bev_predicted = torch.sigmoid(
                    world_state_next)

                world_future_bev_predicted_list.append(
                    world_future_bev_predicted)
                ego_future_location_predicted_list.append(
                    ego_state_next["location"])
                ego_future_yaw_predicted_list.append(ego_state_next["yaw"])
                ego_future_speed_predicted_list.append(ego_state_next["speed"])
                ego_future_action_predicted_list.append(action)
                # Predict the future ego location

                # Update the previous bev
                world_previous_bev = torch.cat(
                    (world_previous_bev[:, 1:], world_future_bev_predicted.unsqueeze(1)), dim=1)

                ego_state_previous = ego_state_next

            world_future_bev_predicted = torch.stack(
                world_future_bev_predicted_list, dim=1)

            ego_future_location_predicted = torch.stack(
                ego_future_location_predicted_list, dim=1)

            ego_future_yaw_predicted = torch.stack(
                ego_future_yaw_predicted_list, dim=1)

            ego_future_speed_predicted = torch.stack(
                ego_future_speed_predicted_list, dim=1)

            ego_future_action_predicted = torch.stack(
                ego_future_action_predicted_list, dim=1)

            cost_dict = cost(ego_future_location_predicted,
                             ego_future_yaw_predicted,
                             ego_future_speed_predicted,
                             world_future_bev_predicted)

            control = ego_future_action_predicted_list[0][0]#torch.mean(ego_future_action_predicted, dim=1)[0]

            cost_canvas = render(
                config.rollout_length,
                world_future_bev_predicted,
                cost_dict,
                ego_future_action_predicted)

            # control = output["action"][0]
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
            occupancy = data["occ"]["occupancy"]
            occupancy = torch.tensor(
                occupancy,
                dtype=torch.float32).unsqueeze(0).to(
                device=device)
            occupancy[occupancy <= 5] = 1
            occupancy[occupancy > 5] = 0

            t1 = time.time()

            target_wrt_ego = torch.matmul(target_location - ego_state["location"], torch.tensor([[math.cos(-ego_state["yaw"]), -math.sin(
                -ego_state["yaw"])], [math.sin(-ego_state["yaw"]), math.cos(-ego_state["yaw"])]], device=device).t())

            c.render(
                predicted_location=ego_state["location"].detach().cpu().numpy(),
                bev=bev,
                cost_canvas=cost_canvas,
                control=control,
                current_state=ego_state,
                target_state=target_location,
                target_wrt_ego=target_wrt_ego,
                occupancy = occupancy,
                counter=counter,
                sim_fps=1 / (t1 - t0),
                seed=config.seed,
                world_forward_model_wandb_link=config.world_forward_model_wandb_link,
                world_forward_model_checkpoint_number=config.world_forward_model_checkpoint_number,
                policy_model_wandb_link=config.policy_model_wandb_link,
                policy_model_checkpoint_number=config.policy_model_checkpoint_number)

            counter += 1

    c.close()


def render(
        num_time_step_future,
        world_future_bev_predicted,
        cost,
        action_pred):

    canvas = _init_canvas(num_time_step_future, 192, 192)
    x1 = 0
    y1 = 0
    for k in range(1):
        for m in range(num_time_step_future - 1):
            bev = world_future_bev_predicted[0, m]
            bev[bev > 0.5] = 1
            bev[bev <= 0.5] = 0
            bev = bev.detach().cpu().numpy()

            mask_car = cost["mask_car"][0, m]
            mask_car = mask_car.detach().cpu().numpy()
            mask_car = (((mask_car -
                          mask_car.min()) /
                         (mask_car.max() -
                          mask_car.min())) *
                        255).astype(np.uint8)

            # mask_side = cost["mask_side"][k, m]
            # mask_side = mask_side.detach().cpu().numpy()
            # mask_side = (((mask_side -
            #                mask_side.min()) /
            #               (mask_side.max() -
            #                mask_side.min())) *
            #              255).astype(np.uint8)

            mask_light = cost["mask_light"][0, m]
            mask_light = mask_light.detach().cpu().numpy()
            mask_light = (((mask_light -
                            mask_light.min()) /
                           (mask_light.max() -
                            mask_light.min())) *
                          255).astype(np.uint8)

            mask_car = cv2.applyColorMap(mask_car, cv2.COLORMAP_JET)
            # mask_side = cv2.applyColorMap(mask_side, cv2.COLORMAP_JET)
            mask_light = cv2.applyColorMap(
                mask_light, cv2.COLORMAP_JET)

            bev = cv2.cvtColor(
                BirdViewProducer.as_rgb_model(
                    np.transpose(
                        bev, (1, 2, 0))), cv2.COLOR_BGR2RGB)

            x2 = x1 + bev.shape[1]
            y2 = y1 + bev.shape[0]
            canvas[y1:y2, x1:x2] = cv2.addWeighted(
                bev, 0.5, mask_car, 0.5, 0)
            # self.canvas_side[y1:y2, x1:x2] = cv2.addWeighted(
            #     bev, 0.5, mask_side, 0.5, 0)
            y1 = y2 + 20
            y2 = y1 + bev.shape[0]
            canvas[y1:y2, x1:x2] = cv2.addWeighted(
                bev, 0.5, mask_light, 0.5, 0)

            # # Draw ground truth action to the left corner of each bev
            # # as vector
            # action = action_gt[0, m]
            # action = action.detach().cpu().numpy()
            # action = action * 50
            # action = action.astype(np.int32)
            # cv2.arrowedLine(
            #     self.canvas_car,
            #     (x1 + 50,
            #      y1 + 50),
            #     (x1 + 50 +
            #      action[1],
            #      y1 + 50 - action[0]),
            #     (255,
            #      255,
            #      255),
            #     1,
            #     tipLength=0.5)

            # Draw predicted action to the left corner of each bev
            # as vector
            action = action_pred[0, m]
            action = action.detach().cpu().numpy()
            action = action * 50
            action = action.astype(np.int32)
            cv2.arrowedLine(
                canvas,
                (x1 + 50,
                    y1 + 50),
                (x1 + 50 +
                    action[1],
                    y1 + 50 - action[0]),
                (0,
                    255,
                    255),
                1,
                tipLength=0.5)
            y1 = 0
            x1 = x2 + 20

        x1 = 0
        y1 = y2 + 20

    x1 = 10
    yy = y1
    yy += 15

    cv2.putText(
        canvas,
        f"lane_cost: {cost['lane_cost']}",
        (x1,
            yy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255,
            255,
            255),
        1,
        cv2.LINE_AA)
    yy += 15
    cv2.putText(
        canvas,
        f"vehicle_cost: {cost['vehicle_cost']}",
        (x1,
            yy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255,
            255,
            255),
        1,
        cv2.LINE_AA)
    yy += 15
    cv2.putText(
        canvas,
        f"offroad_cost: {cost['offroad_cost']}",
        (x1,
            yy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255,
            255,
            255),
        1,
        cv2.LINE_AA)
    yy += 15
    cv2.putText(
        canvas,
        f"red_light_cost: {cost['red_light_cost']}",
        (x1,
            yy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255,
            255,
            255),
        1,
        cv2.LINE_AA)
    yy += 15
    cv2.putText(
        canvas,
        f"green_light_cost: {cost['green_light_cost']}",
        (x1,
            yy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255,
            255,
            255),
        1,
        cv2.LINE_AA)

    return canvas


def _init_canvas(num_time_step_future, bev_width, bev_height):

    width = (num_time_step_future - 1) * (bev_width + 20)
    height = 2 * (bev_height + 20) + 100
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    return canvas


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")

    parser.add_argument("--seed", type=int, default=333)

    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle")

    parser.add_argument("--rollout_length", type=int, default=4)

    parser.add_argument(
        "--world_forward_model_wandb_link",
        type=str,
        default="vaydingul/mbl/bylewhod")

    parser.add_argument(
        "--world_forward_model_checkpoint_number",
        type=int,
        default=94)

    parser.add_argument(
        "--policy_model_wandb_link",
        type=str,
        default="vaydingul/mbl/3ohsztk8")

    parser.add_argument(
        "--policy_model_checkpoint_number",
        type=int,
        default=4)

    config = parser.parse_args()

    main(config)
