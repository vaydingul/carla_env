import numpy as np
import torch
from carla_env.tester.mpc import Tester
from utilities.model_utils import convert_standard_bev_to_model_bev
from utilities.train_utils import apply_torch_func, requires_grad, to

class Evaluator(Tester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self, hero_actor, input_data, bev_image, next_waypoint):
        processed_data = {}

        hero_actor_location = hero_actor.get_location()
        hero_actor_rotation = hero_actor.get_transform().rotation
        hero_actor_velocity = hero_actor.get_velocity()

        ego_previous = {
            "location":
            {
                "x": hero_actor_location.x,
                "y": hero_actor_location.y,
                "z": hero_actor_location.z,
            }
            ,
            "rotation":
            {
                "roll": hero_actor_rotation.roll * np.pi / 180,
                "pitch": hero_actor_rotation.pitch * np.pi / 180,
                "yaw": hero_actor_rotation.yaw * np.pi / 180,
            }
            ,
            "velocity":
            {
                "x": hero_actor_velocity.x,
                "y": hero_actor_velocity.y,
                "z": hero_actor_velocity.z,
            }
        }

        ego_previous = apply_torch_func(ego_previous, torch.tensor)
        ego_previous = apply_torch_func(ego_previous, torch.Tensor.view, (1, 1, -1))

        ego_previous = to(ego_previous, self.device)
        requires_grad(ego_previous, True)

        processed_data["ego_previous"] = ego_previous

        target_location = torch.zeros((1, 1, 2), device=self.device)
        target_yaw = torch.zeros((1, 1, 1), device=self.device)
        target_speed = torch.zeros((1, 1, 1), device=self.device)

        target_location[..., 0] = next_waypoint.location.x
        target_location[..., 1] = next_waypoint.location.y
        target_yaw[..., 0] = next_waypoint.rotation.yaw * np.pi / 180

        # target_speed[..., 0] = 5

        target_location.requires_grad_(True)
        target_yaw.requires_grad_(True)
        target_speed.requires_grad_(True)

        target = {
            "location": target_location,
            "yaw": target_yaw,
            "speed": target_speed,
        }

        processed_data["target"] = target

        bev_tensor = convert_standard_bev_to_model_bev(
            bev_image,
            agent_channel=self.bev_agent_channel,
            vehicle_channel=self.bev_vehicle_channel,
            selected_channels=self.bev_selected_channels,
            calculate_offroad=self.bev_calculate_offroad,
            device=self.device,
        )
        bev_tensor.requires_grad_(True)

        processed_data["bev_tensor"] = bev_tensor

        occupancy = torch.zeros((1, 8), dtype=torch.float32, device=self.device)
        for i in range(8):
            radar_data = input_data[f"radar_{i}"][1][:, 0]
            occupancy[0, i] = float(
                np.nanmin(radar_data) if radar_data.shape[0] > 0 else 10
            )

        occupancy[occupancy > 10] = 10

        processed_data["occupancy"] = occupancy

        return processed_data

    def step(self, ego_previous, world_previous_bev, target):
        return super()._step(ego_previous, world_previous_bev, target)

    def reset(self, initial_guess=None):
        super()._reset(initial_guess=initial_guess)
