import numpy as np
import torch
from carla_env.tester.mpc import Tester
from utils.model_utils import convert_standard_bev_to_model_bev


class Evaluator(Tester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self, hero_actor, input_data, bev_image, next_waypoint):
        processed_data = {}

        hero_actor_location = hero_actor.get_location()
        hero_actor_rotation = hero_actor.get_transform().rotation
        hero_actor_velocity = hero_actor.get_velocity()
        hero_actor_speed = hero_actor_velocity.length()

        ego_previous_location_array = torch.zeros((1, 1, 3), device=self.device)
        ego_previous_rotation_array = torch.zeros((1, 1, 3), device=self.device)
        ego_previous_velocity_array = torch.zeros((1, 1, 3), device=self.device)

        ego_previous_location_array[..., 0] = hero_actor_location.x
        ego_previous_location_array[..., 1] = hero_actor_location.y
        ego_previous_location_array[..., 2] = hero_actor_location.z

        ego_previous_rotation_array[..., 0] = hero_actor_rotation.roll * np.pi / 180
        ego_previous_rotation_array[..., 1] = hero_actor_rotation.pitch * np.pi / 180
        ego_previous_rotation_array[..., 2] = hero_actor_rotation.yaw * np.pi / 180

        ego_previous_velocity_array[..., 0] = hero_actor_velocity.x
        ego_previous_velocity_array[..., 1] = hero_actor_velocity.y
        ego_previous_velocity_array[..., 2] = hero_actor_velocity.z

        ego_previous_location_array.requires_grad_(True)
        ego_previous_rotation_array.requires_grad_(True)
        ego_previous_velocity_array.requires_grad_(True)

        ego_previous = {
            "location_array": ego_previous_location_array,
            "rotation_array": ego_previous_rotation_array,
            "velocity_array": ego_previous_velocity_array,
        }

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
