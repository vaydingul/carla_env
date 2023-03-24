import torch
from torch import nn
import numpy as np
import logging
from utils.cost_utils import *
from utils.train_utils import organize_device


class Cost(nn.Module):
    def __init__(
        self,
        device,
        config,
    ):
        super(Cost, self).__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.coordinate_mask, _, _ = create_coordinate_mask(
            nx=self.image_width,
            ny=self.image_height,
            pixels_per_meter=self.pixels_per_meter,
            device=organize_device(device),
        )

    def build_from_config(self):

        self.image_width = self.config["image_width"]
        self.image_height = self.config["image_height"]

        self.reduction = self.config["reduction"]
        self.decay_factor = self.config["decay_factor"]
        self.vehicle_width = self.config["vehicle_width"]
        self.vehicle_length = self.config["vehicle_length"]
        self.pixels_per_meter = self.config["pixels_per_meter"]

        self.lateral_scaler = self.config["lateral_scaler"]
        self.lateral_offset = self.config["lateral_offset"]
        self.longitudinal_scaler = self.config["longitudinal_scaler"]
        self.longitudinal_offset = self.config["longitudinal_offset"]
        self.longitudinal_speed_scaler = self.config["longitudinal_speed_scaler"]
        self.longitudinal_speed_offset = self.config["longitudinal_speed_offset"]

        self.light_lateral_scaler = self.config["light_lateral_scaler"]
        self.light_lateral_offset = self.config["light_lateral_offset"]
        self.light_longitudinal_scaler = self.config["light_longitudinal_scaler"]
        self.light_longitudinal_offset = self.config["light_longitudinal_offset"]
        self.light_longitudinal_speed_scaler = self.config[
            "light_longitudinal_speed_scaler"
        ]
        self.light_longitudinal_speed_offset = self.config[
            "light_longitudinal_speed_offset"
        ]

        self.light_master_offset_x = self.config["light_master_offset_x"]
        self.light_master_offset_y = self.config["light_master_offset_y"]

        self.mask_alpha = self.config["mask_alpha"]

    def forward(
        self,
        location,
        yaw,
        speed,
        bev,
    ):

        # Create masks
        x, y, yaw_ = rotate_batched(location, yaw)

        speed_ = speed[:, 1:]  # speed[:, 1:, 0:1]
        (mask_car, mask_side, mask_light) = self.create_masks(
            x=x, y=y, yaw=yaw_, speed=speed_
        )

        bev = bev[:, 1:].clone()
        bev[bev > 0.5] = 1
        bev[bev <= 0.5] = 0

        road_channel = bev[:, :, 0]
        lane_channel = bev[:, :, 1]
        vehicle_channel = bev[:, :, 2]
        green_light_channel = bev[:, :, 3]
        red_yellow_light_channel = bev[:, :, 4]
        offroad_channel = bev[:, :, 5]

        # vehicle_channel -= agent_channel

        # Calculate cost

        decay_weight = (
            torch.pow(
                self.decay_factor,
                torch.arange(0, speed_.shape[1], device=speed_.device).float(),
            )
            .reshape(1, -1, 1, 1)
            .repeat(speed_.shape[0], 1, 1, 1)
        )

        road_cost_tensor = road_channel * mask_side * decay_weight
        lane_cost_tensor = lane_channel * mask_side * decay_weight
        vehicle_cost_tensor = vehicle_channel * mask_car * decay_weight
        green_light_cost_tensor = green_light_channel * mask_light * decay_weight
        red_yellow_light_cost_tensor = (
            red_yellow_light_channel * mask_light * decay_weight
        )
        offroad_cost_tensor = offroad_channel * mask_side * decay_weight

        if self.reduction == "sum":
            road_cost = torch.sum(road_cost_tensor)
            lane_cost = torch.sum(lane_cost_tensor)
            vehicle_cost = torch.sum(vehicle_cost_tensor)
            green_light_cost = torch.sum(green_light_cost_tensor)
            red_yellow_light_cost = torch.sum(red_yellow_light_cost_tensor)
            offroad_cost = torch.sum(offroad_cost_tensor)

        elif self.reduction == "mean":
            road_cost = torch.mean(road_cost_tensor)
            lane_cost = torch.mean(lane_cost_tensor)
            vehicle_cost = torch.mean(vehicle_cost_tensor)
            green_light_cost = torch.mean(green_light_cost_tensor)
            red_yellow_light_cost = torch.mean(red_yellow_light_cost_tensor)
            offroad_cost = torch.mean(offroad_cost_tensor)

        elif self.reduction == "none":
            road_cost = road_cost_tensor
            lane_cost = lane_cost_tensor
            vehicle_cost = vehicle_cost_tensor
            green_light_cost = green_light_cost_tensor
            red_yellow_light_cost = red_yellow_light_cost_tensor
            offroad_cost = offroad_cost_tensor

        else:

            raise ValueError(
                "Invalid reduction type. Expected 'sum', 'mean', 'batch-sum', 'batch-mean', or 'none'."
            )

        cost_dict = {
            "road_cost": road_cost,
            "lane_cost": lane_cost,
            "vehicle_cost": vehicle_cost,
            "green_light_cost": green_light_cost,
            "red_yellow_light_cost": red_yellow_light_cost,
            "offroad_cost": offroad_cost,
        }

        mask_dict = {
            "mask_car": mask_car,
            "mask_side": mask_side,
            "mask_light": mask_light,
        }

        return {
            "cost_dict": cost_dict,
            "mask_dict": mask_dict,
        }

    def create_masks(self, x, y, yaw, speed):

        # Repeat the coordinate mask for each time step and batch
        coordinate_mask = self.coordinate_mask.clone().repeat(
            x.shape[0], x.shape[1], 1, 1, 1
        )

        # Align the coordinate mask with the ego vehicle
        aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
            x, y, yaw, coordinate_mask
        )

        # Align the coordinate mask with the ego vehicle for light masks
        aligned_coordinate_mask_light = align_coordinate_mask_with_ego_vehicle(
            x + self.light_master_offset_x,
            y + self.light_master_offset_y,
            yaw,
            coordinate_mask,
        )

        # dx = (self.vehicle_width / 2) + 4
        dx = self.vehicle_width * self.lateral_scaler + self.lateral_offset
        dx_light = (
            self.vehicle_width * self.light_lateral_scaler + self.light_lateral_offset
        )

        # dy = 1.5 * (torch.maximum(torch.tensor(10), speed) + self.vehicle_length) + 1
        dy = (
            speed * self.longitudinal_speed_scaler + self.longitudinal_speed_offset
        ) + (self.vehicle_length * self.longitudinal_scaler + self.longitudinal_offset)

        dy_light = (
            speed * self.light_longitudinal_speed_scaler
            + self.light_longitudinal_speed_offset
        ) + (
            self.vehicle_length * self.light_longitudinal_scaler
            + self.light_longitudinal_offset
        )

        dy = dy.view(*dy.shape, 1, 1)
        dy_light = dy_light.view(*dy_light.shape, 1, 1)

        (mask_car, mask_side) = calculate_mask(
            aligned_coordinate_mask,
            dx,
            dy,
            self.vehicle_width,
            self.vehicle_length,
            self.mask_alpha,
        )
        (mask_light, _) = calculate_mask(
            aligned_coordinate_mask_light,
            dx_light,
            dy_light,
            self.vehicle_width,
            self.vehicle_length,
            self.mask_alpha,
        )

        return (mask_car, mask_side, mask_light)

    def set_default_config(self):

        self.config = {
            "image_width": 192,
            "image_height": 192,
            "reduction": "mean",
            "mask_alpha": 1.1,
            "decay_factor": 0.97,
            "vehicle_width": 2.1,
            "vehicle_length": 4.9,
            "pixels_per_meter": 5.0,
            "longitudinal_offset": 0.125,
            "longitudinal_scaler": 1,
            "longitudinal_speed_offset": 0.125,
            "longitudinal_speed_scaler": 1.0,
            "lateral_offset": 1.0,
            "lateral_scaler": 0.5,
            "light_longitudinal_offset": 0.0,
            "light_longitudinal_scaler": 3.5,
            "light_longitudinal_speed_offset": 0.0,
            "light_longitudinal_speed_scaler": 0.5,
            "light_lateral_offset": 1.0,
            "light_lateral_scaler": 1.0,
            "light_master_offset_x": 12.0,
            "light_master_offset_y": 3.25,
        }

    def append_config(self, config):

        self.config.update(config)
