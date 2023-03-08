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
        (mask_car) = self.create_masks(x=x, y=y, yaw=yaw_, speed=speed_)

        bev = bev[:, 1:].clone()
        bev[bev > 0.5] = 1
        bev[bev <= 0.5] = 0

        # Calculate cost

        decay_weight = (
            torch.pow(
                self.decay_factor,
                torch.arange(0, speed_.shape[1], device=speed_.device).float(),
            )
            .reshape(1, -1, 1, 1, 1)
            .repeat(speed_.shape[0], 1, bev.shape[2], 1, 1)
        )

        mask_car = mask_car.unsqueeze(2).repeat(1, 1, bev.shape[2], 1, 1)

        cost_tensor = bev * mask_car * decay_weight

        if self.reduction == "sum" or self.reduction == "mean":

            if self.reduction == "sum":
                cost = torch.sum(cost_tensor, dim=[0, 1, 3, 4])

            elif self.reduction == "mean":
                cost = torch.mean(cost_tensor, dim=[0, 1, 3, 4])

            cost_dict = {
                "road_cost": cost[0],
                "road_on_cost": cost[1],
                "road_off_cost": cost[2],
                "road_red_yellow_cost": cost[3],
                "road_green_cost": cost[4],
                "lane_cost": cost[5],
                "vehicle_cost": cost[6],
                "offroad_cost": cost[7],
            }
        elif self.reduction == "none":

            cost = cost_tensor

            cost_dict = {
                "road_cost": cost[:, :, 0],
                "road_on_cost": cost[:, :, 1],
                "road_off_cost": cost[:, :, 2],
                "road_red_yellow_cost": cost[:, :, 3],
                "road_green_cost": cost[:, :, 4],
                "lane_cost": cost[:, :, 5],
                "vehicle_cost": cost[:, :, 6],
                "offroad_cost": cost[:, :, 7],
            }

        else:

            raise ValueError(
                f"Reduction {self.reduction} not supported. Supported reductions are 'sum', 'mean', 'none'"
            )

        mask_dict = {
            "mask_car": mask_car[:, :, 0],
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

        # dx = (self.vehicle_width / 2) + 4
        dx = self.vehicle_width * self.lateral_scaler + self.lateral_offset

        # dy = 1.5 * (torch.maximum(torch.tensor(10), speed) + self.vehicle_length) + 1
        dy = (
            speed * self.longitudinal_speed_scaler + self.longitudinal_speed_offset
        ) + (self.vehicle_length * self.longitudinal_scaler + self.longitudinal_offset)

        dy = dy.view(*dy.shape, 1, 1)

        (mask_car) = calculate_mask(
            aligned_coordinate_mask,
            dx,
            dy,
            self.vehicle_width,
            self.vehicle_length,
            self.mask_alpha,
            False,
        )

        return mask_car

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
            "longitudinal_speed_scaler": 2.0,
            "lateral_offset": 1.0,
            "lateral_scaler": 0.5,
        }

    def append_config(self, config):

        self.config.update(config)
