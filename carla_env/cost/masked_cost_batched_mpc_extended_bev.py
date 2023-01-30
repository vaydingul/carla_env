import torch
from torch import nn
import numpy as np
import logging
from utils.cost_utils import *


class Cost(nn.Module):
    def __init__(
        self,
        image_width,
        image_height,
        device,
        decay_factor=0.97,
        vehicle_width=2.1,
        vehicle_length=4.9,
        pixels_per_meter=5,
        light_offset_x=12,
        light_offset_y=3.25,
    ):
        super(Cost, self).__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.device = device

        self.decay_factor = decay_factor
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.pixels_per_meter = pixels_per_meter
        self.light_offset_x = light_offset_x
        self.light_offset_y = light_offset_y

        self.coordinate_mask, _, _ = create_coordinate_mask(
            nx=self.image_width,
            ny=self.image_height,
            pixels_per_meter=self.pixels_per_meter,
            device=self.device,
        )

    def forward(
        self,
        location,
        yaw,
        speed,
        bev,
    ):

        # Create masks
        x, y, yaw_ = rotate_batched(location, yaw)

        speed_ = speed[:, 1:, 0:1]
        mask_car = self.create_masks(x=x, y=y, yaw=yaw_, speed=speed_)

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

        cost = torch.sum(bev * mask_car * decay_weight, dim=(0, 1, 3, 4))
        return {"cost": cost, "mask_car": mask_car[:, :, 0]}

    def create_masks(
        self,
        x,
        y,
        yaw,
        speed,
    ):

        coordinate_mask = self.coordinate_mask.clone().repeat(
            x.shape[0], x.shape[1], 1, 1, 1
        )

        aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
            x, y, yaw, coordinate_mask
        )

        dx = (self.vehicle_width / 2) + 1

        # dy = 1.5 * (torch.maximum(torch.tensor(1), speed) + vehicle_length) + 1
        dy = (speed * 2 + self.vehicle_length) + 0.25
        dy = dy.unsqueeze(-1).unsqueeze(-1)

        mask_car = calculate_mask(
            aligned_coordinate_mask,
            dx,
            dy,
            self.vehicle_width,
            self.vehicle_length,
            1.1,
            False,
        )

        return mask_car
