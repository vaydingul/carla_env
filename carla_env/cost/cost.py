import torch
from torch import nn
import numpy as np
import logging
from utils.cost_utils import *

logging.basicConfig(level=logging.INFO)


class MaskedCostModule(nn.Module):
    def __init__(self, image_width, image_height):
        super(MaskedCostModule, self).__init__()

        self.vehicle_width = 2.1
        self.vehicle_length = 4.9
        self.pixels_per_meter = 18

        self.image_width = image_width
        self.image_height = image_height

    def forward(
            self,
            location,
            yaw,
            speed,
            location_next,
            yaw_next,
            speed_next,
            bev):
        # Create masks
        x, y = location_next[..., 0] - location[...,
                                                0], location_next[..., 1] - location[..., 1]
        yaw_ = yaw_next - yaw
        speed_ = speed_next

        mask_car, mask_side = self.create_mask(
            self.image_width, self.image_height, self.pixels_per_meter, x, y, yaw_, speed_, self.vehicle_width, self.vehicle_length)

        car_channel = torch.where(
            bev[..., -1] == 1, torch.tensor(1), torch.tensor(0))
        offroad_channel = torch.where(
            bev[..., -1] == 2, torch.tensor(1), torch.tensor(0))

        car_cost = torch.sum(car_channel * mask_car)
        side_cost = torch.sum(offroad_channel * mask_side)

        return car_cost, side_cost, car_cost + side_cost

    def create_masks(
            self,
            nx,
            ny,
            pixels_per_meter,
            x,
            y,
            yaw,
            speed,
            vehicle_width,
            vehicle_length):

        coordinate_mask = create_coordinate_mask(nx, ny, pixels_per_meter)
        aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
            x, y, yaw, coordinate_mask)

        dx = 1.5 * (torch.maximum(torch.tensor(5), speed) + vehicle_length) + 1
        dy = (vehicle_width / 2) + 3

        mask_car, mask_side = calculate_mask(
            aligned_coordinate_mask, dx, dy, vehicle_width, vehicle_length, 1)

        return mask_car, mask_side
