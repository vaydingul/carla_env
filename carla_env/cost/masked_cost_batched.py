import torch
from torch import nn
import numpy as np
import logging
from utils.cost_utils import *

logging.basicConfig(level=logging.INFO)


class Cost(nn.Module):
    def __init__(self, image_width, image_height, device, decay_factor=0.97):
        super(Cost, self).__init__()

        self.vehicle_width = 2.1
        self.vehicle_length = 4.9
        self.pixels_per_meter = 5

        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.decay_factor = decay_factor

    def forward(
            self,
            location,
            yaw,
            speed,
            bev,
            agent_mask):

        # Create masks
        x, y, yaw_ = rotate_batched(location, yaw)

        speed_ = speed[1:, 0:1]

        mask_car, mask_side = self.create_masks(nx=self.image_width, ny=self.image_height, pixels_per_meter=self.pixels_per_meter,
                                                x=x, y=y, yaw=yaw_, speed=speed_, vehicle_width=self.vehicle_width, vehicle_length=self.vehicle_length)

        bev = bev.clone()
        bev[bev > 0.5] = 1
        bev[bev <= 0.5] = 0

        agent_mask = torch.tensor(agent_mask, device=self.device).float()

        road_channel = bev[:, 0]
        lane_channel = bev[:, 1]
        vehicle_channel = torch.bitwise_and(
            bev[:, 2].byte(), torch.bitwise_not(agent_mask.byte())).float()
        #agent_channel = bev[..., 3]
        green_light_channel = bev[:, 3]
        yellow_light_channel = bev[:, 4]
        red_light_channel = bev[:, 5]
        pedestrian_channel = bev[:, 6]
        offroad_channel = bev[:, 7]

        #vehicle_channel -= agent_channel

        # Calculate cost
        torch.bitwise_and
        decay_weight = torch.pow(self.decay_factor, torch.arange(
            0, speed_.shape[0], device=speed_.device).float()).view(-1, 1, 1)

        #road_cost = torch.sum(road_channel * mask_car * decay_weight)
        lane_cost = torch.sum(lane_channel * mask_side * decay_weight)
        vehicle_cost = torch.sum(vehicle_channel * mask_car * decay_weight)
        #agent_cost = torch.sum(agent_channel * mask_car * decay_weight)
        green_light_cost = torch.sum(
            green_light_channel * mask_side * decay_weight)
        yellow_light_cost = torch.sum(
            yellow_light_channel * mask_side * decay_weight)
        red_light_cost = torch.sum(
            red_light_channel * mask_side * decay_weight)
        pedestrian_cost = torch.sum(
            pedestrian_channel * mask_car * decay_weight)
        offroad_cost = torch.sum(offroad_channel * mask_side * decay_weight)

        return (
            lane_cost,
            vehicle_cost,
            green_light_cost,
            yellow_light_cost,
            red_light_cost,
            pedestrian_cost,
            offroad_cost,
            mask_car,
            mask_side)

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

        coordinate_mask, _, _ = create_coordinate_mask(
            nx, ny, pixels_per_meter, self.device)
        coordinate_mask = coordinate_mask.repeat(x.shape[0], 1, 1, 1)

        aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
            x, y, yaw, coordinate_mask)

        dx = (vehicle_width / 2) + 1

        # dy = 1.5 * (torch.maximum(torch.tensor(1), speed) + vehicle_length) + 1
        dy = (speed + vehicle_length) + 0.25

        dy = dy.view(-1, 1, 1, 1)

        mask_car, mask_side = calculate_mask(
            aligned_coordinate_mask, dx, dy, vehicle_width, vehicle_length, 1)

        return mask_car, mask_side
