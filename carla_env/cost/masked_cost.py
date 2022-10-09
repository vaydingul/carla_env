import torch
from torch import nn
import numpy as np
import logging
from utils.cost_utils import *

logging.basicConfig(level=logging.INFO)


class MaskedCost(nn.Module):
    def __init__(self, image_width, image_height):
        super(MaskedCost, self).__init__()

        self.vehicle_width = 2.1
        self.vehicle_length = 4.9
        self.pixels_per_meter = 20

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
        # Convert bev to torch tensor
        bev = torch.from_numpy(bev).float().to(location.device)
        # Create masks
        x, y, yaw_ = rotate(location, yaw, location_next, yaw_next)
        speed_ = speed_next

        mask_car, mask_side = self.create_masks(
            self.image_width, self.image_height, self.pixels_per_meter, x, y, yaw_, speed_, self.vehicle_width, self.vehicle_length)

        # Fetch respective channels from bev representation
        offroad_mask = torch.where(torch.all(bev == 0, dim = -1), torch.ones_like(bev[..., 0]), torch.zeros_like(bev[..., 0]))
        bev = torch.cat([bev, offroad_mask.unsqueeze(-1)], dim = -1)
        
        road_channel = bev[..., 0]
        lane_channel = bev[..., 1]
        vehicle_channel = bev[..., 2]
        agent_channel = bev[..., 3]
        green_light_channel = bev[..., 4]
        yellow_light_channel = bev[..., 5]
        red_light_channel = bev[..., 6]
        pedestrian_channel = bev[..., 7]
        offroad_channel = bev[..., 8]

        vehicle_channel -= agent_channel
        # Calculate cost
        #road_cost = torch.sum(road_channel * mask_car)
        lane_cost = torch.sum(lane_channel * mask_side)
        vehicle_cost = torch.sum(vehicle_channel * mask_car)
        #agent_cost = torch.sum(agent_channel * mask_car)
        green_light_cost = torch.sum(green_light_channel * mask_side)
        yellow_light_cost = torch.sum(yellow_light_channel * mask_side)
        red_light_cost = torch.sum(red_light_channel * mask_side)
        pedestrian_cost = torch.sum(pedestrian_channel * mask_car)
        offroad_cost = torch.sum(offroad_channel * mask_side)

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

        coordinate_mask, _, _ = create_coordinate_mask(nx, ny, pixels_per_meter)
        aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
            x, y, yaw, coordinate_mask)

        dx = 1.5 * (torch.maximum(torch.tensor(5), speed) + vehicle_length) + 1
        dy = (vehicle_width / 2) + 1.5

        mask_car, mask_side = calculate_mask(
            aligned_coordinate_mask, dy, dx, vehicle_width, vehicle_length, 1)

        return mask_car, mask_side
