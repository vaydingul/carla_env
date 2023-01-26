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
            reduction="mean",
            decay_factor=0.97,
            vehicle_width=2.1,
            vehicle_length=4.9,
            pixels_per_meter=5,
            light_offset_x=12,
            light_offset_y=3.25):
        super(Cost, self).__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.reduction = reduction
        self.decay_factor = decay_factor
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.pixels_per_meter = pixels_per_meter
        self.light_offset_x = light_offset_x
        self.light_offset_y = light_offset_y

        self.coordinate_mask, _, _ = create_coordinate_mask(
            nx=self.image_width, ny=self.image_height, pixels_per_meter=self.pixels_per_meter, device=self.device)

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
        (mask_car, mask_side, mask_light) = self.create_masks(
            x=x, y=y, yaw=yaw_, speed=speed_)

        bev = bev[:, 1:].clone()
        bev[bev > 0.5] = 1
        bev[bev <= 0.5] = 0

        # print(f"Mask Car Max-Min: {torch.max(mask_car)}-{torch.min(mask_car)}")
        # print(
        # f"Mask Side Max-Min: {torch.max(mask_side)}-{torch.min(mask_side)}")

        #road_channel = bev[:, :, 0]
        lane_channel = bev[:, :, 1]
        vehicle_channel = bev[:, :, 2]
        green_light_channel = bev[:, :, 3]
        yellow_light_channel = bev[:, :, 4]
        red_light_channel = bev[:, :, 5]
        pedestrian_channel = bev[:, :, 6]
        offroad_channel = bev[:, :, 7]

        #vehicle_channel -= agent_channel

        # Calculate cost

        decay_weight = torch.pow(
            self.decay_factor,
            torch.arange(
                0,
                speed_.shape[1],
                device=speed_.device).float()).repeat(
            speed_.shape[0],
            1,
            1,
            1).permute(
                0,
                3,
                1,
            2)

        #road_cost_tensor = road_channel * mask_car * decay_weight
        lane_cost_tensor = lane_channel * mask_side * decay_weight
        vehicle_cost_tensor = vehicle_channel * mask_car * decay_weight
        #agent_cost_tensor = agent_channel * mask_car * decay_weight
        green_light_cost_tensor = green_light_channel * mask_light * decay_weight
        yellow_light_cost_tensor = yellow_light_channel * mask_light * decay_weight
        red_light_cost_tensor = red_light_channel * mask_light * decay_weight
        pedestrian_cost_tensor = pedestrian_channel * mask_car * decay_weight
        offroad_cost_tensor = offroad_channel * mask_side * decay_weight

        if self.reduction == "sum":
            lane_cost = torch.sum(lane_cost_tensor)
            vehicle_cost = torch.sum(vehicle_cost_tensor)
            green_light_cost = torch.sum(green_light_cost_tensor)
            yellow_light_cost = torch.sum(yellow_light_cost_tensor)
            red_light_cost = torch.sum(red_light_cost_tensor)
            pedestrian_cost = torch.sum(pedestrian_cost_tensor)
            offroad_cost = torch.sum(offroad_cost_tensor)

        elif self.reduction == "mean":
            lane_cost = torch.mean(lane_cost_tensor)
            vehicle_cost = torch.mean(vehicle_cost_tensor)
            green_light_cost = torch.mean(green_light_cost_tensor)
            yellow_light_cost = torch.mean(yellow_light_cost_tensor)
            red_light_cost = torch.mean(red_light_cost_tensor)
            pedestrian_cost = torch.mean(pedestrian_cost_tensor)
            offroad_cost = torch.mean(offroad_cost_tensor)

        elif self.reduction == "batch-sum":
            lane_cost = torch.sum(lane_cost_tensor, dim=[2, 3, 4])
            vehicle_cost = torch.sum(vehicle_cost_tensor, dim=[2, 3, 4])
            green_light_cost = torch.sum(
                green_light_cost_tensor, dim=[2, 3, 4])
            yellow_light_cost = torch.sum(
                yellow_light_cost_tensor, dim=[2, 3, 4])
            red_light_cost = torch.sum(red_light_cost_tensor, dim=[2, 3, 4])
            pedestrian_cost = torch.sum(pedestrian_cost_tensor, dim=[2, 3, 4])
            offroad_cost = torch.sum(offroad_cost_tensor, dim=[2, 3, 4])
        elif self.reduction == "batch-mean":
            lane_cost = torch.mean(lane_cost_tensor, dim=[2, 3, 4])
            vehicle_cost = torch.mean(vehicle_cost_tensor, dim=[2, 3, 4])
            green_light_cost = torch.mean(
                green_light_cost_tensor, dim=[2, 3, 4])
            yellow_light_cost = torch.mean(
                yellow_light_cost_tensor, dim=[2, 3, 4])
            red_light_cost = torch.mean(red_light_cost_tensor, dim=[2, 3, 4])
            pedestrian_cost = torch.mean(pedestrian_cost_tensor, dim=[2, 3, 4])
            offroad_cost = torch.mean(offroad_cost_tensor, dim=[2, 3, 4])

        return {
            "lane_cost": lane_cost,
            "vehicle_cost": vehicle_cost,
            "green_light_cost": green_light_cost,
            "yellow_light_cost": yellow_light_cost,
            "red_light_cost": red_light_cost,
            "pedestrian_cost": pedestrian_cost,
            "offroad_cost": offroad_cost,
            "mask_car": mask_car,
            "mask_side": mask_side,
            "mask_light": mask_light,
        }

    def create_masks(
            self,
            x,
            y,
            yaw,
            speed
    ):

        coordinate_mask = self.coordinate_mask.clone().repeat(
            x.shape[0], x.shape[1], 1, 1, 1)

        aligned_coordinate_mask = align_coordinate_mask_with_ego_vehicle(
            x, y, yaw, coordinate_mask)

        aligned_coordinate_mask_light = align_coordinate_mask_with_ego_vehicle(
            x + self.light_offset_x, y + self.light_offset_y, yaw, coordinate_mask)

        # dx = (self.vehicle_width / 2) + 4
        dx = (self.vehicle_width / 2) + 1
        dx_light = (self.vehicle_width) + 1

        # dy = 1.5 * (torch.maximum(torch.tensor(10), speed) + self.vehicle_length) + 1
        dy = (speed + self.vehicle_length) + 0.25
        dy_light = speed * 0.5 + self.vehicle_length * 3
        dy = dy.unsqueeze(-1).unsqueeze(-1)
        dy_light = dy_light.unsqueeze(-1).unsqueeze(-1)

        (mask_car, mask_side) = calculate_mask(aligned_coordinate_mask,
                                               dx, dy, self.vehicle_width, self.vehicle_length, 1.1)
        (mask_light, _) = calculate_mask(aligned_coordinate_mask_light,
                                         dx_light, dy_light, self.vehicle_width, self.vehicle_length, 1.1)

        return (mask_car, mask_side, mask_light)
