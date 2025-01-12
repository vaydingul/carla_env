import torch
from torch import nn
import numpy as np


class ModelPredictiveControl(nn.Module):
    """ModelPredictiveControl controller."""

    def __init__(
            self,
            device,
            action_size,
            rollout_length,
            number_of_optimization_iterations,
            model,
            cost):
        """Initialize."""

        super(ModelPredictiveControl, self).__init__()

        #self.model = model.to(device)
        self.device = device
        self.action_size = action_size
        self.rollout_length = rollout_length
        self.number_of_optimization_iterations = number_of_optimization_iterations
        self.model = model
        self.cost = cost

        self.action = torch.zeros(
            (1, self.rollout_length, self.action_size), device=self.device)
        self.action = nn.Parameter(self.action, requires_grad=True)
        self.optimizer = torch.optim.SGD((self.action, ), lr=.5)

        self.loss_criterion = nn.L1Loss()

    def forward(self, location, rotation, speed):
        """Run a single step of ModelPredictiveControl."""

        location_predicted = []
        rotation_predicted = []
        speed_predicted = []

        for i in range(self.rollout_length - 1):

            action_ = self.action[:, i, :].clone()

            location, rotation, speed = self.model(
                location, rotation, speed, action_)
            location_predicted.append(location)
            rotation_predicted.append(rotation)
            speed_predicted.append(speed)

        location_predicted = torch.cat(location_predicted, dim=1)
        rotation_predicted = torch.cat(rotation_predicted, dim=1)
        speed_predicted = torch.cat(speed_predicted, dim=1)

        return location_predicted.clone(), rotation_predicted.clone(), speed_predicted.clone()

    def step(self, initial_state, target_state):
        """Optimize the action."""

        for _ in range(self.number_of_optimization_iterations):

            self.optimizer.zero_grad()

            location = initial_state[:, :, 0:2].clone()
            rotation = initial_state[:, :, 2:3].clone()
            speed = initial_state[:, :, 3:4].clone()

            location_predicted, rotation_predicted, speed_predicted = self.forward(
                location, rotation, speed)
            
            cost = self._calculate_cost(
                location_predicted,
                rotation_predicted,
                speed_predicted,
                target_state)
            
            cost.backward(retain_graph=True)

            torch.nn.utils.clip_grad_value_(self.action, 0.1)

            self.optimizer.step()

        return list(self.action[0, 0, :].detach().cpu().numpy(
        )), location_predicted[0].detach().cpu().numpy(), cost.item()

    def reset(self):
        """Reset the controller."""

        action = torch.zeros(
            (1,
                self.rollout_length,
                self.action_size),
            device=self.device,
            dtype=torch.float32)
        self.action = nn.Parameter(action)
        self.optimizer = torch.optim.Adam((self.action, ), lr=0.05)

    def _calculate_cost(
            self,
            predicted_location,
            predicted_rotation,
            predicted_speed,
            target_state):
        """Calculate the cost."""

        loss = torch.tensor(0.)

        loss += self.cost(predicted_location[..., :1], target_state[..., :1].expand(
            *(predicted_location[..., :1].shape)))
        loss += self.cost(predicted_location[..., 1:2], target_state[..., 1:2].expand(
            *(predicted_location[..., 1:2].shape)))
        loss += self.cost(torch.cos(predicted_rotation),
                          torch.cos(target_state[...,
                                                 2:3].expand(*(predicted_rotation.shape))))
        loss += self.cost(torch.sin(predicted_rotation),
                          torch.sin(target_state[...,
                                                 2:3].expand(*(predicted_rotation.shape))))
        loss += self.cost(predicted_speed,
                          target_state[...,
                                       3:4].expand(*(predicted_speed.shape)))

        loss += torch.diff(self.action[..., 1], dim=1).square().sum()

        return loss
