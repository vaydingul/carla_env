import torch
from torch import nn
import numpy as np


class MPC(nn.Module):
	"""MPC controller."""

	def __init__(self, device,rollout_length, number_of_optimization_iterations, model):
		"""Initialize."""
		
		super(MPC, self).__init__()

		#self.model = model.to(device)
		self.device = device
		self.rollout_length = rollout_length
		self.number_of_optimization_iterations = number_of_optimization_iterations
		self.model = model

		# self.action = torch.randn((1, self.rollout_length, 3), device=self.device)
		# self.action[..., 2] = torch.randint(0, 2, (1, self.rollout_length), device=self.device, dtype=torch.float32)
		self.action = torch.randn((1, self.rollout_length, 3), device=self.device)
		self.action = nn.Parameter(self.action, requires_grad=True)
		self.optimizer = torch.optim.SGD((self.action, ), lr=0.5)
		
		self.loss_criterion = nn.L1Loss()
		
		
		
	def forward(self, location, rotation, speed):
		"""Run a single step of MPC."""

		location_predicted = []
		rotation_predicted = []
		speed_predicted = []

		
		for i in range(self.rollout_length-1):

			
			action_ = self.action[:, i, :].clone()

			location, rotation, speed = self.model(location, rotation, speed, action_)
			location_predicted.append(location)
			rotation_predicted.append(rotation)
			speed_predicted.append(speed)

		location_predicted = torch.cat(location_predicted, dim=1)
		rotation_predicted = torch.cat(rotation_predicted, dim=1)
		speed_predicted = torch.cat(speed_predicted, dim=1)


		return location_predicted.clone(), rotation_predicted.clone(), speed_predicted.clone()

	def optimize_action(self, initial_state, target_state):
		"""Optimize the action."""

	


		for _ in range(self.number_of_optimization_iterations):
			
			self.optimizer.zero_grad()
			
			location = initial_state[:, :, 0:2].clone()
			rotation = initial_state[:, :, 2:3].clone()
			speed = initial_state[:, :, 3:4].clone()
			#state[:, 0, :] = initial_state[:, 0, :].clone()

			location_predicted, rotation_predicted, speed_predicted = self.forward(location, rotation, speed)
			cost = self._calculate_cost(location_predicted, rotation_predicted, speed_predicted, target_state)

			cost.backward(retain_graph = True)

			torch.nn.utils.clip_grad_value_(self.action, 0.1)
			self.optimizer.step()

		return list(self.action[0, 0, :].detach().cpu().numpy()), location_predicted.detach().cpu().numpy(), rotation_predicted.detach().cpu().numpy(), speed_predicted.detach().cpu().numpy()

	def reset(self, initial_guess = None):
		"""Reset the controller."""
		if initial_guess is None:
			# Reset the action
			# action = torch.randn((1, self.rollout_length, 3), device=self.device)
			# action[..., 2] = torch.randint(0, 2, (1, self.rollout_length), device=self.device, dtype=torch.float32)
			action = torch.randn((1, self.rollout_length, 3), device=self.device, dtype=torch.float32)
			self.action = nn.Parameter(action)
		else:
			self.action = nn.Parameter(initial_guess.clone().detach().requires_grad_(True).repeat(1, self.rollout_length, 1))
		
		self.optimizer = torch.optim.Adam((self.action, ), lr=0.05)

	def _calculate_cost(self, predicted_location, predicted_rotation, predicted_speed, target_state):
		"""Calculate the cost."""
		#return self.loss_criterion(predicted_location, target_location) + (self.action[..., 1:] ** 2).sum() + ((1 - self.action[..., 0]) ** 2).sum()
		loss = torch.tensor(0.)

		loss += self.loss_criterion(predicted_location[..., :1], target_state[..., :1].expand(*(predicted_location[..., :1].shape))) 
		loss += self.loss_criterion(predicted_location[..., 1:2], target_state[..., 1:2].expand(*(predicted_location[..., 1:2].shape)))

		#loss += self.loss_criterion(predicted_rotation, target_state[..., 2:3]) 

		#loss += (self.action[..., 1:2] ** 2).sum()# + ((1 - self.action[..., 0]) ** 2).sum()

		loss += torch.diff(self.action, dim = 1).square().sum() * 0.05

		return loss