import math
import torch
import torch.nn.functional as F
from torch import nn


class KinematicBicycleModel(nn.Module):
	def __init__(self, dt= 0.1):
		super().__init__()

		self.dt = dt

		# Kinematic bicycle model
		self.front_wheelbase = nn.Parameter(
			torch.tensor(1.), requires_grad=True)
		self.rear_wheelbase = nn.Parameter(
			torch.tensor(1.), requires_grad=True)

		self.steer_gain = nn.Parameter(torch.tensor(1.), requires_grad=True)

		self.brake_acceleration = nn.Parameter(
			torch.zeros(1), requires_grad=True)

		self.throttle_acceleration = nn.Sequential(
			nn.Linear(1, 1, bias=False),
		)

	def forward(self, location, yaw, speed, action):
		'''
		One step semi-parametric kinematic bicycle model
		'''

		throttle = action[..., 0:1]
		steer = action[..., 1:2]
		brake = action[..., 2:3].byte()


		acceleration = torch.where(brake == 1, self.brake_acceleration.expand(
			*brake.size()), self.throttle_acceleration(throttle))

		# Transformation from steer to wheel steering angle 
		# to use the kinematic model
		wheel_steer = self.steer_gain * steer

		# beta = atan((l_r * tan(delta_f)) / (l_f + l_r))
		beta = torch.atan(
			self.rear_wheelbase/(self.front_wheelbase+self.rear_wheelbase) * torch.tan(wheel_steer))

		# x_ = x + v * dt
		location_next = location + speed * \
			torch.cat([torch.cos(yaw+beta), torch.sin(yaw+beta)], -1) * self.dt
		
		# speed_ = speed + a * dt
		speed_next = speed + acceleration * self.dt
		
		
		yaw_next = yaw + speed / self.rear_wheelbase * \
			torch.sin(beta) * self.dt

		return location_next, yaw_next, F.relu(speed_next) 
