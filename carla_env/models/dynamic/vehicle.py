import math
import torch
import torch.nn.functional as F
from torch import nn


class KinematicBicycleModel(nn.Module):
    def __init__(self, dt=0.1):
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

        # throttle = torch.clip(action[..., 0:1], 0, 1)
        # steer = torch.clip(action[..., 1:2], -1, 1)
        throttle = action[..., 0:1]
        steer = action[..., 1:2]
        brake = action[..., 2:3].byte()

        acceleration = torch.where(brake == 1, self.brake_acceleration.expand(
            *brake.size()), self.throttle_acceleration(throttle))

        # Transformation from steer to wheel steering angle
        # to use the kinematic model

        wheel_steer = self.steer_gain * steer

        # beta = atan((l_r * tan(delta_f)) / (l_f + l_r))
        beta = torch.atan(self.rear_wheelbase /
                          (self.front_wheelbase +
                           self.rear_wheelbase) *
                          torch.tan(wheel_steer))

        # x_ = x + v * dt
        location_next = location + speed * \
            torch.cat([torch.cos(yaw + beta), torch.sin(yaw + beta)], -1) * self.dt

        # speed_ = speed + a * dt
        speed_next = speed + acceleration * self.dt

        yaw_next = yaw + speed / self.rear_wheelbase * \
            torch.sin(beta) * self.dt

        return location_next, yaw_next, F.relu(speed_next)


class KinematicBicycleModelV2(nn.Module):
    def __init__(self, dt=0.1):
        super().__init__()

        self.dt = dt

        # Kinematic bicycle model
        self.front_wheelbase = nn.Parameter(
            torch.tensor(1.), requires_grad=True)
        self.rear_wheelbase = nn.Parameter(
            torch.tensor(1.), requires_grad=True)

        self.steer_gain = nn.Parameter(torch.tensor(1.), requires_grad=True)

        self.acceleration_encoder = nn.Sequential(
            nn.Linear(1, 1, bias=False),
        )

    def forward(self, ego_state, action):
        '''
        One step semi-parametric kinematic bicycle model
        '''

        location = ego_state["location"]
        yaw = ego_state["yaw"]
        speed = ego_state["speed"]

        acceleration = torch.clip(action[..., 0:1], -1, 1)
        steer = torch.clip(action[..., 1:2], -1, 1)

        acceleration_encoded = self.acceleration_encoder(acceleration)

        # Transformation from steer to wheel steering angle
        # to use the kinematic model

        wheel_steer = self.steer_gain * steer

        # beta = atan((l_r * tan(delta_f)) / (l_f + l_r))
        beta = torch.atan(self.rear_wheelbase /
                          (self.front_wheelbase +
                           self.rear_wheelbase) *
                          torch.tan(wheel_steer))

        # x_ = x + v * dt
        location_next = location + speed * \
            torch.cat([torch.cos(yaw + beta), torch.sin(yaw + beta)], -1) * self.dt

        # speed_ = speed + a * dt
        speed_next = speed + acceleration_encoded * self.dt

        yaw_next = yaw + speed / self.rear_wheelbase * \
            torch.sin(beta) * self.dt

        ego_state_next = {
            "location": location_next,
            "yaw": yaw_next,
            "speed": F.relu(speed_next)
        }

        return ego_state_next

    @classmethod
    def load_model_from_wandb_run(cls, run, checkpoint, device):

        checkpoint = torch.load(
            checkpoint.name,
            map_location=f"cuda:{device}" if isinstance(device, int) else device)
        model = cls(
            dt=run.config["dt"]
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return model


class KinematicBicycleModelWoR(nn.Module):
    def __init__(self, dt=1. / 4):
        super().__init__()

        self.dt = dt

        # Kinematic bicycle model
        self.front_wb = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.rear_wb = nn.Parameter(torch.tensor(1.), requires_grad=True)

        self.steer_gain = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.throt_accel = nn.Sequential(
            nn.Linear(1, 1, bias=False),
        )

    def forward(self, locs, yaws, spds, acts):
        '''
        only plannar
        '''

        steer = acts[..., 1:2]
        throt = acts[..., 0:1]
        brake = acts[..., 2:3].byte()

        accel = torch.where(
            brake == 1,
            self.brake_accel.expand(
                *brake.size()),
            self.throt_accel(throt))
        wheel = self.steer_gain * steer

        beta = torch.atan(self.rear_wb / (self.front_wb +
                          self.rear_wb) * torch.tan(wheel))

        next_locs = locs + spds * \
            torch.cat([torch.cos(yaws + beta), torch.sin(yaws + beta)], -1) * self.dt
        next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
        next_spds = spds + accel * self.dt

        return next_locs, next_yaws, F.relu(next_spds)
