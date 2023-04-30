import math
import torch
import torch.nn.functional as F
from torch import nn
from utils.train_utils import organize_device

G = 9.81


class KinematicBicycleModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        # Kinematic bicycle model
        self.front_wheelbase = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.rear_wheelbase = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.steer_gain = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.acceleration_encoder = nn.Linear(1, 1, bias=False)

    def build_from_config(self):
        self.dt = self.config["dt"]

    def forward(self, ego_state, action):
        """
        One step semi-parametric kinematic bicycle model
        """

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
        beta = torch.atan(
            self.rear_wheelbase
            / (self.front_wheelbase + self.rear_wheelbase)
            * torch.tan(wheel_steer)
        )

        # x_ = x + v * dt
        location_next = (
            location
            + speed
            * torch.cat([torch.cos(yaw + beta), torch.sin(yaw + beta)], -1)
            * self.dt
        )

        # speed_ = speed + a * dt
        speed_next = speed + acceleration_encoded * self.dt

        yaw_next = yaw + speed / self.rear_wheelbase * torch.sin(beta) * self.dt

        ego_state_next = {
            "location": location_next,
            "yaw": yaw_next,
            "speed": F.relu(speed_next),
        }

        return ego_state_next

    def set_default_config(self):
        self.config = {
            "dt": 0.1,
        }

    def append_config(self, config):
        self.config.update(config)

    @classmethod
    def load_model_from_wandb_run(cls, config, checkpoint_path, device):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=organize_device(device),
        )

        model = cls(config)

        model.load_state_dict(checkpoint["model_state_dict"])

        return model


class DynamicBicycleModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        # Kinematic bicycle model
        self.front_wheelbase = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.rear_wheelbase = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.cornering_stiffness_front = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.cornering_stiffness_rear = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.inertia = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.mass = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.frontal_area = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.aerodynamic_drag_coefficient = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.equivalent_aerodynamic_force_height = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.center_of_gravity_height = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.air_density = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.rolling_resistance_coefficient = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.longitudinal_tire_stiffness_front = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )
        self.longitudinal_tire_stiffness_rear = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )

        self.steer_encoder = nn.Sequential(
            nn.Linear(1, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 1, bias=True),
            nn.Tanh(),
        )
        self.acceleration_encoder = nn.Sequential(
            nn.Linear(1, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 1, bias=True),
        )

    def build_from_config(self):
        self.dt = self.config["dt"]

    def forward(self, ego_state, action):
        location_world = ego_state["location_array"]
        location_x_world = location_world[..., 0:1]
        location_y_world = location_world[..., 1:2]
        velocity_world = ego_state["velocity_array"]
        velocity_x_world = velocity_world[..., 0:1]
        velocity_y_world = velocity_world[..., 1:2]
        acceleration_world = ego_state["acceleration_array"]
        acceleration_x_world = acceleration_world[..., 0:1]
        acceleration_y_world = acceleration_world[..., 1:2]
        rotation = ego_state["rotation_array"]
        pitch = torch.deg2rad(ego_state[..., 1:2])
        yaw = torch.deg2rad(ego_state[..., 2:3])
        angular_velocity = ego_state["angular_velocity_array"]
        omega = torch.deg2rad(angular_velocity[..., 2:3])

        velocity_x_body = velocity_x_world * torch.cos(
            -yaw
        ) + velocity_y_world * torch.sin(-yaw)
        velocity_y_body = -velocity_x_world * torch.sin(
            -yaw
        ) + velocity_y_world * torch.cos(-yaw)

        acceleration_x_body = acceleration_x_world * torch.cos(
            -yaw
        ) + acceleration_y_world * torch.sin(-yaw)
        acceleration_y_body = -acceleration_x_world * torch.sin(
            -yaw
        ) + acceleration_y_world * torch.cos(-yaw)

        acceleration = torch.clip(action[..., 0:1], -1, 1)
        steer = torch.clip(action[..., 1:2], -1, 1)

        acceleration_encoded = self.acceleration_encoder(acceleration)
        steer_encoded = self.steer_encoder(steer)

        # TODO: Convert normal dynamical bicycle model to semi-parametric model

        # ----------------------------- LATERAL DYNAMICS ----------------------------- #

        acceleration_body_y_next = (
            (
                -(self.cornering_stiffness_rear + self.cornering_stiffness_front)
                / (self.mass * velocity_x_body)
            )
            * (velocity_y_body)
            + (
                (
                    (self.cornering_stiffness_rear * self.rear_wheelbase)
                    - (self.cornering_stiffness_front * self.front_wheelbase)
                )
                / (self.mass * velocity_x_body)
            )
            * (omega)
            + ((self.cornering_stiffness_front) / (self.mass)) * (steer_encoded)
        )

        velocity_y_body_next = velocity_y_body + acceleration_body_y_next * self.dt

        yaw_next = yaw + omega * self.dt

        omega_next = omega + (
            (
                (
                    (
                        (self.rear_wheelbase * self.cornering_stiffness_rear)
                        - (self.front_wheelbase * self.cornering_stiffness_front)
                    )
                    / (self.inertia * velocity_x_body)
                )
                * (velocity_y_body)
                + (
                    -(
                        ((self.front_wheelbase**2) * self.cornering_stiffness_front)
                        + ((self.front_wheelbase**2) * self.cornering_stiffness_rear)
                    )
                    / (self.inertia * velocity_x_body)
                )
                * (omega)
                + (
                    (self.cornering_stiffness_front * self.front_wheelbase)
                    / (self.inertia)
                )
                * (steer_encoded)
            )
            * self.dt
        )

        # ----------------------------- LONGITUDINAL DYNAMICS ----------------------------- #

        _slope_force = self.mass * G * torch.sin(pitch)
        _aerodynamics_drag_force = (
            0.5
            * self.air_density
            * self.aerodynamic_drag_coefficient
            * self.frontal_area
            * (velocity_x_body**2)
        )

        _normal_force_front = (
            (-_aerodynamics_drag_force * self.equivalent_aerodynamic_force_height)
            + (-self.mass * acceleration_x_body * self.center_of_gravity_height)
            + (-self.mass * G * self.center_of_gravity_height * torch.sin(pitch))
            + (self.mass * G * self.front_wheelbase * torch.cos(pitch))
        ) / (self.front_wheelbase + self.rear_wheelbase)

        _normal_force_rear = (
            (_aerodynamics_drag_force * self.equivalent_aerodynamic_force_height)
            + (self.mass * acceleration_x_body * self.center_of_gravity_height)
            + (self.mass * G * self.center_of_gravity_height * torch.sin(pitch))
            + (self.mass * G * self.front_wheelbase * torch.cos(pitch))
        ) / (self.front_wheelbase + self.rear_wheelbase)

        _rolling_resistance_force_front = (self.rolling_resistance_coefficient) * (
            _normal_force_front * self.rolling_resistance_coefficient
        )
        _rolling_resistance_force_rear = (self.rolling_resistance_coefficient) * (
            _normal_force_rear * self.rolling_resistance_coefficient
        )
        _tire_force_front = (
            self.longitudinal_tire_stiffness_front * acceleration_encoded
        )
        _tire_force_rear = self.longitudinal_tire_stiffness_rear * acceleration_encoded

        acceleration_x_body_next = (
            -_slope_force
            - _aerodynamics_drag_force
            - _rolling_resistance_force_front
            - _rolling_resistance_force_rear
            + _tire_force_front
            + _tire_force_rear
        ) / (self.mass)
        velocity_x_body_next = velocity_x_body + acceleration_x_body_next * self.dt

        # ----------------------------- WORLD LOCATION TRANSFORMATION ----------------------------- #
        velocity_x_world_next = velocity_x_body_next * torch.cos(
            yaw
        ) - velocity_y_body_next * torch.sin(yaw)
        velocity_y_world_next = velocity_x_body_next * torch.sin(
            yaw
        ) + velocity_y_body_next * torch.cos(yaw)
        location_x_world_next = location_x_world + velocity_x_world * self.dt
        location_y_world_next = location_y_world + velocity_y_world * self.dt
        # ----------------------------- UPDATE ----------------------------- #
        ego_state["location_array"][..., 0:2] = torch.cat(
            [location_x_world_next, location_y_world_next], dim=-1
        )
        ego_state["velocity_array"][..., 0:2] = torch.cat(
            [velocity_x_world_next, velocity_y_world_next], dim=-1
        )
        ego_state["acceleration_array"][..., 0:2] = torch.cat(
            [acceleration_x_body_next, acceleration_body_y_next], dim=-1
        )

        ego_state["rotation_array"][..., -1:] = yaw_next
        ego_state["angular_velocity_array"][..., -1:] = omega_next

        return ego_state


# class KinematicBicycleModel(nn.Module):
#     def __init__(self, dt=0.1):
#         super().__init__()

#         self.dt = dt

#         # Kinematic bicycle model
#         self.front_wheelbase = nn.Parameter(
#             torch.tensor(1.), requires_grad=True)
#         self.rear_wheelbase = nn.Parameter(
#             torch.tensor(1.), requires_grad=True)

#         self.steer_gain = nn.Sequential(
#             nn.Linear(1, 10, bias=True),
#             nn.ReLU(),
#             nn.Linear(10, 1, bias=True),
#         )

#         self.acceleration_encoder = nn.Sequential(
#             nn.Linear(1, 10, bias=True),
#             nn.ReLU(),
#             nn.Linear(10, 1, bias=True),
#         )

#     def forward(self, ego_state, action):
#         '''
#         One step semi-parametric kinematic bicycle model
#         '''

#         location = ego_state["location"]
#         yaw = ego_state["yaw"]
#         speed = ego_state["speed"]

#         acceleration = torch.clip(action[..., 0:1], -1, 1)
#         steer = torch.clip(action[..., 1:2], -1, 1)

#         acceleration_encoded = self.acceleration_encoder(acceleration)

#         # Transformation from steer to wheel steering angle
#         # to use the kinematic model

#         wheel_steer = self.steer_gain(steer)

#         # beta = atan((l_r * tan(delta_f)) / (l_f + l_r))
#         beta = torch.atan(self.rear_wheelbase /
#                           (self.front_wheelbase +
#                            self.rear_wheelbase) *
#                           torch.tan(wheel_steer))

#         # x_ = x + v * dt
#         location_next = location + speed * \
#             torch.cat([torch.cos(yaw + beta), torch.sin(yaw + beta)], -1) * self.dt

#         # speed_ = speed + a * dt
#         speed_next = speed + acceleration_encoded * self.dt

#         yaw_next = yaw + speed / self.rear_wheelbase * \
#             torch.sin(beta) * self.dt

#         ego_state_next = {
#             "location": location_next,
#             "yaw": yaw_next,
#             "speed": F.relu(speed_next)
#         }

#         return ego_state_next

#     @classmethod
#     def load_model_from_wandb_run(cls, run, checkpoint, device):

#         checkpoint = torch.load(
#             checkpoint.name,
#             map_location=f"cuda:{device}" if isinstance(
#                 device,
#                 int) else device)
#         model = cls(
#             dt=run.config["dt"]
#         )

#         model.load_state_dict(checkpoint["model_state_dict"])

#         return model


# class KinematicBicycleModel(nn.Module):
#     def __init__(self, dt=0.1):
#         super().__init__()

#         self.dt = dt

#         # Kinematic bicycle model
#         self.front_wheelbase = nn.Parameter(
#             torch.tensor(1.), requires_grad=True)
#         self.rear_wheelbase = nn.Parameter(
#             torch.tensor(1.), requires_grad=True)

#         self.steer_gain = nn.Parameter(torch.tensor(1.), requires_grad=True)

#         self.brake_acceleration = nn.Parameter(
#             torch.zeros(1), requires_grad=True)

#         self.throttle_acceleration = nn.Sequential(
#             nn.Linear(1, 1, bias=False),
#         )

#     def forward(self, location, yaw, speed, action):
#         '''
#         One step semi-parametric kinematic bicycle model
#         '''

#         # throttle = torch.clip(action[..., 0:1], 0, 1)
#         # steer = torch.clip(action[..., 1:2], -1, 1)
#         throttle = action[..., 0:1]
#         steer = action[..., 1:2]
#         brake = action[..., 2:3].byte()

#         acceleration = torch.where(brake == 1, self.brake_acceleration.expand(
#             *brake.size()), self.throttle_acceleration(throttle))

#         # Transformation from steer to wheel steering angle
#         # to use the kinematic model

#         wheel_steer = self.steer_gain * steer

#         # beta = atan((l_r * tan(delta_f)) / (l_f + l_r))
#         beta = torch.atan(self.rear_wheelbase /
#                           (self.front_wheelbase +
#                            self.rear_wheelbase) *
#                           torch.tan(wheel_steer))

#         # x_ = x + v * dt
#         location_next = location + speed * \
#             torch.cat([torch.cos(yaw + beta), torch.sin(yaw + beta)], -1) * self.dt

#         # speed_ = speed + a * dt
#         speed_next = speed + acceleration * self.dt

#         yaw_next = yaw + speed / self.rear_wheelbase * \
#             torch.sin(beta) * self.dt

#         return location_next, yaw_next, F.relu(speed_next)


# class KinematicBicycleModelWoR(nn.Module):
#     def __init__(self, dt=1.0 / 4):
#         super().__init__()

#         self.dt = dt

#         # Kinematic bicycle model
#         self.front_wb = nn.Parameter(torch.tensor(1.0), requires_grad=True)
#         self.rear_wb = nn.Parameter(torch.tensor(1.0), requires_grad=True)

#         self.steer_gain = nn.Parameter(torch.tensor(1.0), requires_grad=True)
#         self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
#         self.throt_accel = nn.Sequential(
#             nn.Linear(1, 1, bias=False),
#         )

#     def forward(self, locs, yaws, spds, acts):
#         """
#         only plannar
#         """

#         steer = acts[..., 1:2]
#         throt = acts[..., 0:1]
#         brake = acts[..., 2:3].byte()

#         accel = torch.where(
#             brake == 1, self.brake_accel.expand(*brake.size()), self.throt_accel(throt)
#         )
#         wheel = self.steer_gain * steer

#         beta = torch.atan(
#             self.rear_wb / (self.front_wb + self.rear_wb) * torch.tan(wheel)
#         )

#         next_locs = (
#             locs
#             + spds
#             * torch.cat([torch.cos(yaws + beta), torch.sin(yaws + beta)], -1)
#             * self.dt
#         )
#         next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
#         next_spds = spds + accel * self.dt

#         return next_locs, next_yaws, F.relu(next_spds)
