"""Policies: abstract base class and concrete implementations."""

import torch as th
import torch.nn as nn
import numpy as np
from gym import spaces
from . import torch_util as tu
from carla_env.models.dynamic.vehicle import KinematicBicycleModelFromParams
from carla_env.models.world.world import (
    WorldBEVModelRepeatedFrames,
    WorldBEVModelPassThrough,
)
from carla_env.cost.masked_cost_batched_extended_bev_with_pedestrian import Cost


class XtMaCNN(nn.Module):
    """
    Inspired by https://github.com/xtma/pytorch_car_caring
    """

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        n_input_channels = observation_space["birdview"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space["birdview"].sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + states_neurons[-1], 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        states_neurons = [observation_space["state"].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(
                nn.Linear(states_neurons[i], states_neurons[i + 1])
            )
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state):
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)

        # latent_state = state.repeat(1, state.shape[1]*256)

        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(
        self,
        observation_space,
        chans=(16, 32, 32, 64, 64),
        states_neurons=[256],
        features_dim=256,
        nblock=2,
        batch_norm=False,
        final_relu=True,
    ):
        # (16, 32, 32)
        super().__init__()
        self.features_dim = features_dim
        self.final_relu = final_relu

        # image encoder
        curshape = observation_space["birdview"].shape
        s = 1 / np.sqrt(len(chans))  # per stack scale
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = tu.CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                scale=s,
                batch_norm=batch_norm,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        # dense after concatenate
        n_image_latent = tu.intprod(curshape)
        self.dense = tu.NormedLinear(
            n_image_latent + states_neurons[-1], features_dim, scale=1.4
        )

        # state encoder
        states_neurons = [observation_space["state"].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(
                tu.NormedLinear(states_neurons[i], states_neurons[i + 1])
            )
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

    def forward(self, birdview, state):
        # birdview: [b, c, h, w]
        # x = x.to(dtype=th.float32) / self.scale_ob

        for layer in self.stacks:
            birdview = layer(birdview)

        x = th.flatten(birdview, 1)
        x = th.relu(x)

        latent_state = self.state_linear(state)

        x = th.cat((x, latent_state), dim=1)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x


class CARLASystem(nn.Module):
    def __init__(self):
        super().__init__()

        self.ego_forward_model = KinematicBicycleModelFromParams()

        self.world_forward_model = WorldBEVModelPassThrough(n=10)

    def forward(self, state, action):
        ego_state = state["ego"]
        world_state = state["world"]

        ego_state = self.ego_forward_model(ego_state, action)
        # World on Rails assumption
        world_state = self.world_forward_model(world_state)

        state["ego"] = ego_state
        state["world"] = world_state

        return state


class CARLACost(Cost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, target, action, cost_dict=None):
        location = state["ego"]["location"]
        speed = state["ego"]["speed"]
        yaw = state["ego"]["yaw"]
        bev = state["world"]

        target_location = target["ego"]["location"]
        target_speed = target["ego"]["speed"]
        target_yaw = target["ego"]["yaw"]


        cost = super().forward(
            location=location,
            yaw=yaw,
            speed=speed,
            bev=bev,
        )

        
