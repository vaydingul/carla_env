from torch import nn
import torch
from carla_env.models.layers.encoder import Encoder, Encoder2D


class Policy(nn.Module):

    def __init__(
            self,
            input_shape_world_state,
            input_shape_ego_state,
            action_size,
            hidden_size=256,
            layers=4,
            dropout=0.1):
        super(Policy, self).__init__()

        self.input_shape_world_state = input_shape_world_state
        self.input_shape_ego_state = input_shape_ego_state
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout

        self.world_state_encoder = Encoder2D(
            input_shape=self.input_shape_world_state,
            output_channel=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout)

        self.world_state_encoder_fc = nn.Linear(
            in_features=self.world_state_encoder.get_output_shape(),
            out_features=self.hidden_size)

        self.ego_state_encoder = Encoder(
            input_shape=self.input_shape_ego_state,
            output_channel=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
            nn.Tanh())

    def forward(self, ego_state, world_state):

        world_state = self.world_state_encoder(world_state)
        world_state = self.world_state_encoder_fc(world_state)
        ego_state = self.ego_state_encoder(ego_state)
        x = torch.cat((world_state, ego_state), dim=1)
        action = self.fc(x)

        return action
