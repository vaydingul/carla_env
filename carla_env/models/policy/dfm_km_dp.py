from torch import nn
import torch
from carla_env.models.layers.encoder import Encoder, Encoder2D


class Policy(nn.Module):

    def __init__(
            self,
            input_shape_world_state,
            input_shape_ego_state,
            action_size,
            command_size=6,
            target_location_size=2,
            hidden_size=256,
            layers=4,
            dropout=0.1):
        super(Policy, self).__init__()

        self.input_shape_world_state = input_shape_world_state
        self.input_shape_ego_state = input_shape_ego_state
        self.action_size = action_size
        self.command_size = command_size
        self.target_location_size = target_location_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout

        self.ego_state_encoder = Encoder(
            input_size=self.input_shape_ego_state,
            output_size=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout)

        self.command_encoder = Encoder(
            input_size=self.command_size,
            output_size=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout)

        self.target_encoder = Encoder(
            input_size=self.target_location_size,
            output_size=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
        )
        self.fc_acceleration = nn.Sequential(nn.Linear(self.hidden_size, 1),
                                             nn.Tanh())
        self.fc_steer = nn.Sequential(nn.Linear(self.hidden_size, 1),
                                      nn.Tanh())

    def forward(self, ego_state, command, target_location):

        ego_state_encoded = self.ego_state_encoder(ego_state)
        command_encoded = self.command_encoder(command)
        target_encoded = self.target_encoder(target_location)
        x = torch.cat(
            (ego_state_encoded,
                command_encoded,
                target_encoded),
            dim=1)
        x_encoded = self.fc(x)
        acceleration = self.fc_acceleration(x_encoded)
        steer = self.fc_steer(x_encoded)
        action = torch.cat((acceleration, steer), dim=1)

        return action


if __name__ == "__main__":

    inp2 = torch.randn(10, 10)
    inp3 = torch.randn(10, 6)
    inp4 = torch.randn(10, 2)
    policy = Policy(input_shape_ego_state=10,
        action_size=2,
        command_size=6,
        target_location_size=2,
        hidden_size=256,
        layers=4,
        dropout=0.1)

    out = policy(inp2, inp3, inp4)
    print(out.shape)
