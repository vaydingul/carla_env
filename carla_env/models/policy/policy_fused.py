from torch import nn
import torch
from carla_env.models.layers.encoder import Encoder, Encoder2D
from utils.cost_utils import create_2x2_rotation_tensor_from_angle_tensor


class Policy(nn.Module):
    def __init__(
        self,
        input_shape_world_state,
        input_ego_location,
        input_ego_yaw,
        input_ego_speed,
        action_size,
        command_size=6,
        target_location_size=2,
        occupancy_size=None,
        hidden_size=256,
        layers=4,
        delta_target=True,
        dropout=0.1,
    ):
        super(Policy, self).__init__()

        self.input_shape_world_state = input_shape_world_state

        self.keys = []
        if input_ego_location > 0:
            self.keys.append("location")
        if input_ego_yaw > 0:
            self.keys.append("yaw")
        if input_ego_speed > 0:
            self.keys.append("speed")

        self.input_shape_ego_state = (
            input_ego_location * 2 + input_ego_yaw + input_ego_speed
        )

        self.action_size = action_size
        self.command_size = command_size
        self.target_location_size = target_location_size
        self.occupancy_size = occupancy_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.delta_target = delta_target
        self.dropout = dropout

        self.world_state_encoder = Encoder2D(
            input_shape=self.input_shape_world_state,
            output_channel=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout,
        )

        self.world_state_encoder_fc = nn.Linear(
            in_features=self.world_state_encoder.get_output_shape().numel(),
            out_features=self.hidden_size,
        )

        dim = 0

        dim += self.input_shape_ego_state
        dim += self.command_size
        dim += self.target_location_size
        dim += self.occupancy_size if self.occupancy_size is not None else 0
        dim += self.hidden_size

        self.action_decoder = nn.Sequential(
            nn.Linear(dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
            nn.Tanh(),
        )

    def forward(self, ego_state, world_state, command, target_location, occupancy=None):

        # Encode the world state
        world_state = world_state.view(
            world_state.shape[0], -1, world_state.shape[-2], world_state.shape[-1]
        )
        world_state_encoded = self.world_state_encoder(world_state)

        # Flatten the encoded world state
        world_state_encoded = self.world_state_encoder_fc(
            world_state_encoded.view(world_state_encoded.size(0), -1)
        )

        if self.delta_target:
            target_location = target_location - ego_state["location"]
            # rot = create_2x2_rotation_tensor_from_angle_tensor(
            #     ego_state["yaw"])
            # target_location = torch.matmul(
            #     rot, target_location.unsqueeze(-1)).squeeze(-1)

        fused = [
            torch.cat([ego_state[k] for k in self.keys], dim=1),
            command,
            target_location,
            world_state_encoded,
        ]

        if self.occupancy_size is not None and self.occupancy_size > 0:
            fused.append(occupancy)

        fused = torch.cat(fused, dim=1)

        action = self.action_decoder(fused)

        return action

    @classmethod
    def load_model_from_wandb_run(cls, run, checkpoint, device):

        checkpoint = torch.load(
            checkpoint.name,
            map_location=f"cuda:{device}" if isinstance(device, int) else device,
        )

        model = cls(
            input_shape_world_state=run.config["input_shape_world_state"],
            input_ego_location=run.config["input_ego_location"],
            input_ego_yaw=run.config["input_ego_yaw"],
            input_ego_speed=run.config["input_ego_speed"],
            action_size=run.config["action_size"],
            hidden_size=run.config["hidden_size"],
            occupancy_size=run.config["occupancy_size"],
            layers=run.config["num_layer"],
            delta_target=run.config["delta_target"],
            
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        return model


if __name__ == "__main__":

    inp1 = torch.randn(10, 10, 192, 192)
    inp2 = torch.randn(10, 10)
    inp3 = torch.randn(10, 6)
    inp4 = torch.randn(10, 2)
    policy = Policy(
        input_shape_world_state=(10, 192, 192),
        input_shape_ego_state=10,
        action_size=2,
        command_size=6,
        target_location_size=2,
        hidden_size=256,
        layers=4,
        dropout=0.1,
    )

    out = policy(inp2, inp1, inp3, inp4)
    print(out.shape)
