from torch import nn
import torch
from carla_env.models.layers.encoder import Encoder, Encoder2D
from utils.cost_utils import create_2x2_rotation_tensor_from_angle_tensor
from utils.train_utils import organize_device


class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

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

        dim = (
            self.hidden_size
            + self.input_shape_ego_state
            + (self.command_size * self.use_command)
            + (self.target_location_size * self.use_target)
            + (self.occupancy_size * self.use_occupancy)
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
            nn.Tanh(),
        )

    def build_from_config(self):
        self.input_shape_world_state = self.config["input_shape_world_state"]

        self.keys = []
        if self.config["input_ego_location"] > 0:
            self.keys.append("location")
        if self.config["input_ego_yaw"] > 0:
            self.keys.append("yaw")
        if self.config["input_ego_speed"] > 0:
            self.keys.append("speed")
        self.input_shape_ego_state = (
            self.config["input_ego_location"]
            + self.config["input_ego_yaw"]
            + self.config["input_ego_speed"]
        )

        # The size of the output action prediction
        # 2 for steer and acceleration
        # 3 for steer, throttle and brake
        self.action_size = self.config["action_size"]

        # Whether to use command encoder in the architecture or not
        self.use_command = self.config["use_command"]

        # The size of the command input
        self.command_size = self.config["command_size"]

        # Whether to use target encoder in the architecture or not
        self.use_target = self.config["use_target"]

        # The size of the target location input
        self.target_location_size = self.config["target_location_size"]

        # Whether to use occupancy encoder in the architecture or not
        self.use_occupancy = self.config["use_occupancy"]

        # The size of the occupancy input
        self.occupancy_size = self.config["occupancy_size"]

        # The size of the hidden layer where everything is combined
        self.hidden_size = self.config["hidden_size"]

        # The number of layers in the encoder(s)
        # If there are multiple encoders, the number of layers in each encoder is the same
        self.layers = self.config["layers"]

        # Whether to use target location as delta state or absolute state
        self.delta_target = self.config["delta_target"]

        # Dropout rate
        self.dropout = self.config["dropout"]

    def forward(
        self, ego_state, world_state, command=None, target_location=None, occupancy=None
    ):
        ego_state_ = {
            "location": ego_state["location_array"][..., :2].squeeze(1),
            "yaw": ego_state["rotation_array"][..., 2:3].squeeze(1),
            "speed": ego_state["velocity_array"].norm(2, -1, True).squeeze(1),
        }

      
        # Encode the world state
        world_state = world_state.view(
            world_state.shape[0], -1, world_state.shape[-2], world_state.shape[-1]
        )
        world_state_encoded = self.world_state_encoder(world_state)

        # Flatten the encoded world state
        world_state_encoded = self.world_state_encoder_fc(
            world_state_encoded.view(world_state_encoded.size(0), -1)
        )

        # Initiate the fused state
        fused = torch.cat([world_state_encoded], dim=1)

        if self.use_target:
            if self.delta_target:
                target_location = target_location - ego_state_["location"]
                # rot = create_2x2_rotation_tensor_from_angle_tensor(
                #     ego_state_["yaw"])
                # target_location = torch.matmul(
                #     rot, target_location.unsqueeze(-1)).squeeze(-1)

            # Concatenate the target location to the fused state
            fused = torch.cat([fused, target_location], dim=1)

        ego_state_ = torch.cat([ego_state_[k] for k in self.keys], dim=1)
        # Concatenate the ego state to the fused state
        fused = torch.cat([fused, ego_state_], dim=1)

        if self.use_command:
            # Concatenate the command to the fused state
            fused = torch.cat([fused, command], dim=1)

        if self.use_occupancy:
            # Concatenate the occupancy to the fused state
            fused = torch.cat([fused, occupancy], dim=1)

        action = self.action_decoder(fused)

        return action

    def get_keys(self):
        return self.keys

    def set_default_config(self):
        self.config = {
            "input_shape_world_state": (8, 192, 192),
            "input_ego_location": 2,
            "input_ego_yaw": 1,
            "input_ego_speed": 1,
            "action_size": 2,
            "use_command": True,
            "command_size": 6,
            "use_target": True,
            "target_location_size": 2,
            "use_occupancy": True,
            "occupancy_size": 8,
            "hidden_size": 256,
            "layers": 4,
            "delta_target": True,
            "dropout": 0.1,
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
