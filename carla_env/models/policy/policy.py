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

        self.ego_state_encoder = Encoder(
            input_size=self.input_shape_ego_state,
            output_size=self.hidden_size,
            layers=self.layers,
            dropout=self.dropout,
        )

        if self.use_command_encoder:

            self.command_encoder = Encoder(
                input_size=self.command_size,
                output_size=self.hidden_size,
                layers=self.layers,
                dropout=self.dropout,
            )

        if self.use_target_encoder:

            self.target_encoder = Encoder(
                input_size=self.target_location_size,
                output_size=self.hidden_size,
                layers=self.layers,
                dropout=self.dropout,
            )

        if self.use_occupancy_encoder:

            self.occupancy_encoder = Encoder(
                input_size=self.occupancy_size,
                output_size=self.hidden_size,
                layers=self.layers,
                dropout=self.dropout,
            )

            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size * 5, self.hidden_size * 3),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.ReLU(),
            )

        else:

            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ReLU(),
            )

        self.fc = nn.Sequential(
            nn.Linear(
                self.hidden_size
                * (
                    2
                    + self.use_command_encoder
                    + self.use_occupancy_encoder
                    + self.use_target_encoder
                ),
                self.hidden_size * 2,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
        )

        self.fc_acceleration = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Tanh())
        self.fc_steer = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Tanh())

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
            self.config["input_ego_location"] * 2
            + self.config["input_ego_yaw"]
            + self.config["input_ego_speed"]
        )

        # The size of the output action prediction
        # 2 for steer and acceleration
        # 3 for steer, throttle and brake
        self.action_size = self.config["action_size"]

        # Whether to use command encoder in the architecture or not
        self.use_command_encoder = self.config["use_command_encoder"]

        # The size of the command input
        self.command_size = self.config["command_size"]

        # Whether to use target encoder in the architecture or not
        self.use_target_encoder = self.config["use_target_encoder"]

        # The size of the target location input
        self.target_location_size = self.config["target_location_size"]

        # Whether to use occupancy encoder in the architecture or not
        self.use_occupancy_encoder = self.config["use_occupancy_encoder"]

        # The size of the occupancy input
        self.occupancy_size = self.config["occupancy_size"]

        # The size of the hidden layer where everything is combined
        self.hidden_size = self.config["hidden_size"]

        # The number of layers in the encoder(s)
        # If there are multiple encoders, the number of layers in each encoder is the same
        self.layers = self.config["layers"]

        # Whether to use target location as delta state or absolute state
        self.delta_target = self.config["delta_target"]

        # Whether to use single world state input or multiple world state inputs (history)
        self.single_world_state_input = self.config["single_world_state_input"]

        # Dropout rate
        self.dropout = self.config["dropout"]

    def forward(
        self, ego_state, world_state, command=None, target_location=None, occupancy=None
    ):

        assert self.use_command_encoder == (command is not None), (
            "Command encoder is used in the architecture, "
            "but no command is provided or vice versa"
        )
        assert self.use_target_encoder == (target_location is not None), (
            "Target encoder is used in the architecture, "
            "but no target location is provided or vice versa"
        )
        assert self.use_occupancy_encoder == (occupancy is not None), (
            "Occupancy encoder is used in the architecture, "
            "but no occupancy is provided or vice versa"
        )

        if self.single_world_state_input:
            world_state = world_state[:, -1:]
        world_state = world_state.view(
            world_state.shape[0], -1, world_state.shape[-2], world_state.shape[-1]
        )

        world_state_encoded = self.world_state_encoder(world_state)

        # Flatten the encoded world state
        world_state_encoded = self.world_state_encoder_fc(
            world_state_encoded.view(world_state_encoded.size(0), -1)
        )

        # Constitute the concatenated feature vector
        x = torch.cat([world_state_encoded], dim=1)

        # Encode the ego state
        ego_state_encoded = self.ego_state_encoder(
            torch.cat([ego_state[k] for k in self.keys], dim=1)
        )

        # Concatenate the ego state encoded vector to the feature vector
        x = torch.cat([x, ego_state_encoded], dim=1)

        if self.use_command_encoder:

            command_encoded = self.command_encoder(command)

            # Concatenate the command encoded vector to the feature vector
            x = torch.cat([x, command_encoded], dim=1)

        if self.use_target_encoder:

            if self.delta_target:
                target_location = target_location - ego_state["location"]
                # rot = create_2x2_rotation_tensor_from_angle_tensor(
                #     ego_state["yaw"])
                # target_location = torch.matmul(
                #     rot, target_location.unsqueeze(-1)).squeeze(-1)

            target_encoded = self.target_encoder(target_location)

            # Concatenate the target encoded vector to the feature vector
            x = torch.cat([x, target_encoded], dim=1)

        if self.use_occupancy_encoder:

            occupancy_encoded = self.occupancy_encoder(occupancy)

            # Concatenate the occupancy encoded vector to the feature vector
            x = torch.cat([x, occupancy_encoded], dim=1)

        x_encoded = self.fc(x)
        acceleration = self.fc_acceleration(x_encoded)
        steer = self.fc_steer(x_encoded)
        action = torch.cat((acceleration, steer), dim=1)

        return action

    def set_default_config(self):

        self.config = {
            "input_shape_world_state": (8, 192, 192),
            "input_ego_location": 2,
            "input_ego_yaw": 1,
            "input_ego_speed": 1,
            "action_size": 2,
            "use_command_encoder": True,
            "command_size": 6,
            "use_target_encoder": True,
            "target_location_size": 2,
            "use_occupancy_encoder": True,
            "occupancy_size": 8,
            "hidden_size": 256,
            "layers": 4,
            "delta_target": True,
            "single_world_state_input": False,
            "dropout": 0.1,
        }

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
