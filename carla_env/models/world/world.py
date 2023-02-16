from typing import List, TypeVar

import torch
from torch import nn
from torch.nn import functional as F
from carla_env.models.layers.encoder import Encoder2D, ProbabilisticEncoder2D
from carla_env.models.layers.decoder import Decoder2D
from utils.train_utils import organize_device


class WorldBEVModel(nn.Module):
    def __init__(self, config) -> None:
        super(WorldBEVModel, self).__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        self.world_previous_bev_encoder = Encoder2D(
            input_shape=self.input_shape_previous,
            output_channel=self.hidden_channel,
            layers=self.num_encoder_layer,
            dropout=self.dropout,
        )

        self.world_future_bev_encoder = Encoder2D(
            input_shape=self.input_shape_future,
            output_channel=self.hidden_channel,
            layers=self.num_encoder_layer,
            dropout=self.dropout,
        )

        self.probabilistic_encoder = ProbabilisticEncoder2D(
            input_shape=self.world_previous_bev_encoder.get_output_shape(),
            output_channel=self.output_channel,
            layers=self.num_probabilistic_encoder_layer,
            latent_size=self.latent_size,
            dropout=self.dropout,
        )

        self.latent_expander = nn.Sequential(
            nn.Linear(
                in_features=self.latent_size,
                out_features=self.world_previous_bev_encoder.get_output_shape().numel(),
            )
        )

        self.world_bev_decoder = Decoder2D(
            input_shape=self.world_previous_bev_encoder.get_output_shape(),
            output_channel=self.input_shape_future[0],
            layers=self.num_encoder_layer,
            dropout=self.dropout,
        )

    def build_from_config(self):

        self.input_shape_previous = self.config["input_shape"].copy()
        self.input_shape_future = self.config["input_shape"].copy()
        self.latent_size = self.config["latent_size"]
        self.hidden_channel = self.config["hidden_channel"]
        self.output_channel = self.config["output_channel"]
        self.num_encoder_layer = self.config["num_encoder_layer"]
        self.num_probabilistic_encoder_layer = self.config[
            "num_probabilistic_encoder_layer"
        ]

        self.num_time_step_previous = self.config["num_time_step_previous"]
        self.num_time_step_future = self.config["num_time_step_future"]

        self.input_shape_previous[0] *= self.num_time_step_previous
        self.input_shape_future[0] *= self.num_time_step_future

        self.dropout = self.config["dropout"]

    def forward(self, world_previous_bev, world_future_bev=None, sample_latent=False):

        world_previous_bev = world_previous_bev.view(
            world_previous_bev.shape[0],
            -1,
            world_previous_bev.shape[-2],
            world_previous_bev.shape[-1],
        )

        if world_future_bev is not None:

            world_future_bev = world_future_bev.view(
                world_future_bev.shape[0],
                -1,
                world_future_bev.shape[-2],
                world_future_bev.shape[-1],
            )

        if not sample_latent:

            world_previous_bev_encoded = self.world_previous_bev_encoder(
                world_previous_bev
            )
            world_future_bev_encoded = self.world_future_bev_encoder(world_future_bev)

            latent_representation, mu, logvar = self.probabilistic_encoder(
                world_previous_bev_encoded + world_future_bev_encoded
            )

            h = self.latent_expander(
                latent_representation
            ) + world_previous_bev_encoded.flatten(start_dim=1)

            world_future_bev_predicted = self.world_bev_decoder(
                h.view(world_previous_bev_encoded.shape)
            )

            return (world_future_bev_predicted, mu, logvar)

        else:

            world_previous_bev_encoded = self.world_previous_bev_encoder(
                world_previous_bev
            )

            latent_representation = torch.randn(
                world_previous_bev_encoded.shape[0], self.latent_size
            ).to(world_previous_bev.device)

            h = self.latent_expander(
                latent_representation
            ) + world_previous_bev_encoded.flatten(start_dim=1)

            world_future_bev_predicted = self.world_bev_decoder(
                h.view(world_previous_bev_encoded.shape)
            )

            return world_future_bev_predicted

    def set_default_config(self):

        self.config = {
            "input_shape": [8, 192, 192],
            "latent_size": 256,
            "hidden_channel": 256,
            "output_channel": 512,
            "num_encoder_layer": 4,
            "num_probabilistic_encoder_layer": 2,
            "num_time_step_previous": 1,
            "num_time_step_future": 1,
            "dropout": 0.2,
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
    model = WorldBEVModel(input_shape=[8, 192, 192])
    print(model)

    inp = torch.rand(1, 10, 8, 192, 192)
    out, _, _ = model(inp, inp)
    print(F.mse_loss(out, inp.squeeze()))
