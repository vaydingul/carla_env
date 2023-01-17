from typing import List, TypeVar

import torch
from torch import nn
from torch.nn import functional as F
from carla_env.models.layers.encoder import Encoder2D, ProbabilisticEncoder2D
from carla_env.models.layers.decoder import Decoder2D

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')


class WorldBEVModel(nn.Module):

    def __init__(self,
                 input_shape: List[int] = [
            8,
            192,
            192],
        latent_size: int = 256,
        hidden_channel: int = 256,
        output_channel: int = 512,
        num_encoder_layer: int = 4,
        num_probabilistic_encoder_layer: int = 2,
        num_time_step: int = 2,
        dropout: float = 0.2
    ) -> None:
        super(WorldBEVModel, self).__init__()

        self.input_shape_previous = input_shape.copy()
        self.input_shape_future = input_shape.copy()
        self.latent_size = latent_size
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.num_encoder_layer = num_encoder_layer
        self.num_probabilistic_encoder_layer = num_probabilistic_encoder_layer

        self.num_time_step_previous = num_time_step - 1
        self.num_time_step_future = 1

        self.input_shape_previous[0] *= self.num_time_step_previous
        self.input_shape_future[0] *= self.num_time_step_future

        self.world_previous_bev_encoder = Encoder2D(
            input_shape=self.input_shape_previous,
            output_channel=hidden_channel,
            layers=self.num_encoder_layer,
            dropout=dropout)

        self.world_future_bev_encoder = Encoder2D(
            input_shape=self.input_shape_future,
            output_channel=hidden_channel,
            layers=self.num_encoder_layer,
            dropout=dropout)

        self.probabilistic_encoder = ProbabilisticEncoder2D(
            input_shape=self.world_previous_bev_encoder.get_output_shape(),
            output_channel=output_channel,
            layers=self.num_probabilistic_encoder_layer,
            latent_size=latent_size,
            dropout=dropout)

        self.latent_expander = nn.Sequential(
            nn.Linear(
                in_features=latent_size,
                out_features=self.world_previous_bev_encoder.get_output_shape().numel()))

        self.world_bev_decoder = Decoder2D(
            input_shape=self.world_previous_bev_encoder.get_output_shape(),
            output_channel=self.input_shape_future[0],
            layers=self.num_encoder_layer,
            dropout=dropout)

    def forward(
            self,
            world_previous_bev,
            world_future_bev=None,
            sample_latent=False):

        world_previous_bev = world_previous_bev.view(
            world_previous_bev.shape[0],
            -1,
            world_previous_bev.shape[-2],
            world_previous_bev.shape[-1])

        if world_future_bev is not None:

            world_future_bev = world_future_bev.view(
                world_future_bev.shape[0],
                -1,
                world_future_bev.shape[-2],
                world_future_bev.shape[-1])

        if not sample_latent:

            world_previous_bev_encoded = self.world_previous_bev_encoder(
                world_previous_bev)
            world_future_bev_encoded = self.world_future_bev_encoder(
                world_future_bev)

            latent_representation, mu, logvar = self.probabilistic_encoder(
                world_previous_bev_encoded + world_future_bev_encoded)

            h = self.latent_expander(latent_representation) + \
                world_previous_bev_encoded.flatten(start_dim=1)

            world_future_bev_predicted = self.world_bev_decoder(
                h.view(world_previous_bev_encoded.shape))

            return (world_future_bev_predicted, mu, logvar)

        else:

            world_previous_bev_encoded = self.world_previous_bev_encoder(
                world_previous_bev)

            latent_representation = torch.randn(
                world_previous_bev_encoded.shape[0],
                self.latent_size).to(
                world_previous_bev.device)

            h = self.latent_expander(latent_representation) + \
                world_previous_bev_encoded.flatten(start_dim=1)

            world_future_bev_predicted = self.world_bev_decoder(
                h.view(world_previous_bev_encoded.shape))

            return world_future_bev_predicted

    @classmethod
    def load_model_from_wandb_run(cls, run, checkpoint, device):

        checkpoint = torch.load(
            checkpoint.name,
            map_location=f"cuda:{device}" if isinstance(device, int) else device)
        model = cls(
            input_shape=run.config["input_shape"],
            hidden_channel=run.config["hidden_channel"],
            output_channel=run.config["output_channel"],
            num_encoder_layer=run.config["num_encoder_layer"],
            num_probabilistic_encoder_layer=run.config[
                "num_probabilistic_encoder_layer"],
            num_time_step=run.config["num_time_step_previous"] + 1,
            dropout=run.config["dropout"],
            latent_size=run.config["latent_size"] if "latent_size" in run.config else 256)

        model.load_state_dict(checkpoint["model_state_dict"])

        return model


if __name__ == "__main__":
    model = WorldBEVModel(input_shape=[8, 192, 192])
    print(model)

    inp = torch.rand(1, 10, 8, 192, 192)
    out, _, _ = model(inp, inp)
    print(F.mse_loss(out, inp.squeeze()))
