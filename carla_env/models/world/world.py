from typing import List, TypeVar

import torch
from torch import nn
from torch.nn import functional as F

from carla_env.models.layers.encoder import Encoder2D, ProbabilisticEncoder2D
from carla_env.models.layers.decoder import Decoder2D

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')


class WorldBEVModel(nn.Module):
    # TODO: Input will be BEV at (t) and (t+1)
    # TODO: Figure out how the BEV at two different time-steps will be combined (sum or concat?) ANSWER: IT IS SUM
    # TODO: Create a function Combine layer to fuse different ideas (sum and concat)
    # TODO: Output will be the reconstruction of the BEV at (t+1)
    # TODO: It will basically be a conditional autoencoder which is conditioned on BEV at (t)
    # TODO: What we care is the latent code to encode to have an idea about
    # the change in the environment

    # ! NOTE: Everything is encoded before it is processed.

    # FIXME: Apart from everything, fix the code! WTF is this?

    def __init__(self,
                 input_shape: List[int],
                 latent_size: int = 256,
                 hidden_channel: int = 32,
                 output_channel: int = 512,
                 num_encoder_layer: int = 4,
                 num_probabilistic_encoder_layer: int = 2,
                 num_time_step: int = 2
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
            layers=self.num_encoder_layer)

        self.world_future_bev_encoder = Encoder2D(
            input_shape=self.input_shape_future,
            output_channel=hidden_channel,
            layers=self.num_encoder_layer)

        self.probabilistic_encoder = ProbabilisticEncoder2D(
            input_shape=self.world_previous_bev_encoder.get_output_shape(),
            output_channel=output_channel,
            layers=self.num_probabilistic_encoder_layer,
            latent_size=latent_size)

        self.latent_expander = nn.Sequential(
            nn.Linear(
                in_features=latent_size,
                out_features=self.world_previous_bev_encoder.get_output_shape().numel()))

        self.world_bev_decoder = Decoder2D(
            input_shape=self.world_previous_bev_encoder.get_output_shape(),
            output_channel=self.input_shape_future[0],
            layers=self.num_encoder_layer)

    def forward(self, world_previous_bev, world_future_bev):

        world_previous_bev = world_previous_bev.view(
            world_previous_bev.shape[0],
            -1,
            world_previous_bev.shape[-2],
            world_previous_bev.shape[-1])

        world_future_bev = world_future_bev.view(
            world_future_bev.shape[0],
            -1,
            world_future_bev.shape[-2],
            world_future_bev.shape[-1])

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

        return world_future_bev_predicted, mu, logvar


if __name__ == "__main__":
    model = WorldBEVModel(input_shape=[7, 192, 192])
    print(model)

    inp = torch.rand(1, 1, 7, 192, 192)
    out, _, _ = model(inp, inp)
    print(F.mse_loss(out, inp.squeeze()))
