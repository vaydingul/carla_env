from typing import List, TypeVar

import torch
from torch import nn
from torch.nn import functional as F
from carla_env.models.layers.vgg import VGG64, VGG128, VGG192
from carla_env.models.layers.dcgan import DCGAN64, DCGAN128, DCGAN192
from carla_env.models.layers.lstm import LSTM, ProbabilisticLSTM
from utilities.train_utils import organize_device


class WorldSVGLPModel(nn.Module):
    def __init__(self, config) -> None:
        super(WorldSVGLPModel, self).__init__()

        self.set_default_config()
        self.append_config(config)
        self.build_from_config()

        encoder_decoder_class = self.get_encoder_decoder()
        self.encoder_decoder = encoder_decoder_class(
            dim=self.encoder_output_decoder_input_size,
            input_channel=self.input_shape[0],
        )

        self.frame_predictor = LSTM(
            input_size=self.encoder_output_decoder_input_size + self.latent_size,
            output_size=self.encoder_output_decoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.frame_predictor_num_layer,
        )

        self.posterior_network = ProbabilisticLSTM(
            input_size=self.encoder_output_decoder_input_size,
            output_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.posterior_network_num_layer,
        )

        self.prior_network = ProbabilisticLSTM(
            input_size=self.encoder_output_decoder_input_size,
            output_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.prior_network_num_layer,
        )

    def build_from_config(self):
        self.input_shape = self.config["input_shape"]
        self.encoder_decoder_type = self.config["encoder_decoder_type"]
        self.num_time_step_previous = self.config["num_time_step_previous"]
        self.num_time_step_future = self.config["num_time_step_future"]
        self.latent_size = self.config["latent_size"]
        self.hidden_size = self.config["hidden_size"]
        self.encoder_output_decoder_input_size = self.config[
            "encoder_output_decoder_input_size"
        ]
        self.frame_predictor_num_layer = self.config["frame_predictor_num_layer"]
        self.posterior_network_num_layer = self.config["posterior_network_num_layer"]
        self.prior_network_num_layer = self.config["prior_network_num_layer"]

    def set_default_config(self):
        self.config = {
            "input_shape": (8, 192, 192),
            "encoder_decoder_type": "vgg192",
            "num_time_step_previous": 10,
            "num_time_step_future": 10,
            "latent_size": 64,
            "hidden_size": 256,
            "encoder_output_decoder_input_size": 128,
            "frame_predictor_num_layer": 3,
            "posterior_network_num_layer": 1,
            "prior_network_num_layer": 1,
        }

    def get_encoder_decoder(self):
        if self.encoder_decoder_type == "vgg64":
            return VGG64
        elif self.encoder_decoder_type == "vgg128":
            return VGG128
        elif self.encoder_decoder_type == "vgg192":
            return VGG192
        elif self.encoder_decoder_type == "dcgan64":
            return DCGAN64
        elif self.encoder_decoder_type == "dcgan128":
            return DCGAN128
        elif self.encoder_decoder_type == "dcgan192":
            return DCGAN192
        else:
            raise NotImplementedError

    def init_hidden(self, batch_size, device):
        self.frame_predictor.init_hidden(batch_size, device)
        self.posterior_network.init_hidden(batch_size, device)
        self.prior_network.init_hidden(batch_size, device)

    def append_config(self, config):
        self.config.update(config)

    def forward(
        self,
        world_previous_bev,
        world_future_bev,
        skip_feature=None,
    ):
        h_previous, skip_previous = self.encoder_decoder.encode(world_previous_bev)
        h_future, _ = self.encoder_decoder.encode(world_future_bev)

        (z_prior, mu_prior, logvar_prior) = self.prior_network(h_previous)
        (z_posterior, mu_posterior, logvar_posterior) = self.posterior_network(h_future)

        h_predicted = self.frame_predictor(torch.cat([h_previous, z_posterior], dim=1))

        skip_feature = skip_previous if skip_feature is None else skip_feature

        world_future_bev_predicted = self.encoder_decoder.decode(
            [h_predicted, skip_feature]
        )

        output = {
            "world_future_bev_predicted": world_future_bev_predicted,
            "z_posterior": z_posterior,
            "mu_posterior": mu_posterior,
            "logvar_posterior": logvar_posterior,
            "z_prior": z_prior,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "skip_feature": skip_previous,
        }

        return output

    def generate(self, world_previous_bev, skip_feature=None):
        h_previous, skip_previous = self.encoder_decoder.encode(world_previous_bev)
        # h_future, _ = self.encoder_decoder.encode(world_future_bev)

        (z_prior, mu_prior, logvar_prior) = self.prior_network(h_previous)
        # (z_posterior, mu_posterior, logvar_posterior) = self.posterior_network(h_future)

        h_predicted = self.frame_predictor(torch.cat([h_previous, z_prior], dim=1))

        skip_feature = skip_previous if skip_feature is None else skip_feature

        world_future_bev_predicted = self.encoder_decoder.decode(
            (h_predicted, skip_previous)
        )

        output = {
            "world_future_bev_predicted": world_future_bev_predicted,
            "z_prior": z_prior,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "skip_feature": skip_previous,
        }

        return output

    @classmethod
    def load_model_from_wandb_run(cls, config, checkpoint_path, device):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=organize_device(device),
        )

        model = cls(config)

        model.load_state_dict(checkpoint["model_state_dict"])

        return model
