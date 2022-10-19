import torch

from torch import nn


class Encoder2D(nn.Module):
    """Encoder for image-like data"""

    def __init__(self, input_shape, output_channel=256, layers=4, dropout=0.2):
        super(Encoder2D, self).__init__()

        (self.input_channel, self.input_height, self.input_width) = input_shape

        assert output_channel % (
            2 ** (layers - 1)) == 0, "Output size must be divisible by 2^(layers - 1)"

        feature_maps = [output_channel // (2 ** i)
                        for i in range(layers - 1, 0, -1)]

        encoder_layers = []

        current_size = self.input_channel
        for i, next_size in enumerate(feature_maps):
            encoder_layers += [
                nn.Conv2d(current_size, next_size, 4, 2, 1),
                nn.Dropout2d(p=dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            current_size = next_size

        encoder_layers.append(nn.Conv2d(current_size, output_channel, 4, 2, 1))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

    def get_output_shape(self):
        """Calculates the output shape of the encoder given image input"""
        with torch.no_grad():
            inp = torch.randn(
                (1, self.input_channel, self.input_height, self.input_width))
            out = self.encoder(inp)
            return out.shape[1:]


class ProbabilisticEncoder2D(Encoder2D):
    """Probabilistic encoder for image-like data"""

    def __init__(
            self,
            input_shape,
            output_channel=256,
            layers=4,
            dropout=0.2,
            latent_size=256):
        super(ProbabilisticEncoder2D, self).__init__(
            input_shape, output_channel, layers, dropout)

        in_features = self.get_output_shape().numel()
        self.fc_mu = nn.Linear(in_features, latent_size)
        self.fc_logvar = nn.Linear(in_features, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    """Encoder for 1D data"""

    def __init__(
            self,
            input_size=5,
            output_channel=256,
            layers=4,
            dropout=0.2):
        super(Encoder, self).__init__()

        assert output_channel % (
            2 ** (layers - 1)) == 0, "Output size must be divisible by 2^(layers - 1)"

        feature_maps = [output_channel // (2 ** i)
                        for i in range(layers - 1, 0, -1)]

        encoder_layers = []

        current_size = input_size
        for i, next_size in enumerate(feature_maps):
            encoder_layers += [
                nn.Linear(current_size, next_size),
                nn.Dropout(p=dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            current_size = next_size

        encoder_layers.append(nn.Linear(current_size, output_channel))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class ProbabilisticEncoder(Encoder):
    """Probabilistic encoder for 1D data"""

    def __init__(
            self,
            input_size=7,
            output_channel=256,
            layers=4,
            dropout=0.2,
            latent_size=256):
        super(ProbabilisticEncoder, self).__init__(
            input_size, output_channel, layers, dropout)

        self.fc_mu = nn.Linear(output_channel, latent_size)
        self.fc_logvar = nn.Linear(output_channel, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)
