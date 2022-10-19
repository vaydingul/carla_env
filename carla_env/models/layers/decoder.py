import torch

from torch import nn


class Decoder2D(nn.Module):
    """Decoder for image-like data"""

    def __init__(
            self,
            input_shape,
            output_channel=7,
            layers=4,
            dropout=0.2):
        super(Decoder2D, self).__init__()

        (self.input_channel, self.input_height, self.input_width) = input_shape

        assert self.input_channel % (
            2 ** (layers - 1)) == 0, "Output size must be divisible by 2^(layers - 1)"

        feature_maps = [self.input_channel // (2 ** i)
                        for i in range(0, layers - 1)]

        decoder_layers = []

        current_size = self.input_channel
        for i, next_size in enumerate(feature_maps):
            decoder_layers += [
                nn.ConvTranspose2d(current_size, next_size, 4, 2, 1),
                nn.Dropout2d(p=dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            current_size = next_size

        decoder_layers.append(
            nn.ConvTranspose2d(
                current_size,
                output_channel,
                4,
                2,
                1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)

    def get_output_shape(self):
        """Calculates the output shape of the decoder given hidden representation"""

        with torch.no_grad():
            inp = torch.randn(
                (1, self.input_channel, self.input_height, self.input_width))
            out = self.decoder(inp)
            return out.shape[1:]
