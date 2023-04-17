import torch
import torch.nn as nn


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.SyncBatchNorm(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class VGG64(nn.Module):
    def __init__(self, dim, input_channel=1):
        super(VGG64, self).__init__()
        self.dim = dim
        # 64 x 64
        self.e_c1 = nn.Sequential(
            vgg_layer(input_channel, 64),
            vgg_layer(64, 64),
        )
        # 32 x 32
        self.e_c2 = nn.Sequential(
            vgg_layer(64, 128),
            vgg_layer(128, 128),
        )
        # 16 x 16
        self.e_c3 = nn.Sequential(
            vgg_layer(128, 256),
            vgg_layer(256, 256),
            vgg_layer(256, 256),
        )
        # 8 x 8
        self.e_c4 = nn.Sequential(
            vgg_layer(256, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 4 x 4
        self.e_c5 = nn.Sequential(
            nn.Conv2d(512, dim, 4, 1, 0), nn.SyncBatchNorm(dim), nn.Tanh()
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 1 x 1 -> 4 x 4
        self.d_c1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 512, 4, 1, 0),
            nn.SyncBatchNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 8 x 8
        self.d_c2 = nn.Sequential(
            vgg_layer(512 * 2, 512), vgg_layer(512, 512), vgg_layer(512, 256)
        )
        # 16 x 16
        self.d_c3 = nn.Sequential(
            vgg_layer(256 * 2, 256), vgg_layer(256, 256), vgg_layer(256, 128)
        )
        # 32 x 32
        self.d_c4 = nn.Sequential(vgg_layer(128 * 2, 128), vgg_layer(128, 64))
        # 64 x 64
        self.d_c5 = nn.Sequential(
            vgg_layer(64 * 2, 64),
            nn.ConvTranspose2d(64, input_channel, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def encode(self, x):
        h1 = self.e_c1(x)
        h2 = self.e_c2(self.mp(h1))
        h3 = self.e_c3(self.mp(h2))
        h4 = self.e_c4(self.mp(h3))
        h5 = self.e_c5(self.mp(h4))
        return h5.view(-1, self.dim), [h1, h2, h3, h4]

    def decode(self, x):
        vec, skip = x
        d1 = self.d_c1(vec.view(-1, self.dim, 1, 1))  # 1 -> 4
        up1 = self.up(d1)  # 4 -> 8
        d2 = self.d_c2(torch.cat([up1, skip[3]], 1))  # 8 x 8
        up2 = self.up(d2)  # 8 -> 16
        d3 = self.d_c3(torch.cat([up2, skip[2]], 1))  # 16 x 16
        up3 = self.up(d3)  # 8 -> 32
        d4 = self.d_c4(torch.cat([up3, skip[1]], 1))  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        output = self.d_c5(torch.cat([up4, skip[0]], 1))  # 64 x 64
        return output


class VGG128(nn.Module):
    def __init__(self, dim, input_channel=1):
        super(VGG128, self).__init__()
        self.dim = dim
        # 128 x 128
        self.e_c1 = nn.Sequential(
            vgg_layer(input_channel, 64),
            vgg_layer(64, 64),
        )
        # 64 x 64
        self.e_c2 = nn.Sequential(
            vgg_layer(64, 128),
            vgg_layer(128, 128),
        )
        # 32 x 32
        self.e_c3 = nn.Sequential(
            vgg_layer(128, 256),
            vgg_layer(256, 256),
            vgg_layer(256, 256),
        )
        # 16 x 16
        self.e_c4 = nn.Sequential(
            vgg_layer(256, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 8 x 8
        self.e_c5 = nn.Sequential(
            vgg_layer(512, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 4 x 4
        self.e_c6 = nn.Sequential(
            nn.Conv2d(512, dim, 4, 1, 0), nn.SyncBatchNorm(dim), nn.Tanh()
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 1 x 1 -> 4 x 4
        self.d_c1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 512, 4, 1, 0),
            nn.SyncBatchNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 8 x 8
        self.d_c2 = nn.Sequential(
            vgg_layer(512 * 2, 512),
            vgg_layer(256, 256),
            vgg_layer(256, 128),
        )
        # 64 x 64
        self.d_c5 = nn.Sequential(vgg_layer(128 * 2, 128), vgg_layer(128, 64))
        # 128 x 128
        self.d_c6 = nn.Sequential(
            vgg_layer(64 * 2, 64),
            nn.ConvTranspose2d(64, input_channel, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def encode(self, x):
        h1 = self.e_c1(x)
        h2 = self.e_c2(self.mp(h1))
        h3 = self.e_c3(self.mp(h2))
        h4 = self.e_c4(self.mp(h3))
        h5 = self.e_c5(self.mp(h4))
        h6 = self.e_c6(self.mp(h5))
        return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]

    def decode(self, x):
        vec, skip = x
        d1 = self.d_c1(vec.view(-1, self.dim, 1, 1))  # 1 -> 4
        up1 = self.up(d1)  # 4 -> 8
        d2 = self.d_c2(torch.cat([up1, skip[4]], 1))  # 8 x 8
        up2 = self.up(d2)  # 8 -> 16
        d3 = self.d_c3(torch.cat([up2, skip[3]], 1))  # 16 x 16
        up3 = self.up(d3)  # 8 -> 32
        d4 = self.d_c4(torch.cat([up3, skip[2]], 1))  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        d5 = self.d_c5(torch.cat([up4, skip[1]], 1))  # 64 x 64
        up5 = self.up(d5)  # 64 -> 128
        output = self.d_c6(torch.cat([up5, skip[0]], 1))  # 128 x 128
        return output


class VGG192(nn.Module):
    def __init__(self, dim, input_channel=1):
        super(VGG192, self).__init__()
        self.dim = dim
        # 192 x 192
        self.e_c1 = nn.Sequential(
            vgg_layer(input_channel, 64),
            vgg_layer(64, 64),
        )
        # 96 x 96
        self.e_c2 = nn.Sequential(
            vgg_layer(64, 128),
            vgg_layer(128, 128),
        )
        # 48 x 48
        self.e_c3 = nn.Sequential(
            vgg_layer(128, 256),
            vgg_layer(256, 256),
            vgg_layer(256, 256),
        )
        # 24 x 24
        self.e_c4 = nn.Sequential(
            vgg_layer(256, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 12 x 12
        self.e_c5 = nn.Sequential(
            vgg_layer(512, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 6 x 6
        self.e_c6 = nn.Sequential(
            vgg_layer(512, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 3 x 3
        self.e_c7 = nn.Sequential(
            nn.Conv2d(512, dim, 3, 1, 0),
            nn.SyncBatchNorm(dim),
            nn.Tanh(),
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 1 x 1 -> 3 x 3
        self.d_c1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 512, 3, 1, 0),
            nn.SyncBatchNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 6 x 6
        self.d_c2 = nn.Sequential(
            vgg_layer(512 * 2, 512),
            vgg_layer(512, 512),
            vgg_layer(512, 512),
        )
        # 12 x 12
        self.d_c3 = nn.Sequential(
            vgg_layer(512 * 2, 512), vgg_layer(512, 512), vgg_layer(512, 512)
        )
        # 24 x 24
        self.d_c4 = nn.Sequential(
            vgg_layer(512 * 2, 256), vgg_layer(256, 256), vgg_layer(256, 256)
        )
        # 48 x 48
        self.d_c5 = nn.Sequential(vgg_layer(256 * 2, 128), vgg_layer(128, 128))
        # 96 x 96
        self.d_c6 = nn.Sequential(vgg_layer(128 * 2, 64), vgg_layer(64, 64))
        # 192 x 192
        self.d_c7 = nn.Sequential(
            vgg_layer(64 * 2, 64),
            nn.ConvTranspose2d(64, input_channel, 3, 1, 1),
            # nn.Sigmoid(),
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def encode(self, x):
        h1 = self.e_c1(x)
        h2 = self.e_c2(self.mp(h1))
        h3 = self.e_c3(self.mp(h2))
        h4 = self.e_c4(self.mp(h3))
        h5 = self.e_c5(self.mp(h4))
        h6 = self.e_c6(self.mp(h5))
        h7 = self.e_c7(self.mp(h6))
        return h7.view(-1, self.dim), [h1, h2, h3, h4, h5, h6]

    def decode(self, x):
        vec, skip = x
        d1 = self.d_c1(vec.view(-1, self.dim, 1, 1))
        up1 = self.up(d1)
        d2 = self.d_c2(torch.cat([up1, skip[5]], 1))
        up2 = self.up(d2)
        d3 = self.d_c3(torch.cat([up2, skip[4]], 1))
        up3 = self.up(d3)
        d4 = self.d_c4(torch.cat([up3, skip[3]], 1))
        up4 = self.up(d4)
        d5 = self.d_c5(torch.cat([up4, skip[2]], 1))
        up5 = self.up(d5)
        d6 = self.d_c6(torch.cat([up5, skip[1]], 1))
        up6 = self.up(d6)
        output = self.d_c7(torch.cat([up6, skip[0]], 1))
        return output


if __name__ == "__main__":
    v1 = VGG64(128, 7)
    v2 = VGG128(128, 7)
    v3 = VGG192(128, 7)
    # Print number of parameters for each model
    print("VGG64: ", sum(p.numel() for p in v1.parameters() if p.requires_grad))
    print("VGG128: ", sum(p.numel() for p in v2.parameters() if p.requires_grad))
    print("VGG192: ", sum(p.numel() for p in v3.parameters() if p.requires_grad))
