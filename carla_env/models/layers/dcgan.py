import torch
import torch.nn as nn


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class DCGAN64(nn.Module):
    def __init__(self, dim, nc=1):
        super(DCGAN64, self).__init__()
        self.dim = dim
        nf = 64

        # Define encoder and decoder part

        # ---------------------------------- ENCODER --------------------------------- #
        # input is (nc) x 64 x 64
        self.e_c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.e_c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.e_c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.e_c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.e_c5 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 4, 1, 0), nn.BatchNorm2d(dim), nn.Tanh()
        )

        # ---------------------------------- DECODER --------------------------------- #
        self.d_c1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (nf*8) x 4 x 4
        self.d_c2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.d_c3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.d_c4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.d_c5 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def encode(self, x):
        h1 = self.e_c1(x)
        h2 = self.e_c2(h1)
        h3 = self.e_c3(h2)
        h4 = self.e_c4(h3)
        h5 = self.e_c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]

    def decode(self, x):
        vec, skip = x
        d1 = self.d_c1(vec.view(-1, self.dim, 1, 1))
        d2 = self.d_c2(torch.cat([d1, skip[3]], 1))
        d3 = self.d_c3(torch.cat([d2, skip[2]], 1))
        d4 = self.d_c4(torch.cat([d3, skip[1]], 1))
        output = self.d_c5(torch.cat([d4, skip[0]], 1))
        return output


class DCGAN128(nn.Module):
    def __init__(self, dim, nc=1):
        super(DCGAN128, self).__init__()
        self.dim = dim
        nf = 64

        # Define encoder and decoder part

        # ---------------------------------- ENCODER --------------------------------- #
        # input is (nc) x 128 x 128
        self.e_c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.e_c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.e_c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.e_c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.e_c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.e_c6 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 4, 1, 0), nn.BatchNorm2d(dim), nn.Tanh()
        )

        # ---------------------------------- DECODER --------------------------------- #
        self.d_c1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (nf*8) x 4 x 4
        self.d_c2 = dcgan_upconv(nf * 8 * 2, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.d_c3 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.d_c4 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.d_c5 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 64 x 64
        self.d_c6 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

    def encode(self, x):
        h1 = self.e_c1(x)
        h2 = self.e_c2(h1)
        h3 = self.e_c3(h2)
        h4 = self.e_c4(h3)
        h5 = self.e_c5(h4)
        h6 = self.e_c6(h5)
        return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]

    def decode(self, x):
        vec, skip = x
        d1 = self.d_c1(vec.view(-1, self.dim, 1, 1))
        d2 = self.d_c2(torch.cat([d1, skip[4]], 1))
        d3 = self.d_c3(torch.cat([d2, skip[3]], 1))
        d4 = self.d_c4(torch.cat([d3, skip[2]], 1))
        d5 = self.d_c5(torch.cat([d4, skip[1]], 1))
        output = self.d_c6(torch.cat([d5, skip[0]], 1))
        return output


class DCGAN192(nn.Module):
    def __init__(self, dim, nc=1):
        super(DCGAN192, self).__init__()
        self.dim = dim
        nf = 64

        # Define encoder and decoder part

        # ---------------------------------- ENCODER --------------------------------- #
        # input is (nc) x 192 x 192
        self.e_c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 96 x 96
        self.e_c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 48 x 48
        self.e_c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 24 x 24
        self.e_c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 12 x 12
        self.e_c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 6 x 6
        self.e_c6 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 3 x 3
        self.e_c7 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 3, 1, 0), nn.BatchNorm2d(dim), nn.Tanh()
        )

        # ---------------------------------- DECODER --------------------------------- #
        self.d_c1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 3, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (nf*8) x 3 x 3
        self.d_c2 = dcgan_upconv(nf * 8 * 2, nf * 8)
        # state size. (nf*8) x 6 x 6
        self.d_c3 = dcgan_upconv(nf * 8 * 2, nf * 8)
        # state size. (nf*8) x 12 x 12
        self.d_c4 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 24 x 24
        self.d_c5 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 48 x 48
        self.d_c6 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 96 x 96
        self.d_c7 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 192 x 192
        )

    def encode(self, x):
        h1 = self.e_c1(x)
        h2 = self.e_c2(h1)
        h3 = self.e_c3(h2)
        h4 = self.e_c4(h3)
        h5 = self.e_c5(h4)
        h6 = self.e_c6(h5)
        h7 = self.e_c7(h6)
        return h7.view(-1, self.dim), [h1, h2, h3, h4, h5, h6]

    def decode(self, x):
        vec, skip = x
        d1 = self.d_c1(vec.view(-1, self.dim, 1, 1))
        d2 = self.d_c2(torch.cat([d1, skip[5]], 1))
        d3 = self.d_c3(torch.cat([d2, skip[4]], 1))
        d4 = self.d_c4(torch.cat([d3, skip[3]], 1))
        d5 = self.d_c5(torch.cat([d4, skip[2]], 1))
        d6 = self.d_c6(torch.cat([d5, skip[1]], 1))
        output = self.d_c7(torch.cat([d6, skip[0]], 1))
        return output
