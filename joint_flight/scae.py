import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
from PIL import Image

################################################################################
# Discriminator
################################################################################

def batch_to_img(batch):
    bs, _, m, n = batch.size()
    bss = int(np.sqrt(bs))
    img = np.zeros((bss * m, bss * n))
    k = 0
    for i in range(bss):
        i_start = m * i
        i_end = i_start + m
        for j in range(bss):
            j_start = n * j
            j_end = j_start + n
            img[i_start : i_end, j_start : j_end] = batch[k, 0, :, :].detach().numpy()
            k += 1
    return img

class Dcae(nn.Module):
    """
    Denoising convolutional autoencode.
    """
    def __init__(self,
                 latent_dim,
                 n_filters = 32):
        """
        Arguments:
            latent_dim: The size of the noise vector used as input for the
                generator
            n_filters: Number of filter in the second-to-last layer.
        """
        super(Dcae, self).__init__()
        self.latent_dim = latent_dim

        #
        # The encoder
        #

        self.encoder = nn.Sequential(
            # 32 x 32 -> 16 x 16
            nn.Sequential(nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
                          nn.LeakyReLU(0.2, inplace=True)),
            # 16 x 16 -> 8 x 8
            nn.Sequential(nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(n_filters * 2),
                          nn.LeakyReLU(0.2, inplace=True)),
            # 8 x 8 -> 4 x 4
            nn.Sequential(nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(n_filters * 4),
                          nn.LeakyReLU(0.2, inplace=True)),
            # 4 x 4 -> 2 x 2
            nn.Sequential(nn.Conv2d(n_filters * 4, n_filters * 4, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(n_filters * 4)),
            # state size. (n_filters*8) x 4 x 4
            # 2 x 2 -> 1 x 1
            nn.Conv2d(n_filters * 4, latent_dim, 2, 1, 0, bias=False)
            )

        #
        # The decoder
        #

        self.decoder = nn.Sequential(
            # 1 x 1 -> 4 x 4
            nn.Sequential(nn.ConvTranspose2d(latent_dim, n_filters * 4, 4, 1, 0, bias=False),
                          nn.BatchNorm2d(n_filters * 4),
                          nn.LeakyReLU(0.2, True)),
            # 4 x 4 -> 8 x 8
            nn.Sequential(nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(n_filters * 2),
                          nn.LeakyReLU(0.2, True)),
            # 8 x 8 -> 16 x 16
            nn.Sequential(nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(n_filters),
                          nn.LeakyReLU(0.2, True)),
            # Output
            # 16 x 16 -> 32 x 32
            nn.Sequential(nn.ConvTranspose2d(n_filters, n_filters, 4, 2, 0, bias=False),
                          nn.BatchNorm2d(n_filters),
                          nn.Conv2d(n_filters, 1, 3, 1, 0, bias=False),
                          nn.Tanh())
        )

        self.smoother = nn.Conv2d(1, 1, 3, 1, 1, bias = False)
        self.smoother.weight.data = torch.ones(1, 1, 3, 3) / 5.0
        self.smoother.weight.requires_grad = False

    def add_noise(self, x, noise_type = "random"):
        if noise_type == "random":
            n = self.smoother(torch.randn(x.size()))
            other = -1.0 * torch.ones(x.size())
            return torch.where(n > 0.0, x, other)
        else:
            if np.random.rand() < 0.5:
                n = x.size()[-2]
                x_n = x.clone().detach()
                if np.random.rand() < 0.5:
                    x_n[:, :, n // 2 :, :] = -1.0
                else:
                    x_n[:, :, : n // 2, :] = -1.0
            else:
                n = x.size()[-1]
                x_n = x.clone().detach()
                if np.random.rand() < 0.5:
                    x_n[:, :, :, n // 2 :] = -1.0
                else:
                    x_n[:, :, :, : n // 2] = -1.0
            return x_n

    def forward(self, x, layer = None):
        if layer is None:
            return self.decoder(self.encoder(x))
        else:
            for l in list(encoder.children())[:layer]:
                x = l(x)
            for l in list(decoder.children())[-layer:]:
                x = l(x)
        return x

    def train(self,
              dataloader,
              lr = 0.001,
              beta1 = 0.5,
              noise_type = "half"):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

        loss = nn.MSELoss()
        self.outputs = []

        for i in range(4):
            print("Training layer {}:".format(i))

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            for j,l in enumerate(self.encoder.children()):
                if i == j:
                    l.requires_grad = True
                else:
                    l.requires_grad = False

            for j,l in enumerate(self.decoder.children()):
                if i == j:
                    l.requires_grad = True
                else:
                    l.requires_grad = False


            for j, x in enumerate(dataloader):

                if j > 1000: break

                x_noisy = self.add_noise(x, noise_type = noise_type)
                y = self.forward(x_noisy)

                if j == 0:
                    self.outputs += [batch_to_img(y)]

                l = loss(y, x)
                l.backward()
                optimizer.step()

                if (j % 50 == 0):
                    s = "Step {} / {}: l = {}".format(j, len(dataloader), l.item())
                    print(s)

    def train_all(self,
                  dataloader,
                  lr = 0.001,
                  beta1 = 0.5,
                  noise_type = "half"):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

        loss = nn.MSELoss()
        self.outputs = []

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        for j, x in enumerate(dataloader):

            x_noisy = self.add_noise(x, noise_type = noise_type)
            x_noisy = x.clone()
            x_noisy.requires_grad = True
            y = self.forward(x_noisy)

            if j == 0:
                self.outputs += [batch_to_img(y)]

            l = loss(y, x)
            l.backward()
            optimizer.step()

            if (j % 50 == 0):
                s = "Step {} / {}: l = {}".format(j, len(dataloader), l.item())
                print(s)
