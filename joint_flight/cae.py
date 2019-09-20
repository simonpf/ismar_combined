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
# Convolutional autoencoder
################################################################################

class Cae(nn.Module):
    """
    (Denoising) convolutional autoencode.
    """
    def __init__(self,
                 latent_dim,
                 loss = "mse",
                 optimizer = "adam",
                 device = "gpu",
                 n_filters = 32):
        """
        Arguments:
            latent_dim: The size of the noise vector used as input for the
                generator
            n_filters: Number of filter in the second-to-last layer.
        """
        super(Cae, self).__init__()
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

        #
        # Device
        #

        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "gpu":
            self.device = torch.device("cuda:0")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise Exception("Unknown device")

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        #
        # Optimizer
        #

        beta1 = 0.5
        lr = 0.0001

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise Exception("Unknown optimizer type.")

        #
        # Loss function
        #

        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            raise Exception("Unknown loss type.")

        self.losses = []
        self.image_list = []

    def add_noise(self, x, noise_type = "random"):
        if noise_type == "random":
            n = self.smoother(torch.randn(x.size(), device = self.device))
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
              lr = 0.001):

        self.optimizer.learning_rate = lr

        for j, x in enumerate(dataloader):

            x = x.to(self.device)

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            y = self.forward(x)
            l = self.loss(y, x)
            l.backward()
            self.optimizer.step()

            if (j % 50 == 0):
                self.losses.append(l.item())
                s = "Step {} / {}: l = {}".format(j, len(dataloader), l.item())
                print(s)

            if (j % 500 == 0):
                self.image_list.append((x, y))

    def save(self, path):
        torch.save({"state" : self.state_dict(),
                    "optimizer_state" : self.optimizer.state_dict(),
                    "losses" : self.losses,
                    "image_list" : self.image_list},
                   path)

    def load(self, path):
        state = torch.load(path)
        self.load_state_dict(state["state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.losses = state["losses"]
        self.image_list = state["image_list"]
