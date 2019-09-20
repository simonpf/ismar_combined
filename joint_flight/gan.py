import numpy as np
import shutil
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
from PIL import Image

def create_mosaic(data,
                  m = 10,
                  n = 10,
                  padding = 1):

    ind = np.random.randint(0, len(data))
    img = data[ind][0]

    h, w = img.shape

    out = np.zeros((m * (h + padding) - padding, n * (w + padding) - padding))
    for i in range(m):

        i_start = i * h
        if i > 0:
            i_start += i * padding
        i_end = i_start + h

        for j in range(n):

            j_start = j * h
            if j > 0:
                j_start += j * padding
            j_end = j_start + w

            ind = np.random.randint(0, len(data))
            out[i_start : i_end, j_start : j_end] = data[ind][0].detach().numpy()

    return out

################################################################################
# Training data
################################################################################

class IceShapes(Dataset):
    def __init__(self,
                 path,
                 discrete = False):
        self.files = glob.glob(os.path.join(path, "cip_image_*.png"))
        self.discrete = discrete

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx])
        x = np.array(im, dtype = np.float32)
        im.close()

        if self.discrete:
            x = x > 0.0
        else:
            x /= x.max()

        x = -1.0 + 2.0 * x

        fx, fy = np.random.rand(2)
        if fx > 0.5:
            x = np.fliplr(x)
            x = np.flipud(x)
            if fy > 0.5:
                x = x.T

        return torch.tensor(np.array(x[np.newaxis, :, :]))

    def extract_images(self, inds, dest):
        for ind in inds:
            shutil.copy(self.files[ind], dest)


################################################################################
# Discriminator
################################################################################

class Generator(nn.Module):
    """
    Generator module that generates particles images from Gaussian noise
    using a convolutional neural network for upsampling.
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

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(latent_dim, n_filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # 32 x 32 -> 32 x 32
            nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # Output
            nn.ConvTranspose2d(n_filters, n_filters, 4, 2, 0, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, 1, 3, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

    def generate(self, n = 1):
        z = torch.randn(n, self.latent_dim, 1, 1, device = self.device)
        return self.forward(z)

################################################################################
# Discriminator
################################################################################

class Discriminator(nn.Module):
    """
    The discriminator that learns to distinguish synthetic particle imges
    from real ones.
    """
    def __init__(self,
                 n_filters = 32,
                 output_filters = 32):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 32 x 32 -> 16 x 16
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 16 -> 8 x 8
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8 -> 4 x 4
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 x 4 -> 2 x 2
            nn.Conv2d(n_filters * 4, output_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*8) x 4 x 4
            # 2 x 2 -> 1 x 1
            nn.Conv2d(output_filters, output_filters, 2, 1, 0, bias=False),
            nn.Conv2d(output_filters, 1, 1, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return self.main(x)

class Gan:
    def __init__(self,
                 latent_dim = 100,
                 n_filters_discriminator = 32,
                 n_filters_generator = 32,
                 features = 32,
                 device = None,
                 optimizer = "adam"):

        self.generator = Generator(latent_dim, n_filters_generator)
        self.discriminator = Discriminator(n_filters = n_filters_discriminator,
                                           output_filters = features)

        # Random input to track progress
        self.fixed_noise = torch.randn(64, self.generator.latent_dim)

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        for l in self.generator.children():
            weights_init(l)

        for l in self.discriminator.children():
            weights_init(l)


        self.generator_losses = []
        self.discriminator_losses = []
        self.image_list = []
        self.input_list = []

        #
        # Create optimizers
        #

        beta1 = 0.5
        lr_gen =  0.0002
        lr_dis = 0.0002
        if optimizer == "adam":
            self.optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=lr_dis,
                                                  betas=(beta1, 0.999))
            self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=lr_gen,
                                                  betas=(beta1, 0.999))
        elif optimizer == "sgd":
            self.optimizer_dis = torch.optim.SGD(self.discriminator.parameters(), lr=lr_dis)
            self.optimizer_gen = torch.optim.SGD(self.generator.parameters(), lr=lr_gen)
        else:
            raise Exception("Unknown optimizer type.")

        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "gpu":
            self.device = torch.device("cuda:0")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise Exception("Unknown device")
        self.generator.device = device



    def train(self,
              dataloader,
              lr_gen =  0.0002,
              lr_dis = 0.0002,
              noise = 0.1):


        self.optimizer_gen.learning_rate = lr_gen
        self.optimizer_dis.learning_rate = lr_dis
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        criterion = nn.BCELoss()
        real_label = 1.0
        fake_label = 0
        iters = 0

        for i, data in enumerate(dataloader, 0):

            self.discriminator.zero_grad()

            real = data.to(self.device)
            real = real + noise * torch.randn(real.size())
            real = torch.clamp(real, -1.0, 1.0)

            bs = real.size(0)
            label = 0.1 * torch.randn((bs,), device = self.device) + real_label

            # Forward pass real batch through D
            output = self.discriminator(real).view(-1)
            err_dis_real = criterion(output, label)
            err_dis_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            fake = self.generator.generate(bs)
            fake = fake + noise * torch.randn(real.size())
            fake = torch.clamp(fake, -1.0, 1.0)

            output = self.discriminator(fake.detach()).view(-1)
            label.fill_(fake_label)
            err_dis_fake = criterion(output, label)
            err_dis_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            err_dis = err_dis_real + err_dis_fake
            self.optimizer_dis.step()

            #
            # Train generator
            #

            self.generator.zero_grad()
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            self.optimizer_gen.step()

            # Check how the generator is doing by saving G's output on fixed_noise
            self.fixed_noise.to(self.device)
            if (iters % 500 == 0):
                with torch.no_grad():
                    fake = self.generator(self.fixed_noise.view(-1, 100, 1, 1)).detach().cpu()
                    self.image_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    self.input_list.append((real, fake))

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (i, len(dataloader),
                        err_dis.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            self.generator_losses.append(errG.item())
            self.discriminator_losses.append(err_dis.item())


            iters += 1

    def get_features(self, x):
        for l in next(self.discriminator.children()).children():
            if x.size()[-1] > 1:
                x = l(x)
            else:
                break
        return x

    def save(self, path):
        torch.save({"discriminator_state" : self.discriminator.state_dict(),
                    "generator_state" : self.generator.state_dict(),
                    "discriminator_opt_state" : self.optimizer_dis.state_dict(),
                    "generator_opt_state" : self.optimizer_gen.state_dict(),
                    "discriminator_losses" : self.discriminator_losses,
                    "generator_losses" : self.generator_losses,
                    "image_list" : self.image_list,
                    "input_list" : self.input_list}, path)

    def load(self, path):
        state = torch.load(path)
        self.discriminator.load_state_dict(state["discriminator_state"])
        self.optimizer_dis.load_state_dict(state["discriminator_opt_state"])
        self.generator.load_state_dict(state["generator_state"])
        self.optimizer_gen.load_state_dict(state["generator_opt_state"])
        self.discriminator_losses = state["discriminator_losses"]
        self.generator_losses = state["generator_losses"]
        self.image_list = state["image_list"]
        self.input_list = state["input_list"]