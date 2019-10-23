import argparse

from joint_flight import path
from joint_flight.gan import InfoGan
from joint_flight.particles import IceShapes
from torchvision import datasets, transforms
import os
import torch

#
# Command line arguments
#

parser = argparse.ArgumentParser(prog = "gan",
                                 description = "Trains ice particle GAN.")
parser.add_argument('name', metavar = 'name', type = str, nargs = 1)
parser.add_argument('device', metavar = 'device', type = str, nargs = 1)
parser.add_argument('nf_gen', metavar = 'nf_gen', type = int, nargs = 1)
parser.add_argument('nf_dis', metavar = 'nf_dis', type = int, nargs = 1)
parser.add_argument('n_cat_dim', metavar = 'n_cat_dim', type = int, nargs = 1)

args = parser.parse_args()
name = args.name[0]
device = args.device[0]
nf_gen = args.nf_gen[0]
nf_dis = args.nf_dis[0]
n_cat_dim = args.n_cat_dim[0]


#
# Training
#
data = IceShapes(os.path.join(path, "data", "shape_images.nc"), mode = "r")
dataloader = torch.utils.data.DataLoader(data, batch_size = 128,
                                         shuffle = True, num_workers = 1)

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.Resize((32, 32)),
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize([0.5], [0.5]) # normalize inputs
                                                          ])), 
                                           batch_size=64, 
                                           shuffle=True)

noise = 0.00
gan = InfoGan(50,
              n_filters_discriminator = nf_dis,
              n_filters_generator = nf_gen,
              n_cat_dim = n_cat_dim,
              device = device)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
gan.train(train_loader, noise = noise)
#gan.train(dataloader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
#gan.train(dataloader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
#gan.train(dataloader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
#gan.train(train_loader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
#gan.train(dataloader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
#gan.train(dataloader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
#gan.train(dataloader, lr_dis = 0.001, lr_gen = 0.001, noise = noise)
gan.save(os.path.join(path, "models", "gan_" + name + ".pt"))
