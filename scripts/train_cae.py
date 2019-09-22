import argparse

from joint_flight import path
from joint_flight.gan import IceShapes
from joint_flight.cae import Cae
import os
import torch

#
# Command line arguments
#

parser = argparse.ArgumentParser(prog = "cae",
                                 description = "Trains ice particle AE.")
parser.add_argument('name', metavar = 'name', type = str, nargs = 1)
parser.add_argument('device', metavar = 'device', type = str, nargs = 1)
parser.add_argument('optimizer', metavar = 'optimizer', type = str, nargs = 1)
parser.add_argument('nf', metavar = 'nf', type = int, nargs = 1)
parser.add_argument('latent_dim', metavar = 'latent_dim', type = int, nargs = 1)
parser.add_argument('loss', metavar = 'loss', type = str, nargs = 1)

args = parser.parse_args()
name = args.name[0]
device = args.device[0]
opt = args.optimizer[0]
nf = args.nf[0]
latent_dim = args.latent_dim[0]
loss = args.loss[0]


#
# Training
#
data = IceShapes(os.path.join(path, "data", "shape_images_15"))
dataloader = torch.utils.data.DataLoader(data, batch_size = 256,
                                         shuffle = False, num_workers = 4)
cae = Cae(n_filters = nf,
          latent_dim = latent_dim,
          loss = loss,
          device = device)
cae.train(dataloader, lr = 0.01)
#cae.train(dataloader, lr = 0.01)
#cae.train(dataloader, lr = 0.005)
#cae.train(dataloader, lr = 0.005)
#cae.train(dataloader, lr = 0.001)
#cae.train(dataloader, lr = 0.001)
cae.save(os.path.join(path, "models", "cae_" + name + ".pt"))
