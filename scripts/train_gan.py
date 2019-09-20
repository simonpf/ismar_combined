import argparse

from joint_flight import path
from joint_flight.gan import IceShapes, create_mosaic, Gan
import os
import torch

data = IceShapes("/home/simonpf/src/joint_flight/data/shape_images_15")
dataloader = torch.utils.data.DataLoader(data, batch_size = 128,
                                         shuffle = True, num_workers = 1)

#
# Command line arguments
#

parser = argparse.ArgumentParser(prog = "gan",
                                 description = "Trains ice particle GAN.")
parser.add_argument('name', metavar = 'name', type = str, nargs = 1)
parser.add_argument('device', metavar = 'device', type = str, nargs = 1)
parser.add_argument('optimizer', metavar = 'optimizer', type = str, nargs = 1)
parser.add_argument('nf_gen', metavar = 'nf_gen', type = int, nargs = 1)
parser.add_argument('nf_dis', metavar = 'nf_dis', type = int, nargs = 1)
parser.add_argument('features', metavar = 'features', type = int, nargs = 1)

args = parser.parse_args()
name = args.name[0]
device = args.device[0]
opt = args.optimizer[0]
nf_gen = args.nf_gen[0]
nf_dis = args.nf_dis[0]
features = args.features[0]


#
# Training
#
data = IceShapes(os.path.join(path, "data", "shape_images_15"))
dataloader = torch.utils.data.DataLoader(data, batch_size = 64,
                                         shuffle = False, num_workers = 1)
gan = Gan(n_filters_discriminator = nf_dis,
          n_filters_generator = nf_gen,
          features = features,
          device = device)
gan.train(dataloader, lr_dis = 0.01, lr_gen = 0.01, noise = 0.1)
#gan.train(dataloader, lr_dis = 0.01, lr_gen = 0.01, noise = 0.1)
gan.save(os.path.join(path, "models", "gan_" + name + ".pt"))
