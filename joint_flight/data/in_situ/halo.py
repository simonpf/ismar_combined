""""
HALO in situ data
=================

This module provides an interface to the dropsondes dropped by
the HALO aircraft during the joint flight.

Attributes:
"""
import glob
import os
from netCDF4      import Dataset
from datetime     import datetime
from geopy        import distance as dist
import numpy as np

from joint_flight import path
from joint_flight.data.hamp import lat_r, lon_r

data_path = os.path.join(path, "data")
ds = Dataset(glob.glob(os.path.join(data_path, "halo_nawd_sonde*"))[0], "r")

lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
t   = ds.variables["ta"]
rh  = ds.variables["hur"]
z   = ds.variables["height"][:]

d = []
for i in range(lat.shape[0]):
    mask = np.logical_not(np.logical_or(lat.mask[i, :], lon.mask[i, :]))
    d  += [dist.vincenty((lat_r, lon_r), (lat[i, mask][0], lon[i, mask][0])).km]

