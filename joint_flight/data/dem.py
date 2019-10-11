""""
Digital elevation model data
----------------------------

This module provides surface elevation data for the surface beneath
the joint flight path.

Attributes:

    z(:code:`numpy.array`): 2D array containing the surface altitude
        w.r.t. the reference ellipsoid.

    lat(:code:`numpy.array`): 1D array containing the latitude values
        corresponding to first dimension of the altitude grid.

    lat(:code:`numpy.array`): 1D array containing the latitude values
        corresponding to second dimension of the altitude grid.
"""
import rasterio
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from joint_flight import path
dem_data = rasterio.open(os.path.join(path, "data", "dem.tif"))

west  = -9
south = 55.5
east  = -4
north = 60.5

z_full = dem_data.read()[0, :, :]
height, width = z_full.shape
lon_full = np.linspace(west, east, width)
lat_full = np.linspace(north, south, height)

z = dem_data.read()[0, ::2, ::2]
height, width = z.shape
lon = np.linspace(west, east, width)
lat = np.linspace(north, south, height)
