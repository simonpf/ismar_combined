""""
ERA5 Data
=========

This module provides ERA5 surface data that is required to run
the joint flight retrieval.

The gridded ERA5 data is interpolated to the HAMP flight path.

Attributes:

    z(:code:`numpy.ndarray`): The altitude grid that is used for the
       retrieval.

    p(:code:`numpy.ndarray`): 2D array containing the pressure field
       along the flight path.

    t(:code:`numpy.ndarray`): 2D array containing the temperature field
       along the flight path.

    wind_speed(:code:`numpy.ndarray`): 1D array containing the wind speed
        in :math:`m/s` along the HAMP flight path

    sst(:code:`numpy.ndarray`): 1D array containing the sea surface temperature
        in :math:`K` along the HAMP flight path.

    skt(:code:`numpy.ndarray`): 1D array containing the surface skin temperature
        along the HAMP flight path.
"""
import os
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator, interp1d

from netCDF4 import Dataset

import joint_flight.data.hamp as hamp
from joint_flight import path
data_path = os.path.join(path, "data")

#
# Atmospheric variables
#

era_5      = Dataset(os.path.join(data_path, "era_5_atmosphere.nc"), "r")
era_5_lon  = era_5.variables["longitude"][:]
era_5_lat  = era_5.variables["latitude"][:]
era_5_p    = era_5.variables["level"][:]
era_5_phy_full  = era_5.variables["z"][:]
era_5_t_full    = era_5.variables["t"][:]
era_5_rh_full   = era_5.variables["r"][:]

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]),
                            era_5_t_full[0, ::-1, ::-1, :].T)
era_5_t = f((hamp.lon + 360.0, hamp.lat))

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]),
                            era_5_rh_full[0, ::-1, ::-1, :].T)
era_5_rh = f((hamp.lon + 360.0, hamp.lat))

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]),
                            era_5_phy_full[0, ::-1, ::-1, :].T)
era_5_z = f((hamp.lon + 360.0, hamp.lat)) / 9.80665

z = np.linspace(0, 13e3, 66)
p = np.zeros((hamp.lon.size, z.size))
t = np.zeros((hamp.lon.size, z.size))
rh = np.zeros((hamp.lon.size, z.size))

for i in range(hamp.lon.size):
    f = interp1d(era_5_z[i, :], era_5_p[::-1], fill_value = "extrapolate")
    p[i, :] = f(z) * 100.0
    f = interp1d(era_5_z[i, :], era_5_t[i, :], fill_value = "extrapolate")
    t[i, :] = f(z)
    f = interp1d(era_5_z[i, :], era_5_rh[i, :], fill_value = "extrapolate")
    rh[i, :] = f(z)
#
# Surface variables
#

era_5      = Dataset(os.path.join(data_path, "era_5_surface.nc"), "r")
era_5_lon  = era_5.variables["longitude"][:]
era_5_lat  = era_5.variables["latitude"][:]
era_5_u_10 = era_5.variables["u10"][:]
era_5_v_10 = era_5.variables["v10"][:]
era_5_sst  = era_5.variables["sst"][:]
era_5_skt  = era_5.variables["skt"][:]
era_5.close()

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]), era_5_u_10[0, ::-1, :].T)
u_10 = f((hamp.lon + 360.0, hamp.lat))

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]), era_5_v_10[0, ::-1, :].T)
v_10 = f((hamp.lon + 360.0, hamp.lat))

wind_speed = np.sqrt(u_10 ** 2 + v_10 ** 2)

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]), era_5_sst[0, ::-1, :].T)
sst = f((hamp.lon + 360.0, hamp.lat))

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]), era_5_skt[0, ::-1, :].T)
skt = f((hamp.lon + 360.0, hamp.lat))
