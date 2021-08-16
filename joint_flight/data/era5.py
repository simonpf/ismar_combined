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
from typhon.physics.atmosphere import relative_humidity2vmr, vmr2relative_humidity

from netCDF4 import Dataset

import joint_flight.data.hamp as hamp
from joint_flight import path
data_path = os.path.join(path, "data")

def p_eq(t):
    """
    WV saturation pressure as used in ARTS.

    Arguments:
        t: Array containing atmospheric temperatures for which to compute
            the water vapor saturation pressure.

    Return:
        Array containing the WV saturation pressure.
    """
    p = np.zeros(t.shape)

    inds = t >= 273.15
    tt = np.exp(54.842763 - 6763.22 / t - 4.21 * np.log(t) + 0.000367 * t + \
                np.tanh(0.0415 * (t - 218.8))
                * ( 53.878 - 1331.22/t - 9.44523 * np.log(t) + 0.014025 * t))
    p[inds] = tt[inds]
    inds = t < 273.15
    tt = np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t) - 0.00728332 * t)
    p[inds] = tt[inds]
    return p


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

# Convert to WV saturation pressure assumptions
era_5_h2o = relative_humidity2vmr(era_5_rh, era_5_p.reshape(1, -1), era_5_t)
era_5_rh = vmr2relative_humidity(era_5_h2o, era_5_p.reshape(1, -1), era_5_t, e_eq = p_eq)

f = RegularGridInterpolator((era_5_lon, era_5_lat[::-1]),
                            era_5_phy_full[0, ::-1, ::-1, :].T)
era_5_z = f((hamp.lon + 360.0, hamp.lat)) / 9.80665

z = np.linspace(0, 12e3, 61)
p = np.zeros((hamp.lon.size, z.size))
t = np.zeros((hamp.lon.size, z.size))
rh = np.zeros((hamp.lon.size, z.size))
h2o = np.zeros((hamp.lon.size, z.size))

for i in range(hamp.lon.size):
    f = interp1d(era_5_z[i, :], era_5_p[::-1], fill_value = "extrapolate")
    p[i, :] = f(z) * 100.0
    f = interp1d(era_5_z[i, :], era_5_t[i, :], fill_value = "extrapolate")
    t[i, :] = f(z)
    f = interp1d(era_5_z[i, :], era_5_rh[i, :], fill_value = "extrapolate")
    rh[i, :] = f(z)
    f = interp1d(era_5_z[i, :], era_5_h2o[i, :], fill_value = "extrapolate")
    h2o[i, :] = f(z)

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
