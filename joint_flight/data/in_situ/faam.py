""""
FAAM in situ data
=================

This module provides an interface to the in-situ measurement collected
by the FAAM aircraft during the joint flight campaign.

Attributes:

    d(:code:`numpy.ndarray`): 1D array containing the position of the FAAM
        aircraft along the joint flight path during its descent into the
        clouds.

    dbz(:code:`numpy.ndarray`): 2D array containing profiles of measured
        RADAR reflectivities along the flight path.

    z(:code:`numpy.ndarray`): 1D array containing the altitudes corresponding
       to the RADAR profiles in `dbz`.

    zsl(:code:`numpy.ndarray`): 1D array containing height of the HALO aircraft
        along the flight path.

    zs(:code:`numpy.ndarray`): 1D array containing surface elevation
        along the flight path.

    lat(:code:`numpy.ndarray`): 1D array containing the latitude coordinates
        of the HAMP observations along the flight path.

    lon(:code:`numpy.ndarray`): 1D array containing the longitude coordinates
        of the HAMP observations along the flight path.

    bt(:code:`numpy.ndarray`): 2D array containing the passive HAMP observation
        along the flight path.

    channels(:code:`numpy.ndarray`) 1D array containing the corresponding channel
        numbers for all observed bands in :code:`bt`.
"""
import glob
import os
from netCDF4      import Dataset
from datetime     import datetime
from geopy        import distance as dist
from scipy.optimize import least_squares
import numpy as np

from joint_flight import path
from joint_flight.data.hamp import lat_r, lon_r

data_path = os.path.join(path, "data")
faam_core   = Dataset(glob.glob(os.path.join(data_path, "core_faam_*_v004_r1*"))[0], "r")
faam_nev    = Dataset(glob.glob(os.path.join(data_path, "*nevzorov*"))[0], "r")

faam_time = faam_core["Time"][:]
i_start = np.where(faam_time / 3600 > 10.64)[0][0]
i_end   = np.where(faam_time / 3600 >  11.183)[0][0]
faam_z   = faam_core["ALT_GIN"][i_start : i_end]
faam_lat = faam_core["LAT_GIN"][i_start : i_end]
faam_lon = faam_core["LON_GIN"][i_start : i_end]
faam_time = faam_time[i_start : i_end]


d = np.zeros(faam_lat.shape)
for i in range(d.size):
    d[i] = dist.vincenty((lat_r, lon_r), (faam_lat[i], faam_lon[i])).km

################################################################################
# Nevzorov data
################################################################################

nev_time   = faam_nev["TIME"][:]
j_start    = np.where(nev_time / 3600 > 10.6)[0][0]
j_end      = np.where(nev_time / 3600 >  11.183)[0][0]
twc_ice    = faam_nev["TWC_Q_ice"][j_start : j_end]
lwc_liquid = faam_nev["LWC_Q_liq"][j_start : j_end]
nev_z      = np.interp(nev_time[j_start : j_end], faam_time, faam_z)
nev_d      = np.interp(nev_time[j_start : j_end], faam_time, d)
nev_time   = faam_nev["TIME"][j_start : j_end]

################################################################################
# CIP data
################################################################################

faam_cip_15  = Dataset(glob.glob(os.path.join(data_path, "*cip15*"))[0], "r")

cip_15_bins   = faam_cip_15["BIN_EDGES"][:]
cip_15_time   = faam_cip_15["TIME"][:]
j_start       = np.where(cip_15_time / 3600 > 10.6)[0][0]
j_end         = np.where(cip_15_time / 3600 >  11.183)[0][0]
cip_15_counts = faam_cip_15["COUNTS"][:][:, j_start : j_end]
d_cip_15      = 0.5 * (cip_15_bins[1:] + cip_15_bins[:-1])
dndd_cip_15   = cip_15_counts / np.diff(cip_15_bins).reshape(-1, 1)
iwc_cip_15       = np.interp(cip_15_time[j_start : j_end], nev_time, twc_ice)
cip_15_d      = np.interp(cip_15_time[j_start : j_end], faam_time, d)
cip_15_z      = np.interp(cip_15_time[j_start : j_end], faam_time, faam_z)

faam_cip_100 = Dataset(glob.glob(os.path.join(data_path, "*cip100*"))[0], "r")

cip_100_bins   = faam_cip_100["BIN_EDGES"][:]
cip_100_time   = faam_cip_100["TIME"][:]
j_start       = np.where(cip_100_time / 3600 > 10.6)[0][0]
j_end         = np.where(cip_100_time / 3600 >  11.183)[0][0]
cip_100_counts = faam_cip_100["COUNTS"][:][:, j_start : j_end]
d_cip_100      = 0.5 * (cip_100_bins[1:] + cip_100_bins[:-1])
dndd_cip_100   = cip_100_counts / np.diff(cip_100_bins).reshape(-1, 1)

dndd  = dndd_cip_15
d_max = d_cip_15
z_cip = np.interp(cip_15_time[j_start : j_end], faam_time, faam_z)

#
# Mass-size relation
#

#def iwc_fun(x, i_start, i_end):
#    alpha, beta = x
#    print(alpha, beta)
#    s = d_cip_15.reshape(-1, 1) * 1e-3
#    print(dndd_cip_15[:, i_start : i_end].shape)
#    iwc = np.trapz(alpha * s ** beta * dndd_cip_15[:, i_start : i_end],
#                   x = s,
#                   axis = 0)
#    iwc_r = np.maximum(iwc_cip_15[i_start : i_end], 1e-6) * 1e-9
#    print(iwc_r)
#    print(iwc)
#    #return iwc - iwc_r
#    return iwc - iwc_r
#
#def iwc_fun_jac(x, i_start, i_end):
#    alpha, beta = x
#    s = d_cip_15.reshape(-1, 1)
#    diwc_dalpha = np.trapz(s ** beta * dndd_cip_15[:, i_start : i_end],
#                           x = s,
#                           axis = 0)
#    diwc_dbeta = np.trapz(alpha * np.log(s) * s ** beta * dndd_cip_15[:, i_start : i_end],
#                          x = s,
#                          axis = 0)
#    dfdx = np.stack([diwc_dalpha, diwc_dbeta], axis = 1)
#    #dfdx = dfdx / iwc_fun((alpha, beta), i_start, i_end).reshape(-1, 1)
#    return dfdx
#
#def numjac(x, i_start, i_end, dx = 0.001):
#    alpha, beta = x
#
#    x_2 = (alpha + dx * alpha, beta)
#    x_1 = (alpha - dx * alpha, beta)
#
#    iwc_2 = iwc_fun(x_2, i_start, i_end)
#    iwc_1 = iwc_fun(x_1, i_start, i_end)
#    d_iwc_1 = (iwc_2 - iwc_1) / (2 * dx * alpha)
#
#    x_2 = (alpha, beta + dx * beta)
#    x_1 = (alpha, beta - dx * beta)
#
#    iwc_2 = iwc_fun(x_2, i_start, i_end)
#    iwc_1 = iwc_fun(x_1, i_start, i_end)
#    d_iwc_2 = (iwc_2 - iwc_1) / (2 * dx * beta)
#
#    return np.stack([d_iwc_1, d_iwc_2], axis = 1)
#
#step = 20
#n = iwc_cip_15.size // step
#alpha = np.zeros(n)
#beta  = np.zeros(n)
#ms_z  = np.zeros(n)
#ms_d  = np.zeros(n)
#
#for i in range(iwc_cip_15.size // step):
#    i_start = i * step
#    i_end   = (i + 1) * step
#
#    res = least_squares(fun = iwc_fun,
#                        x0 = np.array([1e3, 2.5]),
#                        jac = iwc_fun_jac,
#                        bounds = [np.array([0.0, 2.0]), np.array([1e9, 3.0])],
#                        #method = "lm",
#                        args = (i_start, i_end))
#    a, b = res["x"]
#    alpha[i] = a
#    beta[i] = b
#    ms_z = np.mean(cip_15_z[i_start : i_end])
#    ms_d = np.mean(cip_15_d[i_start : i_end])


################################################################################
# Dropsondes
################################################################################

files = glob.glob(os.path.join(data_path, "*dropsonde*"))
files.sort()
ds_rh  = []
ds_lat = []
ds_lon = []
ds_z   = []
ds_d   = []
ds_t   = []
for f in files:
    ds = Dataset(f)
    ds_rh += [ds["rh"][:]]
    ds_lat += [ds["lat"][:]]
    ds_lon += [ds["lon"][:]]
    ds_z  += [ds["alt"][:]]
    ds_t  += [ds["tdry"][:] - 273.15]

    lat  = ds["lat"][:]
    lon  = ds["lon"][:]
    mask = np.logical_not(np.logical_or(lat.mask, lon.mask))
    ds_d  += [dist.vincenty((lat_r, lon_r), (lat[mask][0], lon[mask][0])).km]
    ds.close()

