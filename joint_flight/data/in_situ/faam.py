""""
FAAM in situ data
=================

This module provides an interface to the in-situ measurements collected
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
faam_t = faam_core["TAT_DI_R"][i_start : i_end]


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

#
# CIP15
#
# PSD data with 15um resolution
#

cip_15 = {}

cip_15_time   = faam_cip_15["TIME"][:]

j_start      = np.where(cip_15_time / 3600 > 10.6)[0][0]
j_end       = np.where(cip_15_time / 3600 >  11.183)[0][0]

psd_iwc_nev = np.interp(cip_15_time[j_start : j_end], nev_time, twc_ice)
psd_d       = np.interp(cip_15_time[j_start : j_end], faam_time, d)
psd_z       = np.interp(cip_15_time[j_start : j_end], faam_time, faam_z)
psd_t       = np.interp(cip_15_time[j_start : j_end], faam_time, faam_t)

cip_15["bins"] = faam_cip_15["BIN_EDGES"][:]
cip_15["x"]    = 0.5 * (cip_15["bins"][1:] + cip_15["bins"][:-1])
cip_15["n"]    = faam_cip_15["SPEC"][:][:, j_start : j_end] * 1e6
cip_15["dndd"] = cip_15["n"] / (np.diff(cip_15["bins"]).reshape(-1, 1))

#
# CIP100
#

faam_cip_100 = Dataset(glob.glob(os.path.join(data_path, "*cip100*"))[0], "r")
cip_100 = {}
cip_100["bins"] = faam_cip_100["BIN_EDGES"][:]
cip_100["x"]    = 0.5 * (cip_100["bins"][1:] + cip_100["bins"][:-1])
cip_100["n"]    = faam_cip_100["SPEC"][:][:, j_start : j_end] * 1e6
cip_100["dndd"] = cip_100["n"] / (np.diff(cip_100["bins"]).reshape(-1, 1))


start_15, end_15   = 5,54
start_100, end_100 = 9, 54
psd_x = np.concatenate([cip_15["x"][start_15 : end_15],
                        cip_100["x"][start_100 : end_100]])
psd_y  = np.concatenate([cip_15["dndd"][start_15 : end_15],
                         cip_100["dndd"][start_100 : end_100]])


def psd_to_iwc(counts, a, b):
    x = psd_x.reshape(-1, 1)
    return np.trapz(counts * a * x ** b, x = x, axis = 0)
#
# Mass-size relation
#

def iwc_fun(x, i_start, i_end):
   alpha, beta = x
   s = psd_x.reshape(-1, 1)
   y = psd_y[:, i_start : i_end]
   iwc = np.trapz(alpha * s ** beta * y, x = s, axis = 0) * 1e6
   iwc_r = psd_iwc_nev[i_start : i_end]
   return iwc - iwc_r

def iwc_fun_jac(x, i_start, i_end):
   alpha, beta = x
   s = psd_x.reshape(-1, 1)
   y = psd_y[:, i_start : i_end]
   diwc_dalpha = np.trapz(s ** beta * y, x = s, axis = 0)
   diwc_dbeta = np.trapz(alpha * np.log(s) * s ** beta * y, x = s,
                         axis = 0)
   dfdx = np.stack([diwc_dalpha, diwc_dbeta], axis = 1) * 1e6
   #dfdx = dfdx / iwc_fun((alpha, beta), i_start, i_end).reshape(-1, 1)
   return dfdx

def numjac(x, i_start, i_end, dx = 0.001):
   alpha, beta = x

   x_2 = (alpha + dx * alpha, beta)
   x_1 = (alpha - dx * alpha, beta)

   iwc_2 = iwc_fun(x_2, i_start, i_end)
   iwc_1 = iwc_fun(x_1, i_start, i_end)
   d_iwc_1 = (iwc_2 - iwc_1) / (2 * dx * alpha)

   x_2 = (alpha, beta + dx * beta)
   x_1 = (alpha, beta - dx * beta)

   iwc_2 = iwc_fun(x_2, i_start, i_end)
   iwc_1 = iwc_fun(x_1, i_start, i_end)
   d_iwc_2 = (iwc_2 - iwc_1) / (2 * dx * beta)

   return np.stack([d_iwc_1, d_iwc_2], axis = 1)

step = 200
n = psd_iwc_nev.size // step
alpha = np.zeros(n)
beta  = np.zeros(n)
ms_z  = np.zeros(n)
ms_d  = np.zeros(n)
ms_iwc    = np.zeros(n)
ms_iwc_f  = np.zeros(n)


for i in range(psd_iwc_nev.size // step):
    i_start = i * step
    i_end   = (i + 1) * step
    res = least_squares(fun = iwc_fun,
                        x0 = np.array([0.005, 2.4]),
                        jac = iwc_fun_jac,
                        #bounds = [np.array([0.0, 1.0]), np.array([1e9, 3.0])],
                        method = "lm",
                        args = (i_start, i_end))
    print(res["success"])
    a, b = res["x"]
    alpha[i] = a
    beta[i] = b

iwc_psd = psd_to_iwc(psd_y, 0.0013, 1.5) * 1e6

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

