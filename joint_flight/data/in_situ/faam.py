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
psd_time    = cip_15_time[j_start : j_end]

cip_15["bins"] = faam_cip_15["BIN_EDGES"][:]
cip_15["x"]    = 0.5 * (cip_15["bins"][1:] + cip_15["bins"][:-1])
cip_15["n"]    = faam_cip_15["SPEC"][:][:, j_start : j_end] * 1e6
cip_15["dndd"] = cip_15["n"] / (np.diff(cip_15["bins"]).reshape(-1, 1))

cip_15_s = {}
cip_15_s["bins"] = faam_cip_15["BIN_EDGES"][::2]
cip_15_s["x"]    = 0.5 * (cip_15_s["bins"][1:] + cip_15_s["bins"][:-1])
cip_15_s["n"] = faam_cip_15["SPEC"][:][::2, j_start : j_end] * 1e6
cip_15_s["n"] += faam_cip_15["SPEC"][:][1::2, j_start : j_end] * 1e6
cip_15_s["dndd"] = cip_15_s["n"] / (np.diff(cip_15_s["bins"]).reshape(-1, 1))

#
# CIP100
#

faam_cip_100 = Dataset(glob.glob(os.path.join(data_path, "*cip100*"))[0], "r")
cip_100 = {}
cip_100["bins"] = faam_cip_100["BIN_EDGES"][:]
cip_100["x"]    = 0.5 * (cip_100["bins"][1:] + cip_100["bins"][:-1])
cip_100["n"]    = faam_cip_100["SPEC"][:][:, j_start : j_end] * 1e6
cip_100["dndd"] = cip_100["n"] / (np.diff(cip_100["bins"]).reshape(-1, 1))

cip_100_s = {}
cip_100_s["bins"] = faam_cip_100["BIN_EDGES"][::2]
cip_100_s["x"] = 0.5 * (cip_100_s["bins"][1:] + cip_100_s["bins"][:-1])
cip_100_s["n"] = faam_cip_100["SPEC"][:][::2, j_start : j_end] * 1e6
cip_100_s["n"] += faam_cip_100["SPEC"][:][1::2, j_start : j_end] * 1e6
cip_100_s["dndd"] = cip_100_s["n"] / (np.diff(cip_100_s["bins"]).reshape(-1, 1))

start_15 = 5
end_15   = np.where(cip_15["x"] > 500)[0][0]
start_100 = np.where(cip_100["x"] > 500)[0][0]
end_100   = -5
psd_x = np.concatenate([cip_15["x"][start_15 : end_15],
                        cip_100["x"][start_100 : end_100]])
psd_y  = np.concatenate([cip_15["dndd"][start_15 : end_15],
                         cip_100["dndd"][start_100 : end_100]])

start_15 = 3
end_15   = np.where(cip_15_s["x"] > 500)[0][0]
start_100 = np.where(cip_100_s["x"] > 500)[0][0]
end_100   = -3
psd_x_s = np.concatenate([cip_15_s["x"][start_15 : end_15],
                        cip_100_s["x"][start_100 : end_100]])
psd_y_s  = np.concatenate([cip_15_s["dndd"][start_15 : end_15],
                         cip_100_s["dndd"][start_100 : end_100]])




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

