""""
FAAM CIP data
=============

This module provides a convenience interface to the PSD data collected
by the FAAM aircraft.
"""
import glob
import os
from netCDF4 import Dataset
from datetime import datetime
from geopy import distance as dist
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt

from joint_flight import path
from joint_flight.data.hamp import lat_r, lon_r

data_path = os.path.join(path, "data")

faam_core = Dataset(glob.glob(os.path.join(data_path, "core_faam_*_v004_r1*"))[0], "r")
faam_cip_15 = Dataset(glob.glob(os.path.join(data_path, "*cip15*"))[0], "r")
faam_cip_100 = Dataset(glob.glob(os.path.join(data_path, "*cip100*"))[0], "r")

faam_time = faam_core["Time"][:]
i_start = np.where(faam_time / 3600 > 10.64)[0][0]
i_end = np.where(faam_time / 3600 > 11.183)[0][0]
faam_z = faam_core["ALT_GIN"][i_start:i_end]
faam_lat = faam_core["LAT_GIN"][i_start:i_end]
faam_lon = faam_core["LON_GIN"][i_start:i_end]
faam_time = faam_time[i_start:i_end]
d = np.zeros(faam_lat.shape)
for i in range(d.size):
    d[i] = dist.vincenty((lat_r, lon_r), (faam_lat[i], faam_lon[i])).km

################################################################################
# CIP 15
################################################################################

cip_15 = {}

cip_15_time = faam_cip_15["TIME"][:]

j_start = np.where(cip_15_time / 3600 > 10.6)[0][0]
j_end = np.where(cip_15_time / 3600 > 11.183)[0][0]

# psd_iwc_nev = np.interp(cip_15_time[j_start : j_end], nev_time, twc_ice)
psd_d = np.interp(cip_15_time[j_start:j_end], faam_time, d)
psd_z = np.interp(cip_15_time[j_start:j_end], faam_time, faam_z)

cip_15["bins"] = faam_cip_15["BIN_EDGES"][:] / 1e4
cip_15["x"] = 0.5 * (cip_15["bins"][1:] + cip_15["bins"][:-1])
cip_15["n"] = faam_cip_15["SPEC"][:][:, j_start:j_end]
cip_15["dndd"] = cip_15["n"] / (np.diff(cip_15["bins"]).reshape(-1, 1))

#
# CIP100
#

cip_100 = {}
cip_100["bins"] = faam_cip_100["BIN_EDGES"][:] / 1e4
cip_100["x"] = 0.5 * (cip_100["bins"][1:] + cip_100["bins"][:-1])
cip_100["n"] = faam_cip_100["SPEC"][:][:, j_start:j_end]
cip_100["dndd"] = cip_100["n"] / (np.diff(cip_100["bins"]).reshape(-1, 1))


start_15, end_15 = 0, 64
start_100, end_100 = 9, 64
x = np.concatenate([cip_15["x"][start_15:end_15], cip_100["x"][start_100:end_100]])
y = np.concatenate(
    [cip_15["dndd"][start_15:end_15], cip_100["dndd"][start_100:end_100]]
)

n_avg = 50
y_avg = np.zeros((y.shape[0], y.shape[1] // n_avg))
z = np.zeros(y.shape[1] // n_avg)
d = np.zeros(y.shape[1] // n_avg)
for i in range(y.shape[1] // n_avg):
    start = i * n_avg
    end = (i + 1) * n_avg
    y_avg[:, i] = np.mean(y[:, i * n_avg : (i + 1) * n_avg], axis=1)
    z[i] = np.mean(psd_z[start:end])
    d[i] = np.mean(psd_d[start:end])

bins = np.concatenate(
    [
        np.linspace(0, 1e-2, 6),
        np.linspace(0, 1e-1, 11)[2:],
        np.linspace(0, 1e-0, 11)[2:],
    ]
)
x_r = 0.5 * (bins[1:] + bins[:-1])
xx = np.broadcast_to(x.reshape(-1, 1), y.shape)
y_r = np.zeros((x_r.size, y.shape[1] // n_avg))
for i in range(y.shape[1] // n_avg):
    start = i * n_avg
    end = (i + 1) * n_avg
    y_r[:, i], _ = np.histogram(
        xx[:, start:end].ravel(), weights=y[:, start:end].ravel(), bins=bins
    )
