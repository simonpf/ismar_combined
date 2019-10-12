""""
RASTA joint flight data
-----------------------

This module provides the co-located RASTA data for the joint flight
observations. For the Falcon aircraft, the joined leg of the flight campaign
lasted from 9:50 to 10:21 UTC.

Attributes:

    d(:code:`numpy.array`): 1D array containing the distance along
        the joint flight path.

    dbz(:code:`numpy.array`): 2D array containing profiles of measured
        RADAR reflectivities along the flight path.

    z(:code:`numpy.array`): 1D array containing the altitudes corresponding
       to the RADAR profiles in `dbz`.
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
rasta_radar = Dataset(glob.glob(os.path.join(data_path, "NAWDEX*RASTA*.nc"))[0], "r")

#
# Determine time range of joint observations.
#

t_start = 9  + 50 / 60
t_end   = 10 + 21 / 60
rasta_times = rasta_radar["time"][:]
i_start = np.where(rasta_times >= t_start)[0][0]
i_end   = np.where(rasta_times >  t_end)[0][0]


time = rasta_times[i_start : i_end] * 3600

dbz = rasta_radar["Z"][i_start : i_end, :]
lat = rasta_radar["latitude"][i_start : i_end]
lon = rasta_radar["longitude"][i_start : i_end]
altitude  = rasta_radar.variables["altitude"][i_start : i_end] * 1e3
z   = rasta_radar.variables["height_2D"][i_start : i_end, :] * 1e3
d = np.zeros(lat.shape)
for i in range(d.size):
    d[i] = dist.vincenty((lat_r, lon_r), (lat[i], lon[i])).km

i_start = np.where(rasta_times >= t_start - 0.5)[0][0]
i_end   = np.where(rasta_times >  t_end + 0.5)[0][0]
lat_e = rasta_radar["latitude"][i_start : i_end]
lon_e = rasta_radar["longitude"][i_start : i_end]

"""
Surface altitude
----------------

To determine the surface altitude below the RADAR observations, the DEM data
is interpolated to the Falcon flight path.
"""
import scipy
from scipy.interpolate import RegularGridInterpolator
from joint_flight.data import dem

f  = RegularGridInterpolator((dem.lon_full, dem.lat_full[::-1]), dem.z_full.T,
                             bounds_error = False, fill_value = 0.0)
zs = np.maximum(f((lon, lat)), 0.0)

