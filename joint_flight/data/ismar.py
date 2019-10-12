""""
ISMAR joint flight data
-----------------------

This module provides the co-located ISMAR observations for the joint flight
observations. For the FAAM aircraft, the joined leg of the flight campaign
lasted from about 9:48 to 10:21 UTC.

From the ISMAR observation along the joint flight path only observations
with a sensor view angle within :math:`[-5^\circ, +5^\circ]` are used.

Attributes:

    angles(:code:`numpy.array`): 1D array containing the sensor viewing angles
        for all observations along the joint flight path. 

    d(:code:`numpy.array`): 1D array containing the distance along
        for all nadir-looking observations along the joint flight path.

    zsl(:code:`numpy.array`): 1D array containing height above surface of
        the ISMAR sensor for all nadir-looking observations along the flight
        path.

    lat(:code:`numpy.array`): 1D array containing the latitude coordinates
        of the ISMAR nadir-looking observations along the flight path.

    lon(:code:`numpy.array`): 1D array containing the longitude coordinates
        of the ISMAR nadir-looking observations along the flight path.

    bt(:code:`numpy.array`): 2D array containing the passive, nadir-looking
        ISMAR observations along the flight path.

    nedt(:code:`numpy.array`): 2D array containing the estimated random error
        for the nadir-looking observations along the flight path.

    channels(:code:`numpy.array`) 1D array containing the corresponding channel
        numbers for all observed bands in :code:`bt`.
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
faam_ismar  = Dataset(glob.glob(os.path.join(data_path, "metoffice-ismar_faam*.nc"))[0], "r")

ismar_time = faam_ismar.variables["time"][:]
i_start = np.where(ismar_time / 3600 > 9.81)[0][0]
i_end   = np.where(ismar_time / 3600 >  10.35)[0][0]

#
# Select NADIR views.
#

angles = faam_ismar.variables["sensor_view_angle"][i_start : i_end]
indices = np.logical_and(angles > -5, angles < 5)

channel_names = ["".join([np.bytes_(c).decode() for c in cs]) for cs in faam_ismar.variables["channel"]]
lat  = faam_ismar.variables["latitude"][i_start : i_end][indices]
lon  = faam_ismar.variables["longitude"][i_start : i_end][indices]
tbs  = faam_ismar.variables["brightness_temperature"][i_start : i_end, :][indices]
nedt = faam_ismar.variables["brightness_temperature_random_error"][i_start : i_end, :][indices]
zsl  = faam_ismar.variables["altitude"][i_start : i_end][indices]
altitude = zsl
channels = np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 6, 6, 8, 8])
time = ismar_time[i_start : i_end][indices]

d = np.zeros(lat.shape)
for i in range(d.size):
    d[i] = dist.vincenty((lat_r, lon_r), (lat[i], lon[i])).km

i_start = np.where(ismar_time / 3600 > 9.81)[0][0]
i_end   = np.where(ismar_time / 3600 > 11.183)[0][0]
angles = faam_ismar.variables["sensor_view_angle"][i_start : i_end]
indices = np.logical_and(angles > -5, angles < 5)
time_e = ismar_time[i_start : i_end][indices]
lat_e = faam_ismar["latitude"][i_start : i_end][indices]
lon_e = faam_ismar["longitude"][i_start : i_end][indices]
zsl_e = faam_ismar.variables["altitude"][i_start : i_end][indices]
d_e = np.zeros(lat_e.shape)
for i in range(d_e.size):
    d_e[i] = dist.vincenty((lat_r, lon_r), (lat_e[i], lon_e[i])).km
