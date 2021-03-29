""""
HAMP joint flight data
======================

This module provides the co-located HAMP data for the joint flight
observations. For the HALO aircraft, the joined leg of the flight campaign
lasted from 9:51 to 10:16 UTC.

The locations of the HAMP observations along the flight track are used as
reference coordinates for the combined retrieval. All other observations are
interpolated to match the locations of the HAMP observations.

Attributes:

    d(:code:`numpy.ndarray`): 1D array containing the distance along
        the joint flight path.

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
from joint_flight import path
from datetime     import datetime
from geopy        import distance as dist
import numpy as np

data_path = os.path.join(path, "data")
halo_mw     = Dataset(glob.glob(os.path.join(data_path, "*nawd*mwr*.nc"))[0],    "r")
halo_radar  = Dataset(glob.glob(os.path.join(data_path, "*nawd*cr*.nc"))[0],     "r")
halo_sonde  = Dataset(glob.glob(os.path.join(data_path, "*nawd*sonde*.nc"))[0],  "r")

#
# Determine time range of joint observations.
#

# Reference time used for HALO data.
t0 = datetime(year = 1970, month = 1, day = 1, hour = 0, minute = 0, second = 0)
# Start time of the joint flight.
t1 = datetime(year = 2016, month = 10, day = 14, hour = 9, minute = 51, second = 30)
# End time of the joint flight.
t2 = datetime(year = 2016, month = 10, day = 14, hour = 10, minute = 15, second = 30)

dt_start = (t1 - t0).total_seconds()
dt_end   = (t2 - t0).total_seconds()

halo_times = halo_radar["time"][:]
i_start = np.where(halo_times >= dt_start)[0][0]
i_end   = np.where(halo_times >  dt_end)[0][0]

tr = datetime(year = 2016, month = 10, day = 14, hour = 0, minute = 0, second = 0)
dt = (tr - t0).total_seconds()
time = halo_times[i_start : i_end] - dt
#
# Data attributes.
#

dbz = halo_radar["dbz"][i_start : i_end]
lat = halo_radar["lat"][i_start : i_end]
lon = halo_radar["lon"][i_start : i_end]
zsl = halo_radar["zsl"][i_start : i_end]
z   = halo_radar.variables["height"][:]
bt  = halo_mw.variables["tb"][i_start : i_end, :]
channels = np.array([1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2,
                     3,
                     4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5, 5])
i_start = np.where(halo_times >= dt_start - 1800)[0][0]
i_end   = np.where(halo_times >  dt_end + 1800)[0][0]
lat_e = halo_radar["lat"][i_start : i_end]
lon_e = halo_radar["lon"][i_start : i_end]
"""
Reference coordinates
---------------------

The latitude and longitude coordinates from the first HAMP observation from
the joint flight are used as reference coordinates for the co-location of
the remaining observations. They are available as the :code:`lat_r` and
`lon_r` this module.

"""

lat_r = lat[0]
lon_r = lon[0]

d = np.zeros(lat.shape)
for i in range(d.size):
    d[i] = dist.distance((lat_r, lon_r), (lat[i], lon[i])).km

"""
Surface altitude
----------------

To determine the surface altitude below the RADAR observations, the DEM data
is interpolated to the HAMP flight path.
"""
import scipy
from scipy.interpolate import RegularGridInterpolator
from joint_flight.data import dem

f  = RegularGridInterpolator((dem.lon_full, dem.lat_full[::-1]), dem.z_full.T[:, ::-1],
                             bounds_error = False, fill_value = 0.0)
zs = f((lon, lat))

# Surface mask
from scipy.signal import convolve
k = np.ones(5) / 5.0
land = zs > 0.0
land_mask = (convolve(land, k, "same") > 0.0).astype(np.float)

