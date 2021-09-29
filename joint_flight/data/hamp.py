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
from pathlib import Path

from netCDF4 import Dataset
from joint_flight import PATH
from datetime import datetime
from geopy import distance as dist
import numpy as np
import xarray as xr
import scipy as sp
from scipy.signal import convolve
import typhon
from typhon.geodesy import great_circle_distance

from joint_flight.utils import centers_to_edges

data_path = os.path.join(PATH, "data")
halo_mw = Dataset(glob.glob(os.path.join(data_path, "*nawd*mwr*.nc"))[0], "r")
halo_radar = Dataset(glob.glob(os.path.join(data_path, "*nawd*cr*.nc"))[0], "r")
halo_sonde = Dataset(glob.glob(os.path.join(data_path, "*nawd*sonde*.nc"))[0], "r")

#
# Determine time range of joint observations.
#

# Reference time used for HALO data.
t0 = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
# Start time of the joint flight.
t1 = datetime(year=2016, month=10, day=14, hour=9, minute=51, second=30)
# End time of the joint flight.
t2 = datetime(year=2016, month=10, day=14, hour=10, minute=15, second=30)

dt_start = (t1 - t0).total_seconds()
dt_end = (t2 - t0).total_seconds()

halo_times = halo_radar["time"][:]
i_start = np.where(halo_times >= dt_start)[0][0]
i_end = np.where(halo_times > dt_end)[0][0]

tr = datetime(year=2016, month=10, day=14, hour=0, minute=0, second=0)
dt = (tr - t0).total_seconds()
time = halo_times[i_start:i_end] - dt
#
# Data attributes.
#

dbz = halo_radar["dbz"][i_start:i_end]
lat = halo_radar["lat"][i_start:i_end]
lon = halo_radar["lon"][i_start:i_end]
zsl = halo_radar["zsl"][i_start:i_end]
z = halo_radar.variables["height"][:]
bt = halo_mw.variables["tb"][i_start:i_end, :]
channels = np.array(
    [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5]
)
i_start = np.where(halo_times >= dt_start - 1800)[0][0]
i_end = np.where(halo_times > dt_end + 1800)[0][0]
lat_e = halo_radar["lat"][i_start:i_end]
lon_e = halo_radar["lon"][i_start:i_end]
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

f = RegularGridInterpolator(
    (dem.lon_full, dem.lat_full[::-1]),
    dem.z_full.T[:, ::-1],
    bounds_error=False,
    fill_value=0.0,
)
zs = f((lon, lat))

# Surface mask
from scipy.signal import convolve

k = np.ones(5) / 5.0
land = zs > 0.0
land_mask = (convolve(land, k, "same") > 0.0).astype(np.float)


def load_radar_data(data_path=None):
    """
    Loads the hamp radar data into a xarray datasets.
    """
    if data_path is None:
        data_path = Path(PATH) / "data"
    else:
        data_path = Path(data_path)

    file = next(iter(data_path.glob("*nawd*cr*.nc")))
    radar_data = xr.open_dataset(file)

    times = radar_data["time"].data
    t1 = np.datetime64("2016-10-14T09:51:30", "ns")
    t2 = np.datetime64("2016-10-14T10:15:00", "ns")

    indices = (times >= t1) * (times < t2)

    radar_data = radar_data.loc[{"time": indices}]

    # Subsample radar data horizontally and vertically
    dbz = radar_data["dbz"]
    m = 3
    n = 7
    k = np.ones((m, n)) / (m * n)

    dbz = dbz.data
    dbz[np.isnan(dbz)] = -30
    dbz = np.maximum(dbz, -30)
    dbz = 10.0 * np.log(convolve(np.exp(dbz / 10.0), k, mode="valid"))[::m, ::n]
    # print(dbz)
    # dbz = dbz.data[slice(1, -1, 3), slice(3, -3, 7)]

    radar_data = radar_data[{"time": slice(1, -1, 3), "height": slice(3, -3, 7)}]
    radar_data["dbz"] = (("time", "height"), dbz)
    radar_data["dbz"][{"height": radar_data.height.data > 9.0e3}] = -30

    lats = radar_data["lat"].data
    lons = radar_data["lon"].data

    f = RegularGridInterpolator(
        (dem.lon_full, dem.lat_full[::-1]),
        dem.z_full.T[:, ::-1],
        bounds_error=False,
        fill_value=0.0,
    )
    zs = f((lons, lats))

    for i in range(dbz.shape[0]):
        ind = np.where(radar_data.height > zs[i])[0][0] + 3
        dbz[i, :ind] = dbz[i, ind]

    dx = great_circle_distance(
        lats[:-1], lons[:-1], lats[1:], lons[1:], r=typhon.constants.earth_radius
    )
    d = np.pad(np.cumsum(dx), (1, 0), "constant", constant_values=0)

    radar_data["d"] = ("time", d)
    radar_data["surface_height"] = ("time", zs)
    radar_data["range_bins"] = (
        ("height_1",),
        centers_to_edges(radar_data.height.data, axis=0),
    )
    radar_data = radar_data.rename(
        {
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    x = np.broadcast_to(d.reshape(-1, 1), (d.size, radar_data.height.size + 1))
    x = centers_to_edges(x, axis=0)

    y = np.broadcast_to(
        radar_data.height.data.reshape(1, -1), (x.shape[0], radar_data.height.size)
    )
    y = centers_to_edges(y, axis=1)

    radar_data["x"] = (("time_1", "height_1"), x)
    radar_data["y"] = (("time_1", "height_1"), y)

    return radar_data
