"""
======================
joint_flight.data.c161
======================

This module provides access to the preprocessed data from flight c159 of the
FAAM aircraft.
"""
from datetime import timedelta

import numpy as np
from pansat.products.satellite.cloud_sat import l2b_geoprof
from pansat.formats.hdf4 import HDF4File
from pyresample import kd_tree, geometry
import xarray
from scipy.signal import convolve

from joint_flight import PATH
from joint_flight.faam import resample_observations
from joint_flight.cloud_sat import load_radar_data
from joint_flight.era5 import resample_era5_data


START_TIME = np.datetime64("2019-03-22T12:55")
END_TIME = np.datetime64("2019-03-22T13:50")

###############################################################################
# CloudSat
###############################################################################

CLOUDSAT_FILENAME = "2019081121105_68702_CS_2B-GEOPROF_GRANULE_P1_R05_E08_F03.hdf"
i_start = 24159 + 10
i_end = 24631 - 40
RADAR = load_radar_data(PATH / "data" / CLOUDSAT_FILENAME, i_start, i_end)


###############################################################################
# ISMAR
###############################################################################

ISMAR_FILE = PATH / "data" / "metoffice-ismar_faam_20190322_r002_c161.nc"
ISMAR = resample_observations(
    ISMAR_FILE, RADAR["longitude"].data, RADAR["latitude"].data, START_TIME, END_TIME
)
ISMAR_30 = resample_observations(
    ISMAR_FILE,
    RADAR["longitude"].data,
    RADAR["latitude"].data,
    START_TIME,
    END_TIME,
    angle_limits=(27.5, 32.5),
)
###############################################################################
# MARSS
###############################################################################

MARSS_FILE = PATH / "data" / "metoffice-marss_faam_20190322_r002_c161.nc"
MARSS = resample_observations(
    MARSS_FILE, RADAR["longitude"].data, RADAR["latitude"].data, START_TIME, END_TIME
)

###############################################################################
# ERA5
###############################################################################

ERA5_PRESSURE_FILE = (
    PATH / "data" / "c161" / "reanalysis-era5-pressure-levels_2019032212_geopotential"
    "-temperature-cloud liquid water content-u-v-relative "
    "humidity40-65--10-10.nc"
)
ERA5_SURFACE_FILE = (
    PATH / "data" / "c161" / "reanalysis-era5-single-"
    "levels_2019032212_sst-skt-10u-10v40-65--10-10.nc"
)
ATMOSPHERE = resample_era5_data(
    ERA5_PRESSURE_FILE,
    ERA5_SURFACE_FILE,
    RADAR["longitude"].data,
    RADAR["latitude"].data,
    RADAR["height"].data,
)

SURFACE_MASK = RADAR.surface_height > 0
