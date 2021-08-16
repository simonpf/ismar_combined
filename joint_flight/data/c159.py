"""
======================
joint_flight.data.c159
======================

This module provides access to the preprocessed data from flight c159 of the
FAAM aircraft.
"""
from datetime import timedelta

import numpy as np
from pansat.products.satellite.cloud_sat import l2b_geoprof
from pyresample import kd_tree, geometry
import xarray

from joint_flight import PATH, ISMAR_INDICES
from joint_flight.data.atmosphere import get_oxygen, get_ozone, get_nitrogen
from joint_flight.utils import centers_to_edges
from joint_flight.faam import resample_observations
from joint_flight.faam.in_situ import load_cip_data, load_nevzorov_data
from joint_flight.cloud_sat import load_radar_data
from joint_flight.era5 import resample_era5_data

START_TIME = np.datetime64("2019-03-19T12:27")
END_TIME = np.datetime64("2019-03-19T13:15")

###############################################################################
# CloudSat
###############################################################################

CLOUDSAT_FILENAME = "2019078115538_68658_CS_2B-GEOPROF_GRANULE_P1_R05_E08_F03.hdf"

i_start = 24691
i_end = 25111 - 24
RADAR = load_radar_data(PATH / "data" / CLOUDSAT_FILENAME,
                        i_start,
                        i_end)

###############################################################################
# ISMAR
###############################################################################

ISMAR_FILE = PATH / "data" / "metoffice-ismar_faam_20190319_r002_c159.nc"
ISMAR = resample_observations(ISMAR_FILE,
                              RADAR["longitude"].data,
                              RADAR["latitude"].data,
                              START_TIME,
                              END_TIME)
tbs = ISMAR["brightness_temperatures"]

MARSS_FILE = PATH / "data" / "metoffice-marss_faam_20190319_r002_c159.nc"
MARSS = resample_observations(MARSS_FILE,
                              RADAR["longitude"].data,
                              RADAR["latitude"].data,
                              START_TIME,
                              END_TIME)

###############################################################################
# ERA5
###############################################################################

ERA5_PRESSURE_FILE = (PATH / "data" / "c159" /
                      "reanalysis-era5-pressure-levels_2019031912_geopotential"
                      "-temperature-cloud liquid water content-u-v-relative "
                      "humidity40-65--10-10.nc")
ERA5_SURFACE_FILE = (PATH / "data" / "c159" /
                     "reanalysis-era5-single-levels_2016101409_sea_surface_"
                     "temperature-skin temperature-10m_u_component_of_wind-"
                     "10m_v_component_of_wind40-65--10-10.nc")

ATMOSPHERE = resample_era5_data(ERA5_PRESSURE_FILE,
                                ERA5_SURFACE_FILE,
                                RADAR["longitude"].data,
                                RADAR["latitude"].data,
                                RADAR["height"].data)

SURFACE_MASK = RADAR.surface_height > 0

START_TIME_IN_SITU = np.datetime64("2019-03-19T13:20:00")
END_TIME_IN_SITU = np.datetime64("2019-03-19T14:45:00")
NEVZOROV = load_nevzorov_data(PATH / "data" / "c159" / "c159_nevzorov_20190319_1hz_r1.nc",
                              PATH / "data" / "c159" / "core_faam_20190319_v004_r0_c159_1hz.nc",
                              START_TIME_IN_SITU,
                              END_TIME_IN_SITU,
                              reference=RADAR)
PSD = load_cip_data(PATH / "data" / "c159" / "c159_cip15_20190319_1hz_r0.nc",
                    PATH / "data" / "c159" / "c159_cip100_20190319_1hz_r0.nc",
                    PATH / "data" / "c159" / "core_faam_20190319_v004_r0_c159_1hz.nc",
                    START_TIME_IN_SITU,
                    END_TIME_IN_SITU,
                    reference=RADAR)
