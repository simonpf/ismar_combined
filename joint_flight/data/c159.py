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

from joint_flight import path
from joint_flight.data.atmosphere import get_oxygen, get_ozone, get_nitrogen
from joint_flight.utils import centers_to_edges


start_time = np.datetime64("2019-03-19T12:27")
end_time = np.datetime64("2019-03-19T13:15")

###############################################################################
# CloudSat
###############################################################################

cloudsat_filename = "2019078115538_68658_CS_2B-GEOPROF_GRANULE_P1_R05_E08_F03.hdf"
cloudsat_data = l2b_geoprof.open(path / "data" / cloudsat_filename)
i_start = 24691
i_end = 25111 - 24

# Basic CloudSat data
time_cs = l2b_geoprof.filename_to_date(cloudsat_filename)
lons_cs = cloudsat_data["longitude"][i_start:i_end]
lats_cs = cloudsat_data["latitude"][i_start:i_end]
dbz = cloudsat_data["radar_reflectivity"][i_start:i_end, ::-1]
height_cs = cloudsat_data["height"][i_start:i_end, ::-1]

z_grid = np.mean(height_cs, axis=0)
j_start = np.where(z_grid > 0e3)[0][0]
j_end = np.where(z_grid > 10e3)[0][0]
z_grid = z_grid[j_start-1:j_end+1]
z_grid[0] = 0.0
z_grid[-1] = 10e3

y_cloudsat = dbz[:, j_start:j_end]
for i in range(y_cloudsat.shape[0]):
    z = height_cs[i, j_start:j_end]
    index = np.where(z > 1.0e3)[0][0] - 2
    y_cloudsat[i, :index] = y_cloudsat[i, index]
y_cloudsat = y_cloudsat / 100.0
y_cloudsat = np.maximum(y_cloudsat, -26)

range_bins_cloudsat = centers_to_edges(height_cs.data[:, j_start:j_end], axis=1)
range_bins_cloudsat = range_bins_cloudsat.astype(np.float)
range_bins_cloudsat_b = centers_to_edges(range_bins_cloudsat, axis=0)

lats_cs_r = np.broadcast_to(lats_cs.data.reshape(-1, 1), height_cs.shape)
lats_cs_b = np.broadcast_to(lats_cs.data.reshape(-1, 1), range_bins_cloudsat.shape)
lats_cs_b = centers_to_edges(lats_cs_b, axis=0)

dt = timedelta(seconds=cloudsat_data["time_since_start"].data[i_start])
start_time_cs  = time_cs + dt
dt = timedelta(seconds=cloudsat_data["time_since_start"].data[i_end])
end_time_cs  = time_cs + dt

dts = [timedelta(seconds=cloudsat_data["time_since_start"].data[i])
       for i in range(i_start, i_end)]
time_cs = [time_cs + dt for dt in dts]

cloudsat_swath = geometry.SwathDefinition(lons=lons_cs, lats=lats_cs)

###############################################################################
# ISMAR
###############################################################################

ismar_data = xarray.load_dataset(path / "data" / "metoffice-ismar_faam_20190319_r002_c159.nc")
ismar_channels = ismar_data.channel

time = ismar_data["time"]
indices = ((time > start_time)
           * (time <= end_time)
           * (np.isfinite(np.all(ismar_data["brightness_temperature"].data, axis=-1))))
angles = ismar_data["sensor_view_angle"]
indices *= np.abs(np.abs(angles) - 0.0) < 5.0


time = ismar_data["time"][indices]
lats_ismar = ismar_data["latitude"][indices]
lons_ismar = ismar_data["longitude"][indices]
lats_ismar_all = ismar_data["latitude"]
lons_ismar_all = ismar_data["longitude"]
altitude_ismar = ismar_data["altitude"][indices]
ismar_swath = geometry.SwathDefinition(lons=lons_ismar, lats=lats_ismar)

tbs_ismar_raw = ismar_data["brightness_temperature"][indices]
lons_raw = ismar_data["longitude"][indices]
lats_raw = ismar_data["latitude"][indices]
tbs_ismar = kd_tree.resample_nearest(ismar_swath,
                                   tbs_ismar_raw.data,
                                   cloudsat_swath,
                                   #sigmas=[2e3] * tbs_ismar.shape[1],
                                   fill_value=None,
                                   radius_of_influence=5e3)
random_errors_ismar = ismar_data["brightness_temperature_random_error"][indices]
random_errors_ismar, _, n = kd_tree.resample_gauss(ismar_swath,
                                                   random_errors_ismar.data,
                                                   cloudsat_swath,
                                                   sigmas=[2e3] * tbs_ismar.shape[1],
                                                   fill_value=None,
                                                   radius_of_influence=5e3,
                                                   with_uncert=True)
random_errors_ismar /= np.sqrt(n)
errors_ismar = np.maximum(ismar_data["brightness_temperature_positive_error"][indices].data,
                          ismar_data["brightness_temperature_negative_error"][indices].data,)
errors_ismar = kd_tree.resample_gauss(ismar_swath,
                                      errors_ismar,
                                      cloudsat_swath,
                                      sigmas=[2e3] * tbs_ismar.shape[1],
                                      fill_value=None,
                                      radius_of_influence=5e3)
errors_ismar = np.sqrt(errors_ismar ** 2 + random_errors_ismar ** 2)
time_ismar = kd_tree.resample_nearest(ismar_swath,
                                      time.data,
                                      cloudsat_swath,
                                      fill_value=np.datetime64("nat", "ns"),
                                      radius_of_influence=5e3)
altitude_ismar = kd_tree.resample_gauss(ismar_swath,
                                        altitude_ismar.data,
                                        cloudsat_swath,
                                        sigmas=2e3,
                                        fill_value=None,
                                        radius_of_influence=5e3)

start_time_ismar = np.nanmin(time_ismar)
end_time_ismar = np.nanmax(time_ismar)

###############################################################################
# Marss
###############################################################################

marss_data = xarray.load_dataset(path / "data" / "metoffice-marss_faam_20190319_r002_c159.nc")
marss_channels = marss_data.channel

time = marss_data["time"]
indices = ((time > start_time)
           * (time <= end_time)
           * (np.isfinite(np.all(marss_data["brightness_temperature"].data, axis=-1))))
angles = marss_data["sensor_view_angle"]
indices *= np.abs(np.abs(angles) - 0.0) < 5.0

time = marss_data["time"][indices]
lats_marss = marss_data["latitude"][indices]
lons_marss = marss_data["longitude"][indices]
marss_swath = geometry.SwathDefinition(lons=lons_marss, lats=lats_marss)

tbs_marss_raw = marss_data["brightness_temperature"][indices]
tbs_marss = kd_tree.resample_gauss(marss_swath,
                                   tbs_marss_raw.data,
                                   cloudsat_swath,
                                   sigmas=[1e3] * tbs_marss_raw.shape[1],
                                   fill_value=None,
                                   radius_of_influence=5e3)

time_marss = kd_tree.resample_nearest(marss_swath,
                                      time.data,
                                      cloudsat_swath,
                                      fill_value=np.datetime64("nat", "ns"),
                                      radius_of_influence=5e3)

tbs_marss = marss_data["brightness_temperature"][indices]
tbs_marss = kd_tree.resample_gauss(marss_swath,
                                   tbs_marss.data,
                                   cloudsat_swath,
                                   sigmas=[2e3] * tbs_marss.shape[1],
                                   fill_value=None,
                                   radius_of_influence=5e3)
errors_marss = np.maximum(marss_data["brightness_temperature_positive_error"][indices].data,
                          marss_data["brightness_temperature_negative_error"][indices].data,)
errors_marss = kd_tree.resample_gauss(marss_swath,
                                      errors_marss,
                                      cloudsat_swath,
                                      sigmas=[2e3] * tbs_marss.shape[1],
                                      fill_value=None,
                                      radius_of_influence=5e3)

start_time_marss = np.nanmin(time_marss)
end_time_marss = np.nanmax(time_marss)

###############################################################################
# ERA5
###############################################################################

era5_pressure = xarray.load_dataset(path / "data" / "reanalysis-era5-pressure-levels_2019031912_geopotential-temperature-relative_humidity59-64--6-5.nc")
era5_pressure["altitude"] = era5_pressure["z"] / 9.80665
era5_i = era5_pressure.interp(latitude=lats_cs, longitude=lons_cs)

p = np.zeros((lats_cs.rays.size, z_grid.size))
t = np.zeros((lats_cs.rays.size, z_grid.size))
r = np.zeros((lats_cs.rays.size, z_grid.size))

for i in range(lats_cs.size):
    z_cs = height_cs[i, :].data
    p_era = era5_i.level.data[::-1] * 100.0
    t_era = era5_i["t"].data[0, ::-1, i]
    r_era = era5_i["r"].data[0, ::-1, i]
    z_era = era5_i["altitude"].data[0, ::-1, i]
    p[i] = np.interp(z_grid, z_era, p_era)
    t[i] = np.interp(z_grid, z_era, t_era)
    r[i] = np.interp(z_grid, z_era, r_era)

mask = (p[:, 1] == p[:, 0])
p[mask, 1] -= 1.0

cloud_sat_range_bins = height_cs[i_start:i_end, j_start:j_end]
o2 = np.broadcast_to(get_oxygen(z_grid).data.reshape(1, -1), p.shape)
o3 = np.broadcast_to(get_ozone(z_grid).data.reshape(1, -1), p.shape)
n2 = np.broadcast_to(get_nitrogen(z_grid).data.reshape(1, -1), p.shape)


era5_surface = xarray.load_dataset(path / "data" / "reanalysis-era5-single-levels_2019031912_sst-skin_temperature-10m_u_component_of_wind-10m_v_component_of_wind59-64--6-0.nc")
era5_surface = era5_surface.interp(latitude=lats_cs, longitude=lons_cs)
