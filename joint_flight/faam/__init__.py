"""
=================
joint_flight.faam
=================

This module provides functions for the preprocessing of radiometer
observation from the FAAM aircraft.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import joint_flight
from pyresample import kd_tree, geometry

def resample_observations(input_file,
                          target_longitudes,
                          target_latitudes,
                          start_time,
                          end_time,
                          angle_limits=(-5, 5)):
    """
    Resample observations to 1D target swath.

    Args:
         input_file: Path to the file containing the ISMAR or MARSS
              oberstaions.
         target_longitudes: Array of longitudes to which to resample the
              radiometer observations in 'input_file'.
         target_latitudes: The latitude corresponding to 'target_longitudes'.
         start_time: Start time of the relevant observations.
         end_time: End time of the relevant observations.

    Returns:
         xarray.Dataset containing the brightness temperatures remapped to the
         given geolocations.
    """
    target_swath = geometry.SwathDefinition(lons=target_longitudes,
                                            lats=target_latitudes)
    data = xr.load_dataset(input_file)

    time = data["time"]
    indices = ((time > start_time)
               * (time <= end_time)
               * (np.isfinite(np.all(data["brightness_temperature"].data, axis=-1))))
    angles = data["sensor_view_angle"]
    indices *= (angles > angle_limits[0]) * (angles < angle_limits[1])

    time = data["time"][indices]
    lats = data["latitude"][indices]
    lons = data["longitude"][indices]
    altitude = data["altitude"][indices]
    source_swath = geometry.SwathDefinition(lons=lons, lats=lats)

    tbs = data["brightness_temperature"][indices]
    tbs = kd_tree.resample_nearest(source_swath,
                                   tbs.data,
                                   target_swath,
                                   fill_value=None,
                                   radius_of_influence=5e3)
    errors = np.maximum(
        data["brightness_temperature_positive_error"][indices].data,
        data["brightness_temperature_negative_error"][indices].data
    )
    errors = kd_tree.resample_nearest(source_swath,
                                      errors,
                                      target_swath,
                                      fill_value=None,
                                      radius_of_influence=5e3)
    if "brightness_temperature_random_error" in data.variables:
        random_errors = data["brightness_temperature_random_error"][indices]
        random_errors  = kd_tree.resample_nearest(source_swath,
                                                  random_errors.data,
                                                  target_swath,
                                                  fill_value=None,
                                                  radius_of_influence=5e3)
    else:
        random_errors = np.zeros_like(errors)

    time_s = kd_tree.resample_nearest(source_swath,
                                      time.data.astype(np.int64),
                                      target_swath,
                                      fill_value=None,
                                      radius_of_influence=5e3)
    time = time_s.astype("datetime64[ns]")

    lats = data["latitude"][indices]
    lats = kd_tree.resample_nearest(source_swath,
                                    lats.data,
                                    target_swath,
                                    fill_value=None,
                                    radius_of_influence=5e3)
    lons = data["longitude"][indices]
    lons = kd_tree.resample_nearest(source_swath,
                                    lons.data,
                                    target_swath,
                                    fill_value=None,
                                    radius_of_influence=5e3)

    errors = np.sqrt(errors ** 2 + random_errors ** 2)
    altitude = kd_tree.resample_nearest(source_swath,
                                        altitude.data,
                                        target_swath,
                                        fill_value=None,
                                        radius_of_influence=5e3)

    results = xr.Dataset({
        "latitude": ("rays", lats),
        "longitude": (("rays"), lons),
        "brightness_temperatures": (("rays", "channel"), tbs),
        "errors": (("rays", "channel"), errors),
        "altitude": (("rays",), altitude),
        "channel": data.channel.data,
        "time": (("rays",), time)
    })
    return results
