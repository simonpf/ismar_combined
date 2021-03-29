"""
=========================
joint_flight.faam.in_situ
=========================

This module provides functions to load and pre-process particle probe data
from FAMM BAE air craft.
"""
from pathlib import Path

import numpy as np
import xarray as xr


def read_psds(cip_file_1,
              cip_file_2,
              core_file,
              start_time=None,
              end_time=None):
    """
    Calculate PSDs from FAAM in-insitu data.
    """
    cip_x5 = xr.load_dataset(cip_file_1)
    cip_size = int(cip_file_1.name[-5:-3])

    cip_100 = xr.load_dataset(cip_file_2)
    latitude = cip_x5["latitude"]
    longitude = cip_x5["longitude"]
    altitude = cip_x5["altitude"]
    time = cip_x5["time"]

    start_x5, end_x5 = 0, 64
    start_100, end_100 = 9, 64
    dx = (np.concatenate([cip_x5[f"cip{cip_size}_bin_width"][start_x5: end_x5],
                          cip_100["cip100_bin_width"][start_100: end_100]])
          * 1e-6)
    x = (np.concatenate([cip_x5[f"cip{cip_size}_bin_centre"][start_x5: end_x5],
                        cip_100["cip100_bin_centre"][start_100: end_100]])
                       * 1e-6)

    cip_x5 = xr.load_dataset(cip_file_1, group="raw_group")
    cip_100 = xr.load_dataset(cip_file_2, group="raw_group")

    if start_time is None:
        start_time = time[0]

    if end_time is None:
        end_time = time[-1]

    indices = ((start_time < time) * (time < end_time))

    time = time[{"time": indices}]
    latitude = latitude[{"time": indices}]
    longitude = longitude[{"time": indices}]
    altitude = altitude[{"time": indices}]

    cip_x5 = cip_x5[{"time": indices}]
    cip_100 = cip_100[{"time": indices}]

    print(cip_x5[f"cip{cip_size}_bin_centre"].size)

    y = np.concatenate([cip_x5[f"cip{cip_size}_conc_psd"][:, start_x5: end_x5],
                        cip_100["cip100_conc_psd"][:, start_100: end_100]],
                       axis=1)

    core = xr.load_dataset(core_file)

    try:
        twc = core["NV_TWC_C"].interp(Time=time)
        lwc_1 = core["NV_LWC1_C"].interp(Time=time)
        lwc_2 = core["NV_LWC2_C"].interp(Time=time)
        lwc = 0.5 * (lwc_1 + lwc_2)
        iwc = twc - lwc
    except KeyError:
        twc = core["NV_TWC_U"].interp(Time=time)
        lwc = core["NV_LWC_U"].interp(Time=time)
        iwc = twc - lwc

    dims = ("time", "bins")
    data = {
        "time": (dims[:1], time),
        "latitude": (dims[:1], latitude),
        "longitude": (dims[:1], longitude),
        "altitude": (dims[:1], altitude),
        "psd": (dims, y),
        "particle_size": ("bins", x),
        "bin_width": ("bin_widths", dx),
        "iwc": ("time", iwc),
        "lwc": ("time", lwc),
        "twc": ("time", twc),
    }

    return xr.Dataset(data)


path = Path("/home/simon/src/joint_flight/data/c159/")
c159_cip15_file = path / "core-cloud-phy_faam_20190319_v003_r0_c159_cip15.nc"
c159_cip100_file = path / "core-cloud-phy_faam_20190319_v003_r0_c159_cip100.nc"
c159_core_file = path / "core_faam_20190319_v004_r0_c159_1hz.nc"
C159 = read_psds(c159_cip15_file,
                 c159_cip100_file,
                 c159_core_file,
                 np.datetime64("2019-03-19T13:10:00", "ns"),
                 np.datetime64("2019-03-19T14:45:00", "ns"))

path = Path("/home/simon/src/joint_flight/data/c161/")
c161_cip15_file = path / "core-cloud-phy_faam_20190322_v003_r0_c161_cip15.nc"
c161_cip100_file = path / "core-cloud-phy_faam_20190322_v003_r0_c161_cip100.nc"
c161_core_file = path / "core_faam_20190322_v004_r0_c161_1hz.nc"
C161 = read_psds(c161_cip15_file,
                 c161_cip100_file,
                 c161_core_file)
                 #np.datetime64("2019-03-22T13:59:00", "ns"),
                 #np.datetime64("2019-03-22T14:23:00", "ns"))

path = Path("/home/simon/src/joint_flight/data/")
jf_cip25_file = path / "core-cloud-phy_faam_20161014_v002_r0_b984_cip25.nc"
jf_cip100_file = path / "core-cloud-phy_faam_20161014_v002_r0_b984_cip100.nc"
jf_core_file = path / "core_faam_20161014_v004_r0_b984_1hz.nc"

JF = read_psds(jf_cip25_file,
               jf_cip100_file,
               jf_core_file,
               np.datetime64("2016-10-14T10:30:00", "ns"),
               np.datetime64("2016-10-14T11:02:00", "ns"))

#C159 = read_psds(c159_cip15_file,
#                 c159_cip100_file,
#                 np.datetime64("2019-03-19T13:10:00", "ns"),
#                 np.datetime64("2019-03-19T14:45:00", "ns"))
