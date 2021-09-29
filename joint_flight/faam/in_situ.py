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
import pandas as pd

import joint_flight
from pyresample import kd_tree, geometry
import typhon
from typhon.geodesy import great_circle_distance


def read_psds(cip_file_1, cip_file_2, core_file, start_time=None, end_time=None):
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
    dx = (
        np.concatenate(
            [
                cip_x5[f"cip{cip_size}_bin_width"][start_x5:end_x5],
                cip_100["cip100_bin_width"][start_100:end_100],
            ]
        )
        * 1e-6
    )
    x = (
        np.concatenate(
            [
                cip_x5[f"cip{cip_size}_bin_centre"][start_x5:end_x5],
                cip_100["cip100_bin_centre"][start_100:end_100],
            ]
        )
        * 1e-6
    )

    cip_x5 = xr.load_dataset(cip_file_1, group="raw_group")
    cip_100 = xr.load_dataset(cip_file_2, group="raw_group")

    if start_time is None:
        start_time = time[0]

    if end_time is None:
        end_time = time[-1]

    indices = (start_time < time) * (time < end_time)

    time = time[{"time": indices}]
    latitude = latitude[{"time": indices}]
    longitude = longitude[{"time": indices}]
    altitude = altitude[{"time": indices}]

    cip_x5 = cip_x5[{"time": indices}]
    cip_100 = cip_100[{"time": indices}]

    y = np.concatenate(
        [
            cip_x5[f"cip{cip_size}_conc_psd"][:, start_x5:end_x5],
            cip_100["cip100_conc_psd"][:, start_100:end_100],
        ],
        axis=1,
    )

    core = xr.load_dataset(core_file)

    try:
        twc = core["NV_TWC_C"].interp(Time=time)
        lwc_1 = core["NV_LWC1_C"].interp(Time=time)
        lwc_2 = core["NV_LWC2_C"].interp(Time=time)
        lwc = lwc_2
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


path = Path(joint_flight.PATH) / "data" / "c159"
c159_cip15_file = path / "core-cloud-phy_faam_20190319_v003_r0_c159_cip15.nc"
c159_cip100_file = path / "core-cloud-phy_faam_20190319_v003_r0_c159_cip100.nc"
c159_core_file = path / "core_faam_20190319_v004_r0_c159_1hz.nc"
C159 = read_psds(
    c159_cip15_file,
    c159_cip100_file,
    c159_core_file,
    np.datetime64("2019-03-19T13:10:00", "ns"),
    np.datetime64("2019-03-19T14:45:00", "ns"),
)

path = Path(joint_flight.PATH) / "data" / "c161"
c161_cip15_file = path / "core-cloud-phy_faam_20190322_v003_r0_c161_cip15.nc"
c161_cip100_file = path / "core-cloud-phy_faam_20190322_v003_r0_c161_cip100.nc"
c161_core_file = path / "core_faam_20190322_v004_r0_c161_1hz.nc"
C161 = read_psds(c161_cip15_file, c161_cip100_file, c161_core_file)
# np.datetime64("2019-03-22T13:59:00", "ns"),
# np.datetime64("2019-03-22T14:23:00", "ns"))

path = Path(joint_flight.PATH) / "data"
jf_cip25_file = path / "core-cloud-phy_faam_20161014_v002_r0_b984_cip25.nc"
jf_cip100_file = path / "core-cloud-phy_faam_20161014_v002_r0_b984_cip100.nc"
jf_core_file = path / "core_faam_20161014_v004_r0_b984_1hz.nc"

JF = read_psds(
    jf_cip25_file,
    jf_cip100_file,
    jf_core_file,
    np.datetime64("2016-10-14T10:30:00", "ns"),
    np.datetime64("2016-10-14T11:02:00", "ns"),
)


def load_drop_sonde_data(path, results=None):

    data = []
    files = Path(PATH).glob("faam-dropsonde*.nc")

    for f in files:
        ds_data = xr.load_dataset(f)

        valid = (-90 <= ds_data["lat"].data) * (90 >= ds_data["lat"].data)
        ds_data = ds_data.loc[{"time": valid}]

        if results:
            lons = results["longitude"].data
            lats = results["latitude"].data
            retrieval_swath = geometry.SwathDefinition(lons=lons, lats=lats)
            lons = ds_data["lon"].data
            lats = ds_data["lat"].data
            ds_swath = geometry.SwathDefinition(lons=lons, lats=lats)
            ni = kd_tree.get_neighbour_info(
                retrieval_swath, ds_swath, radius_of_influence=100e3, neighbours=1
            )
            (valid_input_index, valid_output_index, index_array, distance_array) = ni

            n = ds_data.time.size
            n_levels = results.z.size

            t_r = np.zeros(n)
            t_a = np.zeros(n)
            h2o_r = np.zeros(n)
            h2o_a = np.zeros(n)

            t_z = np.zeros((n, n_levels))
            t_a_z = np.zeros((n, n_levels))
            h2o_z = np.zeros((n, n_levels))
            h2o_a_z = np.zeros((n, n_levels))
            z = np.zeros((n, n_levels))
            d = np.zeros((n))

            lats_r = kd_tree.get_sample_from_neighbour_info(
                "nn",
                (n,),
                results["latitude"].data,
                valid_input_index,
                valid_output_index,
                index_array,
                fill_value=np.nan,
            )
            lons_r = kd_tree.get_sample_from_neighbour_info(
                "nn",
                (n,),
                results["longitude"].data,
                valid_input_index,
                valid_output_index,
                index_array,
                fill_value=np.nan,
            )

            d = kd_tree.get_sample_from_neighbour_info(
                "nn",
                (n,),
                results["d"].data,
                valid_input_index,
                valid_output_index,
                index_array,
                fill_value=np.nan,
            )

            for i in range(n_levels):
                # t_z[:, i] = kd_tree.get_sample_from_neighbour_info(
                #    "nn",
                #    (n,),
                #    results["temperature"][:, i].data,
                #    valid_input_index,
                #    valid_output_index,
                #    index_array,
                #    fill_value=np.nan)
                # t_a_z[:, i] = kd_tree.get_sample_from_neighbour_info(
                #    "nn",
                #    (n,),
                #    results["temperature_a_priori"][:, i].data,
                #    valid_input_index,
                #    valid_output_index,
                #    index_array,
                #    fill_value=np.nan)
                h2o_z[:, i] = kd_tree.get_sample_from_neighbour_info(
                    "nn",
                    (n,),
                    results["H2O"][:, i].data,
                    valid_input_index,
                    valid_output_index,
                    index_array,
                    fill_value=np.nan,
                )
                h2o_a_z[:, i] = kd_tree.get_sample_from_neighbour_info(
                    "nn",
                    (n,),
                    results["H2O_a_priori"][:, i].data,
                    valid_input_index,
                    valid_output_index,
                    index_array,
                    fill_value=np.nan,
                )
                z[:, i] = kd_tree.get_sample_from_neighbour_info(
                    "nn",
                    (n,),
                    results["altitude"][:, i].data,
                    valid_input_index,
                    valid_output_index,
                    index_array,
                    fill_value=np.nan,
                )

            for i in range(n):
                if np.isnan(ds_data["alt"][i]):
                    t_r[i] = np.nan
                    t_a[i] = np.nan
                    h2o_r[i] = np.nan
                    h2o_a[i] = np.nan
                    continue

                t_r[i] = np.interp(ds_data["alt"][i], z[i, :], t_z[i, :])
                t_a[i] = np.interp(ds_data["alt"][i], z[i, :], t_a_z[i, :])
                h2o_r[i] = np.interp(ds_data["alt"][i], z[i, :], h2o_z[i, :])
                h2o_a[i] = np.interp(ds_data["alt"][i], z[i, :], h2o_a_z[i, :])

            ds_data["t_retrieved"] = (("time",), t_r)
            ds_data["t_a_priori"] = (("time",), t_a)
            ds_data["h2o_retrieved"] = (("time",), h2o_r)
            ds_data["h2o_a_priori"] = (("time",), h2o_a)
            ds_data["lons_r"] = (("time"), lons_r)
            ds_data["lats_r"] = (("time"), lats_r)
            ds_data["d"] = (("time"), d)
        data.append(ds_data)
    return data


def load_nevzorov_data(nevzorov_file, core_file, start_time, end_time, reference=None):
    data = xr.load_dataset(nevzorov_file, decode_times=False)

    time = pd.Timestamp(start_time)
    year = time.year
    month = time.month
    day = time.day
    time_0 = np.datetime64(f"{year}-{month:02}-{day}T00:00:00")
    time = time_0 + data["TIME"].data * np.timedelta64(1, "s")

    j_start = np.where(time > start_time)[0][0]
    j_end = np.where(time > end_time)[0][0]
    time = time[j_start:j_end]

    data = data[{"TIME": slice(j_start, j_end)}]

    core = xr.load_dataset(core_file, decode_times=False)
    core = core.interp(Time=data["TIME"])

    altitude = core["ALT_GIN"]
    latitude = core["LAT_GIN"]
    longitude = core["LON_GIN"]

    twc = data["TWC_Q_ice"]
    lwc = data["LWC_Q_liq"]
    iwc = twc - lwc

    data = {
        "time": (("time",), time),
        "twc": (("time",), twc),
        "lwc": (("time",), lwc),
        "iwc": (("time",), iwc),
        "altitude": (("time",), altitude),
        "latitude": (("time",), latitude),
        "longitude": (("time",), longitude),
    }

    if reference:
        lons = reference["longitude"].data
        lats = reference["latitude"].data
        d = reference["d"].data
        reference_swath = geometry.SwathDefinition(lons=lons, lats=lats)
        cip_swath = geometry.SwathDefinition(lons=longitude, lats=latitude)
        d = kd_tree.resample_nearest(
            reference_swath, d, cip_swath, 20e3, fill_value=np.nan
        )

    data["d"] = (("time",), d)

    return xr.Dataset(data)


# C159 = read_psds(c159_cip15_file,
#                 c159_cip100_file,
#                 np.datetime64("2019-03-19T13:10:00", "ns"),
#                 np.datetime64("2019-03-19T14:45:00", "ns"))


def load_cip_data(
    cip_15_file, cip_100_file, core_file, start_time, end_time, reference
):

    cip_15 = xr.load_dataset(cip_15_file, decode_times=False)

    time = pd.Timestamp(start_time)
    year = time.year
    month = time.month
    day = time.day
    time_0 = np.datetime64(f"{year}-{month:02}-{day}T00:00:00")
    time = time_0 + cip_15["TIME"].data * np.timedelta64(1, "s")

    j_start = np.where(time > start_time)[0][0]
    j_end = np.where(time > end_time)[0][0]
    time = time[j_start:j_end]

    core = xr.load_dataset(core_file, decode_times=False)

    cip_15 = cip_15[{"TIME": slice(j_start, j_end)}]
    core = core.interp(Time=cip_15["TIME"])
    altitude = core["ALT_GIN"]
    latitude = core["LAT_GIN"]
    longitude = core["LON_GIN"]

    bins = cip_15["BIN_EDGES"][:] / 1e6
    x_15 = 0.5 * (bins[1:] + bins[:-1])
    n_15 = cip_15["SPEC"][:] * 1e6
    dndd_15 = n_15 / (np.diff(bins).reshape(-1, 1))

    cip_100 = xr.load_dataset(cip_100_file, decode_times=False)
    cip_100 = cip_100[{"TIME": slice(j_start, j_end)}]

    bins = cip_100["BIN_EDGES"][:] / 1e6
    x_100 = 0.5 * (bins[1:] + bins[:-1])
    n_100 = cip_100["SPEC"][:] * 1e6
    dndd_100 = n_100 / (np.diff(bins).reshape(-1, 1))

    start_15, end_15 = 0, 64
    start_100, end_100 = 9, 64
    x = np.concatenate([x_15[start_15:end_15], x_100[start_100:end_100]])
    y = np.concatenate([dndd_15[start_15:end_15], dndd_100[start_100:end_100]])
    n = np.concatenate([n_15[start_15:end_15], n_100[start_100:end_100]])

    data = {
        "time": (("time",), time),
        "diameter": (("diameter",), x),
        "dndd": (
            (
                "time",
                "diameter",
            ),
            y.T,
        ),
        "n": (
            (
                "time",
                "diameter",
            ),
            n.T,
        ),
        "altitude": (("time",), altitude),
        "latitude": (("time",), latitude),
        "longitude": (("time",), longitude),
    }

    if reference:
        lons = reference["longitude"].data
        lats = reference["latitude"].data
        d = reference["d"].data
        reference_swath = geometry.SwathDefinition(lons=lons, lats=lats)
        cip_swath = geometry.SwathDefinition(lons=longitude, lats=latitude)
        d = kd_tree.resample_nearest(
            reference_swath, d, cip_swath, 10e3, fill_value=np.nan
        )

    data["d"] = (("time",), d)

    return xr.Dataset(data)


# C159 = read_psds(c159_cip15_file,
#                 c159_cip100_file,
#                 np.datetime64("2019-03-19T13:10:00", "ns"),
#                 np.datetime64("2019-03-19T14:45:00", "ns"))
