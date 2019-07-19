import glob
import os
import numpy as np
import scipy as sp
from netCDF4 import Dataset
from joint_flight import path


results = {}

for p in ["8_column_aggregate", "large_plate_aggregate", "large_column_aggregate", "combined"]:
    data_path = os.path.join(path, "data", p)
    files = glob.glob(os.path.join(data_path, "output_?.nc"))
    files.sort()

    ice_md  = []
    ice_n0  = []
    snow_md = []
    snow_n0 = []
    rain_md = []
    rain_n0 = []
    cw      = []
    rh      = []

    y_hamp_radar  = []
    yf_hamp_radar = []
    y_hamp_passive  = []
    yf_hamp_passive = []
    y_ismar  = []
    yf_ismar = []

    diagnostics = []

    for f in files:
        ds = Dataset(f)
        g  = ds["All quantities"]
        ice_md +=  [g.variables["ice_md"][:]]
        ice_n0 +=  [g.variables["ice_n0"][:]]
        if p == "combined":
            snow_md +=  [g.variables["snow_md"][:]]
            snow_n0 +=  [g.variables["snow_n0"][:]]
        rain_md += [g.variables["rain_md"][:]]
        rain_n0 += [g.variables["rain_n0"][:]]
        cw      += [g.variables["cloud_water"][:]]
        rh      += [g.variables["H2O"][:]]

        y_hamp_radar    += [g.variables["y_hamp_radar"][:]]
        yf_hamp_radar   += [g.variables["yf_hamp_radar"][:]]
        y_hamp_passive  += [g.variables["y_hamp_passive"][:]]
        yf_hamp_passive += [g.variables["yf_hamp_passive"][:]]
        y_ismar  += [g.variables["y_ismar"][:]]
        yf_ismar += [g.variables["yf_ismar"][:]]

        diagnostics += [g.variables["diagnostics"][:]]
        ds.close()

    ds = Dataset(os.path.join(data_path, "output_radar_only.nc"))
    g  = ds["Radar only"]
    ice_md_ro  = g.variables["ice_md"][:]
    rain_md_ro = g.variables["rain_md"][:]
    ds.close()

    res = {}
    res["ice_md"]  = np.concatenate(ice_md, axis = 0)
    res["ice_n0"]  = np.concatenate(ice_n0, axis = 0)
    if p == "combined":
        res["snow_md"]  = np.concatenate(snow_md, axis = 0)
        res["snow_n0"]  = np.concatenate(snow_n0, axis = 0)
    res["rain_md"] = np.concatenate(rain_md, axis = 0)
    res["rain_n0"] = np.concatenate(rain_n0, axis = 0)
    res["cw"]      = np.concatenate(cw, axis = 0)
    res["rh"]      = np.concatenate(rh, axis = 0)
    res["y_hamp_radar"]    = np.concatenate(y_hamp_radar, axis = 0)
    res["yf_hamp_radar"]   = np.concatenate(yf_hamp_radar, axis = 0)
    res["y_hamp_passive"]  = np.concatenate(y_hamp_passive, axis = 0)
    res["yf_hamp_passive"] = np.concatenate(yf_hamp_passive, axis = 0)
    res["y_ismar"]  = np.concatenate(y_ismar, axis = 0)
    res["yf_ismar"] = np.concatenate(yf_ismar, axis = 0)
    res["diagnostics"] = np.concatenate(diagnostics, axis = 0)
    res["ice_md_ro"]   = ice_md_ro
    res["rain_md_ro"]  = rain_md_ro
    results[p] = res
