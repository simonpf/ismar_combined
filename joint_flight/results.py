import glob
import os
import numpy as np
import scipy as sp
from netCDF4 import Dataset
from joint_flight import path

data_path = os.path.join(path, "data")
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
    snow_md += [g.variables["snow_md"][:]]
    snow_n0 += [g.variables["snow_n0"][:]]
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

ice_md  = np.concatenate(ice_md, axis = 0)
ice_n0  = np.concatenate(ice_n0, axis = 0)
snow_md = np.concatenate(snow_md, axis = 0)
snow_n0 = np.concatenate(snow_n0, axis = 0)
rain_md = np.concatenate(rain_md, axis = 0)
rain_n0 = np.concatenate(rain_n0, axis = 0)
cw      = np.concatenate(cw, axis = 0)
rh      = np.concatenate(rh, axis = 0)

y_hamp_radar    = np.concatenate(y_hamp_radar, axis = 0)
yf_hamp_radar   = np.concatenate(yf_hamp_radar, axis = 0)
y_hamp_passive  = np.concatenate(y_hamp_passive, axis = 0)
yf_hamp_passive = np.concatenate(yf_hamp_passive, axis = 0)
y_ismar  = np.concatenate(y_ismar, axis = 0)
yf_ismar = np.concatenate(yf_ismar, axis = 0)

diagnostics = np.concatenate(diagnostics, axis = 0)
