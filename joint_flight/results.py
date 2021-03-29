import glob
import os
import numpy as np
import scipy as sp
from netCDF4 import Dataset
import joint_flight

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def iwc(n0, dm):
    return np.pi * 917.0 * dm ** 4 * n0 / 4 ** 4

def rwc(n0, dm):
    return np.pi * 1000.0 * dm ** 4 * n0 / 4 ** 4

def get_results(variables = ["ice_dm", "ice_n0"],
                config = "combined"):
    """
    This function returns a dictionary of retrieval results
    for all particle types that the retrieval has been
    performed for.

    Args:

        variables: List of variables to extract from file.

    Returns:

        A dict of dict holding the extracted variables for all the
        particle types found in $JOINT_FLIGHT_PATH/data/combined
    """
    path = os.path.join(joint_flight.path, "data",  config)
    results = {}

    pattern = os.path.join(path, "output_" + "*.nc")
    files = glob.glob(pattern)

    for f in files:
        print(f)
        splits = os.path.basename(f).split("_")
        habit = splits[-1].split(".")[0]
        results[habit] = {}

        file = Dataset(f, mode = "r")
        for v in variables:
            try:
                k = list(file.groups.keys())[-1]
                results[habit][v] = file.groups[k][v][:]
            except:
                results[habit][v] = None

        if "ice_n0" in variables and "ice_dm" in variables:
            results[habit]["ice_md"] = iwc(results[habit]["ice_n0"], results[habit]["ice_dm"])

        if "rain_n0" in variables and "rain_dm" in variables:
            results[habit]["rain_md"] = rwc(results[habit]["rain_n0"], results[habit]["rain_dm"])

    return results

