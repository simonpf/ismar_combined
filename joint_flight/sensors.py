from parts.sensor import ActiveSensor, PassiveSensor
import os
import numpy as np
from netCDF4 import Dataset
import scipy as sp
import scipy.sparse

class HampRadar(ActiveSensor):

    def __init__(self, stokes_dimension = 1):

        path = os.path.dirname(__file__)
        ds = Dataset(os.path.join(path, "..", "data", "input.nc"))
        z  = ds.variables["altitude"][0, :]

        range_bins = np.zeros(z.size + 1)
        range_bins[1:-1] = 0.5 * (z[1:] + z[:-1])
        range_bins[0]  = 2 * range_bins[1] - range_bins[2]
        range_bins[-1] = 2 * z[-1] - z[-2]
        ds.close()

        super().__init__(name = "hamp_radar",
                         f_grid = [35.564e9],
                         range_bins = range_bins,
                         stokes_dimension = stokes_dimension)

        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position      = np.array([12500.0])
        self.y_min = -30.0

    @property
    def nedt(self):
        return 0.5 * np.ones(self.range_bins.size - 1)

class RastaRadar(ActiveSensor):

    def __init__(self, stokes_dimension = 1):

        super().__init__(name = "rasta",
                         f_grid = [95e9],
                         stokes_dimension = stokes_dimension)

        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position      = np.array([12500.0])
        self.y_min = -16.0

    @property
    def nedt(self):
        return 0.5 * np.ones(self.range_bins.size - 1)

class HampPassive(PassiveSensor):

    channels = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40,
                         50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00,
                         90,
                         118.75 - 8.5, 118.75 - 4.2, 118.75 - 2.3, 118.75 - 1.4,
                         118.75 + 1.4, 118.75 + 2.3, 118.75 + 4.2, 118.75 + 8.5,
                         183.31 - 12.5, 183.31 - 7.5, 183.31 - 5.0, 183.31 - 3.5,
                         183.31 - 2.5, 183.31 - 1.5, 183.31 - 0.6,
                         183.31 + 0.6, 183.31 + 1.5, 183.31 + 2.5, 183.31 + 3.5,
                         183.31 + 5.0, 183.31 + 7.5, 183.31 + 12.5]) * 1e9

    _nedt = np.array([0.1] * 7 + [0.2] * 7 + [0.25] + [0.6] * 4 + [0.6] * 7)

    def __init__(self, stokes_dimension = 1):
        super().__init__(name = "hamp_passive",
                         f_grid = HampPassive.channels,
                         stokes_dimension = stokes_dimension)
        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position     = np.array([12500.0])

        self.sensor_response_f    = self.f_grid[:-11]
        self.sensor_response_pol  = self.f_grid[:-11]
        self.sensor_response_dlos = self.f_grid[:-11, np.newaxis]

        data = 15 * [1.0] + [0.5] * 22
        i    = list(range(15)) + 2 * list(range(15, 19)) + 2 * list(range(19, 26))
        j    = list(range(37))
        self.sensor_response = sp.sparse.coo_matrix((data, (i, j)))
        self.sensor_f_grid   = self.f_grid[:-11]


    @property
    def nedt(self):
        return HampPassive._nedt
