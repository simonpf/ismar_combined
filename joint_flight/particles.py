import torch
import numpy as np
import netCDF4
from torch.utils.data import Dataset
################################################################################
# Training data
################################################################################

class IceShapes(Dataset):
    def __init__(self,
                 path,
                 discrete = False,
                 group = None):
        print(path)
        self.file_handle = netCDF4.Dataset(path)
        self.discrete = discrete
        self.n = self.file_handle.dimensions["particle_index"].size

        if group is None:
            self.images = self.file_handle["particle_images"]
        else:
            g = self.file_handle[g]
            self.images = g["particle_images"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        x = np.array(self.images[idx, :, :])

        if self.discrete:
            x = x > 0.0
        else:
            x /= x.max()

        x = -1.0 + 2.0 * x
        fx, fy = np.random.rand(2)
        if fx > 0.5:
            x = np.fliplr(x)
            x = np.flipud(x)
            if fy > 0.5:
                x = x.T

        return torch.tensor(np.array(x[np.newaxis, :, :]))

    def extract_images(self, inds, path):

        #
        # Create output file.
        #

        if os.path.isfile(path):
            self._file_handle = netCDF4.Dataset(path, mode = "a")
            fh = self._file_handle
            variables = ["particle_images", "year", "month", "day", "hour",
                            "minute", "second", "millisecond"]
            print(fh.variables.keys())
            if not all([v in fh.variables.keys() for v in variables]):
                raise Exception("Error appending to existing netCDF file: Variables "
                                "are inconsistent.")
            ii = fh.dimensions["particle_index"].size
            print(self._image_index)
        else:
            # path must be a valid filename
            if os.path.isdir(path):
                raise Exception("For netcdf output mode the path must be "
                                "a valid file name.")
            # output_size must be given
            if output_size is None:
                raise Exception("For netcdf output the output size must be fixed.")

            fh = netCDF4.Dataset(path, mode = "w")
            fh = self._file_handle
            fh.createDimension("particle_index", size = None)
            fh.createDimension("width", size = output_size[0])
            fh.createDimension("height", size = output_size[1])
            fh.createVariable("particle_images", "f4",
                            dimensions = ["particle_index", "height", "width"])
            fh.createVariable("year", "i4", dimensions = ["particle_index"])
            fh.createVariable("month", "i4", dimensions = ["particle_index"])
            fh.createVariable("day", "i4", dimensions = ["particle_index"])
            fh.createVariable("hour", "i4", dimensions = ["particle_index"])
            fh.createVariable("minute", "i4", dimensions = ["particle_index"])
            fh.createVariable("second", "i4", dimensions = ["particle_index"])
            fh.createVariable("millisecond", "i4", dimensions = ["particle_index"])

            ii = 0

        #
        # Copy data
        #

        for ind in inds:
            fh_in = self.file_handle
            fh["particle_images"][ii, :, :] = fh_in["particle_images"][ind, :, :]
            fh["year"][ii, :, :] = fh_in["year"][ind, :, :]
            fh["month"][ii, :, :] = fh_in["month"][ind, :, :]
            fh["day"][ii, :, :] = fh_in["day"][ind, :, :]
            fh["hour"][ii, :, :] = fh_in["hour"][ind, :, :]
            fh["minute"][ii, :, :] = fh_in["minute"][ind, :, :]
            fh["second"][ii, :, :] = fh_in["second"][ind, :, :]
            fh["millisecond"][ii, :, :] = fh_in["millisecond"][ind, :, :]
            ii += 1
