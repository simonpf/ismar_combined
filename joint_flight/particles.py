import torch
import numpy as np
import netCDF4
from torch.utils.data import Dataset
import os
################################################################################
# Training data
################################################################################

class IceShapes(Dataset):
    def __init__(self,
                 path,
                 discrete = False,
                 group = None,
                 mode = "a",
                 unclassified_only = False):

        self.file_handle = netCDF4.Dataset(path, mode = mode)
        self.discrete = discrete

        if not "class_index" in self.file_handle.variables:
            self.file_handle.createVariable("class_index", "i4", dimensions = ["particle_index"])
            self.file_handle["class_index"][:] = -1

        if unclassified_only:
            self.inds = np.where(np.array(self.file_handle["class_index"]) == -1)[0]
        else:
            self.inds = np.arange(self.file_handle.dimensions["particle_index"].size)
        self.n = self.inds.size

        if group is None:
            self.images = self.file_handle["particle_images"]
        else:
            g = self.file_handle[g]
            self.images = g["particle_images"]

    def close(self):
        self.file_handle.close()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = self.inds[idx]
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

    def classify_images(self, inds, class_index):
        inds = self.inds[inds]
        fh = self.file_handle
        fh["class_index"][inds] = class_index

    def extract_images(self, inds, path, class_index = None):

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

            w = self.file_handle.dimensions["width"].size
            h = self.file_handle.dimensions["height"].size

            fh = netCDF4.Dataset(path, mode = "w")
            fh.createDimension("particle_index", size = None)
            fh.createDimension("width", size = w)
            fh.createDimension("height", size = h)
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

        if not "class_index" in fh.variables:
            fh.createVariable("class_index", "i4", dimensions = ["particle_index"])

        if not "reference_index" in fh.variables:
            fh.createVariable("reference_index", "i4", dimensions = ["particle_index"])

        if class_index is None:
            class_index = np.max(fh["class_index"]) + 1

        #
        # Copy data
        #


        fh_in = self.file_handle

        for ind in inds:

            if "reference_index" in fh_in:
                ind_ref = fh_in["reference_index"][ind]
            else:
                ind_ref = ind

            fh["particle_images"][ii, :, :] = fh_in["particle_images"][ind, :, :]
            fh["year"][ii] = fh_in["year"][ind, :, :]
            fh["month"][ii] = fh_in["month"][ind, :, :]
            fh["day"][ii] = fh_in["day"][ind, :, :]
            fh["hour"][ii] = fh_in["hour"][ind, :, :]
            fh["minute"][ii] = fh_in["minute"][ind, :, :]
            fh["second"][ii] = fh_in["second"][ind, :, :]
            fh["millisecond"][ii] = fh_in["millisecond"][ind, :, :]
            fh["class_index"][ii] = class_index
            fh["reference_index"][ii] = ind_ref
            ii += 1
        fh.close()

def create_mosaic(data,
                  m = 10,
                  n = 10,
                  padding = 1):

    ind = np.random.randint(0, len(data))
    img = data[ind][0]

    h, w = img.shape

    out = np.zeros((m * (h + padding) - padding, n * (w + padding) - padding))
    for i in range(m):

        i_start = i * h
        if i > 0:
            i_start += i * padding
        i_end = i_start + h

        for j in range(n):

            j_start = j * h
            if j > 0:
                j_start += j * padding
            j_end = j_start + w

            ind = np.random.randint(0, len(data))
            out[i_start : i_end, j_start : j_end] = data[ind][0].detach().numpy()

    return out

