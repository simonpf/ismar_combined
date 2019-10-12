import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib        import gridspec
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def grid_to_edges(grid):
    new_grid = np.zeros((grid.shape[0]+ 1, grid.shape[1] + 1))
    new_grid[1:-1, 1:-1] = 0.25 * (grid[1:, 1:] + grid[1:, :-1] +
                                   grid[:-1, 1:] + grid[:-1, :-1])

    new_grid[0, 1:-1] = (grid[0, :-1] + grid[0, 1:]) - new_grid[1, 1:-1]
    new_grid[-1, 1:-1] = (grid[-1, :-1] + grid[-1, 1:]) - new_grid[-2, 1:-1]
    new_grid[1:-1, 0] = (grid[1:, 0] + grid[:-1, 0]) - new_grid[1:-1, 1]
    new_grid[1:-1, -1] = (grid[:-1, -1] + grid[1:, -1]) - new_grid[1:-1, -2]

    new_grid[0, 0] = 0.5 * (new_grid[0, 1] + new_grid[1, 0])
    new_grid[-1, 0] = 0.5 * (new_grid[-1, 1] + new_grid[-2, 0])
    new_grid[0, -1] = 0.5 * (new_grid[1, -1] + new_grid[0, -2])
    new_grid[-1, -1] = 0.5 * (new_grid[-1, -2] + new_grid[-2, -1])

    new_grid[0, 0]   = new_grid[1, 0] + new_grid[0, 1] - new_grid[1, 1]
    new_grid[0, -1]  = new_grid[0, -2] + new_grid[1, -1] - new_grid[1, -2]
    new_grid[-1, 0]  = new_grid[-2, 0] + new_grid[-1, 1] - new_grid[-2, 1]
    new_grid[-1, -1]  = new_grid[-2, -1] + new_grid[-1, -2] - new_grid[-2, -2]

    return new_grid

def bins_to_centers(bins):
    return 0.5 * (bins[1:] + bins[:-1])

def make_palette(n):
    cs = ["#00363b", "#194285", "#643ebb", "#be3fbb", "#f9598f", "#fd8e69", "#e0cb77"]
    i_start = (len(cs) - n) // 2
    return cs[i_start : i_start + n]
