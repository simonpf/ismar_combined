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

