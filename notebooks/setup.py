import numpy as np

from IPython import get_ipython
ip = get_ipython()

ip.magic("%matplotlib inline")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.style.use("/home/simon/.config/matplotlib/notebook")

import sys
sys.path.insert(0, "..")
