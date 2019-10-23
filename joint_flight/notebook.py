import os
import numpy as np
import sys

# IPython magic
from IPython import get_ipython
ip = get_ipython()
ip.magic("%load_ext autoreload")
ip.magic("%autoreload 2")
ip.magic("%matplotlib inline")
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
sns.reset_defaults()

sys.path.insert(0, "..")

from joint_flight import path
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(os.path.join(path, "misc", "matplotlib_style.rc"))
