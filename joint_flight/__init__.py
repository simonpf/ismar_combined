"""
============
joint_flight
============

The ``joint_flight`` package contains the code that was used for the processing
and analysis of combined radar/sub-mm radiometer retrievals from the flights
of the FAAM research aircraft.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import seaborn as sns

from joint_flight.utils import remove_x_ticks
if "JOINT_FLIGHT_PATH" in os.environ:
    PATH = Path(os.environ["JOINT_FLIGHT_PATH"])
else:
    PATH = Path(__file__).parent.parent
    print(
        f"No environment variable JOINT_FLIGHT_PATH found. Will look in {PATH}"
        f"for the joint flight data."
    )

######################################################################
# Definitions
######################################################################

ISMAR_INDICES = (
    [0, 1, 2, 3, 4] +
    [5] +
    [7, 8, 9] +
    [14, 15, 16] +
    [17] +
    [20]
)
SURFACE_CHANNELS_MARSS = [0, 1]
SURFACE_CHANNELS_ISMAR = [0, 1, 2, 3, 4]
BAND_INDICES_MARSS = [
    [0],
    [1],
    [2, 3, 4],
]
BAND_INDICES_ISMAR = [
    [0, 1, 2, 3, 4],
    [5],
    [7, 8, 9],
    [14, 15, 16],
    [17],
    [20]
]
RESULT_INDICES_ISMAR = {
    "b984": [
        [0, 1, 2, 3, 4],
        [5],
        [6, 7, 8],
        [],
        [9],
        [],
    ],
    "cxxx": [
        [0, 1, 2, 3, 4],
        [5],
        [6],
        [7, 8, 9],
        [10],
        [11],
    ]
}
RESULT_INDICES_MARSS = [
    [0],
    [1],
    [2, 3, 4]
]


######################################################################
# Helper functions.
######################################################################


def add_surface_shading(ax, x, surface_mask):
    """
    Add light-grey background where observations are over land
    surface.

    Args:
        ax: The matplotlib.axes.Axes object to which to add the
            surface shading.
    """
    limits = np.where(np.abs(np.diff(surface_mask)) > 0.0)[0]
    if surface_mask[0]:
        limits = np.concatenate([[0], limits])
    if surface_mask[-1]:
        limits = np.concatenate([limits, [-1]])
    y_min, y_max = ax.get_ylim()
    for i in range(len(limits) // 2):
        l = limits[2 * i]
        r = limits[2 * i + 1]
        ax.fill_betweenx(np.linspace(-1000, 1000, 301), x[l], x[r],
                         color="gainsboro")


def get_colors(n):
    """
    Create a color sequence of given length.

    Args:
        n: The number of colors in the sequence.
    """
    N = n
    n_l = (N) // 2 - n // 2
    n_r = n_l + n
    colors = sns.color_palette("magma", N)
    colors = [sns.desaturate(c, 0.9) for c in colors]
    #colors = sns.cubehelix_palette(n, start=i * 0.2, rot=-0.1, dark=0.1, light=0.7)
    return colors[n_l:n_r]


######################################################################
# Plotting functions.
######################################################################


def plot_atmosphere(radar,
                    atmosphere):
    """
    Plot atmospheric background together with radar observations.

    Args:
        radar: ``xarray.Dataset`` containing the radar observations.
        atmosphere: ``xarray.Dataset`` containing the atmosphere
            background.
    """
    f, axs = plt.subplots(4, 1, figsize=(10, 10))

    x = radar["x"] * 1e-3
    y = radar["y"] * 1e-3

    ax = axs[0]
    norm = Normalize(-30, 20)
    m = ax.pcolormesh(x, y, radar["dbz"], norm=norm)
    plt.colorbar(m, ax=ax, label="Relative humidity")
    ax.set_xticks([])
    ax.set_ylabel("Altitude [km]")
    ax.set_title("(a) Radar reflectivity", loc="left")

    ax = axs[1]
    norm = Normalize(0, 1.2)
    m = ax.pcolormesh(x, y, atmosphere["relative_humidity"], norm=norm, cmap="coolwarm")
    ax.set_xticks([])
    ax.set_ylabel("Altitude [km]")
    ax.set_title("(b) Relative humidity", loc="left")
    plt.colorbar(m, ax=ax, label="Relative humidity")

    ax = axs[2]
    norm = Normalize(230, 290)
    m = ax.pcolormesh(x, y, atmosphere["temperature"], norm=norm, cmap="coolwarm")
    ax.set_xticks([])
    ax.set_ylabel("Altitude [km]")
    ax.set_title("(c) Temperature", loc="left")
    plt.colorbar(m, ax=ax, label="Temperature [K]")

    ax = axs[3]
    norm = LogNorm(1e-6, 1e-3)
    m = ax.pcolormesh(x, y, atmosphere["cloud_water"], norm=norm, cmap="magma")
    ax.set_xlabel("Along-track distance [km]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("(d) Cloud water content", loc="left")
    plt.colorbar(m, ax=ax, label="Temperature [K]")

    plt.tight_layout()
    plt.show()
    return axs


def plot_observations(ismar,
                      marss,
                      radar,
                      axs=None,
                      legends=None,
                      names=None,
                      missing_channels=None):
    """
    Plot radar and radiometer observations.

    Args:
        ismar: ''xarray.Dataset'' containing the observations from the ISMAR
            radiometer.
        marss: ''xarray.Dataset'' containing the observations from the MARSS
            radiometer.
        radar: ``xarray.Dataset`` containing the radar observations.
        axs: Optional list of ``matplotlib.Axes`` to use to plot the
            observations.
        legends: Optional list of ``matplotlib.Axes`` to use to plot the
            the legends for each plot.
        names: Optional list of ``matplotlib.Axes`` to use to plot the
            the titles for each plot.
        missing_channels: Optional list of channel indices that are missing
            for the given flight.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if missing_channels is None:
        missing_channels = []

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [
            1.0, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.4
        ]
        gs = GridSpec(10, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(10)]


    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    ax.pcolormesh(x, y, np.pad(z, ((0, 1), (0, 1))), cmap="inferno", shading="gouraud")

    ax.spines['left'].set_position(('outward', 10))
    ax.set_ylabel(r"Altitude [km]")
    ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)

    ax.set_xlim([x_min, x_max])

    if legends:
        legends[0].set_axis_off()

    if names:
        ax = names[0]
        ax.text(0.5,
                0.5,
                "Radar reflectivity",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 89 GHz
    #

    x = radar["d"] / 1e3
    y = marss["brightness_temperatures"]

    ax = axs[1]
    colors = get_colors(len(BAND_INDICES_MARSS[0]))
    for ci, i in enumerate(BAND_INDICES_MARSS[0]):
        handle = ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[1]
        ax.set_axis_off()
        ax.legend(handles=handle,
                  labels=["$88.992 \pm \SI{1.075}{\giga \hertz}$"],
                  loc="center right")

    if names:
        ax = names[1]
        ax.text(0.5,
                0.5,
                "89 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 118 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[2]
    colors = get_colors(len(BAND_INDICES_ISMAR[0]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[0]):
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[2]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$118 \pm \SI{1.1}{\giga \hertz}$",
                r"$118 \pm \SI{1.5}{\giga \hertz}$",
                r"$118 \pm \SI{2.1}{\giga \hertz}$",
                r"$118 \pm \SI{3.0}{\giga \hertz}$",
                r"$118 \pm \SI{5.0}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[2]
        ax.text(0.5,
                0.5,
                "118 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 157 GHz
    #

    y = marss["brightness_temperatures"]

    ax = axs[3]
    colors = get_colors(len(BAND_INDICES_MARSS[1]))
    for ci, i in enumerate(BAND_INDICES_MARSS[1]):
        handles = ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[3]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$157 \pm$ GHz",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[3]
        ax.text(0.5,
                0.5,
                "157 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 183 GHz
    #

    y = marss["brightness_temperatures"]

    ax = axs[4]
    colors = get_colors(len(BAND_INDICES_MARSS[2]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_MARSS[2]):
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[4]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$183.248 \pm \SI{0.975}{\giga \hertz}$",
                r"$183.248 \pm \SI{3.0}{\giga \hertz}$",
                r"$183.248 \pm \SI{7.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[4]
        ax.text(0.5, 0.5,
                "183 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 243 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[5]
    colors = get_colors(len(BAND_INDICES_ISMAR[1]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[1]):
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[5]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$\SI{243}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[5]
        ax.text(0.5,
                0.5,
                r"243 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 325 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[6]
    colors = get_colors(len(BAND_INDICES_ISMAR[2]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[2]):
        if ci in missing_channels:
            continue
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[6]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$325 \pm 1.5$ GHz",
                r"$325 \pm 3.5$ GHz",
                r"$325 \pm 9.5$ GHz"
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[6]
        ax.text(0.5,
                0.5,
                "325 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 448 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[7]
    colors = get_colors(len(BAND_INDICES_ISMAR[3]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[3]):
        if i in missing_channels:
            continue
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if all([b in missing_channels for b in BAND_INDICES_ISMAR[5]]):
        ax.set_axis_off()

    if legends:
        ax = legends[7]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$448 \pm \SI{1.4}{\giga \hertz}$",
                r"$448 \pm \SI{3.0}{\giga \hertz}$",
                r"$448 \pm \SI{7.2}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[7]
        ax.text(0.5,
                0.5,
                "448 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 664
    #

    y = ismar["brightness_temperatures"]

    ax = axs[8]
    colors = get_colors(len(BAND_INDICES_ISMAR[4]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[4]):
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")

    if legends:
        ax = legends[8]
        ax.set_axis_off()
        if handles:
            labels = [
                "$664\pm\SI{4.2}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[8]
        ax.text(0.5,
                0.5,
                "664 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 884
    #

    y = ismar["brightness_temperatures"]

    ax = axs[9]
    colors = get_colors(len(BAND_INDICES_ISMAR[5]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[5]):
        handles += ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    if all([b in missing_channels for b in BAND_INDICES_ISMAR[5]]):
        handles[0].set_visible(False)
        ax.yaxis.set_visible(False)

    if legends:
        ax = legends[9]
        ax.set_axis_off()
        if handles:
            if any([b not in missing_channels for b in BAND_INDICES_ISMAR[5]]):
                labels = [
                    r"$874.4 \pm \SI{6.0}{\giga \hertz}$",
                ]
                ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[9]
        ax.text(0.5,
                0.5,
                "874 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    plt.tight_layout()


def plot_observations_marss(marss,
                            radar,
                            surface_mask,
                            axs=None,
                            legends=None,
                            names=None,
                            y_axis=True,
                            missing_channels=None):
    """
    Plot MARSS observations by frequency band.

    Args:
        marss: 'xarray.Dataset' containing the MARSS observations.
        radar: 'xarray.Dataset' containing the radar observations which will
            be plotted in the first row of panels.
        surface_mask: Boolean array indicating which observations were taken
             over land surfaces.
        legends: List of axes to use to plot the legends for each band.
        names: List of axes to use to plot the names for each band.
        y_axis: Whether or not to draw a y axis on the plot.
        missing_channels: List of missing channels.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if missing_channels is None:
        missing_channels = []

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [1.0, 1.0, 1.0, 1.0]
        gs = GridSpec(10, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(10)]

    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    ax.pcolormesh(x, y, np.pad(z, ((0, 1), (0, 1))), cmap="magma", shading="gouraud")
    ax.set_ylim(0, 10)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)

    if y_axis:
        ax.set_ylabel(r"Altitude $[\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)


    ax.set_xlim([x_min, x_max])

    if legends:
        legends[0].set_axis_off()

    if names:
        ax = names[0]
        ax.text(0.5,
                0.5,
                "Radar reflectivity",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 89 GHz
    #

    x = radar["d"] / 1e3
    y = marss["brightness_temperatures"]

    ax = axs[1]
    colors = get_colors(len(BAND_INDICES_MARSS[0]))
    for ci, i in enumerate(BAND_INDICES_MARSS[0]):
        handle = ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_MARSS[0]])
    y_max = np.nanmax(y[:, BAND_INDICES_MARSS[0]])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[1]
        ax.set_axis_off()
        ax.legend(handles=handle,
                  labels=["$88.992 \pm \SI{1.075}{\giga \hertz}$"],
                  loc="center right")

    if names:
        ax = names[1]
        ax.text(0.5,
                0.5,
                "89 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


    #
    # 157 GHz
    #

    y = marss["brightness_temperatures"]

    ax = axs[2]
    colors = get_colors(len(BAND_INDICES_MARSS[1]))
    for ci, i in enumerate(BAND_INDICES_MARSS[1]):
        handles = ax.plot(x, y[:, i], c=colors[ci])

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_MARSS[1]])
    y_max = np.nanmax(y[:, BAND_INDICES_MARSS[1]])
    ax.set_ylim([y_min, y_max])

    if legends:
        ax = legends[2]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$\SI{157}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[2]
        ax.text(0.5,
                0.5,
                "157 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 183 GHz
    #

    y = marss["brightness_temperatures"]

    ax = axs[3]
    colors = get_colors(len(BAND_INDICES_MARSS[2]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_MARSS[2]):
        handles += ax.plot(x, y[:, i], c=colors[ci])

    if y_axis:
        ax.spines['left'].set_position(('outward', 10))
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r"Along track distance [$\si{\kilo \meter}$]")

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_MARSS[2]])
    y_max = np.nanmax(y[:, BAND_INDICES_MARSS[2]])
    ax.set_ylim([y_min, y_max])

    if legends:
        ax = legends[3]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$183.248 \pm \SI{0.975}{\giga \hertz}$",
                r"$183.248 \pm \SI{3.0}{\giga \hertz}$",
                r"$183.248 \pm \SI{7.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[3]
        ax.text(0.5, 0.5,
                "183 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


def plot_observations_ismar(ismar,
                            radar,
                            surface_mask,
                            axs=None,
                            legends=None,
                            names=None,
                            y_axis=True,
                            missing_channels=None):
    """
    Plot ISMAR observations by frequency band.

    Args:
        ISMAR: 'xarray.Dataset' containing the ISMAR observations.
        radar: 'xarray.Dataset' containing the radar observations which will
            be plotted in the for row of panels.
        surface_mask: Boolean array indicating which observations were taken
             over land surfaces.
        axs: List of axes to use to plot the observations.
        legends: List of axes to use to plot the legends for each band.
        names: List of axes to use to plot the names for each band.
        y_axis: Whether or not to draw a y axis on the plot.
        missing_channels: List of missing channels.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    lw = 1

    if missing_channels is None:
        missing_channels = []

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [1.0, 1.0, 1.0, 1.0]
        gs = GridSpec(10, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(10)]

    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    ax.pcolormesh(x, y, np.pad(z, ((0, 1), (0, 1))), cmap="magma", shading="gouraud")
    ax.set_ylim(0, 10)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)

    if y_axis:
        ax.set_ylabel(r"Altitude $[\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)


    ax.set_xlim([x_min, x_max])

    if legends:
        legends[0].set_axis_off()

    if names:
        ax = names[0]
        ax.text(0.5,
                0.5,
                "Radar reflectivity",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 118 GHz
    #

    x = radar["d"] / 1e3
    y = ismar["brightness_temperatures"]

    ax = axs[1]
    colors = get_colors(len(BAND_INDICES_ISMAR[0]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[0]):
        handle = ax.plot(x, y[:, i], c=colors[ci], lw=lw)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_ISMAR[0]])
    y_max = np.nanmax(y[:, BAND_INDICES_ISMAR[0]])
    ax.set_ylim([y_min, y_max])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[1]
        ax.set_axis_off()
        ax.legend(handles=handle,
                  labels=[
                      r"$118\pm \SI{1.1}{\giga \hertz}$",
                      r"$118\pm \SI{1.5}{\giga \hertz}$",
                      r"$118\pm \SI{2.1}{\giga \hertz}$",
                      r"$118\pm \SI{3.0}{\giga \hertz}$",
                      r"$118\pm \SI{5.0}{\giga \hertz}$",
                  ],
                  loc="center right")

    if names:
        ax = names[1]
        ax.text(0.5,
                0.5,
                "118 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


    #
    # 243 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[2]
    colors = get_colors(len(BAND_INDICES_ISMAR[1]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[1]):
        handles = ax.plot(x, y[:, i], c=colors[ci], lw=lw)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_ISMAR[1]])
    y_max = np.nanmax(y[:, BAND_INDICES_ISMAR[1]])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[2]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$243.2 \pm \SI{2.5}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[2]
        ax.text(0.5,
                0.5,
                r"243 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 325 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[3]
    colors = get_colors(len(BAND_INDICES_ISMAR[2]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[2]):
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)

    if y_axis:
        ax.spines['left'].set_position(('outward', 10))
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_ISMAR[2]])
    y_max = np.nanmax(y[:, BAND_INDICES_ISMAR[2]])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[3]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$325.15 \pm \SI{1.5}{\giga \hertz}$",
                r"$325.15 \pm \SI{3.5}{\giga \hertz}$",
                r"$325.15 \pm \SI{9.5}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[3]
        ax.text(0.5, 0.5,
                "325 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


    #
    # 448 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[4]
    colors = get_colors(len(BAND_INDICES_ISMAR[3]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[3]):
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)

    if y_axis:
        ax.spines['left'].set_position(('outward', 10))
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_ISMAR[3]])
    y_max = np.nanmax(y[:, BAND_INDICES_ISMAR[3]])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[4]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$448.0 \pm \SI{1.4}{\giga \hertz}$",
                r"$448.0 \pm \SI{3.0}{\giga \hertz}$",
                r"$448.0 \pm \SI{7.2}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[4]
        ax.text(0.5, 0.5,
                "448 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 664 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[5]
    colors = get_colors(len(BAND_INDICES_ISMAR[4]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[4]):
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)

    if y_axis:
        ax.spines['left'].set_position(('outward', 10))
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_ISMAR[4]])
    y_max = np.nanmax(y[:, BAND_INDICES_ISMAR[4]])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[5]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$664.0 \pm \SI{4.2}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[5]
        ax.text(0.5, 0.5,
                "664 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 874.4 GHz
    #

    y = ismar["brightness_temperatures"]

    ax = axs[6]
    colors = get_colors(len(BAND_INDICES_ISMAR[5]))
    handles = []
    for ci, i in enumerate(BAND_INDICES_ISMAR[5]):
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)

    if y_axis:
        ax.spines['left'].set_position(('outward', 10))
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r"Along track distance [$\si{\kilo \meter}$]")

    add_surface_shading(ax, x, surface_mask)
    y_min = np.nanmin(y[:, BAND_INDICES_ISMAR[5]])
    y_max = np.nanmax(y[:, BAND_INDICES_ISMAR[5]])
    try:
        ax.set_ylim([y_min, y_max])
    except:
        pass

    if legends:
        ax = legends[6]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$874.4 \pm \SI{6.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center right")

    if names:
        ax = names[6]
        ax.text(0.5, 0.5,
                r"874 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


def plot_psds(radar,
              psds,
              nevzorov,
              results=None,
              axs=None,
              legends=None,
              names=None,
              y_axis=True):
    """
    Plot in-situ-measured IWC and PSDs.

    Args:
        radar: 'xarray.Dataset' containing the radar observations which will
            be plotted in the first row of panels.
        psds: 'xarray.Dataset' containing the in-situ-measured PSDs.
        nevzorov: 'xarray.Dataset' from the Nevzorov probe.
        axs: List of axes to use to plot the observations.
        legends: List of axes to use to plot the legends for each band.
        names: List of axes to use to plot the names for each band.
        y_axis: Whether or not to draw a y axis on the plot.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [1.0, 1.0, 1.0, 1.0]
        gs = GridSpec(10, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(10)]

    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    ax.pcolormesh(x, y, np.pad(z, ((0, 1), (0, 1))), cmap="magma", shading="gouraud")
    ax.set_ylim(0, 10)

    d = nevzorov["d"] / 1e3
    alt = nevzorov["altitude"] / 1e3
    iwc = nevzorov["twc"] / 1e3

    m = ScalarMappable(norm=LogNorm(1e-5, 1e-3), cmap="coolwarm")

    for i in range(d.size - 1):
        c = m.to_rgba(0.5 * (iwc[i] + iwc[i + 1]))
        ax.plot(d[[i, i+1]], alt[[i, i+1]], c=c, lw=6)

    if legends:
        plt.colorbar(m, cax=legends[0], label=r"IWC [$kg/m^3$]")

    ax.set_yticks(np.arange(2, 9))

    ax.spines['left'].set_position(('outward', 10))
    ax.set_xlabel(r"Along track distance [$\si{\kilo \meter}$]")

    ax.yaxis.grid(True)
    if y_axis:
        ax.set_ylabel(r"Altitude $[\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.spines["left"].set_visible(False)


    ax.set_xlim([x_min, x_max])

    if names:
        ax = names[0]
        ax.text(0.5,
                0.5,
                "Radar reflectivity",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 8 - 7 Kilometers
    #

    n_samples = 100

    for i in range(6):

        ax = axs[i + 1]
        alt_max = 8e3 - i * 1e3
        alt_min = alt_max - 1e3

        indices = (psds.altitude >= alt_min) * (psds.altitude < alt_max)
        data = psds[{"time": indices}]
        x = data["diameter"]
        y = data["dndd"]
        y_mean = y.mean("time")

        x_min = x.min()
        x_max = x.max()



        ax.set_xlim([x_min, x_max])
        ax.set_ylim([1e4, 1e12])
        for _ in range(n_samples):
            index = np.random.randint(data.time.size)
            ax.plot(x, y[index], lw=1, c="grey", alpha=0.1)
        ax.plot(x, y_mean)

        ax.spines['left'].set_position(('outward', 10))

        if i < 4:
            ax.spines['bottom'].set_visible(False)
            remove_x_ticks(ax)
        else:
            ax.spines['bottom'].set_position(('outward', 10))
            ax.set_xlim([x_min, x_max])
            ax.set_xlabel(r"$D_\text{MAX}$ [$\si{\meter}$]")

        ax.set_yscale("log")
        ax.set_xscale("log")


        if y_axis:
            ax.set_ylabel(r"$\frac{dN}{dD_\text{max}}\ [\si{\per  \meter \tothe{4} }]$")
        else:
            ax.yaxis.set_ticks_position('none')
            for l in ax.yaxis.get_ticklabels():
                l.set_visible(False)
            ax.spines["left"].set_visible(False)

        if names:
            ax = names[i + 1]
            ax.text(0.5,
                    0.5,
                    rf"${(alt_min/1e3):1.0f} - \SI{{{(alt_max/1e3):1.0f}}}{{\kilo \meter}}$",
                    rotation="vertical",
                    rotation_mode="anchor",
                    transform=ax.transAxes,
                weight="bold",
                    ha="center",
                    va="center")
            ax.set_axis_off()

        if legends:
            legends[i + 1].set_axis_off()


def plot_residuals(radar,
                   results,
                   flight,
                   surface_mask,
                   axs=None,
                   legend_axs=None,
                   name_axs=None,
                   y_axis=True,
                   title=None):
    """
    Plot retrieval residuals.

    Args:
        radar: ``xarray.Dataset`` containing the radar observations.
        results: ``xarray.Dataset`` contatining the retrieval results.
        surface_mask: Surface mask indicating which observations were performed
            over land surfaces.
        axs: List of 10 ``matplotlib.Axes`` object to use to plot the residuals.
        legend_axs: List of 10 ``matplotlib.Axes`` objects to use to display the legends
             for all plots.
        name_axs: List of 10 ``matplolib.Axes`` object to display titles for all rows in
             the plots.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    flight = flight.lower()

    yf_ismar = results["yf_ismar"].data
    y_ismar = results["y_ismar"].data
    dy_ismar = yf_ismar - y_ismar

    yf_marss = results["yf_marss"].data
    y_marss = results["y_marss"].data
    dy_marss = yf_marss - y_marss

    if flight == "b984":
        yf_radar = results["yf_hamp_radar"].data
        y_radar = results["y_hamp_radar"].data
        dy_radar = yf_radar - y_radar
    else:
        yf_radar = results["yf_cloud_sat"].data
        y_radar = results["y_cloud_sat"].data
        dy_radar = yf_radar - y_radar

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [
            1.0, 1.0, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.4
        ]
        gs = GridSpec(11, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(11)]

    band_indices_ismar = RESULT_INDICES_ISMAR[flight]
    band_indices_marss = RESULT_INDICES_MARSS

    #
    # RADAR reflectivities
    #

    norm = Normalize(-25, 20)

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    sm = ax.pcolormesh(x, y, z, cmap="magma", shading="auto", norm=norm)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)

    ax.set_ylim([0, 10])
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    ax.set_title(title, fontsize=20, pad=10)

    if legend_axs:
        plt.colorbar(sm, cax=legend_axs[0], label=r"Reflectivity [$\text{dBZ}$]")

    if name_axs:
        ax = name_axs[0]
        ax.text(0.5,
                0.5,
                "Radar (observed)",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


    #
    # RADAR deviations
    #

    ax = axs[1]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    sm = ax.pcolormesh(x, y,
                       dy_radar,
                       cmap="coolwarm",
                       norm=Normalize(-3, 3),
                       shading="auto")

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    ax.set_aspect("auto")
    ax.set_ylim([0, 10])
    ax.set_yticklabels([-10, 0, 10])
    if y_axis:
        ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        plt.colorbar(sm, cax=legend_axs[1], label=r"Residual [$\text{dB}$]")

    if name_axs:
        ax = name_axs[1]
        ax.text(0.5,
                0.5,
                "Radar (residuals)",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # Plot settings
    #

    lw = 1

    #
    # 89 GHz
    #

    x = radar["d"] / 1e3
    y = dy_marss.copy()
    y[surface_mask, :] = np.nan

    ax = axs[2]
    colors = get_colors(len(band_indices_marss[0]))
    for ci, i in enumerate(band_indices_marss[0]):
        #handle = [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handle = ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])
    ax.set_aspect("auto")
    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[2]
        ax.set_axis_off()
        ax.legend(handles=handle,
                  labels=["$88.992 \pm \SI{1.075}{\giga \hertz}$"],
                  loc="center")

    if name_axs:
        ax = name_axs[2]
        ax.text(0.5,
                0.5,
                "89 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 118 GHz
    #

    y = dy_ismar.copy()
    y[surface_mask, 1:5] = np.nan

    ax = axs[3]
    colors = get_colors(len(band_indices_ismar[0]))
    handles = []
    for ci, i in enumerate(band_indices_ismar[0]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[3]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$118 \pm \SI{1.1}{\giga \hertz}$",
                r"$118 \pm \SI{1.5}{\giga \hertz}$",
                r"$118 \pm \SI{2.1}{\giga \hertz}$",
                r"$118 \pm \SI{3.0}{\giga \hertz}$",
                r"$118 \pm \SI{5.0}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[3]
        ax.text(0.5,
                0.5,
                "118 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 157 GHz
    #

    y = dy_marss.copy()
    y[surface_mask, :] = np.nan

    ax = axs[4]
    colors = get_colors(len(band_indices_marss[1]))
    handles = []
    for ci, i in enumerate(band_indices_marss[1]):
        #handles = [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[4]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$157 \pm \SI{2.6}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[4]
        ax.text(0.5,
                0.5,
                "157 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 183 GHz
    #

    y = dy_marss

    ax = axs[5]
    colors = get_colors(len(band_indices_marss[2]))
    handles = []
    for ci, i in enumerate(band_indices_marss[2]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[5]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$183.248 \pm \SI{0.975}{\giga \hertz}$",
                r"$183.248 \pm \SI{3.0}{\giga \hertz}$",
                r"$183.248 \pm \SI{7.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[5]
        ax.text(0.5, 0.5,
                "183 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 243 GHz
    #

    y = dy_ismar

    ax = axs[6]
    colors = get_colors(len(band_indices_ismar[1]))
    handles = []
    for ci, i in enumerate(band_indices_ismar[1]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])
    ax.set_aspect("auto")
    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[6]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$243.2 \pm \SI{2.5}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[6]
        ax.text(0.5,
                0.5,
                r"243 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 325 GHz
    #

    y = dy_ismar

    ax = axs[7]
    colors = get_colors(len(BAND_INDICES_ISMAR[2]))
    for ci, i in enumerate(band_indices_ismar[2]):
        ci += (len(BAND_INDICES_ISMAR[2]) - len(band_indices_ismar[2])) // 2
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    handles = []
    colors = get_colors(len(BAND_INDICES_ISMAR[3]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[3]):
        handles += ax.plot(x, -40 * np.ones(x.shape), c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[7]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$325.15 \pm \SI{1.5}{\giga \hertz}$",
                r"$325.15 \pm \SI{3.5}{\giga \hertz}$",
                r"$325.15 \pm \SI{9.5}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[7]
        ax.text(0.5,
                0.5,
                "325 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 448 GHz
    #

    y = dy_ismar

    ax = axs[8]
    colors = get_colors(len(band_indices_ismar[3]))
    for ci, i in enumerate(band_indices_ismar[3]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    handles = []
    colors = get_colors(len(BAND_INDICES_ISMAR[3]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[3]):
        handles += ax.plot(x, -40 * np.ones(x.shape), c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    #if not band_indices_ismar[3]:
    #    ax.set_axis_off()

    if legend_axs:
        ax = legend_axs[8]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$448 \pm \SI{1.4}{\giga \hertz}$",
                r"$448 \pm \SI{3.0}{\giga \hertz}$",
                r"$448 \pm \SI{7.2}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[8]
        ax.text(0.5,
                0.5,
                "448 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 664
    #

    y = dy_ismar

    ax = axs[9]
    colors = get_colors(len(band_indices_ismar[4]))
    handles = []
    for ci, i in enumerate(band_indices_ismar[4]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-10, 10])

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[9]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$664.0 \pm \SI{4.2}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[9]
        ax.text(0.5,
                0.5,
                "664 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 874
    #

    y = dy_ismar

    ax = axs[10]
    colors = get_colors(len(band_indices_ismar[5]))
    for ci, i in enumerate(band_indices_ismar[5]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        ax.plot(x, y[:, i], c=colors[ci], lw=lw)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    handles = []
    colors = get_colors(len(BAND_INDICES_ISMAR[5]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[5]):
        handles += ax.plot(x, -40 * np.ones(x.shape), c=colors[ci])

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r"Along track distance [$\si{\kilo \meter}$]")
    ax.set_ylim([-10, 10])

    if legend_axs:
        ax = legend_axs[10]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$874.4 \pm \SI{6.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[10]
        ax.text(0.5,
                0.5,
                "874 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    plt.tight_layout()

def plot_residuals_lf(radar,
                      results,
                      flight,
                      surface_mask,
                      axs=None,
                      legend_axs=None,
                      name_axs=None,
                      y_axis=True,
                      title=None):
    """
    Plot retrieval residuals.

    Args:
        radar: ``xarray.Dataset`` containing the radar observations.
        results: ``xarray.Dataset`` contatining the retrieval results.
        surface_mask: Surface mask indicating which observations were performed
            over land surfaces.
        axs: List of 10 ``matplotlib.Axes`` object to use to plot the residuals.
        legend_axs: List of 10 ``matplotlib.Axes`` objects to use to display the legends
             for all plots.
        name_axs: List of 10 ``matplolib.Axes`` object to display titles for all rows in
             the plots.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    flight = flight.lower()

    yf_ismar = results["yf_ismar"].data
    y_ismar = results["y_ismar"].data
    dy_ismar = yf_ismar - y_ismar

    yf_marss = results["yf_marss"].data
    y_marss = results["y_marss"].data
    dy_marss = yf_marss - y_marss

    if flight == "b984":
        yf_radar = results["yf_hamp_radar"].data
        y_radar = results["y_hamp_radar"].data
        dy_radar = yf_radar - y_radar
    else:
        yf_radar = results["yf_cloud_sat"].data
        y_radar = results["y_cloud_sat"].data
        dy_radar = yf_radar - y_radar

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [
            1.0, 1.0, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.4
        ]
        gs = GridSpec(11, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(11)]

    band_indices_ismar = RESULT_INDICES_ISMAR[flight]
    band_indices_marss = RESULT_INDICES_MARSS

    #
    # RADAR reflectivities
    #

    norm = Normalize(-25, 20)

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    sm = ax.pcolormesh(x, y, z, cmap="magma", shading="auto", norm=norm)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)

    ax.set_ylim([0, 10])
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    ax.set_title(title, fontsize=20, pad=10)

    if legend_axs:
        plt.colorbar(sm, cax=legend_axs[0], label=r"Reflectivity [$\text{dBZ}$]")

    if name_axs:
        ax = name_axs[0]
        ax.text(0.5,
                0.5,
                "Radar (observed)",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


    #
    # RADAR deviations
    #

    ax = axs[1]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    sm = ax.pcolormesh(x, y,
                       dy_radar,
                       cmap="coolwarm",
                       norm=Normalize(-2, 2),
                       shading="auto")

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    ax.set_aspect("auto")
    ax.set_ylim([0, 10])
    if y_axis:
        ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        plt.colorbar(sm, cax=legend_axs[1], label=r"Residual [$\text{dB}$]")

    if name_axs:
        ax = name_axs[1]
        ax.text(0.5,
                0.5,
                "Radar (residuals)",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 89 GHz
    #

    x = radar["d"] / 1e3
    y = dy_marss.copy()
    y[surface_mask, :] = np.nan

    ax = axs[2]
    colors = get_colors(len(band_indices_marss[0]))
    for ci, i in enumerate(band_indices_marss[0]):
        #handle = [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handle = ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")
    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[2]
        ax.set_axis_off()
        ax.legend(handles=handle, labels=["89 GHz"], loc="center")

    if name_axs:
        ax = name_axs[2]
        ax.text(0.5,
                0.5,
                "89 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 118 GHz
    #

    y = dy_ismar.copy()
    y[surface_mask, :5] = np.nan

    ax = axs[3]
    colors = get_colors(len(band_indices_ismar[0]))
    handles = []
    for ci, i in enumerate(band_indices_ismar[0]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=2)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[3]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$118 \pm 1.1$ GHz",
                r"$118 \pm 1.5$ GHz",
                r"$118 \pm 2.1$ GHz",
                r"$118 \pm 3.0$ GHz",
                r"$118 \pm 5.0$ GHz"
            ]
            ax.legend(handles=handles, labels=labels, loc="center")

    if name_axs:
        ax = name_axs[3]
        ax.text(0.5,
                0.5,
                "118 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 157 GHz
    #

    y = dy_marss.copy()
    y[surface_mask, :] = np.nan

    ax = axs[4]
    colors = get_colors(len(band_indices_marss[1]))
    handles = []
    for ci, i in enumerate(band_indices_marss[1]):
        #handles = [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[4]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$157 \pm$ GHz",
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[4]
        ax.text(0.5,
                0.5,
                "157 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 183 GHz
    #

    y = dy_marss

    ax = axs[5]
    colors = get_colors(len(band_indices_marss[2]))
    handles = []
    for ci, i in enumerate(band_indices_marss[2]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[5]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$183.248 \pm \SI{0.975}{\giga \hertz}$",
                r"$183.248 \pm \SI{3.0}{\giga \hertz}$",
                r"$183.248 \pm \SI{7.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[5]
        ax.text(0.5, 0.5,
                "183 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

def plot_residuals_hf(radar,
                      results,
                      flight,
                      surface_mask,
                      axs=None,
                      legend_axs=None,
                      name_axs=None,
                      y_axis=True,
                      title=None):
    """
    Plot retrieval residuals.

    Args:
        radar: ``xarray.Dataset`` containing the radar observations.
        results: ``xarray.Dataset`` contatining the retrieval results.
        surface_mask: Surface mask indicating which observations were performed
            over land surfaces.
        axs: List of 10 ``matplotlib.Axes`` object to use to plot the residuals.
        legend_axs: List of 10 ``matplotlib.Axes`` objects to use to display the legends
             for all plots.
        name_axs: List of 10 ``matplolib.Axes`` object to display titles for all rows in
             the plots.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    flight = flight.lower()

    yf_ismar = results["yf_ismar"].data
    y_ismar = results["y_ismar"].data
    dy_ismar = yf_ismar - y_ismar

    yf_marss = results["yf_marss"].data
    y_marss = results["y_marss"].data
    dy_marss = yf_marss - y_marss

    if flight == "b984":
        yf_radar = results["yf_hamp_radar"].data
        y_radar = results["y_hamp_radar"].data
        dy_radar = yf_radar - y_radar
    else:
        yf_radar = results["yf_cloud_sat"].data
        y_radar = results["y_cloud_sat"].data
        dy_radar = yf_radar - y_radar

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [
            1.0, 1.0, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.4
        ]
        gs = GridSpec(11, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(11)]

    band_indices_ismar = RESULT_INDICES_ISMAR[flight]
    band_indices_marss = RESULT_INDICES_MARSS

    #
    # RADAR reflectivities
    #

    norm = Normalize(-25, 20)

    x = radar["d"] / 1e3
    x_min = x.min()
    x_max = x.max()

    ax = axs[0]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    sm = ax.pcolormesh(x, y, z, cmap="magma", shading="auto", norm=norm)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)

    ax.set_ylim([0, 10])
    ax.set_xlim([x_min, x_max])

    if y_axis:
        ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    ax.set_title(title, fontsize=20, pad=10)

    if legend_axs:
        plt.colorbar(sm, cax=legend_axs[0], label=r"Reflectivity [$\text{dBZ}$]")

    if name_axs:
        ax = name_axs[0]
        ax.text(0.5,
                0.5,
                "Radar (observed)",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()


    #
    # RADAR deviations
    #

    ax = axs[1]
    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    z = radar["dbz"]
    sm = ax.pcolormesh(x, y,
                       dy_radar,
                       cmap="coolwarm",
                       norm=Normalize(-2, 2),
                       shading="auto")

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])

    ax.set_aspect("auto")
    ax.set_ylim([0, 10])
    if y_axis:
        ax.set_ylabel(r"$\text{Alt.} [\si{\kilo \meter}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        plt.colorbar(sm, cax=legend_axs[1], label=r"Residual [$\text{dB}$]")

    if name_axs:
        ax = name_axs[1]
        ax.text(0.5,
                0.5,
                "Radar (residuals)",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 89 GHz
    #

    x = radar["d"] / 1e3
    y = dy_marss.copy()
    y[surface_mask, :] = np.nan

    ax = axs[2]
    colors = get_colors(len(band_indices_marss[0]))
    for ci, i in enumerate(band_indices_marss[0]):
        #handle = [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handle = ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")
    if y_axis:
        ax.set_ylabel(r"$T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[2]
        ax.set_axis_off()
        ax.legend(handles=handle, labels=["89 GHz"], loc="center left")

    if name_axs:
        ax = name_axs[2]
        ax.text(0.5,
                0.5,
                "89 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 118 GHz
    #

    y = dy_ismar.copy()
    y[surface_mask, :4] = np.nan

    ax = axs[3]
    colors = get_colors(len(band_indices_ismar[0]))
    handles = []
    for ci, i in enumerate(band_indices_ismar[0]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci], lw=2)
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[3]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$118 \pm 1.1$ GHz",
                r"$118 \pm 1.5$ GHz",
                r"$118 \pm 2.1$ GHz",
                r"$118 \pm 3.0$ GHz",
                r"$118 \pm 5.0$ GHz"
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[3]
        ax.text(0.5,
                0.5,
                "118 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 157 GHz
    #

    y = dy_marss.copy()
    y[surface_mask, :] = np.nan

    ax = axs[4]
    colors = get_colors(len(band_indices_marss[1]))
    handles = []
    for ci, i in enumerate(band_indices_marss[1]):
        #handles = [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[4]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$157 \pm$ GHz",
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[4]
        ax.text(0.5,
                0.5,
                "157 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 183 GHz
    #

    y = dy_marss

    ax = axs[5]
    colors = get_colors(len(band_indices_marss[2]))
    handles = []
    for ci, i in enumerate(band_indices_marss[2]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[5]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$183.248 \pm \SI{0.975}{\giga \hertz}$",
                r"$183.248 \pm \SI{3.0}{\giga \hertz}$",
                r"$183.248 \pm \SI{7.0}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[5]
        ax.text(0.5, 0.5,
                "183 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 243 GHz
    #

    y = dy_ismar

    ax = axs[6]
    colors = get_colors(len(band_indices_marss[1]))
    handles = []
    for ci, i in enumerate(band_indices_marss[1]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")
    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[6]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$\SI{243.2}{\giga \hertz}$",
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[6]
        ax.text(0.5,
                0.5,
                r"$243.2 \pm \SI{2.5}{\giga \hertz}$",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 325 GHz
    #

    y = dy_ismar

    ax = axs[7]
    colors = get_colors(len(BAND_INDICES_ISMAR[2]))
    for ci, i in enumerate(band_indices_ismar[2]):
        ci += (len(BAND_INDICES_ISMAR[2]) - len(band_indices_ismar[2])) // 2
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    handles = []
    colors = get_colors(len(BAND_INDICES_ISMAR[3]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[3]):
        handles += ax.plot(x, -40 * np.ones(x.shape), c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])
    ax.set_aspect("auto")

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[7]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$325.15 \pm 1.5$ GHz",
                r"$325.15 \pm 3.5$ GHz",
                r"$325.15 \pm 9.5$ GHz"
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[7]
        ax.text(0.5,
                0.5,
                "325 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    #
    # 448 GHz
    #

    y = dy_ismar

    ax = axs[8]
    colors = get_colors(len(band_indices_ismar[3]))
    for ci, i in enumerate(band_indices_ismar[3]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    handles = []
    colors = get_colors(len(BAND_INDICES_ISMAR[3]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[3]):
        handles += ax.plot(x, -40 * np.ones(x.shape), c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    #if not band_indices_ismar[3]:
    #    ax.set_axis_off()

    if legend_axs:
        ax = legend_axs[8]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$448 \pm \SI{1.4}{\giga \hertz}$",
                r"$448 \pm \SI{3.0}{\giga \hertz}$",
                r"$448 \pm \SI{7.2}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[8]
        ax.text(0.5,
                0.5,
                "448 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 664
    #

    y = dy_ismar

    ax = axs[9]
    colors = get_colors(len(band_indices_ismar[4]))
    handles = []
    for ci, i in enumerate(band_indices_ismar[4]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        handles += ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_visible(False)
    remove_x_ticks(ax)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-15, 15])

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    if legend_axs:
        ax = legend_axs[9]
        ax.set_axis_off()
        if handles:
            labels = [
                "$664\pm\SI{4.2}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[9]
        ax.text(0.5,
                0.5,
                "664 GHz",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()
    #
    # 874
    #

    y = dy_ismar

    ax = axs[10]
    colors = get_colors(len(band_indices_ismar[5]))
    for ci, i in enumerate(band_indices_ismar[5]):
        #handles += [ax.scatter(x, y[:, i], color=colors[ci], s=5, zorder=10)]
        ax.plot(x, y[:, i], c=colors[ci])
    ax.axhline(x_min, x_max, 0.0, ls="--", c="k")
    handles = []
    colors = get_colors(len(BAND_INDICES_ISMAR[5]))
    for ci, i in enumerate(BAND_INDICES_ISMAR[5]):
        handles += ax.plot(x, -40 * np.ones(x.shape), c=colors[ci])

    add_surface_shading(ax, x, surface_mask)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    if y_axis:
        ax.set_ylabel(r"$\Delta T_B\ [\si{\kelvin}]$")
    else:
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r"Along track distance [$\si{\kilo \meter}$]")
    ax.set_ylim([-15, 15])

    if legend_axs:
        ax = legend_axs[10]
        ax.set_axis_off()
        if handles:
            labels = [
                r"$874.4 \pm \SI{6.0}{\giga \hertz}$"
            ]
            ax.legend(handles=handles, labels=labels, loc="center left")

    if name_axs:
        ax = name_axs[10]
        ax.text(0.5,
                0.5,
                r"$874.4 \pm \SI{6.0}{\giga \hertz}$",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center")
        ax.set_axis_off()

    plt.tight_layout()
