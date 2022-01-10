"""
====================
joint_flight.results
====================

This module provides functions to display and analyse the retrieval
results from the combind radar/radiometer retrievals.
"""
import glob
import os
from pathlib import Path
import re

import numpy as np
import scipy as sp
from netCDF4 import Dataset
import joint_flight
import seaborn as sns
from mcrf.psds import D14NDmIce
import scipy
from scipy.signal import convolve
from scipy.stats import linregress

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import pandas as pd
import xarray as xr

from joint_flight.utils import remove_x_ticks, remove_y_ticks, PARTICLE_NAMES
from joint_flight.ssdb import load_habit_meta_data

sns.reset_orig()


def iwc(n0, dm):
    """
    Calculate ice-water content from D14 PSD moments.

    Args:
        n0: Array containing the :math:`N_0^*` values of the PSD.
        dm: Array containing the :math:`D_m` values of the PSD.

    Return:
        Array containing the corresponding ice water content in
        :math:`kg/m^3`.
    """
    return np.pi * 917.0 * dm ** 4 * n0 / 4 ** 4


def rwc(n0, dm):
    """
    Calculate rain-water content from D14 PSD moments.

    Args:
        n0: Array containing the :math:`N_0^*` values of the PSD.
        dm: Array containing the :math:`D_m` values of the PSD.

    Return:
        Array containing the corresponding rain water content in
        :math:`kg/m^3`.
    """
    return np.pi * 1000.0 * dm ** 4 * n0 / 4 ** 4


def get_results(flight, config="", group="All quantities"):
    """
    This function loads the retrieval results from a given flight
    Results are automatically augmented with the retrieved iwc.

    Args:
        flight: The flight name, i.e. b984, c159 or c161
        config: Name of retrieval configuration that defines from which sub-folder
            (if any) the data is loaded.
        group: The NetCDF4 group containing the results.
    """
    flight = flight.lower()
    path = os.path.join(joint_flight.PATH, "data", "old")
    if config != "":
        path = os.path.join(path, config)
    pattern = re.compile(f"output_{flight}_([\w-]*).nc")

    results = {}

    psd = D14NDmIce()

    for f in glob.glob(os.path.join(path, "*")):
        match = re.match(pattern, os.path.basename(f))
        if match is None:
            continue
        shape = match.group(1)
        data = xr.load_dataset(os.path.join(path, f), group=group)
        dm = data["ice_dm"]
        n0 = data["ice_n0"]
        psd.mass_weighted_diameter = dm
        psd.intercept_parameter = n0
        wc = psd.get_mass_density()

        k = np.ones((5, 1)) / 5.0
        wc_s = convolve(wc, k, mode="same")

        nd = psd.get_moment(0)
        nd.data[wc < 5e-6] = 0.0
        data["ice_water_content"] = (dm.dims, wc.data)
        data["ice_water_content_smooth"] = (dm.dims, wc_s.data)
        data["number_density"] = (dm.dims, nd.data)
        results[shape] = data
    return results


def get_distance_mask(radar, nevzorov, d_max=500.0):

    x = radar["x"].data
    y = radar["y"].data
    d = 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])
    z = 0.25 * (y[:-1, :-1] + y[:-1, 1:] + y[1:, :-1] + y[1:, 1:])

    z_n = nevzorov["altitude"].data
    d_n = nevzorov["d"].data

    c = np.expand_dims(np.stack([z, d], axis=2), 2)
    c_n = np.stack([z_n, d_n], 1)[np.newaxis, np.newaxis]
    d_c = np.min(np.sqrt(np.sum((c - c_n) ** 2, axis=-1)), axis=-1)

    mask = d_c < d_max
    return mask


def get_domain_mask(radar, d_start, d_end, nevzorov):

    x = radar["x"].data
    d = 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])
    mask = (d >= d_start) * (d < d_end)
    return mask


def match_bulk_properties(results, mask, radar, nevzorov):

    x = radar["x"].data
    y = radar["y"].data
    d = 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])
    z = 0.25 * (y[:-1, :-1] + y[:-1, 1:] + y[1:, :-1] + y[1:, 1:])

    shapes = list(results.keys())
    ice_water_content = []
    number_density = []
    for s in results:
        r = results[s]
        ice_water_content.append(r["ice_water_content"].data[mask])
        number_density.append(r["number_density"].data[mask])
    ice_water_content = np.stack(ice_water_content)
    number_density = np.stack(number_density)

    data = {
        "altitude": (("samples",), z[mask]),
        "d": (("samples",), d[mask]),
        "shapes": (("shapes",), shapes),
        "ice_water_content": (
            (
                "shapes",
                "samples",
            ),
            ice_water_content,
        ),
        "number_density": (
            (
                "shapes",
                "samples",
            ),
            number_density,
        ),
    }

    return xr.Dataset(data)


COLORS = {
    0: "darkred",
    1: "navy",
    2: "darkgreen",
    3: "darkorange",
    4: "crimson",
    5: "gold",
}


def get_cmap(index):
    return sns.color_palette(f"light:{COLORS[index]}", as_cmap=True)


def plot_residual_distributions(ax, radar, results, flight, shapes=None):
    """
    Plot distributions of retrieval residuals for all habits for
    a given flight.

    Args:
        radar: 'xarray.Dataset' containing the radar observations for the flight.
        results: 'xarray.Dataset' containing the results for the flight.
        flight: The name of the flight as string.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)
    if shapes is None:
        shapes = ["LargePlateAggregate", "LargeColumnAggregate", "8-ColumnAggregate"]

    dys = []
    sources = []
    habits = []

    for s in shapes:
        rs = results[s]
        iwc = rs["ice_water_content"].data
        y = radar["y"] / 1e3
        dy = np.diff(y, axis=-1) * 1e3
        dy = 0.5 * (dy[1:] + dy[:-1])
        #iwp = np.sum(dy * iwc, axis=-1)
        #indices = iwp > 1e-1
        #rs = rs[{"profile": indices}]

        if "yf_cloud_sat" in rs.variables:
            name = "cloud_sat"
            y = rs[f"y_{name}"].data
            y_f = rs[f"yf_{name}"].data
            altitude = radar["height"].data
            mask = (altitude > 2e3) * (altitude < 9e3) * (y > -20)
            dy_radar = (y_f[mask] - y[mask]).ravel()
        else:
            name = "hamp_radar"
            y = rs[f"y_{name}"].data
            y_f = rs[f"yf_{name}"].data
            altitude = radar["height"].data
            print(altitude.shape, y.shape, rs[f"y_{name}"].data.shape)
            mask = ((altitude > 2e3) * (altitude < 10e3)).reshape(1, -1) * (y > -20)
            dy_radar = (y_f[mask] - y[mask]).ravel()
            print(mask.sum())

        source = ["Radar"] * dy_radar.size

        dy_183 = (rs["yf_marss"].data[:, 2:] - rs["y_marss"].data[:, 2:]).ravel()
        source += [
            r"$183.248 \pm \SI{1}{\giga \hertz}$",
            r"$183.248 \pm \SI{3}{\giga \hertz}$",
            r"$183.248 \pm \SI{7}{\giga \hertz}$",
        ] * (dy_183.size // 3)

        dy_243 = (rs["yf_ismar"].data[:, 5:6] - rs["y_ismar"].data[:, 5:6]).ravel()
        source += [r"$\SI{243.2}{\giga \hertz}$"] * (dy_243.size)

        if flight == "b984":
            dy_325 = (rs["yf_ismar"].data[:, 6:9] - rs["y_ismar"].data[:, 6:9]).ravel(
                order="f"
            )
        else:
            dy_325 = (rs["yf_ismar"].data[:, 6:7] - rs["y_ismar"].data[:, 6:7]).ravel(
                order="f"
            )
            dy_325 = np.concatenate(
                [
                    np.array([np.nan] * dy_325.size),
                    dy_325,
                    np.array([np.nan] * dy_325.size),
                ]
            )
        source += (
            [r"$325.15 \pm \SI{1.5}{\giga \hertz}$"] * (dy_325.size // 3)
            + [r"$325.15 \pm \SI{3.5}{\giga \hertz}$"] * (dy_325.size // 3)
            + [r"$325.15 \pm \SI{9.5}{\giga \hertz}$"] * (dy_325.size // 3)
        )

        if flight == "b984":
            dy_448 = np.array([np.nan] * dy_325.size)
        else:
            dy_448 = (
                rs["yf_ismar"].data[:, 7:10] - rs["y_ismar"].data[:, 7:10]
            ).ravel()
        source += [
            r"$448 \pm \SI{1.4}{\giga \hertz}$",
            r"$448 \pm \SI{3.0}{\giga \hertz}$",
            r"$448 \pm \SI{7.2}{\giga \hertz}$",
        ] * (dy_448.size // 3)

        if flight == "b984":
            dy_664 = (
                rs["yf_ismar"].data[:, 9:10] - rs["y_ismar"].data[:, 9:10]
            ).ravel()
        else:
            dy_664 = (
                rs["yf_ismar"].data[:, 10:11] - rs["y_ismar"].data[:, 10:11]
            ).ravel()
        source += [r"$\SI{664}{\giga \hertz}$"] * dy_664.size

        if flight == "b984":
            dy_874 = np.array([np.nan] * dy_664.size)
        else:
            dy_874 = (
                rs["yf_ismar"].data[:, 11:12] - rs["y_ismar"].data[:, 11:12]
            ).ravel()

        dy = np.concatenate([dy_radar, dy_183, dy_243, dy_325, dy_448, dy_664, dy_874])
        source += [r"$874.4 \pm \SI{6.0}{\giga \hertz}$ V"] * dy_874.size

        dys.append(dy)
        sources += source
        habits += [s] * len(source)

    dys = np.concatenate(dys)
    data = {"Residual": dys, "Source": sources, "Habit": habits}
    data = pd.DataFrame(data)

    sns.boxplot(
        x="Source",
        y="Residual",
        hue="Habit",
        data=data,
        fliersize=0.5,
        linewidth=1,
        whis=2.0,
        ax=ax,
    )
    return ax


def plot_results(
    radar,
    results,
    atmosphere,
    shapes=None,
    axs=None,
    legends=None,
    y_axis=True,
    names=None,
):
    """
    Plot bulk ice water path and content for range of shapes.
    """

    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if shapes is None:
        shapes = ["LargePlateAggregate", "8-ColumnAggregate", "LargeColumnAggregate"]

    if axs is None:
        figure = plt.figure(figsize=(10, 10))
        height_ratios = [0.5, 1.0, 0.5, 1.0]
        gs = GridSpec(4, 1, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(4)]

    d = radar["d"] / 1e3
    d_min = d.min()
    d_max = d.max()

    #
    # Ice water path
    #

    x = radar["x"] / 1e3
    x_min = x.min()
    x_max = x.max()

    y = radar["y"] / 1e3

    ax = axs[0]

    dy = np.diff(y, axis=-1) * 1e3
    dy = 0.5 * (dy[1:] + dy[:-1])

    handles = []
    labels = []

    for i, s in enumerate(shapes):
        r = results[s]
        iwc = r["ice_water_content_smooth"].data
        iwp = np.sum(dy * iwc, axis=-1)
        # handles += [ax.scatter(d, iwp, c=f"C{i}", lw=1.5, marker="o", s=2, alpha=0.3)]
        handles += ax.plot(d, iwp, c=f"C{i}", lw=1.5)
        labels.append(s)
    # ax.set_yscale("log")
    ax.set_ylim([0, 3])

    ax.set_xlim([d_min, d_max])
    remove_x_ticks(ax)

    if y_axis:
        ax.set_ylabel(r"IWP [$\si{\kilo \gram \per \meter \squared}$]")
        ax.spines["left"].set_position(("outward", 10))
    else:
        ax.spines["left"].set_visible(False)
        remove_y_ticks(ax)

    if legends:
        ax = legends[0]
        ax.set_axis_off()
        ax.legend(handles=handles, labels=labels, loc="center left")

    ax.set_xlim([x_min, x_max])
    ax.spines["bottom"].set_visible(False)
    remove_x_ticks(ax)

    if names:
        names[0].set_axis_off()

    #
    # Ice water content
    #

    for i, s in enumerate(shapes):
        ax = axs[i + 1]
        z = radar["dbz"]
        # ax.pcolormesh(x, y, np.pad(z, ((0, 1), (0, 1))), cmap="Greys", shading="gouraud")

        norm = LogNorm(5e-6, 5e-3)
        levels = np.logspace(np.log10(1e-5), np.log10(1e-2), 7)[:-1]
        # norm = LogNorm(1e-3, 1e1)
        # levels = np.logspace(np.log10(1e-3),
        #                     np.log10(1e1), 7)[:-1]
        r = results[s]
        iwc = r["ice_water_content_smooth"].data
        cmap = get_cmap(i)
        iwc_r = iwc / iwc.max()
        # m = ax.contourf(x[:-1, :-1], y[:-1, :-1], iwc,
        #                levels=levels,
        #                cmap="magma",
        #                linewidths=1.0,
        #                norm=norm)
        m = ax.pcolormesh(
            x[:-1, :-1], y[:-1, :-1], np.maximum(iwc, 1e-6), cmap="magma", norm=norm
        )
        # ax.contour(x[:-1, :-1], y[:-1, :-1], iwc_r,
        #           levels=levels,
        #           cmap="magma",
        #           linewidths=1.0,
        #           norm=norm)
        t = atmosphere["temperature"].data
        cs = ax.contour(x[:-1, :-1], y[:-1, :-1], t, levels=[273.15], colors="white")
        plt.clabel(cs, fontsize=10, fmt=r"%1.2f$\si{\kelvin}$")
        ax.set_xlim([d_min, d_max])
        remove_x_ticks(ax)
        ax.set_ylim([0, 8])
        ax.set_yticks(np.arange(0, 10, 2))

        if y_axis:
            ax.set_ylabel(r"Altitude [$\si{\kilo \meter}$]")
            ax.spines["left"].set_position(("outward", 10))
        else:
            remove_y_ticks(ax)
            ax.spines["left"].set_visible(False)

        ax.set_xlim([x_min, x_max])
        if i < len(shapes) - 1:
            ax.spines["bottom"].set_visible(False)
            remove_x_ticks(ax)
        else:
            ax.set_xlabel(r"Along track distance [$\si{\kilo \meter}$]")
            ax.set_xlim([x_min, x_max])
            ax.spines["bottom"].set_position(("outward", 10))

        if names:
            ax = names[i + 1]
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                PARTICLE_NAMES[s],
                fontsize=10,
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center",
            )

    if legends:
        ax = legends[1]
        cb = plt.colorbar(
            m, cax=ax, label=r"IWC [$\si{\kilo \gram \per \meter \cubed}$]"
        )
        for c in cb.ax.get_children():
            if isinstance(c, LineCollection):
                c.set_linewidths(8)


def plot_bulk_properties(
    nevzorov,
    results,
    radar,
    mask,
    y_axis=True,
    shapes=None,
    axs=None,
    legends=None,
    cbs=None,
):
    """
    Plot in-situ and retrieved bulk properties

    Args:
        nevzorov: 'xarray.Dataset' containing the measurements from the Nevzorov probe.
        results: Dictionary mappnig shape names to 'xarray.Dataset's containing the
             respective retrieval results.
        radar: 'xarray.Dataset' containing the radar observations.
        mask: Binary mask defining the matched observations to the in-situ measurements.
        shapes: List of habit names.
        axs: If provided, these 'Axes' instances will be used to plot the results.
        legends: If provided, these 'Axes' instances will be used to plot the plot
            legends.
        cbs: If provided, these axes are used to plot the colorbars.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if shapes is None:
        shapes = ["LargePlateAggregate", "8-ColumnAggregate", "LargeColumnAggregate"]

    if axs is None:
        figure = plt.figure(figsize=(10, 10))
        height_ratios = [
            0.5,
            1.0,
        ]
        gs = GridSpec(2, 1, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(2)]

    #
    # Radar observations and mask
    #
    ax = axs[0]

    x = radar["x"] / 1e3
    y = radar["y"] / 1e3
    x_c = 0.5 * (x[0, 1:] + x[0, :-1])
    y_c = 0.5 * (y[1:] + y[:-1])
    z = radar["dbz"]
    norm = Normalize(-30, 20)
    m = ax.pcolormesh(x, y, z, cmap="Greys", norm=norm)

    cmap = matplotlib.cm.get_cmap("Blues")
    cmap.set_bad((0, 0, 0, 0))
    z = mask.copy().astype(np.float32)
    z[z < 1.0] = np.nan
    ax.pcolormesh(
        x,
        y,
        z,
        cmap=cmap,
        alpha=0.4,
        norm=Normalize(0, 1),
        rasterized=True,
        antialiased=True,
        linewidths=0,
    )

    d_n = nevzorov["d"] / 1e3
    z_n = nevzorov["altitude"] / 1e3

    handles = []
    handles += ax.plot(d_n, z_n, ls="--")

    ax.set_xlabel("Along-track distance [$\si{\kilo \meter}$]")
    ax.set_ylim([0, 10])

    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if y_axis:
        ax.set_ylabel("Altitude [$\si{km}$]")
    else:
        remove_y_ticks(ax)
        ax.spines["left"].set_visible(False)

    if cbs:
        ax = cbs[0]
        plt.colorbar(m, cax=ax, label=r"Radar reflectivity [$\si{\deci \bel Z}$]")

    if legends:
        labels = ["Flight path", "Matched retrieval results"]
        handles += [Patch(facecolor="navy", alpha=0.4)]
        ax = legends[0]
        ax.set_axis_off()
        ax.legend(labels=labels, handles=handles, loc="upper center")

    #
    # IWC
    #

    ax = axs[1]

    # Background: in-situ data
    alt_bins = np.linspace(2, 8, 13)
    x = np.logspace(-6, -3, 21)
    y = alt_bins
    img, _, _ = np.histogram2d(
        nevzorov["iwc"].data / 1e3, nevzorov["altitude"].data / 1e3, bins=(x, y)
    )
    img = img / img.sum(0, keepdims=True)
    x_c = 0.5 * (x[1:] + x[:-1])
    y_c = 0.5 * (y[1:] + y[:-1])
    norm = Normalize(0, 0.2)
    #m = ax.pcolormesh(x, y, img.T, cmap="Greys", norm=norm)

    #
    # Plot mean and percentiles.
    #

    percs = []
    means = []
    qs = [5, 95]

    for i in range(img.shape[1] - 1, -1, -1):
        l = alt_bins[i]
        u = alt_bins[i + 1]

        alt = nevzorov["altitude"].data / 1e3
        indices = (l <= alt) * (alt < u)
        iwc_l = nevzorov["iwc"].data[indices]
        percs.append(np.percentile(iwc_l, qs))
        means.append(iwc_l.mean())

    percs = [percs[0]] + percs + [percs[-1]]
    means = [means[0]] + means + [means[-1]]
    levels = [alt_bins[0]] + list(0.5 * (alt_bins[1:] + alt_bins[:-1])) + [alt_bins[-1]]
    levels = np.array(levels)[::-1]
    percs = np.stack(percs, axis=0) / 1e3

    handles_1 = []

    handles_1.append(
        ax.fill_betweenx(levels, percs[:, 0], percs[:, 1], facecolor="grey", alpha=0.5, zorder=-10)
    )
    handles_1 += ax.plot(np.array(means) / 1e3, levels, c="k", zorder=-10)

    handles_1 = handles_1[::-1]
    labels_1 = [
        "In situ (mean)",
        "In situ (5th to 95th perc.)"
    ]


    alt_bins = np.linspace(2, 8, 13)

        #ax.fill_between(
        #    x_c, z_b * 0.5 + i * 0.5 + 2, i *0.5 + 2,
        #    edgecolor="w",
        #    facecolor="grey"
        #)

    # Boxes

    d_alt = np.diff(alt_bins).mean()
    width = d_alt / (len(shapes) + 2)
    offsets = np.linspace(-0.5 * d_alt + width, 0.5 * d_alt - width, len(shapes))
    offsets = offsets[::-1]

    handles = []
    labels = []

    for i, s in enumerate(shapes):
        iwc = results.sel(shapes=s)["ice_water_content"].data
        z = results.sel(shapes=s)["altitude"] / 1e3

        data = []
        for j in range(len(alt_bins) - 1):
            z_l = alt_bins[j]
            z_u = alt_bins[j + 1]

            inds = (z > z_l) * (z <= z_u)
            data.append(iwc[inds])
            #y, _, = np.histogram(iwc[inds], bins=x)
            #y = 0.9 * 0.5 * y / y.max()
            #ax.plot(x_c, y + z_l, f"C{i}")


        pos = 0.5 * (alt_bins[1:] + alt_bins[:-1])
        offset = offsets[i]
        props = {"color": "k", "facecolor": f"C{i}", "linewidth": 0, "alpha": 1.0}
        handles += [
            ax.boxplot(
                data,
                vert=False,
                positions=pos + offset,
                sym="",
                notch=False,
                widths=width,
                manage_ticks=False,
                boxprops=props,
                medianprops={"color": f"C{i}", "linewidth": 2},
                patch_artist=True,
                zorder=0
            )["boxes"][0]
        ]
        labels += [PARTICLE_NAMES[s]]
        means = [d.mean() for d in data]
        ax.scatter(means, pos + offset, c=f"C{i}", marker="^", zorder=20, edgecolor="k")

    ax.set_xlim([1e-6, 1e-3])
    ax.set_ylim([3, 8])
    ax.set_xscale("log")

    ax.set_yticks(alt_bins)
    ax.set_xlabel("IWC [$\si{\kg \per \meter \cubed}$]")

    ax.yaxis.grid(True)

    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    if y_axis:
        ax.set_ylabel("Altitude [$\si{\kilo \meter}$]")
    else:
        ax.spines["left"].set_visible(False)
        remove_y_ticks(ax)

    ax.xaxis.grid(False)

    # Color bar
    if cbs:
        ax = cbs[1]
        #plt.colorbar(m, cax=ax, label="Frequency of Nevzorov-probe measurements")

    # Legend

    if legends:
        ax = legends[1]
        ax.set_axis_off()
        ax.legend(
            title="In situ and retrieved IWC",
            handles=handles + handles_1,
            labels=labels + labels_1,
            loc="center"
        )


def calculate_iwp(
    nevzorov,
    results,
    shapes,
    bins=None,
    n_samples=100
):
    """
    Calculate column-integrated ice water path.

    Args:
        nevzorov: 'xarray.Dataset' containing the measurements from the Nevzorov probe.
        results: Dictionary mappnig shape names to 'xarray.Dataset's containing the
             respective retrieval results.
    """
    if bins is None:
        bins = np.linspace(2.0e3, 8e3, 17)

    iwp = {"nevzorov": np.zeros(n_samples)}
    for s in shapes:
        iwp[s] = np.zeros(n_samples)

    n_particles = []
    for i in range(bins.size - 1):
        l = bins[i]
        u = bins[i + 1]

        iwc = nevzorov["iwc"].data
        inds = (nevzorov["altitude"].data >= l) * (nevzorov["altitude"].data < u)
        indcs = inds * (iwc > 0)
        if inds.sum() == 0:
            continue
        iwc = np.random.choice(iwc[inds], size=n_samples)
        iwp["nevzorov"] += iwc / 1e3 * (u - l)


        inds = (results["altitude"].data >= l) * (results["altitude"].data < u)
        n_particles += [inds.sum()]
        for i, s in enumerate(shapes):
            iwc = results["ice_water_content"].data[i]
            if inds.sum() == 0:
                continue
            iwc = np.random.choice(iwc[inds], size=n_samples)
            iwp[s] += iwc * (u - l)

    return iwp



def calculate_psds(results, mask, radar, sizes=None):
    """
    Calculate PSDs from bulk properties.

    Args:
        results: 'xarray.Dataset' containing the retrieval results matched to in-situ
            measurement for all shapes.
        sizes: A size grid at which to evaluate the PSDs.
    """
    if sizes is None:
        sizes = np.logspace(-5, -2, 41)

    psd = D14NDmIce()

    x = radar["x"].data
    y = radar["y"].data
    d = 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])
    z = 0.25 * (y[:-1, :-1] + y[:-1, 1:] + y[1:, :-1] + y[1:, 1:])

    results_new = {}
    for s in results:
        try:
            meta_data = load_habit_meta_data(s)
        except FileNotFoundError as e:
            continue

        x = meta_data["d_eq"].interp(d_max=sizes).data

        r = results[s]
        dm = r["ice_dm"].data[mask]
        n0 = r["ice_n0"].data[mask]

        psd.mass_weighted_diameter = dm
        psd.intercept_parameter = n0
        y = psd.evaluate(x)

        new = {
            "d_max": ("d_max", sizes),
            "psd": (("samples", "d_max"), y.data),
            "altitude": (("samples",), z[mask]),
            "d": (("samples",), d[mask]),
        }
        results_new[s] = xr.Dataset(new)
    return results_new


def plot_psds(
    psds, psds_r, axs=None, legends=None, names=None, y_axis=True, shapes=None
):
    """
    Args:
        radar: 'xarray.Dataset' containing the radar observations which will
            be plotted in the for row of panels.
        psds: 'xarray.Dataset' containing the in-situ-measured PSDs.
        psds_r: 'xarray.Dataset' containing the retrieved PSDs matched to the
            in-situe measure PSDs.
        axs: List of axes to use to plot the observations.
        legends: List of axes to use to plot the legends for each band.
        names: List of axes to use to plot the names for each band.
        y_axis: Whether or not to draw a y axis on the plot.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if axs is None:
        figure = plt.figure(figsize=(10, 20))
        height_ratios = [1.0] * 5
        gs = GridSpec(10, 2, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(9)]

    if shapes is None:
        shapes = ["LargePlateAggregate", "LargeColumnAggregate", "8-ColumnAggregate"]

    #
    # Loop over altitudes.
    #

    n_samples = 100

    for i in range(6):
        ax = axs[i]
        alt_max = 8e3 - i * 1e3
        alt_min = alt_max - 1e3

        indices = (psds.altitude >= alt_min) * (psds.altitude < alt_max)
        data = psds[{"time": indices}]
        x = data["diameter"]
        y = data["dndd"]
        y_mean = y.mean("time")

        x_min = x.min()
        x_max = x.max()

        handles = []
        labels = ["In situ (mean)"]

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([1e4, 1e12])
        handles += ax.plot(x, y_mean, c="k", lw=2)

        if psds_r is not None:
            for j, s in enumerate(shapes):
                r = psds_r[s]
                indices = (r["altitude"].data >= alt_min) * (
                    r["altitude"].data < alt_max
                )
                x = psds_r[s]["d_max"].data
                psd = psds_r[s]["psd"].data[indices]
                indices = np.arange(indices.sum())
                handles += ax.plot(x, np.nanmean(psd, axis=0), c=f"C{j}", lw=2)
                labels += [PARTICLE_NAMES[s]]

        ax.spines["left"].set_position(("outward", 10))

        if i < 5:
            ax.spines["bottom"].set_visible(False)
            remove_x_ticks(ax)
        else:
            ax.spines["bottom"].set_position(("outward", 10))
            ax.set_xlim([x_min, x_max])
            ax.set_xlabel(r"$D_\text{MAX}$ [$\si{\meter}$]")

        ax.set_yscale("log")
        ax.set_xscale("log")

        if y_axis:
            ax.set_ylabel(r"$\frac{dN}{dD_\text{max}}\ [\si{\per \meter \tothe{4}}]$")
        else:
            ax.yaxis.set_ticks_position("none")
            for l in ax.yaxis.get_ticklabels():
                l.set_visible(False)
            ax.spines["left"].set_visible(False)

        if names:
            ax = names[i]
            ax.text(
                0.5,
                0.5,
                rf"${(alt_min/1e3):1.0f} - \SI{{{(alt_max/1e3):1.0f}}}{{\kilo \meter}}$",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center",
            )
            ax.set_axis_off()

        if legends:
            ax = legends[i]
            if i == 2:
                ax.legend(handles=handles, labels=labels, loc="center left")
            ax.set_axis_off()


def plot_psd_mass(
    psds,
    nevzorov,
    axs=None,
    legends=None,
    names=None,
    y_axis=True,
    shapes=None,
    cbs=None,
):
    """
    Plot bulk IWC profiles derived from in-situ-measured PSDs together
    with that of Nevzorov-measured IWC.
    """
    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if shapes is None:
        shapes = ["LargePlateAggregate", "8-ColumnAggregate", "LargeColumnAggregate"]

    if axs is None:
        figure = plt.figure(figsize=(10, 10))
        height_ratios = [
            1.0,
        ]
        gs = GridSpec(1, 1, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(1)]

    #
    # In-situ and derived IWC.
    #

    ax = axs[0]

    # Background: in-situ data
    alt_bins = np.linspace(2, 8, 7)
    x = np.logspace(-6, -3, 41)
    y = alt_bins
    img, _, _ = np.histogram2d(
        nevzorov["iwc"].data / 1e3, nevzorov["altitude"].data / 1e3, bins=(x, y)
    )
    img = img / img.sum(0, keepdims=True)
    x_c = 0.5 * (x[1:] + x[:-1])
    y_c = 0.5 * (y[1:] + y[:-1])
    norm = Normalize(0, 0.2)
    #m = ax.pcolormesh(x, y, img.T, cmap="Greys", norm=norm)

    #
    # Plot mean and percentiles.
    #
    alt_bins = np.linspace(2, 8, 7)

    percs = []
    means = []
    qs = [5, 95]

    for i in range(alt_bins.size - 1):
        l = alt_bins[i]
        u = alt_bins[i + 1]

        alt = nevzorov["altitude"].data / 1e3
        indices = (l <= alt) * (alt < u)
        iwc_l = nevzorov["iwc"].data[indices]
        if indices.sum() == 0:
            percs.append(np.nan * np.ones(len(qs)))
            means.append(np.nan * np.ones(len(qs)))
            continue
        percs.append(np.percentile(iwc_l, qs))
        means.append(iwc_l.mean())

    percs = [percs[0]] + percs + [percs[-1]]
    means = [means[0]] + means + [means[-1]]
    levels = [alt_bins[0]] + list(0.5 * (alt_bins[1:] + alt_bins[:-1])) + [alt_bins[-1]]
    levels = np.array(levels)
    percs = np.stack(percs, axis=0) / 1e3

    handles_1 = []

    handles_1.append(
        ax.fill_betweenx(levels, percs[:, 0], percs[:, 1], facecolor="grey", alpha=0.5, zorder=-10)
    )
    handles_1 += ax.plot(np.array(means) / 1e3, levels, c="k", zorder=-10)

    handles_1 = handles_1[::-1]
    labels_1 = [
        "Nevzorov (mean)",
        "Nevzorov (5th to 95th perc.)"
    ]

    alt_bins = np.linspace(2, 8, 7)
    # Boxes
    handles = []
    labels = []
    d_alt = np.diff(alt_bins).mean()
    width = d_alt / (len(shapes) + 2)
    offsets = np.linspace(-0.5 * d_alt + width, 0.5 * d_alt - width, len(shapes))[::-1]

    for i, s in enumerate(shapes):
        try:
            meta_data = load_habit_meta_data(s)
        except FileNotFoundError:
            continue

        mass = meta_data["mass"].interp(d_max=psds.diameter)
        mass = mass.fillna(0.0)

        iwc = (mass * psds.n).sum(dim="diameter")
        z = psds["altitude"] / 1e3

        data = []
        for j in range(len(alt_bins) - 1):
            z_l = alt_bins[j]
            z_u = alt_bins[j + 1]
            inds = (z > z_l) * (z <= z_u)
            data.append(iwc[inds])

        pos = 0.5 * (alt_bins[1:] + alt_bins[:-1])
        offset = offsets[i]
        props = {"color": "k", "facecolor": f"C{i:02}", "linewidth": 1, "alpha": 0.75}
        handles += [
            ax.boxplot(
                data,
                vert=False,
                positions=pos + offset,
                sym="",
                notch=False,
                widths=width,
                manage_ticks=False,
                boxprops=props,
                medianprops={"color": f"k", "linewidth": 1},
                patch_artist=True,
            )["boxes"][0]
        ]
        labels += [PARTICLE_NAMES[s]]

    #
    # Axes
    #

    ax.set_xlim([1e-6, 1e-3])
    ax.set_ylim([2, 8])
    ax.set_xscale("log")

    ax.set_yticks(alt_bins)
    ax.yaxis.grid(True)
    ax.set_xlabel("IWC [$\si{\kg \per \meter \cubed}$]")

    if y_axis:
        ax.set_ylabel("Altitude [$\si{\kilo \meter}$]")
    else:
        ax.yaxis.set_ticks_position("none")
        for l in ax.yaxis.get_ticklabels():
            l.set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.spines["left"].set_visible(False)

    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Color bar
    if cbs:
        ax = cbs[0]
        #plt.colorbar(m, cax=ax, label="Frequency of Nevzorov-probe measurements")

    # Legend
    if legends:
        ax = legends[0]
        ax.set_axis_off()
        ax.legend(
            title="Nevzorov and PSD-derived IWC",
            handles=handles + handles_1,
            labels=labels + labels_1,
            loc="upper left"
        )


def scatter_residuals(
        radar, results, flight, shapes=None, axs=None, y_axis=True, names=None,
        channel="243"
):
    """
    Create scatter plots of retrieval residuals.

    Args:
        radar: 'xarray.Datset' containing the radar observations for the
            flight.
        results: 'xarray.Dataset' containing the retrieval results for the
             flight.
        flight: Name of the flight.
        shapes: List of the shapes to plot.
        axs: Array of axis object to use for plotting.
    """
    INDICES = {
        "247": (5, 5),
        "325": (7, 6),
        "664": (-1, -2)
    }

    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if shapes is None:
        shapes = ["LargePlateAggregate", "8-ColumnAggregate", "LargeColumnAggregate"]

    if axs is None:
        figure = plt.figure(figsize=(10, 10))
        height_ratios = [0.5, 1.0, 0.5, 1.0]
        gs = GridSpec(4, 1, height_ratios=height_ratios)
        axs = [figure.add_subplot(gs[i, 0]) for i in range(4)]

    d = radar["d"] / 1e3
    d_min = d.min()
    d_max = d.max()

    #
    # Ice water path
    #

    x = radar["x"] / 1e3
    x_min = x.min()
    x_max = x.max()
    y = radar["y"] / 1e3
    dy = np.diff(y, axis=-1) * 1e3
    dy = 0.5 * (dy[1:] + dy[:-1])

    handles = []
    labels = []

    residuals = []
    iwps = []
    shps = []

    #
    # 325
    #

    for i, s in enumerate(shapes):

        rs = results[s]

        # IWP
        iwc = rs["ice_water_content"].data
        iwp = np.sum(dy * iwc, axis=-1)

        # 325 GHz
        i_b, i_c = INDICES[channel]
        if flight == "b984":
            dy_325 = (rs["yf_ismar"].data[:, i_b] - rs["y_ismar"].data[:, i_b]).ravel()
        else:
            dy_325 = (rs["yf_ismar"].data[:, i_c] - rs["y_ismar"].data[:, i_c]).ravel()
        residuals += [dy_325]
        iwps += [iwp]
        shps += [s] * iwp.size

        ax = axs[i]
        ax.scatter(iwp, dy_325, label=s, s=6, alpha=0.3, c=f"C{i}", edgecolor="none")

        x = np.log10(iwp)
        y = dy_325
        slope, intercept, r, p, se = linregress(x, y)
        x = np.linspace(-2, 2, 101)
        ax.text(x=3e0, y=-10, s=f"r: {r:0.3f} \n p: {p:0.3f}", ha="right")
        y_hat = slope * x + intercept
        ax.set_xscale("log")
        ax.axhline(0, c="k", ls="--")
        ax.plot(10 ** x, y_hat, ls="-", c="grey")

        ax.set_ylim([-10, 10])
        ax.set_xlim([1e-2, 3])

        if y_axis:
            ax.spines["left"].set_position(("outward", 10))
            ax.set_ylabel(r"$\Delta y$ [$\si{\kelvin}$]")
        else:
            ax.spines["left"].set_visible(False)
            remove_y_ticks(ax)

        if i < len(shapes) - 1:
            ax.spines["bottom"].set_visible(False)
            remove_x_ticks(ax)
        else:
            ax.set_xlabel("IWP [$\si{\kilo \gram \per \meter \cubed}]$")
            ax.spines["bottom"].set_position(("outward", 10))

        if names:
            ax = names[i]
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                PARTICLE_NAMES[s],
                fontsize=10,
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
                weight="bold",
                ha="center",
                va="center",
            )

def scatter_residuals_b984(
        radar, results, shapes=None
):
    """
    Create scatter plots of retrieval residuals for flight B984

    Args:
        radar: 'xarray.Datset' containing the radar observations for the
            flight.
        results: 'xarray.Dataset' containing the retrieval results for the
             flight.
        flight: Name of the flight.
        shapes: List of the shapes to plot.
        axs: Array of axis object to use for plotting.
    """

    style_file = Path(__file__).parent / ".." / "misc" / "matplotlib_style.rc"
    plt.style.use(style_file)

    if shapes is None:
        shapes = ["LargePlateAggregate", "8-ColumnAggregate", "LargeColumnAggregate"]

    f, axs = plt.subplots(
        2, 6,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [0.8] + [1.0] * 5}
    )
    names = axs[:, :1]
    axs = axs[:, 1:]

    d = radar["d"] / 1e3
    d_min = d.min()
    d_max = d.max()

    #
    # Ice water path
    #

    x = radar["x"] / 1e3
    x_min = x.min()
    x_max = x.max()
    y = radar["y"] / 1e3
    dy = np.diff(y, axis=-1) * 1e3
    dy = 0.5 * (dy[1:] + dy[:-1])

    handles = []
    labels = []

    residuals = []
    iwps = []
    shps = []

    indices = [5, 8]
    channels = [r"$\SI{243.2}{\giga \hertz}$", r"$325.15 \pm \SI{9.5}{\giga \hertz}$"]

    for i, (ind, c) in enumerate(zip(indices, channels)):
        for j, s in enumerate(shapes):

            rs = results[s]

            # IWP
            y = radar["y"] / 1e3
            dy = np.diff(y, axis=-1) * 1e3
            dy = 0.5 * (dy[1:] + dy[:-1])
            iwc = rs["ice_water_content"].data
            iwp = np.sum(dy * iwc, axis=-1)

            # 325 GHz
            dy = (rs["yf_ismar"].data[:, ind] - rs["y_ismar"].data[:, ind]).ravel()
            residuals += [dy]
            iwps += [iwp]
            shps += [s] * iwp.size

            ax = axs[i, j]
            ax.scatter(iwp, dy, label=s, s=6, alpha=0.3, c=f"C{j}", edgecolor="none")

            x = np.log10(iwp)
            y = dy
            slope, intercept, r, p, se = linregress(x, y)
            x = np.linspace(-2, 2, 101)
            ax.text(x=1e0, y=-10, s=f"r: {r:0.3f} \n p: {p:0.3f}", ha="right")
            y_hat = slope * x + intercept
            ax.set_xscale("log")
            ax.axhline(0, c="k", ls="--")
            ax.plot(10 ** x, y_hat, ls="-", c="grey")

            ax.set_ylim([-10, 10])
            ax.set_xlim([1e-2, 1e0 + 0.01])

            if i == 0:
                ax.set_title(PARTICLE_NAMES[s], fontsize=14)

            if j == 0:
                ax.spines["left"].set_position(("outward", 10))
                ax.set_ylabel(r"$\Delta y$ [$\si{\kelvin}$]")
            else:
                ax.spines["left"].set_visible(False)
                remove_y_ticks(ax)

            if i < 1:
                ax.spines["bottom"].set_visible(False)
                remove_x_ticks(ax)
            else:
                ax.set_xlabel("IWP [$\si{\kilo \gram \per \meter \cubed}]$")
                ax.spines["bottom"].set_position(("outward", 10))

        ax = names[i, 0]
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            c,
            fontsize=14,
            rotation="vertical",
            rotation_mode="anchor",
            transform=ax.transAxes,
            weight="bold",
            ha="center",
            va="center",
        )
    return f, axs

