import io
import os
import matplotlib.pyplot as plt
from matplotlib.cm import Greys, ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import scipy as sp
import scipy.signal
import joint_flight
import numpy as np
import ipywidgets as widgets
from artssat.utils.data_providers import NetCDFDataProvider
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from joint_flight.data import hamp

def sample_particles(data, inds, m, n):
    img = np.zeros((m * 32, n * 32))
    for i in range(m):
        i_s = i * 32
        i_e = (i + 1) * 32
        for j in range(n):
            ind = np.random.choice(inds)
            j_s = j * 32
            j_e = (j + 1) * 32
            img[i_s : i_e, j_s : j_e] = data[ind]
    return img

def draw_classes(data, classes, m = 10, n = 20):

    n_classes = classes.max()
    f, axs = plt.subplots(n_classes + 1, 1, figsize = (8, n_classes * 4))

    for i in range(n_classes + 1):
        inds = np.where(classes == i)[0]
        img = sample_particles(data, inds, m, n)

        ax = axs[i]
        ax.matshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Class {} ({})".format(i, inds.size))

def img_to_bytes(img):
    plt.ioff()
    plt.matshow(img, cmap = "bone_r")
    plt.xticks([])
    plt.yticks([])
    data = io.BytesIO()
    plt.savefig(data, format = "png")
    plt.ion()
    plt.close()
    return data.getvalue()

particle_classes = ["Unclassified", "None", "Aggregate", "Spherical", "Pristine", "Irregular"]
particle_classes = [(l, i - 1) for i, l in enumerate(particle_classes)]

def selectah(data, labels, m = 20, n = 20):

    n_classes = labels.max()
    inputs = [widgets.Dropdown(options = particle_classes, value = -1, description = "Class:") \
              for i in range(n_classes + 1)]

    box_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        justify_content='center')
    images = []
    for i in range(n_classes + 1):
        inds = np.where(labels == i)[0]
        img = img_to_bytes(sample_particles(data, inds, 20, 20))
        img = widgets.Image(value = img, format = "png", width = 500, height = 500)
        images += [img]

    box = widgets.VBox([widgets.HBox([img, widgets.VBox([input], layout = box_layout)]) \
                         for img, input in zip(images, inputs)] )

    return box, inputs

def tsne(x, n = 100000):
    from openTSNE import TSNE
    from openTSNE.callbacks import ErrorLogger
    x_in = x[:n, :]
    tsne = TSNE(perplexity=500,
                metric="euclidean",
                callbacks=ErrorLogger(),
                n_iter = 2000,
                n_jobs=4)
    x_embedded = tsne.fit(x_in)
    return x_embedded

def classify(x, n_classes = 20):

    n_cluster = 20000
    clustering = AgglomerativeClustering(n_clusters=n_classes).fit(x[:n_cluster, :])

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(x[:20000], clustering.labels_) 
    classes = neigh.predict(x)
    return classes

def batch_to_img(batch):
    bs, _, m, n = batch.size()
    bss = int(np.sqrt(bs))
    img = np.zeros((bss * m, bss * n))
    k = 0
    for i in range(bss):
        i_start = m * i
        i_end = i_start + m
        for j in range(bss):
            j_start = n * j
            j_end = j_start + n
            img[i_start : i_end, j_start : j_end] = batch[k, 0, :, :].detach().numpy()
            k += 1
    return img

def make_surface_plot(ax1, ax2, lax = None):
    from joint_flight.data import hamp
    palette = ["#34495e", "#9b59b6", "#3498db", "#95a5a6", "#2ecc71", "#e74c3c"]

    zs = np.copy(hamp.zs)
    zs[zs <= 0.0] = -1e3
    handles = []
    handles += [ax1.fill_between(hamp.d, 0.0, -1e3, color = palette[0])]
    handles += [ax1.fill_between(hamp.d, zs, -1e3, color = palette[3])]
    ax1.set_ylabel("Elev. [m]")
    ax1.set_xlabel("Along track distance [km]")
    ax1.set_ylim([-100, 400])
    ax1.set_xlim([hamp.d[0], hamp.d[-1]])
    ax1.grid(False)

    labels = ["Ocean", "Land"]
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.legend(handles = handles, labels = labels, loc = "center left")
    for s in ax2.spines:
        ax2.spines[s].set_visible(False)

def draw_surface_shading(ax, c="grey", alpha = 0.5):
    from joint_flight.data import hamp
    s = 1000.0 * hamp.land_mask
    ax.fill_between(hamp.d, s, -s, color = c, zorder = 10, alpha = alpha, edgecolor = None)

def plot_gp_dist(ax,
                 samples,
                 x,
                 plot_samples=True,
                 palette="Reds",
                 fill_alpha=0.8,
                 samples_alpha=0.1,
                 fill_kwargs=None,
                 samples_kwargs=None):
    """ A helper function for plotting 1D GP posteriors from trace 

        Parameters
    ----------
    ax : axes
        Matplotlib axes.
    samples : trace or list of traces
        Trace(s) or posterior predictive sample from a GP.
    x : array
        Grid of X values corresponding to the samples. 
    plot_samples: bool
        Plot the GP samples along with posterior (defaults True).
    palette: str
        Palette for coloring output (defaults to "Reds").
    fill_alpha : float
        Alpha value for the posterior interval fill (defaults to 0.8).
    samples_alpha : float
        Alpha value for the sample lines (defaults to 0.1).
    fill_kwargs : dict
        Additional arguments for posterior interval fill (fill_between).
    samples_kwargs : dict
        Additional keyword arguments for samples plot.
    Returns
    -------
    ax : Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100-p, axis=1)
        color_val = colors[i]
        ax.fill_betweenx(x, upper, lower, color=cmap(color_val), alpha=fill_alpha, **fill_kwargs)
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(samples[:,idx], x, color=cmap(0.9), lw=1, alpha=samples_alpha,
                **samples_kwargs)

    return ax

def plot_gp_dist_alpha(ax,
                       samples,
                       x,
                       plot_samples=True,
                       c="C0",
                       fill_alpha=0.8,
                       samples_alpha=0.1,
                       fill_kwargs=None,
                       samples_kwargs=None):
    import matplotlib.pyplot as plt

    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}

    percs = np.linspace(5, 45, 8)
    percs = np.concatenate([np.zeros(1), percs])

    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()

    for i, p in enumerate(percs[:-1]):

        alpha = fill_alpha - fill_alpha / (len(percs) - 1) * i
        pn = percs[i + 1]
        # Lower
        left = np.percentile(samples, 50 - pn, axis=1)
        right = np.percentile(samples, 50 - p, axis=1)
        ax.fill_betweenx(x, left, right,
                         color=c,
                         alpha=alpha,
                         **fill_kwargs,
                         lw = 0,
                         zorder = 10,
                         edgecolor = None)

        # Upper
        left = np.percentile(samples, 50 + p, axis=1)
        right = np.percentile(samples, 50 + pn, axis=1)
        ax.fill_betweenx(x, left, right,
                         color=c,
                         alpha=alpha,
                         **fill_kwargs,
                         lw = 0,
                         zorder = 10,
                         edgecolor = None)

    ax.plot(np.median(samples, axis = 1), x, color = c)

    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(samples[:,idx], x, color=c, lw=1, alpha=samples_alpha,
                **samples_kwargs)

    return ax


def particle_to_image(img, cmap = Greys):
    norm = Normalize(-1, 1)
    sm = ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array(img)
    cs = sm.to_rgba(img.ravel())
    cs.resize((img.shape) + (4,))

    inds = img == -1.0
    cs[inds, -1] = 0.0

    return cs


def tsne_plot(x_embedded,
              classes,
              data,
              inds = None,
              ax = None,
              n_samples = 10):

    if inds is None:
        inds = np.arange(x_embedded.shape[0])

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize = (10, 10))
    else:
        f = plt.gcf()
        ax = plt.gca()

    ax.scatter(x_embedded[:, 0], x_embedded[:, 1], c = classes, cmap = "tab20c", zorder =0, s = 0.1)

    maxs = x_embedded.max(axis = 0)
    mins = x_embedded.min(axis = 0)
    dx, dy = maxs - mins
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])

    _, _, w, h = ax.get_position().bounds

    ih = 0.05 * dy * w / h
    iw = 0.05 * dx * h / w

    print(inds)
    n_classes = classes.max()
    for i in range(n_classes + 1):
        j = 0
        tries = 0
        class_inds = np.where(classes == i)[0]
        others = []
        while (j < n_samples) and tries < 1000:
            ind = np.random.choice(class_inds)
            img = particle_to_image(data[inds[ind]][0, :, :].detach().numpy())

            x_0, y_0 = x_embedded[ind, :]
            dists = [np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) for (x, y) in others]

            tries += 1
            if len(dists) > 0:
                if min(dists) < iw or min(dists) < ih:
                    continue

            ext = (x_0 - iw // 2, x_0 + iw // 2, y_0 - ih // 2, y_0 + ih // 2)
            ax.imshow(img, extent = ext, zorder = 10)
            j += 1

            others += [(x_0, y_0)]

    #for i in range(n_classes + 1):

def despine_ax(ax, left = True, bottom = True, d = 10):

    ax.spines['left'].set_position(('outward', d))
    ax.spines['bottom'].set_position(('outward', d))

    if not left:
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        ax.yaxis.set_tick_params(size = 0)
        ax.tick_params(axis='y', which=u'both',length=0)

    if not bottom:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis = "x", width = 0, length = 0)
        ax.tick_params(axis='x', which=u'both',length=0)


def plot_observation_misfit(sensor_name,
                            y,
                            yf,
                            channel_indices,
                            channel_labels,
                            smoothing = 10,
                            palette = "Reds"):
    f = plt.figure(figsize = (12, 8))
    filename = os.path.join(joint_flight.path, "data", "input.nc")
    data_provider = NetCDFDataProvider(filename)

    gs = GridSpec(2, 2, height_ratios = [1.0, 1.0], width_ratios = [1.0, 0.2])

    nedt_getter = getattr(data_provider, "get_y_" + sensor_name + "_nedt")
    nedts = []
    for i in range(hamp.d.size):
        nedts += [nedt_getter(i)]
    nedts = np.array(nedts)

    k = np.ones(int(smoothing)) / smoothing

    ax = plt.subplot(gs[0, 0])
    x = hamp.d
    for i, ind in enumerate(channel_indices):
        dy = y[:, ind] - yf[:, ind]
        dys = sp.signal.convolve(dy, k, "valid")
        xs = sp.signal.convolve(x, k, "valid")
        ax.plot(xs, dys, c = palette[i])
    ax.set_ylim([-5, 5])
    ax.set_xticklabels([])
    ax.set_ylabel(r"$\Delta T_b$ [$K$]")

    despine_ax(ax, left = True, bottom = False)

    ax = plt.subplot(gs[1, 0])
    for i, ind in enumerate(channel_indices):
        dy = y[:, ind] - yf[:, ind]
        dy = dy ** 2 / nedts[:, ind]
        dys = sp.signal.convolve(dy, k, "valid")
        xs = sp.signal.convolve(x, k, "valid")
        ax.plot(xs, dys, c = palette[i])
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1e3])
    ax.set_ylabel(r"$\chi^2$")
    ax.set_xlabel("Distance [km]")

    despine_ax(ax, left = True, bottom = True)

    # legend
    handles = [Line2D([0, 0], [0, 0], c = c) for c in palette]
    ax = plt.subplot(gs[:, 1])
    ax.set_axis_off()
    ax.legend(handles = handles, labels = channel_labels, loc = "center")


def plot_observation_misfit_radar(y, yf, z):

    f = plt.figure(figsize = (10, 10))
    filename = os.path.join(joint_flight.path, "data", "input.nc")

    gs = GridSpec(2, 2, height_ratios = [0.5, 1.0], width_ratios = [1.0, 0.2])

    nedts = np.ones(y.shape)
    dy = y - yf
    x = hamp.d
    y = z / 1e3

    ax = plt.subplot(gs[0, 0])
    norm = Normalize(-30, 15)
    ax.pcolormesh(x, y, dy.T, norm = norm)

    x2 = dy * dy / nedts
    ax = plt.subplot(gs[1, 0])
    ax.pcolormesh(x, y, x2.T)

def iwc(n0, dm):
    return np.pi * 917.0 * dm ** 4 * n0 / 4 ** 4

def rwc(n0, dm):
    return np.pi * 917.0 * dm ** 4 * n0 / 4 ** 4

from mcrf.psds import D14NDmIce
psd = D14NDmIce()

def number_density(n0, dm):
    psd = D14NDmIce()
    psd.mass_weighted_diameter = dm
    psd.intercept_parameter = n0
    return psd.get_moment(0)

def number_density_100(habit, n0, dm):
    psd = D14NDmIce()
    from joint_flight.data import habits
    try:
        pm = getattr(habits, habit)
        ind = np.where(pm.dmax > 1e-4)[0][0]
        x = np.logspace(np.log10(pm.de[ind]), -1, 201)
    except:
        x = np.logspace(-4, -1, 201)
    psd.mass_weighted_diameter = dm
    psd.intercept_parameter = n0
    data = psd.evaluate(x).data
    nd = np.trapz(data, x = x, axis = -1)
    return nd

def centers_to_edges(array, axis=0):

    n_dims = len(array.shape)

    indices = [slice(0, None)] * n_dims
    indices[axis] = slice(0, -1)
    indices_l = tuple(indices)
    indices[axis] = slice(1, None)
    indices_r = tuple(indices)
    indices[axis] = slice(1, -1)
    indices_c = tuple(indices)

    indices[axis] = 0
    indices_l1 = tuple(indices)
    indices[axis] = 1
    indices_l2 = tuple(indices)

    indices[axis] = -1
    indices_r1 = tuple(indices)
    indices[axis] = -2
    indices_r2 = tuple(indices)

    shape = list(array.shape)
    shape[axis] = shape[axis] + 1
    edges = np.zeros(tuple(shape), dtype=array.dtype)

    edges[indices_c] = 0.5 * (array[indices_r] +  array[indices_l])
    edges[indices_l1] = 2.0 * array[indices_l1] - array[indices_l2]
    edges[indices_r1] = 2.0 * array[indices_r1] - array[indices_r2]

    return edges
