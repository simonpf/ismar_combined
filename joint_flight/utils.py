import io
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

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
