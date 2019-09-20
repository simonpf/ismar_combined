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
    f, axs = plt.subplots(n_classes, 1, figsize = (8, n_classes * 4))

    for i in range(n_classes):
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

def selectah(data, labels, m = 20, n = 20):

    n_classes = labels.max()
    check_boxes = [widgets.Checkbox(value = False, description = "Use") for i in range(n_classes)]

    box_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        justify_content='center')
    images = []
    for i in range(n_classes):
        inds = np.where(labels == i)[0]
        img = img_to_bytes(sample_particles(data, inds, 20, 20))
        img = widgets.Image(value = img, format = "png", width = 500, height = 500)
        images += [img]

    box = widgets.VBox([widgets.HBox([img, widgets.VBox([check], layout = box_layout)]) \
                         for img, check in zip(images, check_boxes)] )

    return box, check_boxes

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
