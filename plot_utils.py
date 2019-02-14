__author__ = 'ando'

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from os.path import exists, join as path_join
from os import makedirs
import numpy as np
import itertools


# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)


def node_space_plot_2D(embedding,
                       labels,
                       size=1,
                       color_dict={1:'red', 2:'yellow', 3:'blue', 4:'greed'},
                       legend_dict=None,
                       save=None):

    # plt.style.use('ggplot')
    plt.figure(figsize=(5, 5))
    for label in sorted(color_dict.keys()):
        color_embedding = embedding[np.where(labels == label)]
        color = color_dict[label]
        if legend_dict:
            label = legend_dict[label]
        plt.scatter(color_embedding[:, 0], color_embedding[:, 1], marker='o', s=size, color=color, label=label)

    plt.axis('off')
    # plt.legend(loc='upper left',
    #            numpoints=1,
    #            ncol=2,
    #            fontsize=8,
    #            bbox_to_anchor=(0, 0))
    if save:
        plt.savefig(path_join("/home/ando/Dblp_visualization", save), format="png")
        plt.close()
    else:
        plt.show()


def node_space_plot_2D_elipsoid(embedding, means, covariances, color_values, name):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.


        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)



    for n_idx, node in enumerate(embedding):
        ax.scatter(node[0], node[1], c=color_values[n_idx], marker='o', s=50, cmap="viridis")
        ax.text(node[0], node[1],  '%s' % (str(n_idx)), size=0)

    for mean in means:
        ax.scatter(mean[0], mean[1], marker='x', s=100)

    # if means is not None:
    #     ax.scatter(means[:,0], means[:,1], marker='x', c='r')


    x_max, x_min = 2, -4
    y_max, y_min = 4, -4

    x_step = (x_max - x_min) / 4.0
    y_step = (y_max - y_min) / 4.0

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    x_major_ticks = np.arange(x_min, x_max+0.01, 2*x_step)
    x_minor_ticks = np.arange(x_min, x_max+0.01, x_step)

    y_major_ticks = np.arange(y_min, y_max+0.001, 2*y_step)
    y_minor_ticks = np.arange(y_min, y_max+0.001, y_step)

    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)

    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)

    ax.grid(which='both')
    plt.plot()
    plt.savefig(name+".png")
    plt.close()