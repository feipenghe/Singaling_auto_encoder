import functools
import logging
import operator
from collections import defaultdict
from typing import Iterable, List, Text

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def plot_raw_and_pca(data, masks: List[List[int]], labels: List[Text], title: Text):
    if data.shape[1] > 1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        pca_model = PCA(n_components=2, whiten=True)
        data_pca = pca_model.fit_transform(data)

        for mask, label in zip(masks, labels):
            # plot first two coordinates
            ax[0].scatter(data[mask, 0], data[mask, 1], alpha=0.2, label=label)
            ax[0].axis("equal")
            ax[0].set(
                xlabel="Coordinate 1", ylabel="Coordinate 2", title="First coordinates"
            )

            # plot principal components from PCA
            ax[1].scatter(data_pca[mask, 0], data_pca[mask, 1], alpha=0.2, label=label)
            ax[1].axis("equal")
            ax[1].set(xlabel="Component 1", ylabel="Component 2", title="PCA")

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))

        for mask, label in zip(masks, labels):
            # plot first (only) coordinate
            ax.scatter(data[mask, 0], [0] * len(mask), alpha=0.2, label=label)
            ax.axis("equal")
            ax.set(xlabel="Coordinate 1", ylabel="Dummy", title="First coordinate")

    fig.suptitle(title)
    leg = plt.legend(ncol=2, bbox_to_anchor=(1.1, 0.9))

    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.show()


def plot_clusters(data, labels, title="Clusters"):
    labels_to_idx = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_idx[label].append(i)

    labels, masks = zip(*labels_to_idx.items())
    plot_raw_and_pca(data, masks, labels, title)


# def plot_information(game: Game, exemplars_size = 40):
#     situations = torch.randn(exemplars_size * game.func_size, game.context_size)
#     func_switches = torch.cat([torch.arange(game.func_size) for _ in range(exemplars_size)])
#     targets = game.target(situations, func_switches)
#     targets = targets.numpy()
#
#     masks = []
#     labels = []
#     for fc in range(game.object_size):
#         masks.append([i * game.object_size + fc for i in range(exemplars_size)])
#         labels.append(f"F{fc}")
#
#     plot_raw_and_pca(targets, masks, labels, "Targets")


def plot_bar_list(L, L_labels=None, transform=True):
    if not (L_labels):
        L_labels = np.arange(len(L))
    index = np.arange(len(L))

    if transform:
        COL = ["blue", "red"]
    else:
        COL = "blue"

    plt.bar(index, [x.item() for x in L], color=COL)
    plt.xticks(index, L_labels, fontsize=5)
    plt.xlabel("Functions", fontsize=5)
    plt.ylabel("MSELoss", fontsize=5)
    plt.title("Loss per function")
    plt.show()


def plot_losses(G, losses=None, exemplars_size=200):
    if losses is None:
        with torch.no_grad():
            situations = torch.randn(exemplars_size, G.situation_size)
            func_switches = torch.cat(
                [torch.arange(G.func_size) for _ in range(exemplars_size)]
            )

            loss_func = []
            for ind in range(len(G.functions)):
                loss_func.append(
                    G.loss(
                        situations, torch.ones(exemplars_size, dtype=torch.long) * ind
                    )
                )
            losses = loss_func

    plot_bar_list(losses, transform=G.transform)
    return losses


def plot_pca_3d(x, data, xlabel, ylabel, zlabel, title):
    pca = PCA(2)
    predictions_pca = pca.fit_transform(data)

    zs = predictions_pca[:, 0]
    ys = predictions_pca[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)

    ax.plot(x, ys, zs)
    ax.legend()

    plt.show()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y:%H:%M:%S",
        level=logging.INFO,
    )


def reduce_prod(vals):
    return functools.reduce(operator.mul, vals)


def batch_flatten(x):
    return torch.reshape(x, [-1, reduce_prod(x.shape[1:])])


def join_ints(ints: Iterable[int], s=",") -> Text:
    return s.join(f"{x}" for x in ints)


def str_val(val) -> Text:
    if isinstance(val, int):
        return str(val)
    elif isinstance(val, tuple) or isinstance(val, list):
        return join_ints(val)
    elif isinstance(val, bool):
        return "1" if val else "0"
    return str(val)
