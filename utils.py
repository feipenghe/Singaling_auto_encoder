import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from game import Game, generate_information, generate_situations


def plot_rawPCA(game, components, exemplars_size):
    if components.shape[1] > 1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        pca_model = PCA(n_components=2, whiten=True)
        components_pca = pca_model.fit_transform(components)

        # plot data
        for fc in range(game.information_size):
            MASK = [i * game.information_size + fc for i in range(exemplars_size)]

            # plot first two coordinates
            ax[0].scatter(components[MASK, 0], components[MASK, 1], alpha=0.2, label=f"F{fc}")
            ax[0].axis('equal')
            ax[0].set(xlabel='Coordinate 1', ylabel='Coordinate 2', title='First coordinates')

            # plot principal components from PCA
            ax[1].scatter(components_pca[MASK, 0], components_pca[MASK, 1], alpha=0.2, label=f"F{fc}")
            ax[1].axis('equal')
            ax[1].set(xlabel='Component 1', ylabel='Component 2', title='PCA')

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))
        # plot data
        for fc in range(game.information_size):
            MASK = [i * game.information_size + fc for i in range(exemplars_size)]

            # plot first (only) coordinate
            ax.scatter(components[MASK, 0], [0] * len(MASK), alpha=0.2, label=f"F{fc}")
            ax.axis('equal')
            ax.set(xlabel='Coordinate 1', ylabel='Dummy', title='First coordinate')

    # if G.transform:
    #     for fc in range(int(G.func_size / 2)):
    #         MASK1 = [i * G.func_size + 2 * fc for i in range(exemplars_size)]
    #         MASK2 = [i * G.func_size + 2 * fc - 1 for i in range(exemplars_size)]
    #
    #         X1, Y1 = np.mean(components[MASK1, 0]), np.mean(components[MASK1, 1])
    #         X2, Y2 = np.mean(components[MASK2, 0]), np.mean(components[MASK2, 1])
    #         ax[0].arrow(X1, Y1, X2 - X1, Y2 - Y1, head_width=.1)
    #
    #         X1_pca, Y1_pca = np.mean(components_pca[MASK1, 0]), np.mean(components_pca[MASK1, 1])
    #         X2_pca, Y2_pca = np.mean(components_pca[MASK2, 0]), np.mean(components_pca[MASK2, 1])
    #         ax[1].arrow(X1_pca, Y1_pca, X2_pca - X1_pca, Y2_pca - Y1_pca, head_width=.1)

    leg = plt.legend(ncol=2, bbox_to_anchor=(1.1, .9))

    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.show()


def plot_messages(G, exemplars_size=40):
    situations = torch.randn(exemplars_size * G.func_size, G.situation_size)
    func_switches = torch.cat([torch.arange(G.func_size) for _ in range(exemplars_size)])
    messages = G.message(situations, func_switches)
    messages = messages.numpy()

    plot_rawPCA(G, messages, exemplars_size)


def plot_information(G, exemplars_size = 40):
    situations = torch.randn(exemplars_size*G.func_size, G.situation_size)
    func_switches = torch.cat([torch.arange(G.func_size) for _ in range(exemplars_size)])
    targets = G.target(situations, func_switches)
    targets = targets.numpy()

    plot_rawPCA(G, targets, exemplars_size)


def plot_messages_information(game: Game, exemplars_size=40):
    with torch.no_grad():
        batch_size = exemplars_size * game.information_size  # TODO: why?
        situations = generate_situations(batch_size, game.situation_size)
        information = generate_information(batch_size, game.information_size)

        messages = game.message(situations, information)
        messages = messages.numpy()
        plot_rawPCA(game, messages, exemplars_size)

        targets = game.target(situations, information)
        # targets = targets.numpy()
        # plot_rawPCA(game, targets, exemplars_size)


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
    plt.xlabel('Functions', fontsize=5)
    plt.ylabel('MSELoss', fontsize=5)
    plt.title('Loss per function')
    plt.show()


def get_loss_per_function(G, exemplars_size=200):
    with torch.no_grad():
        situations = torch.randn(exemplars_size, G.situation_size)
        func_switches = torch.cat([torch.arange(G.func_size) for _ in range(exemplars_size)])

        loss_per_function = []
        for ind in range(len(G.functions)):
            loss_per_function.append(G.loss(situations, torch.ones(exemplars_size, dtype=torch.long) * ind))

    return loss_per_function


def plot_losses(G, losses=None, exemplars_size=200):
    if losses is None:
        with torch.no_grad():
            situations = torch.randn(exemplars_size, G.situation_size)
            func_switches = torch.cat([torch.arange(G.func_size) for _ in range(exemplars_size)])

            loss_func = []
            for ind in range(len(G.functions)):
                loss_func.append(G.loss(situations, torch.ones(exemplars_size, dtype=torch.long) * ind))
            losses = loss_func

    plot_bar_list(losses, transform=G.transform)
    return losses
