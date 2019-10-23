import logging
import math
from collections import defaultdict
from typing import List, Text

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from game import Game


def plot_raw_and_pca(data, masks: List[List[int]], labels: List[Text], title: Text):
    if data.shape[1] > 1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        pca_model = PCA(n_components=2, whiten=True)
        data_pca = pca_model.fit_transform(data)

        for mask, label in zip(masks, labels):
            # plot first two coordinates
            ax[0].scatter(data[mask, 0], data[mask, 1], alpha=0.2, label=label)
            ax[0].axis('equal')
            ax[0].set(xlabel='Coordinate 1', ylabel='Coordinate 2', title='First coordinates')

            # plot principal components from PCA
            ax[1].scatter(data_pca[mask, 0], data_pca[mask, 1], alpha=0.2, label=label)
            ax[1].axis('equal')
            ax[1].set(xlabel='Component 1', ylabel='Component 2', title='PCA')

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))

        for mask, label in zip(masks, labels):
            # plot first (only) coordinate
            ax.scatter(data[mask, 0], [0] * len(mask), alpha=0.2, label=label)
            ax.axis('equal')
            ax.set(xlabel='Coordinate 1', ylabel='Dummy', title='First coordinate')

    fig.suptitle(title)
    leg = plt.legend(ncol=2, bbox_to_anchor=(1.1, .9))

    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.show()


def plot_clusters(data, labels, title="Clusters"):
    labels_to_idx = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_idx[label].append(i)

    labels, masks = zip(*labels_to_idx.items())
    plot_raw_and_pca(data, masks, labels, title)


def plot_information(game: Game, exemplars_size = 40):
    situations = torch.randn(exemplars_size * game.func_size, game.situation_size)
    func_switches = torch.cat([torch.arange(game.func_size) for _ in range(exemplars_size)])
    targets = game.target(situations, func_switches)
    targets = targets.numpy()

    masks = []
    labels = []
    for fc in range(game.information_size):
        masks.append([i * game.information_size + fc for i in range(exemplars_size)])
        labels.append(f"F{fc}")

    plot_raw_and_pca(targets, masks, labels, "Targets")


def generate_information_situations_messages(game: Game, exemplars_size):
    batch_size = exemplars_size * game.information_size
    situations = game.generate_situations(batch_size)

    information = torch.zeros((batch_size, game.information_size))
    for i in range(batch_size):
        information[i, i % game.information_size] = 1.0

    messages = game.message(situations, information)

    return information, situations, messages


def predict_information_from_messages(game: Game, exemplars_size=40) -> float:
    information, situations, messages = generate_information_situations_messages(game, exemplars_size)
    batch_size = information.shape[0]

    train_test_ratio = 0.7
    num_train_samples = math.ceil(batch_size * train_test_ratio)

    train_messages, test_messages = messages[:num_train_samples], messages[num_train_samples:]
    train_information, test_information = information[:num_train_samples], information[num_train_samples:]

    classifier_hidden_size = 32
    model = torch.nn.Sequential(
        torch.nn.Linear(game.message_size, classifier_hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(classifier_hidden_size, game.information_size),
        torch.nn.Softmax()
    )
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000
    for epoch in range(num_epochs):
        y_pred = model(train_messages)
        loss = loss_func(y_pred, train_information)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch > 0 and epoch % 100 == 0:
            logging.info(f"Epoch {epoch+1}:\t{loss.item():.2e}")

    with torch.no_grad():
        test_predicted = model(test_messages).numpy()

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(np.argmax(test_information.numpy(), axis=1), np.argmax(test_predicted, axis=1))
    logging.info(f"Information prediction accuracy: {accuracy}")
    return accuracy


def clusterize_messages(game: Game, exemplars_size=40):
    num_clusters = game.information_size

    information, situations, messages = generate_information_situations_messages(game, exemplars_size)

    k_means = KMeans(n_clusters=num_clusters)
    labels = k_means.fit_predict(messages)

    plot_clusters(messages, labels, "Training messages clusters")

    # Find cluster for each message.
    message_distance_from_centers = k_means.transform(messages)
    representative_message_idx_per_cluster = message_distance_from_centers.argmin(axis=0)
    message_num_per_cluster = information[representative_message_idx_per_cluster,:].argmax(axis=1)
    cluster_label_to_message_num = {cluster_num: message_num for cluster_num, message_num in enumerate(message_num_per_cluster)}

    # Sample unseen messages from clusters.
    _, test_situations, test_messages = generate_information_situations_messages(game, exemplars_size)
    cluster_label_per_test_message = k_means.predict(test_messages)

    batch_size = test_messages.shape[0]
    information_by_message_cluster = torch.zeros((batch_size, game.information_size))
    for i, cluster_label in enumerate(cluster_label_per_test_message):
        information_by_message_cluster[i, cluster_label_to_message_num[cluster_label]] = 1.0

    plot_clusters(test_messages, cluster_label_per_test_message, "Test message clusters")

    predictions_by_unseen_messages = game.predict_by_message(test_messages, test_situations)
    with torch.no_grad():
        predictions_by_inferred_information, _ = game.forward(test_situations, information_by_message_cluster)

    loss_func = torch.nn.MSELoss()
    loss = loss_func(predictions_by_unseen_messages, predictions_by_inferred_information).item()
    logging.info(f"Loss for unseen message/information: {loss}")

    return loss


def plot_messages_information(game: Game, exemplars_size=40):
    with torch.no_grad():
        batch_size = exemplars_size * game.information_size
        situations = game.generate_situations(batch_size)

        information = torch.zeros((batch_size, game.information_size))
        for i in range(batch_size):
            information[i, i % game.information_size] = 1

        messages = game.message(situations, information)
        messages = messages.numpy()

        message_masks = []
        message_labels = []
        for fc in range(game.information_size):
            message_masks.append([i * game.information_size + fc for i in range(exemplars_size)])
            message_labels.append(f"F{fc}")

        plot_raw_and_pca(messages, message_masks, message_labels, "Messages")

        targets = game.target(situations, information)
        plot_raw_and_pca(targets.numpy(), message_masks, message_labels, "Targets")


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


def plot_pca_3d(x, data, xlabel, ylabel, zlabel, title):
    pca = PCA(2)
    predictions_pca = pca.fit_transform(data)

    zs = predictions_pca[:, 0]
    ys = predictions_pca[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)

    ax.plot(x, ys, zs)
    ax.legend()

    plt.show()
