import json
import pathlib
import pickle
from collections import defaultdict
from typing import Dict, Optional, Text

import matplotlib.pyplot as plt
import numpy as np

import game
import simulations


def plot_simulation(simulation_name: Text, element_to_plot: Text):
    simulation = simulations.load_simulation(simulation_name)
    # TODO load from json

    accuracy_mean = [np.mean(vals) for vals in supervised_clustering_accuracies]
    accuracy_error = [np.std(vals) for vals in supervised_clustering_accuracies]

    loss_mean = [np.mean(vals) for vals in unsupervised_clustering_losses]
    loss_error = [np.std(vals) for vals in unsupervised_clustering_losses]

    x_labels = []
    if isinstance(simulation.context_size, tuple):
        context_size = simulation.context_size[0]  # TODO: Is this correct?
    else:
        context_size = simulation.context_size
    for message_size in simulation.message_sizes:
        label = str(message_size)
        if message_size == context_size:
            label += "\nContext size"
        if message_size == simulation.object_size:
            label += "\nObject size"
        x_labels.append(label)

    x_ticks = range(len(simulation.message_sizes))
    plt.bar(x_ticks, accuracy_mean, yerr=accuracy_error, capsize=10)
    plt.xticks(x_ticks, x_labels, fontsize=5)
    plt.axvline(
        x=simulation.message_sizes.index(context_size), linestyle="--", color="gray"
    )
    plt.axvline(
        x=simulation.message_sizes.index(simulation.object_size),
        linestyle="--",
        color="gray",
    )
    plt.xlabel("Message dimensionality", fontsize=5)
    plt.ylabel("F prediction accuracy", fontsize=5)
    plt.title("F prediction accuracy")
    plt.savefig(f"./simulations/{simulation_name}_supervised_classification.png")
    plt.show()

    plt.bar(x_ticks, loss_mean, yerr=loss_error, capsize=10)
    plt.xticks(x_ticks, x_labels, fontsize=5)
    plt.axvline(
        x=simulation.message_sizes.index(context_size), linestyle="--", color="gray"
    )
    plt.axvline(
        x=simulation.message_sizes.index(simulation.object_size),
        linestyle="--",
        color="gray",
    )
    plt.xlabel("Message dimensionality", fontsize=5)
    plt.ylabel("Clustering loss", fontsize=5)
    plt.title("Clustering loss")
    plt.savefig(f"./simulations/{simulation_name}_unsupervised_clustering.png")
    plt.show()


def plot_simulation_training_loss(
    simulation_display_name_to_file_name: Dict[Text, Text],
    plot_type: Text = "line",  # "bar" or "line"
    max_epochs: Optional[int] = None,
    epoch_interval: int = 100,
    label_interval: int = 10,
):
    display_name_to_simulation: Dict[Text, simulations.Simulation] = {
        display_name: simulations.load_simulation(file_name)
        for display_name, file_name in simulation_display_name_to_file_name.items()
    }

    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(f"Training losses (MSE)")

    global_max_loss = 0.0

    for s, (simulation_display_name, simulation) in enumerate(
        display_name_to_simulation.items()
    ):
        epoch_nums = np.array(simulation.epoch_nums)

        for m, message_size in enumerate(simulation.message_sizes):
            # TODO use message size information

            training_losses_per_trial = np.array(
                [
                    simulation.evaluations[message_size][i]["training_losses"]
                    for i in range(simulation.num_trials)
                ]
            )

            if max_epochs is not None:
                epoch_nums = epoch_nums[:max_epochs]
                training_losses_per_trial = training_losses_per_trial[:, :max_epochs]

            losses_mean_per_epoch = np.mean(training_losses_per_trial, axis=0)
            losses_err_per_epoch = np.std(training_losses_per_trial, axis=0)

            global_max_loss = max(global_max_loss, losses_mean_per_epoch.max())

            epoch_idxs = np.arange(0, epoch_nums[-1] + 1, epoch_interval)
            # idxs = np.append(idxs, -1)  # Always plot last epoch
            epoch_nums = epoch_nums[epoch_idxs]
            losses_mean_per_epoch = losses_mean_per_epoch[epoch_idxs]
            losses_err_per_epoch = losses_err_per_epoch[epoch_idxs]

            x_labels = [
                str(epoch_nums[i])
                if i % label_interval == 0 or i + 1 == len(epoch_nums)
                else ""
                for i in range(len(epoch_nums))
            ]
            x_tick_idxs = np.arange(0, len(epoch_nums))
            color = f"C{s}"

            if plot_type == "bar":
                bar_width = 0.5
                x = x_tick_idxs + (bar_width * (-0.5 if s == 0 else 0.5))

                ax.bar(
                    x,
                    losses_mean_per_epoch,
                    width=bar_width,
                    yerr=losses_err_per_epoch,
                    capsize=1.5,
                    label=simulation_display_name,
                    color=color,
                )

            elif plot_type == "line":
                x = x_tick_idxs
                _, caps, bars = ax.errorbar(
                    x,
                    losses_mean_per_epoch,
                    yerr=losses_err_per_epoch,
                    capsize=2,
                    marker=".",
                    linestyle="solid",
                    label=simulation_display_name,
                    color=color,
                )
                [bar.set_alpha(0.3) for bar in bars]
                [cap.set_alpha(0.3) for cap in caps]

            ax.set_xticks(x_tick_idxs)
            ax.set_xticklabels(x_labels)

    y_interval = round(
        global_max_loss / 10, int(np.abs(np.floor(np.log10(global_max_loss / 10))))
    )
    ax.set_yticks(np.arange(0.0, global_max_loss + y_interval, y_interval))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")

    ax.legend(ncol=1, loc="lower right", bbox_to_anchor=(1, 1))
    plt.show()


def plot_simulation_set_strict(simulation_set_name, element_to_plot):
    # TODO load from json

    with pathlib.Path(f"./simulations/{simulation_set_name}.json").open("r") as f:
        sims = [simulations.Simulation.from_dict(x) for x in json.load(f)]

    num_functions = list(sorted(set(x.num_functions for x in sims)))
    context_sizes = [
        tuple(x) if isinstance(x, list) else x for x in (x.context_size for x in sims)
    ]
    context_sizes = list(sorted(set(context_sizes)))
    object_sizes = list(sorted(set(x.object_size for x in sims)))
    num_objects = list(sorted(set(x[0] for x in context_sizes)))

    titles = {
        "functions": "F",
        "object_by_context": "f(c)",
        "object_by_decoder_context": "f(c')",
        "context": "C",
        "decoder_context": "C'",
    }

    if element_to_plot == "functions":
        metric = "accuracy"
    else:
        metric = "loss"

    fig, ax = plt.subplots(1, 1, figsize=(18, 12), squeeze=False)
    fig.suptitle(f"M -> {titles[element_to_plot]} {metric}")
    # fig.suptitle(f"Network output loss")

    global_loss_max = 0.0

    for simulation in sims:
        simulation_path = pathlib.Path(f"./simulations/{simulation.name}/")

        with simulation_path.joinpath("network_losses.pickle").open("rb") as f:
            network_losses = pickle.load(f)
        with simulation_path.joinpath("prediction_by_messages_losses.pickle").open(
            "rb"
        ) as f:
            prediction_by_messages_losses = pickle.load(f)
        with simulation_path.joinpath("unsupervised_clustering_loss.pickle").open(
            "rb"
        ) as f:
            unsupervised_clustering_losses = pickle.load(f)

        element_losses = [
            [x[element_to_plot] for x in trials]
            for trials in prediction_by_messages_losses
        ]

        losses_mean = np.array([np.mean(vals) for vals in element_losses])
        losses_err = [np.std(vals) for vals in element_losses]

        curr_ax = ax[
            0,
            0
            # num_functions.index(simulation.num_functions),
            # 0
            # context_sizes.index(
            #     tuple(simulation.context_size)
            #     if isinstance(simulation.context_size, list)
            #     else simulation.context_size
            # ),
        ]

        x_ticks = np.arange(len(simulation.message_sizes))
        x_labels = []
        for message_size in simulation.message_sizes:
            label = str(message_size)
            if message_size == simulation.context_size:
                label += "\nC"
            x_labels.append(label)

        bar_width = 0.3
        x = x_ticks + (bar_width * (object_sizes.index(simulation.object_size) - 1))
        curr_ax.set(
            xlabel=f"M", ylabel=metric,
        )
        curr_ax.bar(
            x,
            losses_mean,
            width=bar_width,
            label=f"O={simulation.object_size}, F={simulation.num_functions}, C={tuple(simulation.context_size)}",
            yerr=losses_err,
            capsize=4,
            color=f"C{object_sizes.index(simulation.object_size)}",
        )
        curr_ax.set_xticks(x_ticks)
        curr_ax.set_xticklabels(x_labels)

        global_loss_max = max(global_loss_max, losses_mean.max())
        try:
            curr_ax.axvline(
                x=simulation.message_sizes.index(simulation.context_size),
                linestyle="--",
                color="gray",
            )
        except ValueError:
            pass

    y_interval = round(
        global_loss_max / 10, int(np.abs(np.floor(np.log10(global_loss_max / 10))))
    )
    for _, curr_ax in np.ndenumerate(ax):
        curr_ax.set_yticks(np.arange(0.0, global_loss_max + y_interval, y_interval))

    ax[0, -1].legend(ncol=1, loc="lower right", bbox_to_anchor=(1, 1))
    plt.show()


def plot_simulation_set_non_strict(simulation_set_name, element_to_plot):
    with pathlib.Path(f"./simulations/{simulation_set_name}.json").open("r") as f:
        sims = [simulations.Simulation.from_dict(x) for x in json.load(f)]

    num_functions = list(sorted(set(x.num_functions for x in sims)))
    context_sizes = [
        tuple(x) if isinstance(x, list) else x for x in (x.context_size for x in sims)
    ]
    context_sizes = list(sorted(set(context_sizes)))
    object_sizes = list(sorted(set(x.object_size for x in sims)))
    num_objects = list(sorted(set(x[0] for x in context_sizes)))

    titles = {
        "functions": "F",
        "object_by_context": "f(c)",
        "object_by_decoder_context": "f(c')",
        "context": "C",
        "decoder_context": "C'",
    }

    if element_to_plot == "functions":
        metric = "accuracy"
    else:
        metric = "loss"

    fig, ax = plt.subplots(1, len(num_objects), figsize=(20, 6), squeeze=False)
    fig.suptitle(f"M -> {titles[element_to_plot]} {metric}")
    # fig.suptitle(f"Network output loss")

    global_loss_max = 0.0

    for simulation in sims:
        simulation_path = pathlib.Path(f"./simulations/{simulation.name}/")

        with simulation_path.joinpath("network_losses.pickle").open("rb") as f:
            network_losses = pickle.load(f)
        with simulation_path.joinpath("prediction_by_messages_losses.pickle").open(
            "rb"
        ) as f:
            prediction_by_messages_losses = pickle.load(f)
        with simulation_path.joinpath("unsupervised_clustering_loss.pickle").open(
            "rb"
        ) as f:
            unsupervised_clustering_losses = pickle.load(f)

        element_losses = [
            [x[element_to_plot] for x in trials]
            for trials in prediction_by_messages_losses
        ]

        losses_mean = np.array([np.mean(vals) for vals in element_losses])
        losses_err = [np.std(vals) for vals in element_losses]

        curr_ax = ax[
            0,
            num_objects.index(simulation.context_size[0])
            # num_functions.index(simulation.num_functions),
            # 0
            # context_sizes.index(
            #     tuple(simulation.context_size)
            #     if isinstance(simulation.context_size, list)
            #     else simulation.context_size
            # ),
        ]

        x_ticks = np.arange(len(simulation.message_sizes))
        x_labels = []
        for message_size in simulation.message_sizes:
            label = str(message_size)
            if message_size == simulation.context_size:
                label += "\nC"
            x_labels.append(label)

        bar_width = 0.3
        x = x_ticks + (bar_width * (object_sizes.index(simulation.object_size) - 1))
        curr_ax.set(
            xlabel=f"M", ylabel=metric, title=f"C=({simulation.context_size[0]},O)",
        )
        curr_ax.bar(
            x,
            losses_mean,
            width=bar_width,
            label=f"O={simulation.object_size}, F={simulation.num_functions}",
            yerr=losses_err,
            capsize=4,
            color=f"C{object_sizes.index(simulation.object_size)}",
        )
        curr_ax.set_xticks(x_ticks)
        curr_ax.set_xticklabels(x_labels)

        global_loss_max = max(global_loss_max, losses_mean.max())
        try:
            curr_ax.axvline(
                x=simulation.message_sizes.index(simulation.context_size),
                linestyle="--",
                color="gray",
            )
        except ValueError:
            pass

    y_interval = round(
        global_loss_max / 10, int(np.abs(np.floor(np.log10(global_loss_max / 10))))
    )
    for _, curr_ax in np.ndenumerate(ax):
        curr_ax.set_yticks(np.arange(0.0, global_loss_max + y_interval, y_interval))

    ax[0, -1].legend(ncol=1, loc="lower right", bbox_to_anchor=(1, 1))
    plt.show()
