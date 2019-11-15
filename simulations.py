import dataclasses
import itertools
import json
import logging
import multiprocessing
import pathlib
import pickle
from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import dataclasses_json
import game
import utils

utils.setup_logging()

ContextSizeType = Union[int, Tuple[int, int]]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Simulation:
    name: Text
    context_size: ContextSizeType
    object_size: int
    num_functions: int
    message_sizes: Iterable[int]
    target_function: Callable
    context_generator: Callable
    use_context: bool = True
    shared_context: bool = True
    shuffle_decoder_context: bool = False

    num_trials: int = 3
    mini_batch_size: int = 64
    num_batches: int = 1000

    epoch_nums: List[int] = dataclasses.field(default_factory=list)

    """ Results """

    # {Message size -> Trial x {parameter -> loss}}
    prediction_by_message_losses: Dict[
        int, List[Dict[Text, float]]
    ] = dataclasses.field(default_factory=dict)
    # {Message size -> Trial x Epoch x loss}
    training_losses: Dict[int, List[List[float]]] = dataclasses.field(
        default_factory=dict
    )
    # {Message size -> Trial x loss}
    unsupervised_clustering_losses: Dict[int, List[float]] = dataclasses.field(
        default_factory=dict
    )


def _load_simulation(simulation_name: Text) -> Simulation:
    return Simulation.from_dict(
        json.load(pathlib.Path(f"./simulations/{simulation_name}.json").open())
    )


def _save_simulation(simulation: Simulation):
    # Can't serialize functions.
    simulation.target_function = None
    simulation.context_generator = None

    simulation_path = pathlib.Path(f"./simulations/")
    simulation_path.mkdir(parents=True, exist_ok=True)
    with simulation_path.joinpath(f"{simulation.name}.json").open("w") as f:
        json.dump(simulation.to_dict(), f, indent=1)


def run_simulation(
    simulation: Simulation, visualize: bool = False
) -> List[List[game.Game]]:
    logging.info(f"Running simulation: {simulation}")

    # [Message size x Trial x game]
    games: List[List[game.Game]] = []

    for message_size in simulation.message_sizes:
        # [Trial x epoch x loss]
        training_losses_per_trial: List[List[float]] = []
        # [Trial x {parameter name: loss}]
        prediction_loss_per_trial: List[Dict[Text, float]] = []
        # [Trial x loss]
        unsupervised_loss_per_trial = []

        game_per_trial: List[game.Game] = []

        for _ in range(simulation.num_trials):
            current_game: game.Game = game.Game(
                context_size=simulation.context_size,
                object_size=simulation.object_size,
                message_size=message_size,
                num_functions=simulation.num_functions,
                use_context=simulation.use_context,
                shared_context=simulation.shared_context,
                shuffle_decoder_context=simulation.shuffle_decoder_context,
                target_function=simulation.target_function,
                context_generator=simulation.context_generator,
            )
            current_game.play(
                num_batches=simulation.num_batches,
                mini_batch_size=simulation.mini_batch_size,
            )
            if visualize:
                current_game.visualize()

            element_losses = {
                element: current_game.predict_element_by_messages(element)
                for element in (
                    "functions",
                    "min_max",
                    "dimension",
                    "sanity",
                    "object_by_context",
                    "object_by_decoder_context",
                    "context",
                    "decoder_context",
                )
            }
            prediction_loss_per_trial.append(element_losses)
            unsupervised_loss_per_trial.append(
                current_game.clusterize_messages(visualize=visualize)
            )
            training_losses_per_trial.append(current_game.loss_per_epoch)

            game_per_trial.append(current_game)

        simulation.training_losses[message_size] = training_losses_per_trial
        simulation.prediction_by_message_losses[
            message_size
        ] = prediction_loss_per_trial

        simulation.unsupervised_clustering_losses[
            message_size
        ] = unsupervised_loss_per_trial

        games.append(game_per_trial)

    simulation.epoch_nums = games[0][0].epoch_nums
    _save_simulation(simulation)
    return games


def run_simulation_set(
    simulation_name: Text,
    simulation_factory: Callable,
    message_sizes: Tuple[int, ...],
    num_processes: Optional[int] = None,
    **kwargs,
):
    keys, values = zip(*kwargs.items())

    simulations_grid = list(itertools.product(*values))

    logging.info(f"Running {len(simulations_grid) * len(message_sizes)} total games")

    simulations = []
    for grid_values in simulations_grid:
        kw = {k: v for k, v in zip(keys, grid_values)}

        simulation = simulation_factory(message_sizes=message_sizes, **kw)
        simulations.append(simulation)

    simulation_set_name = f"{simulation_name}_simulations__" + "__".join(
        f"{key}_{utils.str_val(val)}" for key, val in kwargs.items()
    )

    with pathlib.Path(f"./simulations/{simulation_set_name}.json").open("w") as f:
        json.dump(
            [
                dataclasses.replace(
                    x, target_function=None, context_generator=None
                ).to_dict()
                for x in simulations
            ],
            f,
            indent=2,
        )

    if num_processes is not None:
        pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=1)
        pool.map(run_simulation, simulations)
    else:
        for simulation in simulations:
            run_simulation(simulation)


def plot_simulation(simulation_name: Text):
    simulation_path = pathlib.Path("./simulations/{simulation.name}/")
    with simulation_path.joinpath("supervised_clustering_accuracies.pickle").open(
        "rb"
    ) as f:
        supervised_clustering_accuracies = pickle.load(f)

    with simulation_path.joinpath("unsupervised_clustering_loss.pickle").open(
        "rb"
    ) as f:
        unsupervised_clustering_losses = pickle.load(f)

    with simulation_path.joinpath(f"{simulation_name}.pickle").open(
        "rb"
    ) as f:  # TODO use json
        simulation: Simulation = pickle.load(f)

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
    max_epochs: Optional[int] = None,
    epoch_interval: Optional[int] = 100,
    label_interval: int = 10,
):
    display_name_to_simulation: Dict[Text, Simulation] = {
        display_name: _load_simulation(file_name)
        for display_name, file_name in simulation_display_name_to_file_name.items()
    }

    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(f"Training losses (MSE)")

    global_max_loss = 0.0

    for s, (simulation_display_name, simulation) in enumerate(
        display_name_to_simulation.items()
    ):
        simulation = Simulation.from_dict(
            json.load(pathlib.Path(f"./simulations/{simulation.name}.json").open())
        )

        epoch_nums = np.array(simulation.epoch_nums)

        for m, message_size in enumerate(simulation.message_sizes):
            # TODO use message size information

            training_losses_per_trial: List[List[float]] = simulation.training_losses[
                message_size
            ]

            losses_mean_per_epoch = np.mean(np.array(training_losses_per_trial), axis=0)
            losses_err_per_epoch = np.std(np.array(training_losses_per_trial), axis=0)

            global_max_loss = max(global_max_loss, losses_mean_per_epoch.max())

            if max_epochs is not None:
                epoch_nums = epoch_nums[:max_epochs]
                losses_mean_per_epoch = losses_mean_per_epoch[:max_epochs]
                losses_err_per_epoch = losses_err_per_epoch[:max_epochs]

            if epoch_interval is not None:
                idxs = np.arange(0, epoch_nums[-1], epoch_interval)
                # idxs = np.append(idxs, -1)  # Always plot last epoch
                epoch_nums = epoch_nums[idxs]
                losses_mean_per_epoch = losses_mean_per_epoch[idxs]
                losses_err_per_epoch = losses_err_per_epoch[idxs]

            bar_width = 0.5

            x_tick_idxs = np.arange(0, len(epoch_nums))

            x = x_tick_idxs + (bar_width * (-0.5 if s == 0 else 0.5))

            x_labels = [
                str(epoch_nums[i])
                if i % label_interval == 0 or i + 1 == len(epoch_nums)
                else ""
                for i in range(len(epoch_nums))
            ]

            ax.bar(
                x,
                losses_mean_per_epoch,
                width=bar_width,
                yerr=losses_err_per_epoch,
                capsize=1.5,
                label=simulation_display_name,
                color=f"C{s}",
            )
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
    with pathlib.Path(f"./simulations/{simulation_set_name}.json").open("r") as f:
        simulations = [Simulation.from_dict(x) for x in json.load(f)]

    num_functions = list(sorted(set(x.num_functions for x in simulations)))
    context_sizes = [
        tuple(x) if isinstance(x, list) else x
        for x in (x.context_size for x in simulations)
    ]
    context_sizes = list(sorted(set(context_sizes)))
    object_sizes = list(sorted(set(x.object_size for x in simulations)))
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

    for simulation in simulations:
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
        simulations = [Simulation.from_dict(x) for x in json.load(f)]

    num_functions = list(sorted(set(x.num_functions for x in simulations)))
    context_sizes = [
        tuple(x) if isinstance(x, list) else x
        for x in (x.context_size for x in simulations)
    ]
    context_sizes = list(sorted(set(context_sizes)))
    object_sizes = list(sorted(set(x.object_size for x in simulations)))
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

    for simulation in simulations:
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
