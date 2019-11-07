import dataclasses
import itertools
import logging
import pathlib
import pickle
from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

import game
import utils

utils.setup_logging()

ContextSizeType = Union[Tuple[int, int], int]


@dataclasses.dataclass()
class Simulation:
    name: Text
    context_size: ContextSizeType
    object_size: int
    num_functions: int
    message_sizes: Iterable[int]
    target_function: Optional[Callable] = None
    context_generator: Optional[Callable] = None
    use_context: bool = True
    shared_context: bool = True
    decoder_shuffle_context: bool = False

    num_trials: int = 3
    mini_batch_size: int = 64
    num_batches: int = 1000


def make_belief_update_simulation(
    context_size, object_size, num_functions, message_sizes, shared_context
):
    return Simulation(
        name=f"belief_update_game_c{context_size}_o{object_size}_f{num_functions}_m{utils.join_ints(message_sizes)}_sharedcontext{int(shared_context)}",
        context_size=context_size,
        object_size=object_size,
        num_functions=num_functions,
        shared_context=shared_context,
        message_sizes=message_sizes,
    )


belief_update_simulation = make_belief_update_simulation(
    context_size=10,
    object_size=10,
    num_functions=4,
    message_sizes=(1, 2, 4, 6, 8, 10),
    shared_context=True,
)


def make_referential_game_simulation(
    object_size, context_size, num_functions, message_sizes
):
    # TODO how to choose the number of functions?
    functions = torch.randn(num_functions, context_size)

    def referential_game_target_function(context, function_selectors):
        # TODO Check correctness.
        selected_functions = torch.matmul(function_selectors.unsqueeze(1), functions)
        objects = torch.matmul(selected_functions, context).squeeze(1)
        return objects

    return Simulation(
        name="referential_game",
        object_size=object_size,
        num_functions=num_functions,
        context_size=(context_size, object_size),
        message_sizes=message_sizes,
        num_trials=3,
        target_function=referential_game_target_function,
    )


referential_game_simulation = make_referential_game_simulation(
    object_size=2, context_size=10, num_functions=4, message_sizes=(1, 2, 4, 6, 10)
)


def visualize_game(game_: game.Game):
    game_.plot_messages_information()
    game_.clusterize_messages(visualize=True)


def run_simulation(simulation: Simulation, visualize: bool = False):
    logging.info(f"Running simulation: {simulation}")
    network_losses: List[List[float]] = []
    prediction_by_messages_losses: List[List[Dict[Text, float]]] = []
    unsupervised_clustering_losses: List[List[float]] = []

    for message_size in simulation.message_sizes:
        network_losses_per_trial = []
        prediction_losses_per_trial = []
        unsupervised_losses_per_trial = []

        for _ in range(simulation.num_trials):
            current_game: game.Game = game.Game(
                context_size=simulation.context_size,
                object_size=simulation.object_size,
                message_size=message_size,
                num_functions=simulation.num_functions,
                use_context=simulation.use_context,
                shared_context=simulation.shared_context,
                decoder_shuffle_context=simulation.decoder_shuffle_context,
                target_function=simulation.target_function,
                context_generator=simulation.context_generator,
            )
            current_game.play(
                num_batches=simulation.num_batches,
                mini_batch_size=simulation.mini_batch_size,
            )
            if visualize:
                visualize_game(current_game)

            element_losses = {
                element: current_game.predict_element_by_messages(element)
                for element in (
                    "functions",
                    "object_by_context",
                    "object_by_decoder_context",
                    "context",
                    "decoder_context",
                )
            }
            prediction_losses_per_trial.append(element_losses)
            unsupervised_losses_per_trial.append(
                current_game.clusterize_messages(visualize=visualize)
            )
            network_losses_per_trial.append(current_game.loss_list[-1][1])

        network_losses.append(network_losses_per_trial)
        prediction_by_messages_losses.append(prediction_losses_per_trial)
        unsupervised_clustering_losses.append(unsupervised_losses_per_trial)

    simulation_path = pathlib.Path(f"./simulations/{simulation.name}/")
    simulation_path.mkdir(parents=True, exist_ok=True)

    with simulation_path.joinpath("network_losses.pickle").open("wb") as f:
        pickle.dump(network_losses, f)

    with simulation_path.joinpath("prediction_by_messages_losses.pickle").open(
        "wb"
    ) as f:
        pickle.dump(prediction_by_messages_losses, f)

    with simulation_path.joinpath("unsupervised_clustering_loss.pickle").open(
        "wb"
    ) as f:
        pickle.dump(unsupervised_clustering_losses, f)

    with simulation_path.joinpath(f"{simulation.name}.pickle").open(
        "wb"
    ) as f:  # TODO use json
        # Can't pickle nested functions
        del simulation.target_function
        del simulation.context_generator
        pickle.dump(simulation, f)


def run_simulation_set(
    simulation_name: Text,
    simulation_factory: Callable,
    message_sizes: Tuple[int, ...],
    **kwargs,
):
    keys, values = zip(*kwargs.items())

    simulations_grid = list(itertools.product(*values))

    print(f"Running {len(simulations_grid)} total simulations")

    simulation_set_name = f"{simulation_name}_simulations__" + "__".join(
        f"{key}_{utils.str_val(val)}" for key, val in kwargs.items()
    )

    simulations = []
    for grid_values in simulations_grid:
        kw = {k: v for k, v in zip(keys, grid_values)}

        simulation = simulation_factory(message_sizes=message_sizes, **kw)
        run_simulation(simulation)
        simulations.append(simulation)

        with pathlib.Path(f"./simulations/{simulation_set_name}.pickle").open(
            "wb"
        ) as f:
            pickle.dump(simulations, f)


def plot_simulation(simulation_name):
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


def plot_simulation_set(simulation_set_name, element_to_plot):
    with pathlib.Path(f"./simulations/{simulation_set_name}.pickle").open("rb") as f:
        simulations: List[Simulation] = pickle.load(f)

    num_functions = list(sorted(x.num_functions for x in simulations))
    context_sizes = list(sorted(x.context_size for x in simulations))
    object_sizes = list(sorted(x.object_size for x in simulations))

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

    fig, ax = plt.subplots(
        len(num_functions), len(context_sizes), figsize=(18, 12), squeeze=False
    )
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
            num_functions.index(simulation.num_functions),
            context_sizes.index(simulation.context_size),
        ]

        x_ticks = np.arange(len(simulation.message_sizes))
        x_labels = []
        for message_size in simulation.message_sizes:
            label = str(message_size)
            if message_size == simulation.context_size:
                label += "\nC"
            x_labels.append(label)

        bar_width = 0.3
        x = x_ticks + (bar_width * object_sizes.index(simulation.object_size))
        curr_ax.set(
            xlabel=f"M",
            ylabel=metric,
            title=f"F={simulation.num_functions}, C={simulation.context_size}",
        )
        curr_ax.bar(
            x,
            losses_mean,
            width=bar_width,
            label=f"O={simulation.object_size}",
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
