import dataclasses
import pickle
from typing import Callable, Iterable, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

import game
import utils

utils.setup_logging()


@dataclasses.dataclass()
class Simulation:
    name: Text
    context_size: Union[Tuple, int]
    object_size: int
    num_functions: int
    message_sizes: Iterable[int]
    target_function: Optional[Callable] = None
    context_generator: Optional[Callable] = None
    use_context: bool = True
    shared_context: bool = True
    num_trials: int = 3


belief_update_game = Simulation(name="belief_game",
                                context_size=2,
                                object_size=2,
                                num_functions=2,
                                message_sizes=(1, 2, 4, 6, 8))


def make_referential_game_simulation(object_size, context_size, num_functions, message_sizes):
    # TODO how to choose the number of functions?
    functions = torch.randn(num_functions, context_size)

    def referential_game_target_function(context, function_selectors):
        selected_functions = torch.matmul(function_selectors.unsqueeze(1), functions)
        objects = torch.matmul(selected_functions, context).squeeze(1)
        return objects

    return Simulation(name="referential_game",
                      object_size=object_size,
                      num_functions=num_functions,
                      context_size=(context_size, object_size),
                      message_sizes=message_sizes,
                      num_trials=3,
                      target_function=referential_game_target_function)


referential_game_simulation = make_referential_game_simulation(object_size=2,
                                                               context_size=10,
                                                               num_functions=4,
                                                               message_sizes=(1, 2, 4, 6, 10))


def make_extremity_game_simulation(object_size, message_sizes):
    num_objects = 2 * object_size
    num_functions = num_objects
    context_size = (num_objects, object_size)

    def extremity_game_context_generator(batch_size, context_shape: Tuple[int, ...]):
        contexts = []
        for _ in range(batch_size):
            context = torch.randn(*context_shape)

            argmins = context.argmin(dim=0)
            argmaxs = context.argmax(dim=0)

            for p, argmin, argmax in zip(range(object_size), argmins, argmaxs):
                temp = context[p * 2, p]  # min
                context[p * 2, p] = context[argmin, p]
                context[argmin, p] = temp

                temp = context[p * 2 + 1, p]  # max
                context[p * 2 + 1, p] = context[argmax, p]
                context[argmax, p] = temp

            context = context[torch.randperm(context.shape[0])]  # Shuffle rows
            contexts.append(context)

        return torch.stack(contexts)

    def extremity_game_target_function(context, function_selectors):
        # TODO Make more efficient+readable.

        func_idxs = function_selectors.argmax(dim=1)
        func_min_or_max = func_idxs % 2
        param_idxs = func_idxs // context.shape[2]  # Number of params.

        min_per_param = context.argmin(dim=1)
        max_per_param = context.argmax(dim=1)

        targets = []
        for batch in range(context.shape[0]):
            if func_min_or_max[batch] == 0:
                targets.append(context[batch][min_per_param[batch][param_idxs[batch]]])
            else:
                targets.append(context[batch][max_per_param[batch][param_idxs[batch]]])
        return torch.stack(targets)

    return Simulation(name="extremity_game",
                      object_size=object_size,
                      num_functions=num_functions,
                      context_size=context_size,
                      message_sizes=message_sizes,
                      num_trials=1,
                      context_generator=extremity_game_context_generator,
                      target_function=extremity_game_target_function)


extremity_game_simulation = make_extremity_game_simulation(object_size=2, message_sizes=(1, 2, 3, 4, 5, 6))


def visualize_game(game_: game.Game):
    game_.play()
    game_.plot_messages_information()
    game_.predict_functions_from_messages()
    game_.clusterize_messages(visualize=True)


def run_simulation(simulation: Simulation):
    supervised_clustering_accuracies: List[List[float]] = []
    unsupervised_clustering_losses: List[List[float]] = []
    for message_size in simulation.message_sizes:
        supervised_accuracies = []
        unsupervised_losses = []

        for _ in range(simulation.num_trials):
            current_game: game.Game = game.Game(context_size=simulation.context_size,
                                                object_size=simulation.object_size,
                                                message_size=message_size,
                                                num_functions=simulation.num_functions,
                                                use_context=simulation.use_context,
                                                shared_context=simulation.shared_context,
                                                target_function=simulation.target_function,
                                                context_generator=simulation.context_generator)
            current_game.play()
            supervised_accuracies.append(current_game.predict_functions_from_messages())
            unsupervised_losses.append(current_game.clusterize_messages())

        supervised_clustering_accuracies.append(supervised_accuracies)
        unsupervised_clustering_losses.append(unsupervised_losses)

    with open(f"./simulations/{simulation.name}_supervised_clustering_accuracies.pickle", "wb") as f:
        pickle.dump(supervised_clustering_accuracies, f)

    with open(f"./simulations/{simulation.name}_unsupervised_clustering_loss.pickle", "wb") as f:
        pickle.dump(unsupervised_clustering_losses, f)

    with open(f"./simulations/{simulation.name}.pickle", "wb") as f:  # TODO use json
        # Can't pickle nested functions
        del simulation.target_function
        del simulation.context_generator
        pickle.dump(simulation, f)


def plot_simulation(simulation_name):
    with open(f"./simulations/{simulation_name}_supervised_clustering_accuracies.pickle", "rb") as f:
        supervised_clustering_accuracies = pickle.load(f)

    with open(f"./simulations/{simulation_name}_unsupervised_clustering_loss.pickle", "rb") as f:
        unsupervised_clustering_losses = pickle.load(f)

    with open(f"./simulations/{simulation_name}.pickle", "rb") as f:  # TODO use json
        simulation: Simulation = pickle.load(f)

    accuracy_mean = [np.mean(vals) for vals in supervised_clustering_accuracies]
    accuracy_error = [np.std(vals) for vals in supervised_clustering_accuracies]

    loss_mean = [np.mean(vals) for vals in unsupervised_clustering_losses]
    loss_error = [np.std(vals) for vals in unsupervised_clustering_losses]

    x_labels = []
    if isinstance(simulation.context_size, tuple):
        context_size = simulation.context_size[0]  #TODO: Is this correct?
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
    plt.axvline(x=simulation.message_sizes.index(context_size), linestyle="--", color="gray")
    plt.axvline(x=simulation.message_sizes.index(simulation.object_size), linestyle="--", color="gray")
    plt.xlabel('Message dimensionality', fontsize=5)
    plt.ylabel('F prediction accuracy', fontsize=5)
    plt.title('F prediction accuracy')
    plt.savefig(f"./simulations/{simulation_name}_supervised_classification.png")
    plt.show()

    plt.bar(x_ticks, loss_mean, yerr=loss_error, capsize=10)
    plt.xticks(x_ticks, x_labels, fontsize=5)
    plt.axvline(x=simulation.message_sizes.index(context_size), linestyle="--", color="gray")
    plt.axvline(x=simulation.message_sizes.index(simulation.object_size), linestyle="--", color="gray")
    plt.xlabel('Message dimensionality', fontsize=5)
    plt.ylabel('Clustering loss', fontsize=5)
    plt.title('Clustering loss')
    plt.savefig(f"./simulations/{simulation_name}_unsupervised_clustering.png")
    plt.show()
