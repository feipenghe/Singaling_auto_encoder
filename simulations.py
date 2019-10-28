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


def referential_game_target_function(context, function_selectors):
    """Selects one row of context (i.e. an object).
    Output shape: (batch_size, object_size)."""
    return torch.matmul(function_selectors.unsqueeze(1), context).squeeze(dim=1)


referential_game = Simulation(name="referential_game",
                              object_size=10,
                              num_functions=6,
                              context_size=(6, 10),
                              message_sizes=range(1, 11),
                              target_function=referential_game_target_function)


def extremity_game_target_function(context, function_selectors: torch.Tensor):
    """Selects one row of context (i.e. an object).
    Output shape: (batch_size, object_size)."""
    # TODO
    funcs = function_selectors.argmax(dim=1)
    parity = funcs % 2
    if parity == 0:
        pass
    else:
        pass

    return torch.matmul(function_selectors.unsqueeze(1), context).squeeze(dim=1)


def extremity_game_context_generator(batch_size, context_shape):
    # TODO
    num_objects = context_shape[0]
    num_properties = context_shape[1]

    assert num_objects <= 2 * num_properties

    contexts = torch.randn(batch_size, *context_shape)

    for i in range(batch_size):
        context = contexts[i]
        for obj_idx in range(0, num_objects, 2):
            prop_vals = context[:, obj_idx]
            where_min = torch.argmin(prop_vals)
            where_max = torch.argmax(prop_vals)

            # context[obj_idx, ???]

            where_min = torch.argmin(context[:,obj_idx])
            context[obj_idx, where_min]
            where_max = torch.argmax(context[:,obj_idx+1])


extremity_game = Simulation(name="extremity_game",
                            object_size=3,
                            num_functions=6,
                            context_size=(6, 3),
                            message_sizes=range(1, 11),
                            target_function=extremity_game_target_function)


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
