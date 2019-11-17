import dataclasses
import itertools
import json
import logging
import multiprocessing
import pathlib
from typing import Callable, Dict, Iterable, List, Optional, Text, Tuple, Union


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
    context_generator: Callable = None
    use_context: bool = True
    shared_context: bool = True
    shuffle_decoder_context: bool = False

    num_trials: int = 3
    mini_batch_size: int = 64
    num_batches: int = 10_000

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


def load_simulation(simulation_name: Text) -> Simulation:
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
