from typing import Optional, Tuple

import numpy as np
import torch

import simulations
import utils


def strict_context_generator(
    batch_size: int, context_shape: Tuple[int, int]
) -> torch.Tensor:
    object_size = context_shape[1]
    num_objects = object_size * 2

    context = np.random.random(size=(batch_size, *context_shape))

    argmins = context.argmin(axis=1)
    argmaxs = context.argmax(axis=1)

    batch_indexing = np.concatenate([[x] * object_size for x in range(batch_size)] * 2)

    extreme_idxs = (
        batch_indexing,
        np.concatenate((argmins.reshape(-1), argmaxs.reshape(-1))),
        list(range(object_size)) * 2 * batch_size,
    )

    goal_idxs = (
        batch_indexing,
        (list(range(object_size)) * batch_size)
        + (list(range(object_size, num_objects)) * batch_size),
        list(range(object_size)) * 2 * batch_size,
    )

    context[extreme_idxs], context[goal_idxs] = (
        context[goal_idxs],
        context[extreme_idxs],
    )

    # """Correctness test. """
    # for b in range(batch_size):
    #     for row in range(num_objects):
    #         if row // object_size == 0:
    #             assert (
    #                 context[b, row, row % object_size]
    #                 == context[b, :, row % object_size].in()
    #             )
    #         else:
    #             assert (
    #                 context[b, row, row % object_size]
    #                 == context[b, :, row % object_size].max()
    #             )

    context = context[:, np.random.permutation(num_objects), :]  # Shuffle objects.
    return torch.from_numpy(context).float()


def extremity_game_target_function(
    context: torch.Tensor, function_selectors: torch.Tensor
) -> torch.Tensor:
    func_idxs = function_selectors.argmax(dim=1)
    func_min_or_max = func_idxs % 2
    param_idxs = func_idxs // 2

    min_obj_per_param = context.argmin(dim=1)
    max_obj_per_param = context.argmax(dim=1)

    targets = []
    for batch in range(context.shape[0]):
        if func_min_or_max[batch] == 0:
            targets.append(context[batch, min_obj_per_param[batch][param_idxs[batch]]])
        else:
            targets.append(context[batch, max_obj_per_param[batch][param_idxs[batch]]])
    return torch.stack(targets)


def make_extremity_game_simulation(
    object_size: int,
    message_sizes: Tuple[int, ...],
    shared_context: bool,
    strict_context: bool = True,
    num_objects: Optional[int] = None,
    **kwargs,
) -> simulations.Simulation:
    if strict_context:
        num_objects = 2 * object_size
    else:
        assert num_objects is not None, "Must set num_objects if context is not strict."

    context_size = (num_objects, object_size)
    num_functions = 2 * object_size

    return simulations.Simulation(
        name=f"extremity_game__o_{object_size}__c_{context_size}__m_{utils.join_vals(message_sizes)}__sharedcontext_{int(shared_context)}__strict_context_{int(strict_context)}",
        object_size=object_size,
        num_functions=num_functions,
        context_size=context_size,
        shared_context=shared_context,
        shuffle_decoder_context=True,
        message_sizes=message_sizes,
        num_trials=1,
        context_generator=strict_context_generator if strict_context else None,
        target_function=extremity_game_target_function,
        num_batches=10_000,
        mini_batch_size=32,
    )


if __name__ == "__main__":
    # extremity_sim = make_extremity_game_simulation(
    #     object_size=3, message_sizes=(6,), strict_context=True, shared_context=True,
    # )
    # simulations.run_simulation(extremity_sim, visualize=True)

    simulations.run_simulation_set(
        "extremity_game",
        make_extremity_game_simulation,
        num_processes=4,
        message_sizes=(1, 2, 4, 6, 8, 10),
        object_size=(2, 3, 4,),
        strict_context=(True,),
        shared_context=(True,),
    )
