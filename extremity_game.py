import torch

import simulations
import utils


def strict_context_generator(batch_size, context_shape):
    object_size = context_shape[1]

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

        contexts.append(context)

    batch = torch.stack(contexts)
    # Shuffle objects.
    batch = batch[:, torch.randperm(batch.shape[1]), :]
    return batch


def extremity_game_target_function(context, function_selectors):
    func_idxs = function_selectors.argmax(dim=1)
    func_min_or_max = func_idxs % 2
    param_idxs = func_idxs // context.shape[2]  # Number of params.

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
    object_size,
    message_sizes,
    shared_context,
    strict_context=True,
    num_objects=None,
    **kwargs,
):
    if strict_context:
        num_objects = 2 * object_size

    context_size = (num_objects, object_size)
    num_functions = 2 * object_size

    return simulations.Simulation(
        name=f"extremity_game__o_{object_size}__c_{context_size}__m_{utils.join_ints(message_sizes)}__sharedcontext_{int(shared_context)}__strict_context_{int(strict_context)}",
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
        message_sizes=(1, 2, 4, 6, 8, 10),
        object_size=(2, 3, 4,),
        strict_context=(True,),
        shared_context=(True,),
    )
