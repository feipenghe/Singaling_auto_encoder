import simulations
import utils


def make_belief_update_simulation(
    context_size, object_size, num_functions, message_sizes, shared_context
):
    return simulations.Simulation(
        name=f"belief_update_game_c{context_size}_o{object_size}_f{num_functions}_m{utils.join_vals(message_sizes)}_sharedcontext{int(shared_context)}",
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
