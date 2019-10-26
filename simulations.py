import torch

import game
import utils

utils.setup_logging()


def belief_update_game():
    context_size = 10
    object_size = context_size  # In update game, O=C
    num_functions = 4
    message_size = 2

    my_game: game.Game = game.Game(context_size, object_size, message_size, num_functions, use_context=True)
    my_game.play()
    my_game.plot_messages_information()
    my_game.predict_functions_from_messages()
    my_game.clusterize_messages()


def referential_game():
    object_size = 6
    num_functions = 4
    message_size = 2
    context_size = (num_functions, object_size)

    def target_function(context, function_selectors):
        """Selects one row of context (i.e. an object).
        Output shape: (batch_size, object_size)."""
        return torch.matmul(function_selectors.unsqueeze(1), context).squeeze(dim=1)

    ref_game: game.Game = game.Game(context_size, object_size, message_size, num_functions, target_function=target_function, use_context=True)
    ref_game.play()
    ref_game.plot_messages_information()
    ref_game.predict_functions_from_messages()
    ref_game.clusterize_messages()


if __name__ == "__main__":
    referential_game()
