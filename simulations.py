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
    my_game.plot_messages_information(exemplars_size=40)
    my_game.predict_functions_from_messages(exemplars_size=40)


def referential_game():
    pass


if __name__ == "__main__":
    belief_update_game()
