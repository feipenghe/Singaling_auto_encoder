import logging

import numpy as np
import torch

from game import Game, play_game
from utils import (clusterize_messages, get_loss_per_function, plot_losses,
                   plot_messages_information,
                   predict_information_from_messages)


def game1():
    situation_size, information_size, message_size, prediction_size, hidden_sizes = 10, 4, 1, 10, (64, 64)
    game = Game(situation_size, information_size, message_size, prediction_size, hidden_sizes, use_situation=True)
    print_first = True
    for lr in [.01, .001, .0001]:
        play_game(game, num_epochs=1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game2():
    situation_size, information_size, message_size, prediction_size, hidden_sizes = 10, 4, 2, 10, (64, 64)
    game = Game(situation_size, information_size, message_size, prediction_size, hidden_sizes, use_situation=True)
    print_first = True

    for lr in [.01, .001, .0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")

    plot_messages_information(game)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game3():
    situation_size, information_size, message_size, prediction_size, hidden_sizes = 10, 4, 2, 2, (64, 64)
    game = Game(situation_size, information_size, message_size, prediction_size, hidden_sizes, use_situation=True)
    print_first = True
    for lr in [.01, .001, .0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game, 40)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game3b():
    situation_size, message_size, prediction_size, func_size, hidden_size = 10, 2, 2, 4, 64
    game = Game(situation_size, message_size, prediction_size, func_size, hidden_size)
    print_first = True
    for lr in [.01, .001, .0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game4():
    situation_size, message_size, prediction_size, func_size, hidden_size = 10, 2, 10, 4, 64
    game = Game(situation_size, message_size, prediction_size, func_size, hidden_size, -1)
    print_first = True
    for lr in [.01, .001, .0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game, 40)


def game5():
    situation_size, message_size, prediction_size, func_size, hidden_size = 10, 2, 10, 20, 64
    game = Game(situation_size, message_size, prediction_size, func_size, hidden_size, 1.2)
    print_first = True
    for lr in [.01, .001, .0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game, 40)


def game6():
    situation_size, message_size, prediction_size, func_size, hidden_size = 10, 2, 10, 20, 64

    num_reproductions = 10
    all_losses = []

    for _ in range(num_reproductions):
        game = Game(situation_size, message_size, prediction_size, func_size, hidden_size, -1)
        print_first = True
        for lr in [.01, .001, .0001]:
            play_game(game, 1000, learning_rate=lr, func_out_training=[0, 1, 2, 3, 5, 7])
            if print_first:
                logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
                print_first = False
            logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")

        all_losses.append(get_loss_per_function(game))
        # plot_messages_information(game, 40)

    all_losses = np.array(all_losses)
    loss_average_per_func = np.average(all_losses, axis=0)

    A = plot_losses(game, loss_average_per_func)


def game7():
    situation_size, message_size, prediction_size, func_size, hidden_size = 10, 2, 10, 4, 64
    game = Game(situation_size, message_size, prediction_size, func_size, hidden_size)
    print_first = True
    for lr in [.01, .001, .0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game, 40)

    # Compute the average messages
    game.average_messages(100)

    replications_per_func = 10
    situations = torch.randn(replications_per_func * game.func_size, game.situation_size)
    func_switches = torch.cat([torch.arange(game.func_size) for _ in range(replications_per_func)])
    targets = game.target(situations, func_switches)

    LOSS = game.criterion(game.discrete_forward(situations, func_switches), targets).item()
    logging.info(f"Loss: {LOSS:.2e}")


if __name__ == "__main__":
    pass
