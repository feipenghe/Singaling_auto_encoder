import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from game import Game
from utils import (
    clusterize_messages,
    generate_information_situations_messages,
    plot_losses,
    plot_messages_information,
    plot_pca_3d,
    predict_information_from_messages,
)


def plot_categorical_transition():
    situation_size, information_size, message_size, prediction_size = 10, 4, 1, 10
    game: Game = Game(
        situation_size,
        information_size,
        message_size,
        prediction_size,
        use_context=True,
    )
    game.play()

    _, situations, messages = generate_information_situations_messages(game, 1)
    message_1 = messages[0]
    situation_1 = situations[0]
    message_2 = messages[1]
    situation_2 = situations[1]

    # Categorical transition as M changes

    ts = []
    predictions = []
    for t in np.linspace(0, 1, 100_000):
        message = (t * message_1) + ((1 - t) * message_2)
        situation = (t * situation_1) + ((1 - t) * situation_2)

        prediction = game.output_by_message(
            torch.unsqueeze(message, dim=0), torch.unsqueeze(situation, dim=0)
        )

        ts.append(t)
        predictions.append(prediction.view(-1).numpy())

    predictions = np.array(predictions)
    plot_pca_3d(
        ts,
        predictions,
        xlabel="t",
        ylabel="Component 2",
        zlabel="Component 1",
        title="S+I PCA by M(t)",
    )

    # Linear transition as S+I changes

    ts = []
    predictions = []
    prediction_1 = game.output_by_message(
        torch.unsqueeze(message_1, dim=0), torch.unsqueeze(situation_1, dim=0)
    )
    prediction_2 = game.output_by_message(
        torch.unsqueeze(message_2, dim=0), torch.unsqueeze(situation_2, dim=0)
    )

    for t in np.linspace(0, 1, 100_000):
        prediction = (t * prediction_1) + ((1 - t) * prediction_2)
        ts.append(t)
        predictions.append(prediction.view(-1).numpy())

    predictions = np.array(predictions)
    plot_pca_3d(
        ts,
        predictions,
        xlabel="t",
        ylabel="Component 2",
        zlabel="Component 1",
        title="S+I PCA by S+I(t)",
    )


def plot_loss_by_message_information_ratio():
    start = 1
    end = 10

    situation_size = prediction_size = 10

    ratios = []
    loss_per_ratio = []

    for message_size, information_size in zip(
        range(start, end + 1), range(end, start - 1, -1)
    ):
        ratio = message_size / information_size
        logging.info(
            f"Training information size: {information_size}, message size: {message_size}, ratio: {ratio} "
        )

        game: Game = Game(
            situation_size, information_size, message_size, prediction_size
        )
        game.play()

        loss = clusterize_messages(game, exemplars_size=50)
        logging.info(f"Loss: {loss}")

        ratios.append(ratio)
        loss_per_ratio.append(loss)

        fig, ax = plt.subplots()
        ax.plot(ratios, loss_per_ratio)

        ax.set(
            xlabel="M/I ratio",
            ylabel="Prediction loss (MSE)",
            title="Prediction loss (MSE) by M/I ratio",
        )

        ax.grid()
        plt.show()


def game1():
    situation_size, information_size, message_size, prediction_size, hidden_sizes = (
        10,
        4,
        1,
        10,
        (64, 64),
    )
    game = Game(
        situation_size,
        information_size,
        message_size,
        prediction_size,
        hidden_sizes,
        use_context=True,
    )
    game.play()
    plot_messages_information(game)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game2():
    situation_size, information_size, message_size, prediction_size, hidden_sizes = (
        10,
        4,
        2,
        10,
        (64, 64),
    )
    game = Game(
        situation_size,
        information_size,
        message_size,
        prediction_size,
        hidden_sizes,
        use_context=True,
    )
    game.play()
    plot_messages_information(game)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game3():
    situation_size, information_size, message_size, prediction_size, hidden_sizes = (
        10,
        4,
        2,
        2,
        (64, 64),
    )
    game = Game(
        situation_size,
        information_size,
        message_size,
        prediction_size,
        hidden_sizes,
        use_context=True,
    )
    game.play()
    plot_messages_information(game, 40)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game3b():
    situation_size, message_size, prediction_size, func_size, hidden_size = (
        10,
        2,
        2,
        4,
        64,
    )
    game = Game(situation_size, message_size, prediction_size, func_size, hidden_size)
    game.play()
    plot_messages_information(game)
    predict_information_from_messages(game)
    clusterize_messages(game)


def game4():
    situation_size, message_size, prediction_size, func_size, hidden_size = (
        10,
        2,
        10,
        4,
        64,
    )
    game = Game(
        situation_size, message_size, prediction_size, func_size, hidden_size, -1
    )
    game.play()
    plot_messages_information(game, 40)


def game5():
    situation_size, message_size, prediction_size, func_size, hidden_size = (
        10,
        2,
        10,
        20,
        64,
    )
    game = Game(
        situation_size, message_size, prediction_size, func_size, hidden_size, 1.2
    )
    game.play()
    plot_messages_information(game, 40)


def game6():
    situation_size, message_size, prediction_size, func_size, hidden_size = (
        10,
        2,
        10,
        20,
        64,
    )

    num_reproductions = 10
    all_losses = []

    for _ in range(num_reproductions):
        game = Game(
            situation_size, message_size, prediction_size, func_size, hidden_size, -1
        )
        print_first = True
        for lr in [0.01, 0.001, 0.0001]:
            play_game(
                game, 1000, learning_rate=lr, func_out_training=[0, 1, 2, 3, 5, 7]
            )
            if print_first:
                logging.info(
                    f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}"
                )
                print_first = False
            logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")

        all_losses.append(get_loss_per_function(game))
        # plot_messages_information(game, 40)

    all_losses = np.array(all_losses)
    loss_average_per_func = np.average(all_losses, axis=0)

    A = plot_losses(game, loss_average_per_func)


def game7():
    situation_size, message_size, prediction_size, func_size, hidden_size = (
        10,
        2,
        10,
        4,
        64,
    )
    game = Game(situation_size, message_size, prediction_size, func_size, hidden_size)
    print_first = True
    for lr in [0.01, 0.001, 0.0001]:
        play_game(game, 1000, learning_rate=lr)
        if print_first:
            logging.info(f"Epoch {game.loss_list[0][0]}:\t{game.loss_list[0][1]:.2e}")
            print_first = False
        logging.info(f"Epoch {game.loss_list[-1][0]}:\t{game.loss_list[-1][1]:.2e}")
    plot_messages_information(game, 40)

    # Compute the average messages
    game.average_messages(100)

    replications_per_func = 10
    situations = torch.randn(
        replications_per_func * game.func_size, game.situation_size
    )
    func_switches = torch.cat(
        [torch.arange(game.func_size) for _ in range(replications_per_func)]
    )
    targets = game.target(situations, func_switches)

    LOSS = game.criterion(
        game.discrete_forward(situations, func_switches), targets
    ).item()
    logging.info(f"Loss: {LOSS:.2e}")


if __name__ == "__main__":
    pass
