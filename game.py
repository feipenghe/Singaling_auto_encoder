import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class UpdateNetwork(nn.Module):
    def __init__(self, situation_size, information_size, prediction_size, hidden_sizes=(64,), use_situation=True):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.use_situation = use_situation

        if self.use_situation:
            input_size = situation_size + information_size
        else:
            input_size = information_size
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, self.hidden_sizes[0])])
        for i, hidden_size in enumerate(self.hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i], hidden_size))
        self.hidden_layers.append(nn.Linear(self.hidden_sizes[-1], prediction_size))

        logging.info("Update network:")
        logging.info(f"Prediction size: {prediction_size}")
        logging.info(f"Hidden layers:\n{self.hidden_layers}")

    def forward(self, situation, information):
        if self.use_situation:
            input = torch.cat((situation, information), dim=1)
        else:
            input = information

        output = F.relu(self.hidden_layers[0](input))
        for hidden_layer in self.hidden_layers[1:]:
            output = F.relu(hidden_layer(output))
        return output


class Game(nn.Module):
    def __init__(self, situation_size, information_size, message_size, prediction_size, hidden_sizes=(64, 64),
                 update_network_hidden_sizes=(64,), use_situation=True):
        super().__init__()
        self.situation_size = situation_size
        self.information_size = information_size
        self.message_size = message_size
        self.prediction_size = prediction_size
        self.hidden_sizes = hidden_sizes
        self.update_network_hidden_sizes = update_network_hidden_sizes
        self.use_situation = use_situation

        self.update_network = UpdateNetwork(situation_size, information_size, prediction_size, update_network_hidden_sizes, use_situation)

        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.loss_list = []

        if self.use_situation:
            encoder_input_size = self.situation_size + self.information_size
            decoder_input_size = self.message_size + self.situation_size
        else:
            encoder_input_size = self.information_size
            decoder_input_size = self.message_size

        encoder_layer_dimensions = [(encoder_input_size, self.hidden_sizes[0])]
        decoder_layer_dimensions = [(decoder_input_size, self.hidden_sizes[0])]

        for i, hidden_size in enumerate(self.hidden_sizes[1:]):
            hidden_shape = (self.hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)
            decoder_layer_dimensions.append(hidden_shape)
        encoder_layer_dimensions.append((self.hidden_sizes[-1], self.message_size))
        decoder_layer_dimensions.append((self.hidden_sizes[-1], self.prediction_size))

        self.encoder_hidden_layers = nn.ModuleList([nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions])
        self.decoder_hidden_layers = nn.ModuleList([nn.Linear(*dimensions) for dimensions in decoder_layer_dimensions])

        logging.info("Game details:")
        logging.info(f"\nSituation size: {situation_size}\nInformation size: {information_size}\nMessage size: {message_size}\nPrediction size: {prediction_size}")
        logging.info(f"Use situation: {use_situation}")
        logging.info(f"Encoder layers:\n{self.encoder_hidden_layers}")
        logging.info(f"Decoder layers:\n{self.decoder_hidden_layers}")

    def play(self, num_epochs=1000, mini_batch_size=1000, loss_every=None):
        for learning_rate in [.01, .001, .0001]:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                situations = self.generate_situations(mini_batch_size)
                information = self.generate_information(mini_batch_size)

                loss = self.loss(situations, information)
                loss.backward()
                optimizer.step()

                self.epoch += 1

                if loss_every is not None:
                    if epoch % loss_every == 0:
                        self.loss_list.append((self.epoch, loss.item()))
                else:
                    if epoch == 0:
                        self.loss_list.append((self.epoch, loss.item()))

            self.loss_list.append((self.epoch, loss.item()))

            logging.info(f"Epoch {self.loss_list[-1][0]}:\t{self.loss_list[-1][1]:.2e}")

    def _encoder_forward_pass(self, situation, information):
        if self.use_situation:
            encoder_input = torch.cat((situation, information), dim=1)
        else:
            encoder_input = information

        message = encoder_input
        for hidden_layer in self.encoder_hidden_layers[:-1]:
            message = F.relu(hidden_layer(message))
        message = self.encoder_hidden_layers[-1](message)

        return message

    def _decoder_forward_pass(self, message, situation):
        if self.use_situation:
            decoder_input = torch.cat((message, situation), dim=1)
        else:
            decoder_input = message

        prediction = decoder_input
        for hidden_layer in self.decoder_hidden_layers[:-1]:
            prediction = F.relu(hidden_layer(prediction))
        prediction = self.decoder_hidden_layers[-1](prediction)

        return prediction

    def forward(self, situation, information):
        message = self._encoder_forward_pass(situation, information)
        prediction = self._decoder_forward_pass(message, situation)
        return prediction, message

    def predict_by_message(self, message, situation):
        with torch.no_grad():
            return self._decoder_forward_pass(message, situation)

    def target(self, situation, information):
        with torch.no_grad():
            return self.update_network.forward(situation, information)

    def message(self, situation, information):
        with torch.no_grad():
            return self._encoder_forward_pass(situation, information)

    def loss(self, situation, information):
        target = self.target(situation, information)
        prediction, message = self.forward(situation, information)
        return self.criterion(prediction, target)

    def generate_situations(self, batch_size):
        return torch.randn(batch_size, self.situation_size)

    def generate_information(self, batch_size):
        """Generate `batch_size` one-hot vectors of dimension `information_size`."""
        information = torch.zeros((batch_size, self.information_size))
        for i, one_hot_idx in enumerate(torch.randint(self.information_size, (batch_size,))):
            information[i][one_hot_idx] = 1
        return information
