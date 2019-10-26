import logging
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import utils


class UpdateNetwork(nn.Module):
    def __init__(self, context_size, num_functions, hidden_sizes=(64,), use_context=True):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.use_context = use_context

        if self.use_context:
            input_size = context_size + num_functions
        else:
            input_size = num_functions
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, self.hidden_sizes[0])])
        for i, hidden_size in enumerate(self.hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i], hidden_size))
        self.hidden_layers.append(nn.Linear(self.hidden_sizes[-1], context_size))

        logging.info("Update network:")
        logging.info(f"Context size: {context_size}")
        logging.info(f"Num functions: {num_functions}")
        logging.info(f"Hidden layers:\n{self.hidden_layers}")

    def forward(self, contexts, function_selectors):
        """`function_selectors` are one-hot vectors representing functions to be applied."""
        if self.use_context:
            input = torch.cat((contexts, function_selectors), dim=-1)
        else:
            input = function_selectors

        output = F.relu(self.hidden_layers[0](input))
        for hidden_layer in self.hidden_layers[1:]:
            output = F.relu(hidden_layer(output))
        return output


def make_update_network_function(context_size, num_functions, update_network_hidden_sizes, use_context):
    update_network = UpdateNetwork(context_size, num_functions, update_network_hidden_sizes, use_context)

    def func(contexts, function_selectors):
        with torch.no_grad():
            return update_network.forward(contexts, function_selectors)
    return func


class Game(nn.Module):
    def __init__(self, context_size, object_size, message_size, num_functions, use_context=True, separate_contexts=False, target_functions: Optional[Tuple[Callable, ...]] = None, hidden_sizes = (64, 64), update_network_hidden_sizes = (64,)):
        super().__init__()
        self.context_size = context_size
        self.object_size = object_size
        self.message_size = message_size
        self.num_functions = num_functions
        self.hidden_sizes = hidden_sizes
        self.update_network_hidden_sizes = update_network_hidden_sizes
        self.use_context = use_context
        self.separate_contexts = separate_contexts

        if target_functions is not None:
            self.target_function = target_functions
        else:
            self.target_function = make_update_network_function(self.context_size, self.num_functions, self.update_network_hidden_sizes, self.use_context)

        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.loss_list = []

        if self.use_context:
            encoder_input_size = self.context_size + self.num_functions
            decoder_input_size = self.message_size + self.context_size
        else:
            encoder_input_size = self.num_functions
            decoder_input_size = self.message_size

        encoder_layer_dimensions = [(encoder_input_size, self.hidden_sizes[0])]
        decoder_layer_dimensions = [(decoder_input_size, self.hidden_sizes[0])]

        for i, hidden_size in enumerate(self.hidden_sizes[1:]):
            hidden_shape = (self.hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)
            decoder_layer_dimensions.append(hidden_shape)
        encoder_layer_dimensions.append((self.hidden_sizes[-1], self.message_size))
        decoder_layer_dimensions.append((self.hidden_sizes[-1], self.object_size))

        self.encoder_hidden_layers = nn.ModuleList([nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions])
        self.decoder_hidden_layers = nn.ModuleList([nn.Linear(*dimensions) for dimensions in decoder_layer_dimensions])

        logging.info("Game details:")
        logging.info(f"\nContext size: {context_size}\nObject size: {object_size}\nMessage size: {message_size}\nNumber of functions: {num_functions}")
        logging.info(f"Use context: {use_context}")
        logging.info(f"Encoder layers:\n{self.encoder_hidden_layers}")
        logging.info(f"Decoder layers:\n{self.decoder_hidden_layers}")

    def play(self, num_epochs=1000, mini_batch_size=1000):
        for learning_rate in [.01, .001, .0001]:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)

            for minibatch_epoch in range(num_epochs):
                optimizer.zero_grad()
                contexts = self.generate_contexts(mini_batch_size)
                if self.separate_contexts:
                    decoder_contexts = self.generate_contexts(mini_batch_size)
                else:
                    decoder_contexts = None
                function_selectors = self.generate_function_selectors(mini_batch_size)

                loss = self.loss(contexts, function_selectors, decoder_contexts)
                loss.backward()
                optimizer.step()

                if minibatch_epoch == 0:
                    self.loss_list.append((self.epoch, loss.item()))
                    logging.info(f"Epoch {self.loss_list[-1][0]}:\t{self.loss_list[-1][1]:.2e}")
                self.epoch += 1

            self.loss_list.append((self.epoch, loss.item()))

        logging.info(f"Epoch {self.loss_list[-1][0]}:\t{self.loss_list[-1][1]:.2e}")

    def _encoder_forward_pass(self, context, function_selector):
        if self.use_context:
            encoder_input = torch.cat((context, function_selector), dim=1)
        else:
            encoder_input = object

        message = encoder_input
        for hidden_layer in self.encoder_hidden_layers[:-1]:
            message = F.relu(hidden_layer(message))
        message = self.encoder_hidden_layers[-1](message)

        return message

    def _decoder_forward_pass(self, message, context):
        if self.use_context:
            decoder_input = torch.cat((message, context), dim=1)
        else:
            decoder_input = message

        prediction = decoder_input
        for hidden_layer in self.decoder_hidden_layers[:-1]:
            prediction = F.relu(hidden_layer(prediction))
        prediction = self.decoder_hidden_layers[-1](prediction)

        return prediction

    def forward(self, context, function_selector, decoder_context=None):
        message = self._encoder_forward_pass(context, function_selector)
        if decoder_context is None:
            decoder_context = context
        prediction = self._decoder_forward_pass(message, decoder_context)
        return prediction, message  # TODO do we still need to return the message?

    def predict_by_message(self, message, context):
        with torch.no_grad():
            return self._decoder_forward_pass(message, context)

    def target(self, context, function_selector):
        return self.target_function(context, function_selector)

    def message(self, context, function_selector):
        with torch.no_grad():
            return self._encoder_forward_pass(context, function_selector)

    def loss(self, context, function_selectors, decoder_context=None):
        target = self.target(context, function_selectors)
        prediction, message = self.forward(context, function_selectors, decoder_context)
        return self.criterion(prediction, target)

    def generate_contexts(self, batch_size):
        return torch.randn(batch_size, self.context_size)

    def generate_function_selectors(self, batch_size):
        """Generate `batch_size` one-hot vectors of dimension `num_functions`."""
        function_selectors = torch.zeros((batch_size, self.num_functions))
        for i, one_hot_idx in enumerate(torch.randint(self.num_functions, (batch_size,))):
            function_selectors[i][one_hot_idx] = 1
        return function_selectors

    def plot_messages_information(self, exemplars_size=40):
        with torch.no_grad():
            batch_size = exemplars_size * self.num_functions
            contexts = self.generate_contexts(batch_size)

            function_selectors = torch.zeros((batch_size, self.num_functions))
            for i in range(batch_size):
                function_selectors[i, i % self.num_functions] = 1

            messages = self.message(contexts, function_selectors)
            messages = messages.numpy()

            message_masks = []
            message_labels = []
            for func_idx in range(self.num_functions):
                message_masks.append([i * self.num_functions + func_idx for i in range(exemplars_size)])
                message_labels.append(f"F{func_idx}")

            utils.plot_raw_and_pca(messages, message_masks, message_labels, "Messages")

            targets = self.target(contexts, function_selectors)
            utils.plot_raw_and_pca(targets.numpy(), message_masks, message_labels, "Targets")

