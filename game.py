import logging
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import cluster, metrics
from torch import optim

import utils


class UpdateNetwork(nn.Module):
    def __init__(self, context_size, object_size, num_functions, hidden_sizes=(64,), use_context=True):
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


def make_update_network_function(context_size, object_size, num_functions, update_network_hidden_sizes, use_context):
    update_network = UpdateNetwork(context_size, object_size, num_functions, update_network_hidden_sizes, use_context)

    def func(contexts, function_selectors):
        with torch.no_grad():
            return update_network.forward(contexts, function_selectors)
    return func


class Game(nn.Module):
    def __init__(self, context_size, object_size, message_size, num_functions, use_context=True, shared_context=True, target_function: Optional[Callable] = None, context_generator: Optional[Callable] = None, hidden_sizes=(64, 64), update_network_hidden_sizes=(64,)):
        super().__init__()
        self.context_size = context_size
        self.object_size = object_size
        self.message_size = message_size
        self.num_functions = num_functions
        self.hidden_sizes = hidden_sizes
        self.update_network_hidden_sizes = update_network_hidden_sizes
        self.use_context = use_context
        self.shared_context = shared_context
        self.context_generator = context_generator

        if target_function is not None:
            self.target_function = target_function
        else:
            self.target_function = make_update_network_function(self.context_size, self.object_size, self.num_functions,
                                                                self.update_network_hidden_sizes, self.use_context)

        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.loss_list = []

        if isinstance(self.context_size, tuple):
            self.flat_context_size = utils.reduce_prod(self.context_size)
        elif isinstance(self.context_size, int):
            self.flat_context_size = self.context_size
        else:
            raise ValueError(f"context_size must be either a tuple or int")

        if self.use_context:
            encoder_input_size = self.flat_context_size + self.num_functions
            decoder_input_size = self.message_size + self.flat_context_size
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
                if self.shared_context:
                    decoder_contexts = None
                else:
                    decoder_contexts = self.generate_contexts(mini_batch_size)
                function_selectors = self.generate_function_selectors(mini_batch_size, random=True)

                loss = self.loss(contexts, function_selectors, decoder_contexts)
                loss.backward()
                optimizer.step()

                if minibatch_epoch == 0:
                    self.loss_list.append((self.epoch, loss.item()))
                    logging.info(f"Epoch {self.loss_list[-1][0]}:\t{self.loss_list[-1][1]:.2e}")
                self.epoch += 1

            self.loss_list.append((self.epoch, loss.item()))

        logging.info(f"Epoch {self.loss_list[-1][0]}:\t{self.loss_list[-1][1]:.2e}")

    def encoder_forward_pass(self, context, function_selector):
        if self.use_context:
            context_flattened = utils.batch_flatten(context)
            encoder_input = torch.cat((context_flattened, function_selector), dim=1)
        else:
            encoder_input = object

        message = encoder_input
        for hidden_layer in self.encoder_hidden_layers[:-1]:
            message = F.relu(hidden_layer(message))
        message = self.encoder_hidden_layers[-1](message)

        return message

    def decoder_forward_pass(self, message, context):
        if self.use_context:
            context_flattened = utils.batch_flatten(context)
            decoder_input = torch.cat((message, context_flattened), dim=1)
        else:
            decoder_input = message

        prediction = decoder_input
        for hidden_layer in self.decoder_hidden_layers[:-1]:
            prediction = F.relu(hidden_layer(prediction))
        prediction = self.decoder_hidden_layers[-1](prediction)

        return prediction

    def forward(self, context, function_selector, decoder_context=None):
        message = self.encoder_forward_pass(context, function_selector)
        if decoder_context is None:
            decoder_context = context
        prediction = self.decoder_forward_pass(message, decoder_context)
        return prediction, message  # TODO do we still need to return the message?

    def predict_by_message(self, message, context):
        with torch.no_grad():
            return self.decoder_forward_pass(message, context)

    def target(self, context, function_selector):
        return self.target_function(context, function_selector)

    def message(self, context, function_selector):
        with torch.no_grad():
            return self.encoder_forward_pass(context, function_selector)

    def loss(self, context, function_selectors, decoder_context=None):
        target = self.target(context, function_selectors)
        prediction, message = self.forward(context, function_selectors, decoder_context)
        return self.criterion(prediction, target)

    def generate_contexts(self, batch_size):
        if isinstance(self.context_size, int):
            context_shape = (self.context_size, )
        else:
            context_shape = self.context_size

        if self.context_generator is None:
            return torch.randn(batch_size, *context_shape)
        else:
            return self.context_generator(batch_size, context_shape)

    def generate_function_selectors(self, batch_size, random=False):
        """Generate `batch_size` one-hot vectors of dimension `num_functions`."""
        if random:
            function_idxs = torch.randint(self.num_functions, size=(batch_size,))
        else:
            function_idxs = torch.arange(batch_size) % self.num_functions
        return torch.nn.functional.one_hot(function_idxs, num_classes=self.num_functions).float()

    def generate_func_selectors_contexts_messages(self, exemplars_size):
        batch_size = exemplars_size * self.num_functions
        contexts = self.generate_contexts(batch_size)
        function_selectors = self.generate_function_selectors(batch_size, random=False)

        messages = self.message(contexts, function_selectors)

        return function_selectors, contexts, messages

    def plot_messages_information(self, exemplars_size=40):
        with torch.no_grad():
            batch_size = exemplars_size * self.num_functions
            contexts = self.generate_contexts(batch_size)
            function_selectors = self.generate_function_selectors(batch_size, random=False)

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

    def predict_functions_from_messages(self, exemplars_size=40) -> float:
        func_selectors, situations, messages = self.generate_func_selectors_contexts_messages(exemplars_size)
        batch_size = func_selectors.shape[0]

        train_test_ratio = 0.7
        num_train_samples = math.ceil(batch_size * train_test_ratio)

        train_messages, test_messages = messages[:num_train_samples], messages[num_train_samples:]
        train_funcs, test_funcs = func_selectors[:num_train_samples], func_selectors[num_train_samples:]

        classifier_hidden_size = 32
        model = torch.nn.Sequential(
            torch.nn.Linear(self.message_size, classifier_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_hidden_size, self.num_functions),
            torch.nn.Softmax()
        )
        loss_func = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1000
        for epoch in range(num_epochs):
            y_pred = model(train_messages)
            loss = loss_func(y_pred, train_funcs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % 100 == 0:
                logging.info(f"Epoch {epoch + 1}:\t{loss.item():.2e}")

        with torch.no_grad():
            test_predicted = model(test_messages).numpy()

        accuracy = metrics.accuracy_score(np.argmax(test_funcs.numpy(), axis=1), np.argmax(test_predicted, axis=1))
        logging.info(f"Information prediction accuracy: {accuracy}")
        return accuracy

    def clusterize_messages(self, exemplars_size=40, visualize=False):
        num_clusters = self.num_functions

        func_selectors, contexts, messages = self.generate_func_selectors_contexts_messages(exemplars_size)

        k_means = cluster.KMeans(n_clusters=num_clusters)
        labels = k_means.fit_predict(messages)

        if visualize:
            utils.plot_clusters(messages, labels, "Training messages clusters")

        # Find cluster for each message.
        message_distance_from_centers = k_means.transform(messages)
        representative_message_idx_per_cluster = message_distance_from_centers.argmin(axis=0)
        message_num_per_cluster = func_selectors[representative_message_idx_per_cluster, :].argmax(axis=1)
        cluster_label_to_message_num = {cluster_num: message_num for cluster_num, message_num in
                                        enumerate(message_num_per_cluster)}

        # Sample unseen messages from clusters.
        _, test_contexts, test_messages = self.generate_func_selectors_contexts_messages(exemplars_size)
        cluster_label_per_test_message = k_means.predict(test_messages)

        batch_size = test_messages.shape[0]
        func_by_message_cluster = torch.zeros((batch_size, self.num_functions))
        for i, cluster_label in enumerate(cluster_label_per_test_message):
            func_by_message_cluster[i, cluster_label_to_message_num[cluster_label]] = 1.0

        if visualize:
            utils.plot_clusters(test_messages, cluster_label_per_test_message, "Test message clusters")

        predictions_by_unseen_messages = self.predict_by_message(test_messages, test_contexts)
        with torch.no_grad():
            predictions_by_inferred_func, _ = self.forward(test_contexts, func_by_message_cluster)

        loss_func = torch.nn.MSELoss()
        loss = loss_func(predictions_by_unseen_messages, predictions_by_inferred_func).item()
        logging.info(f"Loss for unseen message/information: {loss}")

        return loss
