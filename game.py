import logging
import math
from typing import Callable, Optional, Text

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import cluster, metrics
from torch import optim

import utils


class UpdateNetwork(nn.Module):
    def __init__(
        self,
        context_size,
        object_size,
        num_functions,
        hidden_sizes=(64,),
        use_context=True,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.use_context = use_context

        if self.use_context:
            input_size = context_size + num_functions
        else:
            input_size = num_functions
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, self.hidden_sizes[0])]
        )
        for i, hidden_size in enumerate(self.hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i], hidden_size))
        self.hidden_layers.append(nn.Linear(self.hidden_sizes[-1], object_size))

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


def make_update_network_function(
    context_size, object_size, num_functions, update_network_hidden_sizes, use_context
):
    update_network = UpdateNetwork(
        context_size,
        object_size,
        num_functions,
        update_network_hidden_sizes,
        use_context,
    )

    def func(contexts, function_selectors):
        with torch.no_grad():
            return update_network.forward(contexts, function_selectors)

    return func


class Game(nn.Module):
    def __init__(
        self,
        context_size,
        object_size,
        message_size,
        num_functions,
        use_context=True,
        shared_context=True,
        shuffle_decoder_context=False,
        target_function: Optional[Callable] = None,
        context_generator: Optional[Callable] = None,
        hidden_sizes=(64, 64),
        update_network_hidden_sizes=(64,),
    ):
        super().__init__()
        self.context_size = context_size
        self.object_size = object_size
        self.message_size = message_size
        self.num_functions = num_functions
        self.hidden_sizes = hidden_sizes
        self.update_network_hidden_sizes = update_network_hidden_sizes
        self.use_context = use_context
        self.shared_context = shared_context
        self.shuffle_decoder_context = shuffle_decoder_context
        self.context_generator = context_generator

        if target_function is not None:
            self.target_function = target_function
        else:
            self.target_function = make_update_network_function(
                self.context_size,
                self.object_size,
                self.num_functions,
                self.update_network_hidden_sizes,
                self.use_context,
            )

        self.criterion = nn.MSELoss()
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

        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
        )
        self.decoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in decoder_layer_dimensions]
        )

        logging.info("Game details:")
        logging.info(
            f"\nContext size: {context_size}\nObject size: {object_size}\nMessage size: {message_size}\nNumber of functions: {num_functions}"
        )
        logging.info(f"Use context: {use_context}")
        logging.info(f"Encoder layers:\n{self.encoder_hidden_layers}")
        logging.info(f"Decoder layers:\n{self.decoder_hidden_layers}")

    def play(self, num_batches, mini_batch_size, loss_every=100):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for batch_num in range(num_batches):
            function_selectors = self.generate_function_selectors(
                mini_batch_size, random=True
            )
            contexts = self.generate_contexts(mini_batch_size)
            decoder_contexts = self.get_decoder_context(mini_batch_size, contexts)

            optimizer.zero_grad()
            loss = self.loss(contexts, function_selectors, decoder_contexts)
            loss.backward()
            optimizer.step()

            self.loss_list.append((batch_num, loss.item()))
            if batch_num % loss_every == 0:
                self.loss_list.append((batch_num, loss.item()))
                logging.info(
                    f"Batch {batch_num + (1 if batch_num == 0 else 0)} loss:\t{self.loss_list[-1][1]:.2e}"
                )

    def encoder_forward_pass(self, context, function_selector):
        if self.use_context:
            context_flattened = utils.batch_flatten(context)
            encoder_input = torch.cat((context_flattened, function_selector), dim=1)
        else:
            encoder_input = function_selector

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

    def forward(self, context, function_selector, decoder_context):
        message = self.encoder_forward_pass(context, function_selector)
        prediction = self.decoder_forward_pass(message, decoder_context)
        return prediction, message  # TODO do we still need to return the message?

    def output_by_message(self, message, context):
        with torch.no_grad():
            return self.decoder_forward_pass(message, context)

    def target(self, context, function_selector):
        return self.target_function(context, function_selector)

    def message(self, context, function_selector):
        with torch.no_grad():
            return self.encoder_forward_pass(context, function_selector)

    def loss(self, context, function_selectors, decoder_context):
        target = self.target(context, function_selectors)
        prediction, message = self.forward(context, function_selectors, decoder_context)
        return self.criterion(prediction, target)

    def generate_contexts(self, batch_size):
        if isinstance(self.context_size, int):
            context_shape = (self.context_size,)
        else:
            context_shape = self.context_size

        if self.context_generator is None:
            return torch.randn(batch_size, *context_shape)
        else:
            return self.context_generator(batch_size, context_shape)

    def get_decoder_context(self, batch_size, encoder_context):
        if self.shared_context:
            decoder_context = encoder_context
        else:
            decoder_context = self.generate_contexts(batch_size)

        if self.shuffle_decoder_context:
            decoder_context = decoder_context[
                :, torch.randperm(decoder_context.shape[1]), :
            ]
        return decoder_context

    def generate_function_selectors(self, batch_size, random=False):
        """Generate `batch_size` one-hot vectors of dimension `num_functions`."""
        if random:
            function_idxs = torch.randint(self.num_functions, size=(batch_size,))
        else:
            function_idxs = torch.arange(batch_size) % self.num_functions
        return torch.nn.functional.one_hot(
            function_idxs, num_classes=self.num_functions
        ).float()

    def generate_func_selectors_contexts_messages(self, exemplars_size):
        batch_size = exemplars_size * self.num_functions
        contexts = self.generate_contexts(batch_size)
        function_selectors = self.generate_function_selectors(batch_size, random=False)
        messages = self.message(contexts, function_selectors)
        return function_selectors, contexts, messages

    def visualize(self):
        self.plot_messages_information()
        self.clusterize_messages(visualize=True)

    def plot_messages_information(self, exemplars_size=40):
        with torch.no_grad():
            batch_size = exemplars_size * self.num_functions
            contexts = self.generate_contexts(batch_size)
            function_selectors = self.generate_function_selectors(
                batch_size, random=False
            )

            messages = self.message(contexts, function_selectors).numpy()

            message_masks = []
            message_labels = []
            for func_idx in range(self.num_functions):
                message_masks.append(
                    [i * self.num_functions + func_idx for i in range(exemplars_size)]
                )
                message_labels.append(f"F{func_idx}")

            title_information_row = f"M={self.message_size}, O={self.object_size}, C={self.context_size}, F={self.num_functions}"

            utils.plot_raw_and_pca(
                messages,
                message_masks,
                message_labels,
                f"Messages\n{title_information_row}",
            )

            targets = self.target(contexts, function_selectors)
            utils.plot_raw_and_pca(
                targets.numpy(),
                message_masks,
                message_labels,
                f"Targets\n{title_information_row}",
            )

    def predict_element_by_messages(
        self, element_to_predict: Text, exemplars_size=40
    ) -> float:
        logging.info(f"Predicting {element_to_predict} from messages.")

        (
            func_selectors,
            contexts,
            messages,
        ) = self.generate_func_selectors_contexts_messages(exemplars_size)
        batch_size = func_selectors.shape[0]

        train_test_ratio = 0.7
        num_train_samples = math.ceil(batch_size * train_test_ratio)

        ACCURACY_PREDICTIONS = ("functions", "min_max", "dimension")

        if element_to_predict in ACCURACY_PREDICTIONS:
            # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
            loss_func = torch.nn.NLLLoss()
        else:
            loss_func = torch.nn.MSELoss()

        if element_to_predict == "functions":
            elements = func_selectors
        elif element_to_predict == "min_max":
            if len(contexts.shape) != 3:
                # Requires extremity game context.
                return 0.0
            elements = torch.nn.functional.one_hot(
                func_selectors.argmax(dim=1) % 2, num_classes=2
            )
        elif element_to_predict == "dimension":
            if len(contexts.shape) != 3:
                # Requires extremity game context.
                return 0.0
            num_dimensions = contexts.shape[2]
            elements = torch.nn.functional.one_hot(
                func_selectors.argmax(dim=1) // num_dimensions,
                num_classes=num_dimensions,
            )
        elif element_to_predict == "object_by_context":
            elements = self.target_function(contexts, func_selectors)
        elif element_to_predict == "object_by_decoder_context":
            if self.shared_context:
                logging.info("No decoder context, context is shared.")
                return 0.0
            decoder_contexts = self.generate_contexts(batch_size)
            elements = self.target_function(decoder_contexts, func_selectors)
        elif element_to_predict == "context":
            elements = utils.batch_flatten(contexts)
        elif element_to_predict == "decoder_context":
            if self.shared_context:
                logging.info("No decoder context, context is shared.")
                return 0.0
            elements = utils.batch_flatten(self.generate_contexts(batch_size))
        else:
            raise ValueError("Invalid element to predict")

        train_target, test_target = (
            elements[:num_train_samples],
            elements[num_train_samples:],
        )
        train_messages, test_messages = (
            messages[:num_train_samples],
            messages[num_train_samples:],
        )

        classifier_hidden_size = 32
        layers = [
            torch.nn.Linear(self.message_size, classifier_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_hidden_size, test_target.shape[-1]),
        ]

        if element_to_predict in ACCURACY_PREDICTIONS:
            layers.append(torch.nn.LogSoftmax(dim=1))

        model = torch.nn.Sequential(*layers)
        logging.info(f"Prediction network layers:\n{layers}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1000
        for epoch in range(num_epochs):
            y_pred = model(train_messages)
            if element_to_predict in ACCURACY_PREDICTIONS:
                current_train_target = train_target.argmax(dim=1)
            else:
                current_train_target = train_target
            loss = loss_func(y_pred, current_train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 0 and epoch % 100 == 0:
                logging.info(
                    f"Epoch {epoch + (1 if epoch == 0 else 0)}:\t{loss.item():.2e}"
                )

        with torch.no_grad():
            test_predicted = model(test_messages)

        if element_to_predict in ACCURACY_PREDICTIONS:
            accuracy = metrics.accuracy_score(
                test_target.argmax(dim=1).numpy(), test_predicted.argmax(dim=1).numpy()
            )
            result = accuracy
        else:
            result = loss_func(test_predicted, test_target).item()
        logging.info(f"Prediction network result: {result}")
        return result

    def clusterize_messages(self, exemplars_size=40, visualize=False):
        num_clusters = self.num_functions
        (
            func_selectors,
            contexts,
            messages,
        ) = self.generate_func_selectors_contexts_messages(exemplars_size)

        k_means = cluster.KMeans(n_clusters=num_clusters)
        labels = k_means.fit_predict(messages)

        if visualize:
            utils.plot_clusters(messages, labels, "Training messages clusters")

        # Find cluster for each message.
        message_distance_from_centers = k_means.transform(messages)
        representative_message_idx_per_cluster = message_distance_from_centers.argmin(
            axis=0
        )
        message_num_per_cluster = func_selectors[
            representative_message_idx_per_cluster, :
        ].argmax(axis=1)
        cluster_label_to_message_num = {
            cluster_num: message_num
            for cluster_num, message_num in enumerate(message_num_per_cluster)
        }

        # Sample unseen messages from clusters.
        (
            _,
            test_contexts,
            test_messages,
        ) = self.generate_func_selectors_contexts_messages(exemplars_size)
        cluster_label_per_test_message = k_means.predict(test_messages)

        batch_size = test_messages.shape[0]
        func_by_message_cluster = torch.zeros((batch_size, self.num_functions))
        for i, cluster_label in enumerate(cluster_label_per_test_message):
            func_by_message_cluster[
                i, cluster_label_to_message_num[cluster_label]
            ] = 1.0

        if visualize:
            utils.plot_clusters(
                test_messages, cluster_label_per_test_message, "Test message clusters"
            )

        predictions_by_unseen_messages = self.output_by_message(
            test_messages, test_contexts
        )

        decoder_contexts = self.get_decoder_context(batch_size, test_contexts)
        with torch.no_grad():
            predictions_by_inferred_func, _ = self.forward(
                test_contexts, func_by_message_cluster, decoder_contexts
            )

        loss_func = torch.nn.MSELoss()
        loss = loss_func(
            predictions_by_unseen_messages, predictions_by_inferred_func
        ).item()
        logging.info(f"Loss for unseen message/information: {loss}")

        return loss
