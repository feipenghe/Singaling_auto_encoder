import logging
import math
import random
from typing import Callable, List, Optional, Text

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import cluster, metrics
from torch import optim
from collections import defaultdict

import utils


class Game(nn.Module):
    def __init__(
        self,
        context_size: int,
        object_size: int,
        message_size: int,
        num_functions: int,
        target_function: Callable,
        use_context=True,
        shared_context=True,
        shuffle_decoder_context=False,
        context_generator: Optional[Callable] = None,
        loss_every: int = 1,
        hidden_sizes=(64, 64),
    ):
        super().__init__()
        self.context_size = context_size
        self.object_size = object_size
        self.message_size = message_size
        self.num_functions = num_functions
        self.hidden_sizes = hidden_sizes
        self.use_context = use_context
        self.shared_context = shared_context
        self.shuffle_decoder_context = shuffle_decoder_context
        self.context_generator = context_generator
        self.target_function = target_function
        self.loss_every = loss_every

        self.criterion = nn.MSELoss()
        self.epoch_nums: List[int] = []
        self.loss_per_epoch: List[float] = []

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

    def play(self, num_batches, mini_batch_size):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for batch_num in range(num_batches):
            function_selectors = self._generate_function_selectors(
                mini_batch_size, random=True
            )
            contexts = self._generate_contexts(mini_batch_size)
            decoder_contexts = self._get_decoder_context(mini_batch_size, contexts)

            optimizer.zero_grad()
            loss = self._loss(contexts, function_selectors, decoder_contexts)
            loss.backward()
            optimizer.step()

            if batch_num % self.loss_every == 0 or batch_num == (num_batches - 1):
                self._log_epoch_loss(batch_num, loss.item())

            if batch_num % 100 == 0:
                logging.info(
                    f"Batch {batch_num + (1 if batch_num == 0 else 0)} loss:\t{self.loss_per_epoch[-1]:.2e}"
                )

    def visualize(self):
        self.plot_messages_information()
        self.clusterize_messages(visualize=True)

    def _encoder_forward_pass(self, context, function_selector):
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

    def _decoder_forward_pass(self, message, context):
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

    def _forward(self, context, function_selector, decoder_context):
        message = self._encoder_forward_pass(context, function_selector)
        prediction = self._decoder_forward_pass(message, decoder_context)
        return prediction, message  # TODO do we still need to return the message?

    def _output_by_message(self, message, context):
        with torch.no_grad():
            return self._decoder_forward_pass(message, context)

    def _target(self, context, function_selector):
        return self.target_function(context, function_selector)

    def _message(self, context, function_selector):
        with torch.no_grad():
            return self._encoder_forward_pass(context, function_selector)

    def _loss(self, context, function_selectors, decoder_context):
        target = self._target(context, function_selectors)
        prediction, message = self._forward(
            context, function_selectors, decoder_context
        )
        return self.criterion(prediction, target)

    def _generate_contexts(self, batch_size):
        if isinstance(self.context_size, int):
            context_shape = (self.context_size,)
        else:
            context_shape = self.context_size

        if self.context_generator is None:
            return torch.randn(batch_size, *context_shape)
        else:
            return self.context_generator(batch_size, context_shape)

    def _get_decoder_context(self, batch_size, encoder_context):
        if self.shared_context:
            decoder_context = encoder_context
        else:
            decoder_context = self._generate_contexts(batch_size)

        if self.shuffle_decoder_context:
            decoder_context = decoder_context[
                :, torch.randperm(decoder_context.shape[1]), :
            ]
        return decoder_context

    def _generate_function_selectors(self, batch_size, random=False):
        """Generate `batch_size` one-hot vectors of dimension `num_functions`."""
        if random:
            function_idxs = torch.randint(self.num_functions, size=(batch_size,))
        else:
            function_idxs = torch.arange(batch_size) % self.num_functions
        return torch.nn.functional.one_hot(
            function_idxs, num_classes=self.num_functions
        ).float()

    def _generate_funcs_contexts_messages(self, exemplars_size, random=False):
        batch_size = exemplars_size * self.num_functions
        contexts = self._generate_contexts(batch_size)
        function_selectors = self._generate_function_selectors(
            batch_size, random=random
        )
        messages = self._message(contexts, function_selectors)
        return function_selectors, contexts, messages

    def _log_epoch_loss(self, epoch, loss):
        self.loss_per_epoch.append(loss)
        self.epoch_nums.append(epoch)

    def plot_messages_information(self, exemplars_size=50):
        with torch.no_grad():
            (
                func_selectors,
                contexts,
                messages,
            ) = self._generate_funcs_contexts_messages(exemplars_size, random=False)

            message_masks = []
            message_labels = []
            for func_idx in range(self.num_functions):
                message_masks.append(
                    [i * self.num_functions + func_idx for i in range(exemplars_size)]
                )
                message_labels.append(f"F{func_idx}")

            title_information_row = f"M={self.message_size}, O={self.object_size}, C={self.context_size}, F={self.num_functions}"

            utils.plot_raw_and_pca(
                messages.numpy(),
                message_masks,
                message_labels,
                f"Messages\n{title_information_row}",
            )

            targets = self._target(contexts, func_selectors)
            utils.plot_raw_and_pca(
                targets.numpy(),
                message_masks,
                message_labels,
                f"Targets\n{title_information_row}",
            )

    def predict_element_by_messages(
        self, element_to_predict: Text, exemplars_size: int = 50
    ) -> float:
        logging.info(f"Predicting {element_to_predict} from messages.")

        (func_selectors, contexts, messages,) = self._generate_funcs_contexts_messages(
            exemplars_size, random=False
        )
        batch_size = func_selectors.shape[0]

        train_test_ratio = 0.7
        num_train_samples = math.ceil(batch_size * train_test_ratio)

        ACCURACY_PREDICTIONS = ("functions", "min_max", "dimension", "sanity")

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
                func_selectors.argmax(dim=1) // 2, num_classes=num_dimensions,
            )
        elif element_to_predict == "sanity":
            # Test prediction accuracy of random data. Should be at chance level.
            elements = torch.nn.functional.one_hot(
                torch.randint(0, 2, (batch_size,)), num_classes=2,
            )
        elif element_to_predict == "object_by_context":
            elements = self.target_function(contexts, func_selectors)
        elif element_to_predict == "object_by_decoder_context":
            if self.shared_context:
                logging.info("No decoder context, context is shared.")
                return 0.0
            decoder_contexts = self._generate_contexts(batch_size)
            elements = self.target_function(decoder_contexts, func_selectors)
        elif element_to_predict == "context":
            elements = utils.batch_flatten(contexts)
        elif element_to_predict == "decoder_context":
            if self.shared_context:
                logging.info("No decoder context, context is shared.")
                return 0.0
            elements = utils.batch_flatten(self._generate_contexts(batch_size))
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
        logging.info(f"Prediction result for {element_to_predict}: {result}")
        return result

    def run_compositionality_network(self):
        hidden_size = 64
        num_exemplars = 100
        num_epochs = 10_000

        # Generate functions for all parameters except one.

        left_out_param = random.randrange(self.object_size)

        train_param_idxs = [
            i for i in range(self.object_size) if i != left_out_param
        ] * num_exemplars

        train_function_input_idx = np.array([x * 2 for x in train_param_idxs])
        train_function_target_idx = train_function_input_idx + 1

        train_function_input_selectors = torch.nn.functional.one_hot(
            torch.from_numpy(train_function_input_idx), num_classes=self.num_functions
        ).float()

        train_function_target_selectors = torch.nn.functional.one_hot(
            torch.from_numpy(train_function_target_idx), num_classes=self.num_functions
        ).float()

        batch_size = train_function_input_selectors.shape[0]

        # Test on left-out parameter.

        test_function_input_selectors = torch.nn.functional.one_hot(
            torch.from_numpy(np.array([left_out_param * 2] * batch_size)).long(),
            num_classes=self.num_functions,
        ).float()

        test_function_target_selectors = torch.nn.functional.one_hot(
            torch.from_numpy(np.array([left_out_param * 2 + 1] * batch_size)).long(),
            num_classes=self.num_functions,
        ).float()

        # Network

        layers = [
            torch.nn.Linear(self.message_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.message_size),
        ]

        model = torch.nn.Sequential(*layers)
        logging.info(f"Compositionality network layers:\n{layers}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_func = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            context = self._generate_contexts(batch_size)

            input_messages = self._message(context, train_function_input_selectors)
            target_messages = self._message(context, train_function_target_selectors)

            if epoch == 0:
                test_messages_input = self._message(
                    context, test_function_input_selectors
                )
                test_messages_target = self._message(
                    context, test_function_target_selectors
                )

                messages_to_visualize = np.concatenate(
                    [
                        input_messages,
                        target_messages,
                        test_messages_input,
                        test_messages_target,
                    ],
                    axis=0,
                )
                utils.plot_raw_and_pca(
                    messages_to_visualize,
                    masks=[
                        list(range(i * batch_size, (i + 1) * batch_size))
                        for i in range(4)
                    ],
                    title="Input/Target messages",
                    labels=[
                        "Training input",
                        "Training target",
                        "Test input",
                        "Test target",
                    ],
                )

            y_pred = model(input_messages)

            optimizer.zero_grad()
            loss = loss_func(y_pred, target_messages)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                logging.info(
                    f"Epoch {epoch + (1 if epoch == 0 else 0)}:\t{loss.item():.2e}"
                )

        # Evaluate

        test_context = self._generate_contexts(batch_size)
        test_input_messages = self._message(test_context, test_function_input_selectors)
        test_target_messages = self._message(
            test_context, test_function_target_selectors
        )

        with torch.no_grad():
            test_predicted_messages = model(test_input_messages)

        result = loss_func(test_predicted_messages, test_target_messages).item()
        logging.info(f"Prediction loss: {result:.2e}")

        predicted_objects = self._decoder_forward_pass(
            test_predicted_messages, test_context
        )
        target_objects = self._decoder_forward_pass(test_target_messages, test_context)

        object_prediction_loss = loss_func(predicted_objects, target_objects)
        logging.info(f"Object prediction loss: {object_prediction_loss:.2e}")

        target_for_training = self._message(
            test_context, train_function_target_selectors
        )

        with torch.no_grad():
            prediction_for_training = model(
                self._message(test_context, train_function_input_selectors)
            )

        mse_per_training_sample = (
            ((prediction_for_training - target_for_training) ** 2).mean(dim=1).numpy()
        )

        losses_per_training_func = defaultdict(list)
        for i, train_param in enumerate(train_param_idxs):
            losses_per_training_func[train_param].append(mse_per_training_sample[i])

        mean_loss_per_func = {
            func: np.mean(losses) for func, losses in losses_per_training_func.items()
        }
        mean_loss_per_func[left_out_param] = torch.nn.MSELoss()(
            test_predicted_messages, test_target_messages
        ).numpy()

        mean_loss_per_func[left_out_param] = (
            ((test_predicted_messages - test_target_messages) ** 2).mean().numpy()
        )

        funcs = list(sorted(mean_loss_per_func.keys()))
        losses = [mean_loss_per_func[func] for func in funcs]

        plt.bar(
            funcs,
            losses,
            color=["blue" if func != left_out_param else "red" for func in funcs],
        )
        plt.xticks(list(range(len(funcs))), funcs, fontsize=5)
        plt.xlabel("Parameters", fontsize=5)
        plt.ylabel("MSELoss", fontsize=5)
        plt.title("Prediction loss per parameter k (M_{2k} -> M_{2k+1})")
        plt.show()

        # Visualize prediction

        messages_to_visualize = np.concatenate(
            [test_input_messages, test_predicted_messages, test_target_messages,],
            axis=0,
        )
        utils.plot_raw_and_pca(
            messages_to_visualize,
            masks=[
                list(range(i * batch_size, (i + 1) * batch_size))
                for i in range(messages_to_visualize.shape[0])
            ],
            title="Input/Target messages",
            labels=["Test input messages", "Test predicted messages", "Test target",],
        )

        # Baseline

        prediction_for_training = model(
            self._message(test_context, train_function_input_selectors)
        )
        training_prediction_loss = loss_func(
            prediction_for_training,
            self._message(test_context, train_function_target_selectors),
        )
        logging.info(f"Baseline using training data: {training_prediction_loss:.2e}")

        rand_function_selectors_1 = self._generate_function_selectors(
            batch_size, random=True
        )
        rand_function_selectors_2 = self._generate_function_selectors(
            batch_size, random=True
        )

        random_messages_1 = self._message(test_context, rand_function_selectors_1)
        random_messages_2 = self._message(test_context, rand_function_selectors_2)
        baseline_result = loss_func(random_messages_1, random_messages_2).item()

        baseline_object_prediction_loss = loss_func(
            self._decoder_forward_pass(random_messages_1, test_context),
            self._decoder_forward_pass(random_messages_2, test_context),
        )

        logging.info(f"Baseline loss: {baseline_result:.2e}")
        logging.info(
            f"Object prediction baseline loss: {baseline_object_prediction_loss:.2e}"
        )

        return result

    def clusterize_messages(self, exemplars_size=50, visualize=False):
        num_clusters = self.num_functions
        (func_selectors, contexts, messages,) = self._generate_funcs_contexts_messages(
            exemplars_size, random=False
        )

        k_means = cluster.KMeans(n_clusters=num_clusters)
        labels = k_means.fit_predict(messages)

        if visualize:
            utils.plot_clusters(messages, labels, "Training messages clusters")

        # Align cluster ids with with message ids.
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
        (_, test_contexts, test_messages,) = self._generate_funcs_contexts_messages(
            exemplars_size, random=False
        )
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

        predictions_by_unseen_messages = self._output_by_message(
            test_messages, test_contexts
        )

        decoder_contexts = self._get_decoder_context(batch_size, test_contexts)
        with torch.no_grad():
            predictions_by_inferred_func, _ = self._forward(
                test_contexts, func_by_message_cluster, decoder_contexts
            )

        loss_func = torch.nn.MSELoss()
        loss = loss_func(
            predictions_by_unseen_messages, predictions_by_inferred_func
        ).item()
        logging.info(f"Loss for unseen message/information: {loss}")

        return loss
