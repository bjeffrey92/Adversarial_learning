import logging
import os
import pickle
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch


@lru_cache(maxsize=1)
def load_training_data(data_dir, k=None, to_dense=True):
    if k is not None:
        with open(
            os.path.join(data_dir, f"{k}_convolved_training_features.pkl"),
            "rb",
        ) as a:
            features = pickle.load(a)
    else:
        features = torch.load(os.path.join(data_dir, "training_features.pt"))
        if to_dense:
            features = features.to_dense()
    labels = torch.load(os.path.join(data_dir, "training_labels.pt"))
    return features, labels


@lru_cache(maxsize=1)
def load_testing_data(data_dir, k=None, to_dense=True):
    if k is not None:
        with open(
            os.path.join(data_dir, f"{k}_convolved_testing_features.pkl"), "rb"
        ) as a:
            features = pickle.load(a)
    else:
        features = torch.load(os.path.join(data_dir, "testing_features.pt"))
        if to_dense:
            features = features.to_dense()
    labels = torch.load(os.path.join(data_dir, "testing_labels.pt"))
    return features, labels


@lru_cache(maxsize=1)
def load_metadata(data_dir):
    training_metadata = pd.read_csv(os.path.join(data_dir, "training_metadata.csv"))
    testing_metadata = pd.read_csv(os.path.join(data_dir, "testing_metadata.csv"))
    return training_metadata, testing_metadata


def load_labels_2(data_dir, countries=True, families=False):

    if countries == families:
        raise ValueError("One of countries OR families must be true")
    elif countries:
        name = "countries"
    else:
        name = "families"

    training_labels_2 = torch.load(os.path.join(data_dir, f"training_{name}.pt"))
    testing_labels_2 = torch.load(os.path.join(data_dir, f"testing_{name}.pt"))

    def transform(labels_2):
        labels_2 = labels_2.tolist()
        c_index = labels_2.index(max(labels_2))
        return torch.LongTensor([c_index])

    training_labels_2 = torch.cat(list(map(transform, training_labels_2.unbind(0))))
    testing_labels_2 = torch.cat(list(map(transform, testing_labels_2.unbind(0))))
    return training_labels_2, testing_labels_2


def write_epoch_results(epoch, epoch_results, summary_file):
    if not os.path.isfile(summary_file):
        with open(summary_file, "w") as a:
            a.write(
                "epoch\ttraining_data_loss\ttraining_data_acc\ttesting_data_loss\ttesting_data_acc\tvalidation_data_loss\tvalidation_data_acc\n"
            )

    with open(summary_file, "a") as a:
        line = str(epoch) + "\t" + "\t".join([str(i) for i in epoch_results])
        a.write(line + "\n")


def country_accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    assert len(predictions) == len(
        labels
    ), "Predictions and labels are of unequal lengths"
    predictions = predictions.tolist()
    labels = labels.tolist()
    correct_bool = [
        predictions[i].index(max(predictions[i])) == labels[i]
        for i in range(len(predictions))
    ]
    return len(np.array(labels)[correct_bool]) / len(labels) * 100


def R_or_S(MIC, boundary):
    """
    convert MIC to resistant or sensitive
    Presumes MIC has been transformed by log2
    """
    return [0 if 2 ** i <= boundary else 1 for i in MIC]


class DataGenerator:
    def __init__(
        self,
        features,
        labels,
        labels_2=None,
        auto_reset_generator=True,
        global_node=True,
        pre_convolved=False,
    ):
        assert len(features) == len(
            labels
        ), "Features and labels are of different length"
        if pre_convolved:
            self.features = features
        else:
            self.features = self._parse_features(features, global_node)
        self.labels = labels + min(
            abs(labels)
        )  # make +ve so relu can be used on final layer
        self.n_nodes = self.features[0].shape[0]
        self.n_samples = len(labels)
        self.n = 0
        self._index = list(range(self.n_samples))
        self.auto_reset_generator = auto_reset_generator
        if labels_2 is not None:
            assert len(labels_2) == len(
                labels
            ), "labels_2 and labels are of different length"
        self.labels_2 = labels_2

    def _parse_features(self, features, global_node: bool):
        l = [None] * len(features)
        for i in range(len(features)):
            if global_node:
                l[i] = torch.cat((features[i], torch.Tensor([0.5]))).unsqueeze(1)
            else:
                l[i] = features[i].unsqueeze(1)
        return l

    def _iterate(self):
        self.n += 1
        if self.n == self.n_samples and self.auto_reset_generator:
            if self.labels_2 is not None:
                sample = (
                    self.features[self.n - 1],
                    self.labels[self.n - 1],
                    self.labels_2[self.n - 1],
                )
            else:
                sample = self.features[self.n - 1], self.labels[self.n - 1]
            self.reset_generator()
            yield sample
        else:
            if self.labels_2 is not None:
                yield self.features[self.n - 1], self.labels[self.n - 1], self.labels_2[
                    self.n - 1
                ]
            else:
                yield self.features[self.n - 1], self.labels[self.n - 1]

    def next_sample(self):
        return next(self._iterate())

    def reset_generator(self):
        self.n = 0

    def shuffle_samples(self):
        random.shuffle(self._index)
        self.labels = torch.as_tensor(
            list({i: self.labels[i] for i in self._index}.values())
        )
        self.features = list({i: self.features[i] for i in self._index}.values())
        if self.labels_2 is not None:
            self.labels_2 = torch.as_tensor(
                list({i: self.labels_2[i] for i in self._index}.values())
            )


class MetricAccumulator:
    def __init__(self, gradient_batch=10):
        self.training_data_loss = []
        self.training_data_acc = []
        self.testing_data_loss = []
        self.testing_data_acc = []
        self.training_data_loss_grads = []
        self.training_data_acc_grads = []
        self.testing_data_loss_grads = []
        self.testing_data_acc_grads = []
        self.gradient_batch = gradient_batch

    def add(self, epoch_results):
        self.training_data_loss.append(epoch_results[0])
        self.training_data_acc.append(epoch_results[1])
        self.testing_data_loss.append(epoch_results[2])
        self.testing_data_acc.append(epoch_results[3])
        self._all_grads()

    def _all_grads(self):
        self.training_data_loss_grads.append(
            self.metric_gradient(self.training_data_loss)
        )
        self.training_data_acc_grads.append(
            self.metric_gradient(self.training_data_acc)
        )
        self.testing_data_loss_grads.append(
            self.metric_gradient(self.testing_data_loss)
        )
        self.testing_data_acc_grads.append(self.metric_gradient(self.testing_data_acc))

    def metric_gradient(self, x: list):
        batch = self.gradient_batch
        data = x[-batch:]
        return round((data[-1] - data[0]) / len(data), 2)

    def avg_gradient(self, x: list):
        batch = self.gradient_batch
        data = x[-batch:]
        return sum(data) / len(data)

    def log_gradients(self, epoch):
        avg_grads = [
            self.avg_gradient(self.training_data_loss_grads),
            self.avg_gradient(self.training_data_acc_grads),
            self.avg_gradient(self.testing_data_loss_grads),
            self.avg_gradient(self.testing_data_acc_grads),
        ]
        last_epoch = max(0, epoch - self.gradient_batch)
        logging.info(
            f"Average Gradient Between Epoch {epoch} and {last_epoch}:\n \
            Training Data Loss Gradient = {avg_grads[0]}\n \
            Training Data Accuracy Gradient = {avg_grads[1]}\n \
            Testing Data Loss Gradient = {avg_grads[2]}\n \
            Testing Data Accuracy Gradient = {avg_grads[3]}\n"
        )

# eucast resistance breakpoints for gonno
breakpoints = {"azm": 1, "cfx": 0.125, "cip": 0.06, "cro": 0.125}
