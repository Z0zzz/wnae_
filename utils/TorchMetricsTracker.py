import pandas as pd
import torch
import torchmetrics

from Logger import *


class TorchMetricsTracker():
    def __init__(
            self,
            builtin_metric_names=[],
            external_metric_names=[],
            automatic_epoch_numbering=True,
            automatic_batch_numbering=True,
        ):
        """Constructor of the metrics tracker for torch models.

        Args:
            builtin_metric_names (list[str]): Name of metrics that can be
                computed by torchmetrics
            external_metric_names (list[str]): Names of other metrics that 
                will be provided during training as already calculated
            automatic_epoch_numbering (bool): Set to `True` if you call the
                method `epoch_update` at every epoch in your training loop.
                Otherwise, set to `False` and provide the epoch number in the 
                argument of `epoch_update`.
            automatic_batch_numbering (bool): Set to `True` if you call the
                method `epoch_update` at every epoch in your training loop and
                provide a value for `n_batches_per_epoch`. Otherwise, set to 
                `False` and provide the batch number in the argument of
                `epoch_update`. Must be `True` if `automatic_epoch_numbering` 
                is `True`.
        """

        assert automatic_batch_numbering == automatic_epoch_numbering

        self.builtin_metric_names = builtin_metric_names
        self.external_metric_names = external_metric_names
        self.all_metric_names = builtin_metric_names + external_metric_names
        self.metrics_tracker = {"epoch": [], "batch": []}
        self.metrics_tracker.update({metric_name: [] for metric_name in self.all_metric_names})
        self.builtin_metrics = {
            metric_name: getattr(torchmetrics, self.__metrics_name_to_torchmetrics_name(metric_name))
            for metric_name in builtin_metric_names
        }
        self.data_split_names = {
            metric_name: self.__metric_name_to_data_split_name(metric_name)
            for metric_name in self.all_metric_names
        }
        self.automatic_epoch_numbering = automatic_epoch_numbering
        self.automatic_batch_numbering = automatic_batch_numbering
        self.n_batches_per_epoch = None

        self.n_epochs = 0

    def set_n_batches_per_epoch(self, n_batches_per_epoch):
        """Set the number of batch per epoch for automatic batch numbering.

        Args:
            n_batches_per_epoch (int)
        """

        self.n_batches_per_epoch = n_batches_per_epoch

    @staticmethod
    def __metric_name_to_data_split_name(metric_name):
        split = metric_name.split("_")
        if split[0] in ["training", "validation", "test"]:
            data_split = split[0]
        else:
            data_split = "None"
        return data_split

    @staticmethod
    def __metrics_name_to_torchmetrics_name(metric_name):
        variable_name = metric_name.split("_")[1]
        dict_ = {
            "accuracy": "Accuracy",
            "auc": "AUC",
        }
        if variable_name in dict_.keys():
            return dict_[variable_name]
        else:
            return variable_name

    def batch_update(self,
                     training_predictions=None,
                     training_labels=None,
                     validation_predictions=None,
                     validation_labels=None):

        for metric_name in self.metrics.keys():
            data_split_name = self.data_split_names[metric_name]
            if data_split_name == "training":
                if training_predictions is not None:
                    self.metrics[metric_name].update(training_predictions, training_labels)
            else:
                if validation_predictions is not None:
                    self.metrics[metric_name].update(validation_predictions, validation_labels)


    def epoch_update(self, external_metrics, epoch=None, batch=None):
        """Add metrics to tracker at the end of the epoch.

        Args:
            external_metrics (dict[str, float]): Keys are metrics names and
                values are the value for that metric.
            epoch (int or None): Provide a not `None` value only if
                the `TorchMetricsTracker` instance has been built with 
                `automatic_epoch_numbering=False`.
            batch (int or None): Provide a not `None` value only if
                the `TorchMetricsTracker` instance has been built with 
                `automatic_batch_numbering=False`.
        """

        assert (
            (self.automatic_epoch_numbering and epoch is None)
            or (not self.automatic_epoch_numbering and epoch is not None)
        )
        assert (
            (self.automatic_batch_numbering and batch is None)
            or (not self.automatic_batch_numbering and batch is not None)
        )

        assert (
            not self.automatic_batch_numbering
            or (self.automatic_batch_numbering and self.n_batches_per_epoch is not None)
        )

        if set(external_metrics.keys()) != set(self.external_metric_names):
            log.critical("Not all external metrics can be updated!")
            log.critical("External metrics to update:")
            log.critical(f"{list(external_metrics.keys())}")
            log.critical("External metrics:")
            log.critical(f"{self.external_metric_names}")
            exit(1)

        if self.automatic_epoch_numbering:
            epoch = self.n_epochs
        if self.automatic_batch_numbering:
            batch = (self.n_epochs+1) * self.n_batches_per_epoch

        self.metrics_tracker["epoch"].append(epoch)
        self.metrics_tracker["batch"].append(batch)

        for metric_name in self.builtin_metrics.keys():
            metric_value = self.builtin_metrics[metric_name].compute()
            self.metrics_tracker[metric_name].append(metric_value)
            self.builtin_metrics[metric_name].reset()

        for metric_name, metric_value in external_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                if metric_value.requires_grad:
                    metric_value = metric_value.detach()
                metric_value = metric_value.numpy()
            self.metrics_tracker[metric_name].append(metric_value)

        self.n_epochs += 1

    def write_to_file(self, file_name):
        pd.DataFrame(self.metrics_tracker).to_csv(file_name, index=False)

