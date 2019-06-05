"""This module defines callbacks that descend from
:class:`~hyperparameter_hunter.callbacks.bases.BaseInputWranglerCallback`. Input wrangler callbacks
are concerned with managing the input data chunks of an experiment's datasets. This module acts as a
liaison between :class:~hyperparameter_hunter.experiments.BaseCVExperiment` and the input chunk
classes defined in :mod:`hyperparameter_hunter.data.data_chunks.input_chunks`. Each callback defined
herein is responsible for ensuring the proper execution of precisely one descendant of
:class:`~hyperparameter_hunter.data.data_chunks.input_chunks.BaseInputChunk`, defined in
:mod:`~hyperparameter_hunter.data.data_chunks.input_chunks`.

Input wranglers are quite a bit less interesting than the other wranglers because they kinda
"stop caring" after a while. Input data exists entirely for the purposes of fitting a model and
making predictions - between `on_run_start` and `on_run_end`. For essential operations, we don't
need input data once we hit `on_run_end`, so none of the "...end" methods of input wranglers do
anything.

Related
-------
:mod:`hyperparameter_hunter.data.data_chunks.input_chunks`
    Defines the input data chunk classes, each of which has one counterpart/handler defined in
    :mod:`~hyperparameter_hunter.callbacks.wranglers.input_wranglers`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseInputWranglerCallback
from hyperparameter_hunter.data import TrainDataset, OOFDataset, HoldoutDataset, TestDataset

# TODO: Input chunks/wranglers probably unnecessary, since their 2 jobs are done by `CVExperiment`


class WranglerInputTrain(BaseInputWranglerCallback):
    data_train: TrainDataset

    def on_experiment_start(self):
        self.data_train.input.on_experiment_start()
        super().on_experiment_start()

    def on_fold_start(self):
        self.data_train.input.on_fold_start()
        super().on_fold_start()


class WranglerInputOOF(BaseInputWranglerCallback):
    data_oof: OOFDataset

    def on_experiment_start(self):
        self.data_oof.input.on_experiment_start()
        super().on_experiment_start()

    def on_fold_start(self):
        self.data_oof.input.on_fold_start()
        super().on_fold_start()


class WranglerInputHoldout(BaseInputWranglerCallback):
    """Input wrangler callback responsible for properly invoking callback methods defined by
    :class:`~hyperparameter_hunter.data.data_chunks.input_chunks.HoldoutInputChunk` by way of
    :attr:`data_holdout.input`"""

    data_holdout: HoldoutDataset

    def on_experiment_start(self):
        self.data_holdout.input.on_experiment_start()
        super().on_experiment_start()

    def on_fold_start(self):
        self.data_holdout.input.on_fold_start()
        super().on_fold_start()


class WranglerInputTest(BaseInputWranglerCallback):
    data_test: TestDataset

    def on_experiment_start(self):
        self.data_test.input.on_experiment_start()
        super().on_experiment_start()

    def on_fold_start(self):
        self.data_test.input.on_fold_start()
        super().on_fold_start()
