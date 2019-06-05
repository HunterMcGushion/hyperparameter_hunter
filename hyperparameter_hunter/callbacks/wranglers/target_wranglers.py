"""This module defines callbacks that descend from
:class:`~hyperparameter_hunter.callbacks.bases.BaseTargetWranglerCallback`. Target wrangler
callbacks are concerned with managing the target chunks of an experiment's datasets. This module
acts as a liaison between :class:~hyperparameter_hunter.experiments.BaseCVExperiment` and the
target chunk classes defined in :mod:`hyperparameter_hunter.data.data_chunks.target_chunks`. Each
callback defined herein is responsible for ensuring the proper execution of precisely one descendant
of :class:`~hyperparameter_hunter.data.data_chunks.target_chunks.BaseTargetChunk`, defined in
:mod:`~hyperparameter_hunter.data.data_chunks.target_chunks`.

Target wranglers are a stark contrast to the relatively boring input wranglers. We need target data
for fitting models and for evaluating predictions (which takes place during every "...end" method).
Therefore, target wranglers have some mission-critical task to perform on every callback method,
especially when feature engineering gets thrown in.

Related
-------
:mod:`hyperparameter_hunter.data.data_chunks.target_chunks`
    Defines the target data chunk classes, each of which has one counterpart/handler defined in
    :mod:`~hyperparameter_hunter.callbacks.wranglers.target_wranglers`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseTargetWranglerCallback
from hyperparameter_hunter.data import TrainDataset, OOFDataset, HoldoutDataset


##################################################
# Target Wranglers
##################################################
class WranglerTargetTrain(BaseTargetWranglerCallback):
    data_train: TrainDataset


class WranglerTargetOOF(BaseTargetWranglerCallback):
    data_oof: OOFDataset

    #################### Division Start Points ####################
    def on_experiment_start(self):
        # NOTE: Mirror train targets index, but drop columns because they might change intra-CV
        self.data_oof.target.on_experiment_start(
            self._empty_output_like(self.data_train.target.T.d)
        )
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_oof.target.on_repetition_start(
            self._empty_output_like(self.data_train.target.T.d)
        )
        super().on_repetition_start()

    def on_fold_start(self):
        self.data_oof.target.on_fold_start()
        super().on_fold_start()

    def on_run_start(self):
        self.data_oof.target.on_run_start()
        super().on_run_start()

    #################### Division End Points ####################
    def on_run_end(self):
        self.data_oof.target.on_run_end()
        super().on_run_end()

    def on_fold_end(self):
        self.data_oof.target.on_fold_end(self.validation_index)
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_oof.target.on_repetition_end(self.cv_params["n_splits"])
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_oof.target.on_experiment_end(self.cv_params.get("n_repeats", 1))
        super().on_experiment_end()


class WranglerTargetHoldout(BaseTargetWranglerCallback):
    data_holdout: HoldoutDataset

    #################### Division Start Points ####################
    def on_experiment_start(self):
        self.data_holdout.target.on_experiment_start(
            self._empty_output_like(self.data_holdout.target.T.d)
        )
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_holdout.target.on_repetition_start(
            self._empty_output_like(self.data_holdout.target.T.d)
        )
        super().on_repetition_start()

    def on_fold_start(self):
        self.data_holdout.target.on_fold_start()
        super().on_fold_start()

    def on_run_start(self):
        self.data_holdout.target.on_run_start()
        super().on_run_start()

    #################### Division End Points ####################
    def on_run_end(self):
        self.data_holdout.target.on_run_end()
        super().on_run_end()

    def on_fold_end(self):
        self.data_holdout.target.on_fold_end()
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_holdout.target.on_repetition_end(self.cv_params["n_splits"])
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_holdout.target.on_experiment_end(self.cv_params.get("n_repeats", 1))
        super().on_experiment_end()
