"""This module defines callbacks that descend from
:class:`~hyperparameter_hunter.callbacks.bases.BasePredictorCallback`. Predictor wrangler callbacks
are concerned with managing the prediction chunks of an experiment's datasets. This module acts as a
liaison between :class:~hyperparameter_hunter.experiments.BaseCVExperiment` and the prediction chunk
classes defined in :mod:`hyperparameter_hunter.data.data_chunks.prediction_chunks`. Each callback
defined herein is responsible for ensuring the proper execution of precisely one descendant of
:class:`~hyperparameter_hunter.data.data_chunks.prediction_chunks.BasePredictorChunk`, defined in
:mod:`~hyperparameter_hunter.data.data_chunks.prediction_chunks`.

Predictors are the busiest of all three wrangler callbacks. While we only actually get predictions
when we first hit `on_run_end`, we need to keep track of them across runs, folds and reps, so
predictions need to be cleared out during the "...start" callback methods. There are two
mission-critical tasks for which we need predictions: 1) Evaluation against targets, and
2) Recording - not only to ensure our model is behaving as expected, but also for ensembling.
Ensembling is a real pain if you're trying to do it, using only evaluation metrics as a guide, and
re-running selected experiments so you can save the predictions this time, just to figure out if
the ensemble actually performs in the end.

Once again, feature engineering throws a monkey-wrench into our expectations for the predictor
callbacks. If we're performing any kind of target transformation (which is often the case), then
evaluations need to be made using transformed predictions and targets. Calculating f1-score would
not go well if we tried to give the metric function the stringified iris dataset labels of
"setosa", "versicolor", or "virginica". It's gonna want the transformed, numerical representation
of the targets. Similarly, averaging predictions across divisions uses transformed predictions
because it requires values that can actually be averaged. For the purposes of recording, we may
want either transformed or inverted (original form) prediction - or both. Lots of weird things
start misbehaving in lots of confusing ways if our predictor wranglers aren't carefully managing
predictions across all the experiment's divisions, and in both forms: transformed, and
inverted (original form).

Related
-------
:mod:`hyperparameter_hunter.data.data_chunks.prediction_chunks`
    Defines the prediction data chunk classes, each of which has one counterpart/handler defined in
    :mod:`~hyperparameter_hunter.callbacks.wranglers.predictors`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BasePredictorCallback
from hyperparameter_hunter.data import OOFDataset, HoldoutDataset, TestDataset


##################################################
# Predictor Wranglers
##################################################
class PredictorOOF(BasePredictorCallback):
    data_oof: OOFDataset

    #################### Division Start Points ####################
    def on_experiment_start(self):
        self.data_oof.prediction.on_experiment_start(self._empty_output_like(self.train_dataset))
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_oof.prediction.on_repetition_start(self._empty_output_like(self.train_dataset))
        super().on_repetition_start()

    def on_fold_start(self):  # Nothing
        self.data_oof.prediction.on_fold_start()
        super().on_fold_start()

    def on_run_start(self):  # Nothing
        self.data_oof.prediction.on_run_start()
        super().on_run_start()

    #################### Division End Points ####################
    def on_run_end(self):
        prediction = self.model.predict(self.data_oof.input.T.fold)
        self.data_oof.prediction.on_run_end(
            prediction, self.feature_engineer, self.target_column, self.validation_index
        )
        super().on_run_end()

    def on_fold_end(self):
        self.data_oof.prediction.on_fold_end(self.validation_index, self.experiment_params["runs"])
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_oof.prediction.on_repetition_end()
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_oof.prediction.on_experiment_end(self.cv_params.get("n_repeats", 1))
        super().on_experiment_end()


class PredictorHoldout(BasePredictorCallback):
    data_holdout: HoldoutDataset

    #################### Division Start Points ####################
    def on_experiment_start(self):
        self.data_holdout.prediction.on_experiment_start()
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_holdout.prediction.on_repetition_start()
        super().on_repetition_start()

    def on_fold_start(self):
        self.data_holdout.prediction.on_fold_start()
        super().on_fold_start()

    def on_run_start(self):  # Nothing
        self.data_holdout.prediction.on_run_start()
        super().on_run_start()

    #################### Division End Points ####################
    def on_run_end(self):
        prediction = self.model.predict(self.data_holdout.input.T.fold)
        self.data_holdout.prediction.on_run_end(
            prediction, self.feature_engineer, self.target_column
        )
        super().on_run_end()

    def on_fold_end(self):
        self.data_holdout.prediction.on_fold_end(self.experiment_params["runs"])
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_holdout.prediction.on_repetition_end(self.cv_params["n_splits"])
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_holdout.prediction.on_experiment_end(self.cv_params.get("n_repeats", 1))
        super().on_experiment_end()


class PredictorTest(BasePredictorCallback):
    data_test: TestDataset

    #################### Division Start Points ####################
    def on_experiment_start(self):
        self.data_test.prediction.on_experiment_start()
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_test.prediction.on_repetition_start()
        super().on_repetition_start()

    def on_fold_start(self):
        self.data_test.prediction.on_fold_start()
        super().on_fold_start()

    def on_run_start(self):  # Nothing
        self.data_test.prediction.on_run_start()
        super().on_run_start()

    #################### Division End Points ####################
    def on_run_end(self):
        prediction = self.model.predict(self.data_test.input.T.fold)
        self.data_test.prediction.on_run_end(prediction, self.feature_engineer, self.target_column)
        super().on_run_end()

    def on_fold_end(self):
        self.data_test.prediction.on_fold_end(self.experiment_params["runs"])
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_test.prediction.on_repetition_end(self.cv_params["n_splits"])
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_test.prediction.on_experiment_end(self.cv_params.get("n_repeats", 1))
        super().on_experiment_end()
