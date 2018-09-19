"""This module contains extra callbacks that can add commonly-used functionality to Experiments.
This module also serves as an example for how users can properly construct their own custom
callbacks using :func:`hyperparameter_hunter.callbacks.bases.lambda_callback`

Related
-------
:mod:`hyperparameter_hunter.callbacks.bases`
    This module defines :func:`hyperparameter_hunter.callbacks.bases.lambda_callback`, which is how
    all extra callbacks created in :mod:`hyperparameter_hunter.callbacks.recipes` are created
:mod:`hyperparameter_hunter.environment`
    This module provides the means to use custom callbacks made by
    :func:`hyperparameter_hunter.callbacks.bases.lambda_callback` through the `experiment_callbacks`
    argument of :meth:`hyperparameter_hunter.environment.Environment.__init__`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import lambda_callback
from hyperparameter_hunter.metrics import get_clean_prediction

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


##################################################
# Confusion Matrix Callbacks
##################################################
def confusion_matrix_oof(on_run=True, on_fold=True, on_repetition=True, on_experiment=True):
    """

    Parameters
    ----------
    on_run: Boolean, default=True
        If False, skip making confusion matrices for individual Experiment runs
    on_fold: Boolean, default=True
        If False, skip making confusion matrices for individual Experiment folds
    on_repetition: Boolean, default=True
        If False, skip making confusion matrices for individual Experiment repetitions
    on_experiment: Boolean, default=True
        If False, skip making final confusion matrix for the Experiment

    Returns
    -------
    LambdaCallback
        An uninitialized :class:`LambdaCallback` to generate confusion matrices, produced by
        :func:`hyperparameter_hunter.callbacks.bases.lambda_callback`"""
    # TODO: Add documentation to this and inner helper functions
    # TODO: Note that, unlike `confusion_matrix_holdout`, this allows `lambda_callback` to automatically aggregate stats

    def _on_run_end(fold_validation_target, run_validation_predictions, **kwargs):
        return _confusion_matrix(fold_validation_target, run_validation_predictions)

    def _on_fold_end(fold_validation_target, repetition_oof_predictions, validation_index):
        return _confusion_matrix(
            fold_validation_target, repetition_oof_predictions.iloc[validation_index]
        )

    def _on_repetition_end(train_target_data, repetition_oof_predictions):
        return _confusion_matrix(train_target_data, repetition_oof_predictions)

    def _on_experiment_end(train_target_data, final_oof_predictions):
        return _confusion_matrix(train_target_data, final_oof_predictions)

    return lambda_callback(
        on_run_end=_on_run_end if on_run else None,
        on_fold_end=_on_fold_end if on_fold else None,
        on_repetition_end=_on_repetition_end if on_repetition else None,
        on_experiment_end=_on_experiment_end if on_experiment else None,
        agg_name="confusion_matrix_oof",
    )


def confusion_matrix_holdout(on_run=True, on_fold=True, on_repetition=True, on_experiment=True):
    # TODO: Add documentation to this and inner helper functions
    # TODO: Note that, unlike `confusion_matrix_oof`, this manually handles aggregating stats

    def _on_experiment_start(stat_aggregates):
        stat_aggregates["confusion_matrix_holdout"] = dict(runs=[], folds=[], reps=[], final=None)

    def _on_run_end(stat_aggregates, holdout_target_data, run_holdout_predictions):
        stat_aggregates["confusion_matrix_holdout"]["runs"].append(
            _confusion_matrix(holdout_target_data, run_holdout_predictions)
        )

    def _on_fold_end(stat_aggregates, holdout_target_data, fold_holdout_predictions):
        stat_aggregates["confusion_matrix_holdout"]["folds"].append(
            _confusion_matrix(holdout_target_data, fold_holdout_predictions)
        )

    def _on_repetition_end(stat_aggregates, holdout_target_data, repetition_holdout_predictions):
        stat_aggregates["confusion_matrix_holdout"]["reps"].append(
            _confusion_matrix(holdout_target_data, repetition_holdout_predictions)
        )

    def _on_experiment_end(stat_aggregates, holdout_target_data, final_holdout_predictions):
        stat_aggregates["confusion_matrix_holdout"]["final"] = _confusion_matrix(
            holdout_target_data, final_holdout_predictions
        )

    return lambda_callback(
        on_experiment_start=_on_experiment_start,
        on_run_end=_on_run_end if on_run else None,
        on_fold_end=_on_fold_end if on_fold else None,
        on_repetition_end=_on_repetition_end if on_repetition else None,
        on_experiment_end=_on_experiment_end if on_experiment else None,
        agg_name="confusion_matrix_holdout",
    )


def _confusion_matrix(targets, predictions):
    """Helper function to produce a confusion matrix after properly formatting `predictions`

    Parameters
    ----------
    targets: Array-like
        The target/expected output labels for each of the given `predictions`
    predictions: Array-like
        The predicted values corresponding to each of the elements in `targets`

    Returns
    -------
    Array-like
        A confusion matrix for the given `targets` and `predictions`"""
    return sk_confusion_matrix(targets, get_clean_prediction(targets, predictions))
