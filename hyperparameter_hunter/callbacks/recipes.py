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
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
from uuid import uuid4 as uuid

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import confusion_matrix


##################################################
# Confusion Matrix Callbacks
##################################################
def confusion_matrix_oof(on_run=True, on_fold=True, on_repetition=True, on_experiment=True):
    required_attributes = [
        "train_target_data",
        "fold_validation_target",
        "final_oof_predictions",
        "repetition_oof_predictions",
        "run_validation_predictions",
        "validation_index",
    ]

    def _on_run_end(_0, fold_validation_target, _2, _3, run_validation_predictions, _5):
        return _confusion_matrix_helper(fold_validation_target, run_validation_predictions)

    def _on_fold_end(
        _0, fold_validation_target, _2, repetition_oof_predictions, _4, validation_index
    ):
        return _confusion_matrix_helper(
            fold_validation_target, repetition_oof_predictions.iloc[validation_index]
        )

    def _on_repetition_end(train_target_data, _1, _2, repetition_oof_predictions, _4, _5):
        return _confusion_matrix_helper(train_target_data, repetition_oof_predictions)

    def _on_experiment_end(train_target_data, _1, final_oof_predictions, _3, _4, _5):
        return _confusion_matrix_helper(train_target_data, final_oof_predictions)

    return lambda_callback(
        required_attributes=required_attributes,
        on_run_end=_on_run_end if on_run else None,
        on_fold_end=_on_fold_end if on_fold else None,
        on_repetition_end=_on_repetition_end if on_repetition else None,
        on_experiment_end=_on_experiment_end if on_experiment else None,
        agg_name="confusion_matrix_oof",
    )


def confusion_matrix_holdout():
    required_attributes = [
        "stat_aggregates",
        "holdout_target_data",
        "final_holdout_predictions",
        "repetition_holdout_predictions",
        "fold_holdout_predictions",
        "run_holdout_predictions",
    ]

    def _on_experiment_start(
        stat_aggregates,
        holdout_target_data,
        final_holdout_predictions,
        repetition_holdout_predictions,
        fold_holdout_predictions,
        run_holdout_predictions,
    ):
        stat_aggregates["confusion_matrix_holdout"] = dict(runs=[], folds=[], reps=[], final=None)

    def _on_run_end(
        stat_aggregates,
        holdout_target_data,
        final_holdout_predictions,
        repetition_holdout_predictions,
        fold_holdout_predictions,
        run_holdout_predictions,
    ):
        stat_aggregates["confusion_matrix_holdout"]["runs"].append(
            _confusion_matrix_helper(holdout_target_data, run_holdout_predictions)
        )
        return "foo _on_run_end"

    def _on_fold_end(
        stat_aggregates,
        holdout_target_data,
        final_holdout_predictions,
        repetition_holdout_predictions,
        fold_holdout_predictions,
        run_holdout_predictions,
    ):
        stat_aggregates["confusion_matrix_holdout"]["folds"].append(
            _confusion_matrix_helper(holdout_target_data, fold_holdout_predictions)
        )

    def _on_repetition_end(
        stat_aggregates,
        holdout_target_data,
        final_holdout_predictions,
        repetition_holdout_predictions,
        fold_holdout_predictions,
        run_holdout_predictions,
    ):
        stat_aggregates["confusion_matrix_holdout"]["reps"].append(
            _confusion_matrix_helper(holdout_target_data, repetition_holdout_predictions)
        )

    def _on_experiment_end(
        stat_aggregates,
        holdout_target_data,
        final_holdout_predictions,
        repetition_holdout_predictions,
        fold_holdout_predictions,
        run_holdout_predictions,
    ):
        stat_aggregates["confusion_matrix_holdout"]["final"] = _confusion_matrix_helper(
            holdout_target_data, final_holdout_predictions
        )

    return lambda_callback(
        required_attributes=required_attributes,
        on_experiment_start=_on_experiment_start,
        on_run_end=_on_run_end,
        on_fold_end=_on_fold_end,
        on_repetition_end=_on_repetition_end,
        on_experiment_end=_on_experiment_end,
    )


def _confusion_matrix_helper(targets, predictions):
    return confusion_matrix(targets, get_clean_prediction(targets, predictions))
