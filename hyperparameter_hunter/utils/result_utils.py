"""This module defines default helper functions used during an Experiment's result-saving process

Related
-------
:mod:`hyperparameter_hunter.environment`
    Uses the contents of :mod:`hyperparameter_hunter.utils.result_utils` to set default values to
    help process Experiments' result files if they are not explicitly provided. These values are
    then used by :mod:`hyperparameter_hunter.recorders`
:mod:`hyperparameter_hunter.recorders`
    This module uses certain attributes set by :class:`hyperparameter_hunter.environment.Environment`
    (:attr:`Environment.prediction_formatter`, and :attr:`Environment.do_full_save`) for the purpose
    of formatting and saving Experiment result files. Those attributes are, by default, the
    utilities defined in :mod:`hyperparameter_hunter.utils.result_utils`

Notes
-----
The utilities defined herein are weird for a couple reasons: 1) They don't do much, and 2) Despite
the fact that they don't do much, they are extremely sensitive. Because they are default values for
:class:`Environment` attributes that are included when generating
:attr:`Environment.cross_experiment_key`, any seemingly insignificant change to them is likely to
result in an entirely different cross_experiment_key. This will, in turn, result in Experiments not
matching with other similar Experiments during hyperparameter optimization, despite the fact that
the changes may not have done anything at all. So be careful, here"""
##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd


def format_predictions(
    raw_predictions: np.array, dataset_df: pd.DataFrame, target_column: str, id_column: str = None
):
    """Organize components into a pandas.DataFrame that is properly formatted and ready to save

    Parameters
    ----------
    raw_predictions: np.array
        The actual predictions that were made and that should inhabit the column named
        `target_column` in the result
    dataset_df: pd.DataFrame
        The original data provided that yielded `raw_predictions`. If `id_column` is not None, it
        must be in `dataset_df`. In practice, expect this value to be one of the following:
        :attr:`experiments.BaseExperiment.train_dataset`,
        :attr:`experiments.BaseExperiment.holdout_dataset`, or
        :attr:`experiments.BaseExperiment.test_dataset`
    target_column: str
        The name for the result column containing `raw_predictions`
    id_column: str, or None, default=None
        If not None, must be the name of a column in `dataset_df`, the contents of which will be
        included as a column in the result and are assumed to be sample identifiers of some kind

    Returns
    -------
    predictions: pd.DataFrame
        Dataframe containing the formatted predictions"""
    predictions = pd.DataFrame()
    if id_column is not None:
        predictions[id_column] = dataset_df[id_column]
    predictions[target_column] = raw_predictions
    predictions.reset_index(inplace=True, drop=True)
    return predictions


# def format_predictions(raw_predictions, result_df, target_column):
#     result_df[target_column] = raw_predictions
#     predictions = result_df[[target_column]].copy()
#     predictions.reset_index(inplace=True, drop=True)
#     return predictions


# def format_predictions(raw_predictions, result_df, target_column):
#     result_df[target_column] = np.expm1(result_df[target_column]).clip(lower=0.)
#     predictions = result_df[[target_column]].copy()
#     return predictions


# noinspection PyUnusedLocal
def default_do_full_save(result_description: dict) -> bool:
    """Determines whether an Experiment's full result should be saved based on its Description dict

    Parameters
    ----------
    result_description: dict
        The formatted description of the Experiment's results

    Notes
    -----
    This function is useless. It is included as an example for proper implementation of custom
    `do_full_save` functions"""
    return True
