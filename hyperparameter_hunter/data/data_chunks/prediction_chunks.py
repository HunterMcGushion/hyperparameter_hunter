##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.data.data_core import BaseDataChunk
from hyperparameter_hunter.feature_engineering import FeatureEngineer

##################################################
# Import Miscellaneous Assets
##################################################
from contextlib import suppress
from copy import deepcopy
import numpy as np
import pandas as pd


##################################################
# Prediction Chunks
##################################################
class BasePredictionChunk(BaseDataChunk):
    #################### Division Start Points ####################
    def on_exp_start(self, *args, **kwargs):
        self.final = 0
        self.T.final = 0

    def on_rep_start(self, *args, **kwargs):
        self.rep = 0
        self.T.rep = 0

    def on_fold_start(self, *args, **kwargs):
        self.fold = 0
        self.T.fold = 0

    #################### Division End Points ####################
    def on_run_end(self, prediction, feature_engineer, target_column, *args, **kwargs):
        """...

        Parameters
        ----------
        prediction: Array-like
        feature_engineer: FeatureEngineer
        target_column: List[str]
        *args: Tuple
        **kwargs: Dict"""
        self.T.run = deepcopy(prediction)
        self.run = deepcopy(prediction)

        self.run = _format_prediction(self.run, target_column)
        # `self.run` must be same shape as data transformed by `feature_engineer` prior to inversion
        # TODO: Make sure this doesn't screw up when no `inverse_transform` call
        #  Because then it'll just be two consecutive calls to `_format_predictions` with `self.run`

        with suppress(AttributeError):  # TODO: Drop `suppress` - Was for `feature_engineer={}`
            # NOTE: How does `FeatureEngineer` know these are predictions to invert, not inputs?
            #   Probably need to make an assumption for now, albeit a fairly safe one
            self.run = feature_engineer.inverse_transform(self.run)

        self.run = _format_prediction(self.run, target_column)
        self.T.run = _format_prediction(self.T.run, target_column)

        self.fold += self.run
        self.T.fold += self.T.run
        # TODO: Add `FeatureEngineer` method called after `inverse_transform` to format as DataFrame
        #   Should already know about different column names - Move `_format_prediction` there?
        # FLAG: Need to `_format_prediction` on `self.T.run` although `target_column` may differ
        #   Might be able to use transformed `data_holdout.target` to figure it out - Not pretty

    def on_fold_end(self, runs: int, *args, **kwargs):
        # TODO: For all `/=` ops herein, conditionally do floor div if `self.run` is non-continuous?
        self.fold /= runs
        self.rep += self.fold
        self.T.fold /= runs
        self.T.rep += self.T.fold

    def on_rep_end(self, n_splits: int, *args, **kwargs):
        self.rep /= n_splits
        self.final += self.rep
        self.T.rep /= n_splits
        self.T.final += self.T.rep

    def on_exp_end(self, n_repeats: int):
        self.final /= n_repeats
        self.T.final /= n_repeats


class OOFPredictionChunk(BasePredictionChunk):
    #################### Division Start Points ####################
    def on_exp_start(self, zero_predictions, *args, **kwargs):
        self.final = deepcopy(zero_predictions)
        self.T.final = deepcopy(zero_predictions)

    def on_rep_start(self, zero_predictions, *args, **kwargs):
        self.rep = deepcopy(zero_predictions)
        self.T.rep = deepcopy(zero_predictions)

    #################### Division End Points ####################
    # noinspection PyMethodOverriding
    def on_fold_end(self, validation_index, runs: int, *args, **kwargs):
        self.fold /= runs
        self.rep.iloc[validation_index] += self.fold.values
        self.T.fold /= runs
        self.T.rep.iloc[validation_index] += self.T.fold.values

    def on_rep_end(self, *args, **kwargs):
        self.final += self.rep
        self.T.final += self.T.rep


class HoldoutPredictionChunk(BasePredictionChunk):
    ...


class TestPredictionChunk(BasePredictionChunk):
    ...


##################################################
# Utilities
##################################################
def _format_prediction(predictions, target_column, index=None, dtype=np.float64) -> pd.DataFrame:
    """Organize predictions into a standard format, and one-hot encode predictions as necessary

    Parameters
    ----------
    predictions: Array-like
        A model's predictions for a set of input data
    target_column: List[str]
        Name(s) for the target column(s) in the returned formatted `predictions` DataFrame
    index: Array-like, or None, default=None
        Index to use for the resulting DataFrame. Defaults to `numpy.arange(len(predictions))`
    dtype: Dtype, or None, default=`numpy.float64`
        Datatype to force on `predictions`. If None, datatype will be inferred

    Returns
    -------
    predictions: `pandas.DataFrame`
        Formatted DataFrame containing `predictions` that has been one-hot encoded if necessary

    Examples
    --------
    >>> _format_prediction(np.array([3.2, 14.5, 6.8]), ["y"])
          y
    0   3.2
    1  14.5
    2   6.8
    >>> _format_prediction(np.array([1, 0, 1]), ["y"])
         y
    0  1.0
    1  0.0
    2  1.0
    >>> _format_prediction(np.array([2, 1, 0]), ["y_0", "y_1", "y_2"], dtype=np.int8)
       y_0  y_1  y_2
    0    0    0    1
    1    0    1    0
    2    1    0    0"""
    # `target_column` indicates multidimensional output, but predictions are one-dimensional
    if len(target_column) > 1:
        if (len(predictions.shape) == 1) or (predictions.shape[1] == 1):
            predictions = pd.get_dummies(predictions).values

    return pd.DataFrame(data=predictions, index=index, columns=target_column, dtype=dtype)
