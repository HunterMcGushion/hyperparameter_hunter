##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BasePredictorCallback

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd


class PredictorOOF(BasePredictorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.final_oof_predictions = None
        self.repetition_oof_predictions = None
        self.run_validation_predictions = None
        self.validation_index = None
        self.experiment_params = None
        super().__init__()

    def on_experiment_start(self):
        required_attributes = [
            "final_oof_predictions",
            "repetition_oof_predictions",
            "run_validation_predictions",
            "train_dataset",
            "validation_index",
            "target_column",
            "experiment_params",
        ]
        for attr in required_attributes:
            if not hasattr(self, attr):
                raise AttributeError("Missing required `PredictorOOF` attribute: {}".format(attr))

        self.final_oof_predictions = self.__zeros_df()
        super().on_experiment_start()

    def on_repetition_start(self):
        self.repetition_oof_predictions = self.__zeros_df()
        super().on_repetition_start()

    def on_run_end(self):
        self.run_validation_predictions = self.model.predict(self.fold_validation_input)
        self.run_validation_predictions = _format_predictions(
            self.run_validation_predictions, self.target_column, index=self.validation_index
        )

        self.repetition_oof_predictions.iloc[
            self.validation_index
        ] += self.run_validation_predictions

        super().on_run_end()

    def on_fold_end(self):
        self.repetition_oof_predictions.iloc[self.validation_index] /= self.experiment_params[
            "runs"
        ]
        super().on_fold_end()

    def on_repetition_end(self):
        self.final_oof_predictions += self.repetition_oof_predictions
        super().on_repetition_end()

    def on_experiment_end(self):
        self.final_oof_predictions /= self.cross_validation_params.get("n_repeats", 1)
        super().on_experiment_end()

    def __zeros_df(self):
        return pd.DataFrame(0, index=np.arange(len(self.train_dataset)), columns=self.target_column)


class PredictorHoldout(BasePredictorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.final_holdout_predictions = None
        self.repetition_holdout_predictions = None
        self.fold_holdout_predictions = None
        self.run_holdout_predictions = None
        self.experiment_params = None
        super().__init__()

    def on_experiment_start(self):
        self.final_holdout_predictions = 0
        super().on_experiment_start()

    def on_repetition_start(self):
        self.repetition_holdout_predictions = 0
        super().on_repetition_start()

    def on_fold_start(self):
        self.fold_holdout_predictions = 0
        super().on_fold_start()

    def on_run_end(self):
        self.run_holdout_predictions = self.model.predict(self.holdout_input_data)
        self.run_holdout_predictions = _format_predictions(
            self.run_holdout_predictions, self.target_column
        )

        self.fold_holdout_predictions += self.run_holdout_predictions
        super().on_run_end()

    def on_fold_end(self):
        self.fold_holdout_predictions /= self.experiment_params["runs"]
        self.repetition_holdout_predictions += self.fold_holdout_predictions
        super().on_fold_end()

    def on_repetition_end(self):
        self.repetition_holdout_predictions /= self.cross_validation_params["n_splits"]
        self.final_holdout_predictions += self.repetition_holdout_predictions
        super().on_repetition_end()

    def on_experiment_end(self):
        self.final_holdout_predictions /= self.cross_validation_params.get("n_repeats", 1)
        super().on_experiment_end()


class PredictorTest(BasePredictorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.final_test_predictions = None
        self.repetition_test_predictions = None
        self.fold_test_predictions = None
        self.run_test_predictions = None
        self.experiment_params = None
        super().__init__()

    def on_experiment_start(self):
        self.final_test_predictions = 0
        super().on_experiment_start()

    def on_repetition_start(self):
        self.repetition_test_predictions = 0
        super().on_repetition_start()

    def on_fold_start(self):
        self.fold_test_predictions = 0
        super().on_fold_start()

    def on_run_end(self):
        self.run_test_predictions = self.model.predict(self.test_input_data)
        self.run_test_predictions = _format_predictions(
            self.run_test_predictions, self.target_column
        )

        self.fold_test_predictions += self.run_test_predictions
        super().on_run_end()

    def on_fold_end(self):
        self.fold_test_predictions /= self.experiment_params["runs"]
        self.repetition_test_predictions += self.fold_test_predictions
        super().on_fold_end()

    def on_repetition_end(self):
        self.repetition_test_predictions /= self.cross_validation_params["n_splits"]
        self.final_test_predictions += self.repetition_test_predictions
        super().on_repetition_end()

    def on_experiment_end(self):
        self.final_test_predictions /= self.cross_validation_params.get("n_repeats", 1)
        super().on_experiment_end()


def _format_predictions(predictions, target_column, index=None, dtype=np.float64):
    """Organize predictions into a standard format, and one-hot encode predictions as necessary

    Parameters
    ----------
    predictions: Array-like
        A model's predictions for a set of input data
    target_column: List
        A list of one or more strings corresponding to the name(s) of target output column(s)
    index: Array-like, or None, default=None
        Index to use for the resulting DataFrame. Defaults to `numpy.arange(len(predictions))`
    dtype: Dtype, or None, default=`numpy.float64`
        Datatype to force on `predictions`. If None, datatype will be inferred

    Returns
    -------
    predictions: `pandas.DataFrame`
        Formatted DataFrame containing `predictions` that has been one-hot encoded if necessary"""
    if (len(target_column) > 1) and ((len(predictions.shape) == 1) or (predictions.shape[1] == 1)):
        predictions = pd.get_dummies(predictions).values

    predictions = pd.DataFrame(data=predictions, index=index, columns=target_column, dtype=dtype)
    return predictions


if __name__ == "__main__":
    pass
