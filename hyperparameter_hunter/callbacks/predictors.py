##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BasePredictorCallback

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np


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
                raise AttributeError(
                    "Missing required attribute for `PredictorOOF` class: {}".format(attr)
                )

        self.final_oof_predictions = 0 * self.train_dataset[self.target_column]
        super().on_experiment_start()

    def on_repetition_start(self):
        self.repetition_oof_predictions = 0 * self.train_dataset[self.target_column]
        super().on_repetition_start()

    def on_run_end(self):
        self.run_validation_predictions = self.model.predict(self.fold_validation_input)
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
        self.fold_holdout_predictions += self.run_holdout_predictions
        super().on_run_end()

    def on_fold_end(self):
        try:
            self.fold_holdout_predictions /= self.experiment_params["runs"]
        except TypeError:
            self.fold_holdout_predictions = np.divide(
                self.fold_holdout_predictions, self.experiment_params["runs"]
            )

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
        self.fold_test_predictions += self.run_test_predictions
        super().on_run_end()

    def on_fold_end(self):
        try:
            self.fold_test_predictions /= self.experiment_params["runs"]
        except TypeError:
            self.fold_test_predictions = np.divide(
                self.fold_test_predictions, self.experiment_params["runs"]
            )

        self.repetition_test_predictions += self.fold_test_predictions
        super().on_fold_end()

    def on_repetition_end(self):
        self.repetition_test_predictions /= self.cross_validation_params["n_splits"]
        self.final_test_predictions += self.repetition_test_predictions
        super().on_repetition_end()

    def on_experiment_end(self):
        self.final_test_predictions /= self.cross_validation_params.get("n_repeats", 1)
        super().on_experiment_end()


if __name__ == "__main__":
    pass
