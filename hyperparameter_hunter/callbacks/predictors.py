##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BasePredictorCallback
from hyperparameter_hunter.datasets import Dataset

##################################################
# Import Miscellaneous Assets
##################################################
from contextlib import suppress
import numpy as np
import pandas as pd


class PredictorOOF(BasePredictorCallback):
    data_oof: Dataset
    validation_index: list
    experiment_params = dict

    def on_experiment_start(self):
        self.data_oof.prediction.final = self.__zeros_df()
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_oof.prediction.rep = self.__zeros_df()
        super().on_repetition_start()

    def on_run_end(self):
        self.data_oof.prediction.run = self.model.predict(self.data_oof.input.fold)

        # TODO: TEST INVERSE TARGET TRANSFORM BELOW
        with suppress(AttributeError):
            self.data_oof.prediction.run = self.feature_engineer.inverse_transform(
                self.data_oof.prediction.run
            )
        # TODO: TEST INVERSE TARGET TRANSFORM ABOVE

        self.data_oof.prediction.run = self._format_prediction(
            self.data_oof.prediction.run, index=self.validation_index
        )

        self.data_oof.prediction.rep.iloc[self.validation_index] += self.data_oof.prediction.run

        super().on_run_end()

    def on_fold_end(self):
        self.data_oof.prediction.rep.iloc[self.validation_index] /= self.experiment_params["runs"]
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_oof.prediction.final += self.data_oof.prediction.rep
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_oof.prediction.final /= self.cv_params.get("n_repeats", 1)
        super().on_experiment_end()

    def __zeros_df(self):
        return pd.DataFrame(0, index=np.arange(len(self.train_dataset)), columns=self.target_column)


class PredictorHoldout(BasePredictorCallback):
    data_holdout: Dataset
    experiment_params: dict

    def on_experiment_start(self):
        self.data_holdout.prediction.final = 0
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_holdout.prediction.rep = 0
        super().on_repetition_start()

    def on_fold_start(self):
        self.data_holdout.prediction.fold = 0
        super().on_fold_start()

    def on_run_end(self):
        self.data_holdout.prediction.run = self.model.predict(self.data_holdout.input.fold)

        # TODO: TEST INVERSE TARGET TRANSFORM BELOW
        with suppress(AttributeError):
            self.data_holdout.prediction.run = self.feature_engineer.inverse_transform(
                self.data_holdout.prediction.run
            )
        # TODO: TEST INVERSE TARGET TRANSFORM ABOVE

        self.data_holdout.prediction.run = self._format_prediction(self.data_holdout.prediction.run)
        self.data_holdout.prediction.fold += self.data_holdout.prediction.run
        super().on_run_end()

    def on_fold_end(self):
        self.data_holdout.prediction.fold /= self.experiment_params["runs"]
        self.data_holdout.prediction.rep += self.data_holdout.prediction.fold
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_holdout.prediction.rep /= self.cv_params["n_splits"]
        self.data_holdout.prediction.final += self.data_holdout.prediction.rep
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_holdout.prediction.final /= self.cv_params.get("n_repeats", 1)
        super().on_experiment_end()


class PredictorTest(BasePredictorCallback):
    data_test: Dataset
    experiment_params: dict

    def on_experiment_start(self):
        self.data_test.prediction.final = 0
        super().on_experiment_start()

    def on_repetition_start(self):
        self.data_test.prediction.rep = 0
        super().on_repetition_start()

    def on_fold_start(self):
        self.data_test.prediction.fold = 0
        super().on_fold_start()

    def on_run_end(self):
        self.data_test.prediction.run = self.model.predict(self.data_test.input.fold)

        # TODO: TEST INVERSE TARGET TRANSFORM BELOW
        with suppress(AttributeError):
            self.data_test.prediction.run = self.feature_engineer.inverse_transform(
                self.data_test.prediction.run
            )
        # TODO: TEST INVERSE TARGET TRANSFORM ABOVE

        self.data_test.prediction.run = self._format_prediction(self.data_test.prediction.run)
        self.data_test.prediction.fold += self.data_test.prediction.run
        super().on_run_end()

    def on_fold_end(self):
        self.data_test.prediction.fold /= self.experiment_params["runs"]
        self.data_test.prediction.rep += self.data_test.prediction.fold
        super().on_fold_end()

    def on_repetition_end(self):
        self.data_test.prediction.rep /= self.cv_params["n_splits"]
        self.data_test.prediction.final += self.data_test.prediction.rep
        super().on_repetition_end()

    def on_experiment_end(self):
        self.data_test.prediction.final /= self.cv_params.get("n_repeats", 1)
        super().on_experiment_end()


if __name__ == "__main__":
    pass
