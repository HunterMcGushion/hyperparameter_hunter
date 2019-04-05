"""This module defines data wrangler callback classes, whose purpose is to manage the
experiment's datasets. This includes 1) performing feature engineering; 2) collecting entire
datasets; 3) collecting particular aspects of datasets, such as their targets or the predictions
associated with them. Collecting datasets' targets and predictions also involves accounting for
any transformations that may have been applied during feature engineering by inverting them"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseWranglerCallback

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pandas as pd
from typing import Optional


##################################################
# Target Wranglers
##################################################
class WranglerTargetOOF(BaseWranglerCallback):
    transformed_final_oof_target: Optional[pd.DataFrame]
    transformed_repetition_oof_target: Optional[pd.DataFrame]
    transformed_run_validation_target: Optional[pd.DataFrame]
    validation_index: list
    experiment_params: dict
    __initial_fold_validation_target: pd.DataFrame

    def on_experiment_start(self):
        self.transformed_final_oof_target = None
        super().on_experiment_start()

    def on_repetition_start(self):
        if getattr(self, "transformed_repetition_oof_target", None) is not None:
            self.transformed_repetition_oof_target = self.__zeros_df()
        else:
            self.transformed_repetition_oof_target = None
        super().on_repetition_start()

    # FLAG: `WranglerTarget` classes must be executed before `WranglerFeatureEngineer`, otherwise `fold_validation_target` won't be correct here
    def on_fold_start(self):
        self.__initial_fold_validation_target = self.data_oof.target.fold.copy()
        super().on_fold_start()

    def on_run_start(self):
        self.transformed_run_validation_target = None
        super().on_run_start()

    def on_run_end(self):
        if not self.__initial_fold_validation_target.equals(self.data_oof.target.fold):
            if (
                self.transformed_final_oof_target is None
                or self.transformed_repetition_oof_target is None
            ):
                self.transformed_final_oof_target = self.__zeros_df()
                self.transformed_repetition_oof_target = self.__zeros_df()

            self.transformed_run_validation_target = self.data_oof.target.fold.copy()
            self.transformed_run_validation_target.index = self.validation_index
            self.transformed_repetition_oof_target.iloc[
                self.validation_index
            ] += self.transformed_run_validation_target
        super().on_run_end()

    def on_fold_end(self):
        if self.transformed_run_validation_target is not None:
            self.transformed_repetition_oof_target.iloc[
                self.validation_index
            ] /= self.experiment_params["runs"]
        super().on_fold_end()

    def on_repetition_end(self):
        if self.transformed_run_validation_target is not None:
            self.transformed_final_oof_target += self.transformed_repetition_oof_target
        super().on_repetition_end()

    def on_experiment_end(self):
        if self.transformed_run_validation_target is not None:
            self.transformed_final_oof_target /= self.cv_params.get("n_repeats", 1)
        super().on_experiment_end()

    def __zeros_df(self):
        return pd.DataFrame(0, index=np.arange(len(self.train_dataset)), columns=self.target_column)


class WranglerTargetHoldout(BaseWranglerCallback):
    transformed_final_holdout_target: Optional[pd.DataFrame]
    transformed_repetition_holdout_target: Optional[pd.DataFrame]
    transformed_fold_holdout_target: Optional[pd.DataFrame]
    transformed_run_holdout_target: Optional[pd.DataFrame]
    experiment_params: dict
    cv_params: dict

    def on_experiment_start(self):
        self.transformed_final_holdout_target = 0
        super().on_experiment_start()

    def on_repetition_start(self):
        self.transformed_repetition_holdout_target = 0
        super().on_repetition_start()

    def on_fold_start(self):
        self.transformed_fold_holdout_target = 0
        super().on_fold_start()

    def on_run_start(self):
        self.transformed_run_holdout_target = None
        super().on_run_start()

    def on_run_end(self):
        if not self.data_holdout.target.d.equals(self.data_holdout.target.fold):
            self.transformed_run_holdout_target = self.data_holdout.target.fold.copy()
            self.transformed_fold_holdout_target += self.transformed_run_holdout_target
        super().on_run_end()

    def on_fold_end(self):
        if self.transformed_run_holdout_target is not None:
            self.transformed_fold_holdout_target /= self.experiment_params["runs"]
            self.transformed_repetition_holdout_target += self.transformed_fold_holdout_target
        super().on_fold_end()

    def on_repetition_end(self):
        if self.transformed_run_holdout_target is not None:
            self.transformed_repetition_holdout_target /= self.cv_params["n_splits"]
            self.transformed_final_holdout_target += self.transformed_repetition_holdout_target
        super().on_repetition_end()

    def on_experiment_end(self):
        if self.transformed_run_holdout_target is not None:
            self.transformed_final_holdout_target /= self.cv_params.get("n_repeats", 1)
        super().on_experiment_end()


##################################################
# Prediction Wranglers
##################################################
# TODO: Make wranglers to collect un-transformed predictions
# class WranglerPredictionsOOF(BaseWranglerCallback):
#     ...


# class WranglerPredictionsHoldout(BaseWranglerCallback):
#     ...


# class WranglerPredictionsTest(BaseWranglerCallback):
#     ...


##################################################
# Feature Engineering Wranglers
##################################################
# class WranglerFeatureEngineer(BaseWranglerCallback):
#     ...
