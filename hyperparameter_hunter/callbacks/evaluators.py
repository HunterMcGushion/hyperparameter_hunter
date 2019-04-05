"""This module defines the Evaluator callbacks that perform calls to
:meth:`hyperparameter_hunter.metrics.ScoringMixIn.evaluate` in order to score predictions generated
at various stages of the :class:`hyperparameter_hunter.experiments.BaseExperiment`

Related
-------
:mod:`hyperparameter_hunter.metrics`
    Defines :class:`hyperparameter_hunter.metrics.ScoringMixIn`, which is inherited by
    :class:`hyperparameter_hunter.experiments.BaseExperiment`, and provides the `evaluate` method
    that is called by the classes in :mod:`hyperparameter_hunter.callbacks.evaluators`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseEvaluatorCallback
from hyperparameter_hunter.datasets import Dataset

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
from typing import Union


class EvaluatorOOF(BaseEvaluatorCallback):
    data_oof: Dataset
    validation_index: list

    def on_run_end(self):
        """Evaluate out-of-fold predictions for the run"""
        self.evaluate("oof", self.data_oof.target.fold, self.data_oof.prediction.run)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the fold"""
        self.evaluate(
            "oof",
            self.data_oof.target.fold,
            self.data_oof.prediction.rep.iloc[self.validation_index],
        )
        super().on_fold_end()

    def on_repetition_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the repetition"""
        try:
            self.evaluate("oof", self.data_oof.T.target.rep, self.data_oof.prediction.rep)
        except (TypeError, AttributeError):
            # self.evaluate("oof", self.data_oof.target.d, self.data_oof.prediction.rep)  # TODO: Should use this when collecting transformed targets
            self.evaluate(
                "oof", self.data_train.target.d, self.data_oof.prediction.rep
            )  # FLAG: TEMPORARY
        super().on_repetition_end()

    def on_experiment_end(self):
        """Evaluate final (run/repetition-averaged) out-of-fold predictions"""
        try:
            self.evaluate("oof", self.data_oof.T.target.final, self.data_oof.prediction.final)
        except (TypeError, AttributeError):
            # self.evaluate("oof", self.data_oof.target.d, self.data_oof.prediction.final)  # TODO: Should use this when collecting transformed targets
            self.evaluate(
                "oof", self.data_train.target.d, self.data_oof.prediction.final
            )  # FLAG: TEMPORARY
        super().on_experiment_end()


class EvaluatorHoldout(BaseEvaluatorCallback):
    data_holdout: Dataset

    def on_run_end(self):
        """Evaluate holdout predictions for the run"""
        self.evaluate("holdout", self.data_holdout.target.fold, self.data_holdout.prediction.run)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) holdout predictions for the fold"""
        self.evaluate("holdout", self.data_holdout.target.fold, self.data_holdout.prediction.fold)
        super().on_fold_end()

    def on_repetition_end(self):
        """Evaluate (run-averaged) holdout predictions for the repetition"""
        try:
            # If no target transformation occurred, `transformed_repetition_holdout_target` will
            # be `0`, raising `TypeError`
            self.evaluate(
                "holdout", self.data_holdout.T.target.rep, self.data_holdout.prediction.rep
            )
        except (TypeError, AttributeError):
            self.evaluate("holdout", self.data_holdout.target.d, self.data_holdout.prediction.rep)

        super().on_repetition_end()

    def on_experiment_end(self):
        """Evaluate final (run/repetition-averaged) holdout predictions"""
        try:
            # If no target transformation occurred, `transformed_final_holdout_target` will
            # be `0`, raising `TypeError`
            self.evaluate(
                "holdout",
                self.data_holdout.T.target.d,
                self.data_holdout.prediction.final  # FLAG: TEMPORARY
                # "holdout", self.holdout_data.T.target.final, self.data_holdout.prediction.final  # TODO: Should use this when collecting transformed targets
            )
        except (TypeError, AttributeError):
            self.evaluate(
                "holdout", self.data_holdout.target.d, self.data_holdout.prediction.final
            )  # FLAG: TEMPORARY
            # self.evaluate("holdout", self.data_holdout.target.final, self.data_holdout.prediction.final)  # TODO: Should use this when collecting transformed targets

        super().on_experiment_end()


if __name__ == "__main__":
    pass
