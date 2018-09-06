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


class EvaluatorOOF(BaseEvaluatorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.fold_validation_target = None
        self.final_oof_predictions = None
        self.repetition_oof_predictions = None
        self.run_validation_predictions = None
        self.validation_index = None
        super().__init__()

    def on_run_end(self):
        """Evaluate out-of-fold predictions for the run"""
        self.evaluate("oof", self.fold_validation_target, self.run_validation_predictions)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the fold"""
        self.evaluate(
            "oof",
            self.fold_validation_target,
            self.repetition_oof_predictions.iloc[self.validation_index],
        )
        super().on_fold_end()

    def on_repetition_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the repetition"""
        self.evaluate("oof", self.train_target_data, self.repetition_oof_predictions)
        super().on_repetition_end()

    def on_experiment_end(self):
        """Evaluate final (run/repetition-averaged) out-of-fold predictions"""
        self.evaluate("oof", self.train_target_data, self.final_oof_predictions)
        super().on_experiment_end()


class EvaluatorHoldout(BaseEvaluatorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.holdout_target_data = None
        self.final_holdout_predictions = None
        self.repetition_holdout_predictions = None
        self.fold_holdout_predictions = None
        self.run_holdout_predictions = None
        super().__init__()

    def on_run_end(self):
        """Evaluate holdout predictions for the run"""
        self.evaluate("holdout", self.holdout_target_data, self.run_holdout_predictions)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) holdout predictions for the fold"""
        self.evaluate("holdout", self.holdout_target_data, self.fold_holdout_predictions)
        super().on_fold_end()

    def on_repetition_end(self):
        """Evaluate (run-averaged) holdout predictions for the repetition"""
        self.evaluate("holdout", self.holdout_target_data, self.repetition_holdout_predictions)
        super().on_repetition_end()

    def on_experiment_end(self):
        """Evaluate final (run/repetition-averaged) holdout predictions"""
        self.evaluate("holdout", self.holdout_target_data, self.final_holdout_predictions)
        super().on_experiment_end()


if __name__ == "__main__":
    pass
