"""This module defines Evaluator callbacks to score predictions generated during the different time
divisions of the :class:`~hyperparameter_hunter.experiments.BaseExperiment` by invoking
:meth:`hyperparameter_hunter.metrics.ScoringMixIn.evaluate`

Related
-------
:mod:`hyperparameter_hunter.metrics`
    Defines :class:`~hyperparameter_hunter.metrics.ScoringMixIn`, which is inherited by
    :class:`~hyperparameter_hunter.experiments.BaseExperiment`, and provides the `evaluate` method
    that is called by the classes in :mod:`~hyperparameter_hunter.callbacks.evaluators`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseEvaluatorCallback
from hyperparameter_hunter.data.data_core import BaseDataset
from hyperparameter_hunter.settings import G


class EvaluatorOOF(BaseEvaluatorCallback):
    data_oof: BaseDataset
    validation_index: list

    def on_run_end(self):
        """Evaluate out-of-fold predictions for the run"""
        if G.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.fold, self.data_oof.prediction.T.run)
        else:
            self.evaluate("oof", self.data_oof.target.fold, self.data_oof.prediction.run)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the fold"""
        if G.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.fold, self.data_oof.prediction.T.fold)
        else:
            self.evaluate("oof", self.data_oof.target.fold, self.data_oof.prediction.fold)
        super().on_fold_end()

    def on_repetition_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the repetition"""
        if G.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.rep, self.data_oof.prediction.T.rep)
        else:
            self.evaluate("oof", self.data_oof.target.rep, self.data_oof.prediction.rep)
        super().on_repetition_end()

    def on_experiment_end(self):
        """Evaluate final (run/repetition-averaged) out-of-fold predictions"""
        if G.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.final, self.data_oof.prediction.T.final)
        else:
            self.evaluate("oof", self.data_oof.target.final, self.data_oof.prediction.final)
        super().on_experiment_end()


class EvaluatorHoldout(BaseEvaluatorCallback):
    data_holdout: BaseDataset

    def on_run_end(self):
        """Evaluate holdout predictions for the run"""
        if G.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.run, self.data_holdout.prediction.T.run
            )
        else:
            self.evaluate("holdout", self.data_holdout.target.run, self.data_holdout.prediction.run)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) holdout predictions for the fold"""
        if G.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.fold, self.data_holdout.prediction.T.fold
            )
        else:
            self.evaluate(
                "holdout", self.data_holdout.target.fold, self.data_holdout.prediction.fold
            )
        super().on_fold_end()

    def on_repetition_end(self):
        """Evaluate (run-averaged) holdout predictions for the repetition"""
        if G.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.rep, self.data_holdout.prediction.T.rep
            )
        else:
            self.evaluate("holdout", self.data_holdout.target.rep, self.data_holdout.prediction.rep)
        super().on_repetition_end()

    def on_experiment_end(self):
        """Evaluate final (run/repetition-averaged) holdout predictions"""
        if G.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.final, self.data_holdout.prediction.T.final
            )
        else:
            self.evaluate(
                "holdout", self.data_holdout.target.final, self.data_holdout.prediction.final
            )
        super().on_experiment_end()
