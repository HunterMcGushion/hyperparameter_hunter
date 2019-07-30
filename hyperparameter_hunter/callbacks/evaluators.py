"""This module defines Evaluator callbacks to score predictions generated during the different time
divisions of the :class:`~hyperparameter_hunter.experiments.BaseExperiment` by invoking
:meth:`hyperparameter_hunter.metrics.ScoringMixIn.evaluate`

Related
-------
:mod:`hyperparameter_hunter.metrics`
    Defines :class:`~hyperparameter_hunter.metrics.ScoringMixIn`, which is inherited by
    :class:`~hyperparameter_hunter.experiments.BaseExperiment`, and provides the `evaluate` method
    that is called by the classes in :mod:`~hyperparameter_hunter.callbacks.evaluators`

Notes
-----
Regarding evaluation when `G.Env.save_transformed_metrics` is False, target data will be either
`fold` (for `on_run_end`/`on_fold_end`) or `d` (for `on_rep_end`/`on_exp_end`). Prediction data used
for evaluation in this case does not follow this abnormal pattern. Target data is limited to either
the `fold` or `d` `data_chunks` when `G.Env.save_transformed_metrics` is False because targets for
`run` and `rep` are identical to the targets for `fold` and `d`, respectively. This is still the
case even if performing inverse target transformation via
:class:`~hyperparameter_hunter.feature_engineering.EngineerStep`. Because the target values do not
change between these two pairs of divisions, their values may be unset, so the targets for the
division immediately above are used instead. As noted in
:mod:`hyperparameter_hunter.data.data_chunks.target_chunks`, both itself and
:mod:`hyperparameter_hunter.callback.wranglers.target_wranglers` are concerned only with transformed
targets--not with original targets (or inverted targets). That is because original targets and
inverted targets should be identical. Original targets are updated only in
:meth:`hyperparameter_hunter.experiments.BaseExperiment.on_exp_start` (through
:class:`hyperparameter_hunter.data.data_core.BaseDataset` initialization) and in
:meth:`hyperparameter_hunter.experiments.BaseCVExperiment.on_fold_start`"""
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
        if G.Env.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.fold, self.data_oof.prediction.T.run)
        else:  # See module "Notes"
            self.evaluate("oof", self.data_oof.target.fold, self.data_oof.prediction.run)
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the fold"""
        if G.Env.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.fold, self.data_oof.prediction.T.fold)
        else:  # See module "Notes"
            self.evaluate("oof", self.data_oof.target.fold, self.data_oof.prediction.fold)
        super().on_fold_end()

    def on_rep_end(self):
        """Evaluate (run-averaged) out-of-fold predictions for the repetition"""
        if G.Env.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.rep, self.data_oof.prediction.T.rep)
        else:  # See module "Notes"
            self.evaluate("oof", self.data_oof.target.d, self.data_oof.prediction.rep)
        super().on_rep_end()

    def on_exp_end(self):
        """Evaluate final (run/repetition-averaged) out-of-fold predictions"""
        if G.Env.save_transformed_metrics:
            self.evaluate("oof", self.data_oof.target.T.final, self.data_oof.prediction.T.final)
        else:  # See module "Notes"
            self.evaluate("oof", self.data_oof.target.d, self.data_oof.prediction.final)
        super().on_exp_end()


class EvaluatorHoldout(BaseEvaluatorCallback):
    data_holdout: BaseDataset

    def on_run_end(self):
        """Evaluate holdout predictions for the run"""
        if G.Env.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.run, self.data_holdout.prediction.T.run
            )
        else:  # See module "Notes"
            self.evaluate(
                "holdout", self.data_holdout.target.fold, self.data_holdout.prediction.run
            )
        super().on_run_end()

    def on_fold_end(self):
        """Evaluate (run-averaged) holdout predictions for the fold"""
        if G.Env.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.fold, self.data_holdout.prediction.T.fold
            )
        else:  # See module "Notes"
            self.evaluate(
                "holdout", self.data_holdout.target.fold, self.data_holdout.prediction.fold
            )
        super().on_fold_end()

    def on_rep_end(self):
        """Evaluate (run-averaged) holdout predictions for the repetition"""
        if G.Env.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.rep, self.data_holdout.prediction.T.rep
            )
        else:  # See module "Notes"
            self.evaluate("holdout", self.data_holdout.target.d, self.data_holdout.prediction.rep)
        super().on_rep_end()

    def on_exp_end(self):
        """Evaluate final (run/repetition-averaged) holdout predictions"""
        if G.Env.save_transformed_metrics:
            self.evaluate(
                "holdout", self.data_holdout.target.T.final, self.data_holdout.prediction.T.final
            )
        else:  # See module "Notes"
            self.evaluate("holdout", self.data_holdout.target.d, self.data_holdout.prediction.final)
        super().on_exp_end()
