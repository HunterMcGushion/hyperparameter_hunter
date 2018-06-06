##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseEvaluatorCallback
from hyperparameter_hunter.settings import G


class EvaluatorOOF(BaseEvaluatorCallback):
    def __init__(self):
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        self.fold_validation_target = None
        self.final_oof_predictions = None
        self.repetition_oof_predictions = None
        self.run_validation_predictions = None
        self.validation_index = None
        super().__init__()

    def on_run_end(self):
        self.evaluate('oof', self.fold_validation_target, self.run_validation_predictions)
        super().on_run_end()

    def on_fold_end(self):
        self.evaluate('oof', self.fold_validation_target, self.repetition_oof_predictions.iloc[self.validation_index])
        super().on_fold_end()

    def on_repetition_end(self):
        self.evaluate('oof', self.train_target_data, self.repetition_oof_predictions)
        super().on_repetition_end()

    def on_experiment_end(self):
        self.evaluate('oof', self.train_target_data, self.final_oof_predictions)
        super().on_experiment_end()


class EvaluatorHoldout(BaseEvaluatorCallback):
    def __init__(self):
        """Uncalled - See the 'Notes' section in the documentation of :class:`callbacks.bases.BaseCallback` for details"""
        self.holdout_target_data = None
        self.final_holdout_predictions = None
        self.repetition_holdout_predictions = None
        self.fold_holdout_predictions = None
        self.run_holdout_predictions = None
        super().__init__()

    def on_run_end(self):
        self.evaluate('holdout', self.holdout_target_data, self.run_holdout_predictions)
        super().on_run_end()

    def on_fold_end(self):
        self.evaluate('holdout', self.holdout_target_data, self.fold_holdout_predictions)
        super().on_fold_end()

    def on_repetition_end(self):
        self.evaluate('holdout', self.holdout_target_data, self.repetition_holdout_predictions)
        super().on_repetition_end()

    def on_experiment_end(self):
        self.evaluate('holdout', self.holdout_target_data, self.final_holdout_predictions)
        super().on_experiment_end()


if __name__ == '__main__':
    pass
