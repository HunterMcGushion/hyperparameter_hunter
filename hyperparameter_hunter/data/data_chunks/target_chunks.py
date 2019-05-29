##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.data.data_core import BaseDataChunk

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy


# FLAG: No need to invert target data - Already have normal targets - Just save transformed targets
#  and average them for later evaluation - Target wranglers only concerned with transformed targets
##################################################
# Target Chunks
##################################################
class BaseTargetChunk(BaseDataChunk):
    ...


class TrainTargetChunk(BaseTargetChunk):
    ...


class OOFTargetChunk(BaseTargetChunk):
    #################### Division Start Points ####################
    def on_experiment_start(self, empty_output_frame, *args, **kwargs):
        self.T.final = empty_output_frame

    def on_repetition_start(self, empty_output_frame, *args, **kwargs):
        self.T.rep = empty_output_frame

    def on_fold_start(self, *args, **kwargs):
        ...  # `self.fold` and `self.T.fold` (intra-CV) set by `BaseCVExperiment.on_fold_start`

    def on_run_start(self, *args, **kwargs):
        self.T.run = deepcopy(self.T.fold)

    #################### Division End Points ####################
    def on_run_end(self, *args, **kwargs):
        ...  # `self.T.fold` already set - No need to update

    def on_fold_end(self, validation_index, *args, **kwargs):
        self.T.rep.iloc[validation_index] += self.T.fold

    def on_repetition_end(self, n_splits: int, *args, **kwargs):
        self.T.final += self.T.rep

    def on_experiment_end(self, n_repeats: int, *args, **kwargs):
        self.T.final /= n_repeats


class HoldoutTargetChunk(BaseTargetChunk):
    #################### Division Start Points ####################
    def on_experiment_start(self, empty_output_frame, *args, **kwargs):
        # `self.d` and `self.T.d` (pre-CV) set by `BaseExperiment.on_experiment_start`
        self.T.final = empty_output_frame

    def on_repetition_start(self, empty_output_frame, *args, **kwargs):
        self.T.rep = empty_output_frame

    def on_fold_start(self, *args, **kwargs):
        ...  # `self.fold` and `self.T.fold` (intra-CV) set by `BaseCVExperiment.on_fold_start`

    def on_run_start(self, *args, **kwargs):
        self.T.run = deepcopy(self.T.fold)

    #################### Division End Points ####################
    def on_run_end(self, *args, **kwargs):
        ...  # `self.T.fold` already set - No need to update

    def on_fold_end(self, *args, **kwargs):
        self.T.rep += self.T.fold

    def on_repetition_end(self, n_splits: int, *args, **kwargs):
        self.T.rep /= n_splits
        self.T.final += self.T.rep

    def on_experiment_end(self, n_repeats: int, *args, **kwargs):
        self.T.final /= n_repeats
