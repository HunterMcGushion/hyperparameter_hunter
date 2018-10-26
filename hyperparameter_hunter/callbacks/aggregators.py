##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseAggregatorCallback

##################################################
# Import Miscellaneous Assets
##################################################
from datetime import datetime
import numpy as np


class AggregatorTimes(BaseAggregatorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = dict()
        self._rep = None
        self._fold = None
        self._run = None
        super().__init__()

    def on_experiment_start(self):
        self.stat_aggregates.setdefault(
            "times", dict(runs=[], folds=[], reps=[], total_elapsed=None, start=None, end=None)
        )
        self.stat_aggregates["times"]["start"] = str(datetime.now())
        self.stat_aggregates["times"]["total_elapsed"] = datetime.now()
        super().on_experiment_start()

    def on_repetition_start(self):
        self.stat_aggregates["times"]["reps"].append(datetime.now())
        super().on_repetition_start()

    def on_fold_start(self):
        self.stat_aggregates["times"]["folds"].append(datetime.now())
        super().on_fold_start()

    def on_run_start(self):
        self.stat_aggregates["times"]["runs"].append(datetime.now())
        super().on_run_start()

    def on_run_end(self):
        self.__to_elapsed("runs")
        super().on_run_end()

    def on_fold_end(self):
        self.__to_elapsed("folds")
        super().on_fold_end()

    def on_repetition_end(self):
        self.__to_elapsed("reps")
        super().on_repetition_end()

    def on_experiment_end(self):
        #################### Reshape Run/Fold Aggregates to be of Proper Dimensions ####################
        runs_shape = (self._rep + 1, self._fold + 1, self._run + 1)
        folds_shape = (self._rep + 1, self._fold + 1)

        for (key, shape) in [("runs", runs_shape), ("folds", folds_shape)]:
            new_val = np.reshape(self.stat_aggregates["times"][key], shape).tolist()
            self.stat_aggregates["times"][key] = new_val

        self.stat_aggregates["times"]["end"] = str(datetime.now())
        self.__to_elapsed("total_elapsed")
        super().on_experiment_end()

    def __to_elapsed(self, agg_key):
        # TODO: Add documentation
        start_val = self.stat_aggregates["times"][agg_key]
        now = datetime.now()
        if isinstance(start_val, list):
            self.stat_aggregates["times"][agg_key][-1] = (now - start_val[-1]).total_seconds()
        else:
            self.stat_aggregates["times"][agg_key] = (now - start_val).total_seconds()


class AggregatorEvaluations(BaseAggregatorCallback):
    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = dict()
        self.last_evaluation_results = dict(in_fold=None, oof=None, holdout=None)
        self._rep = None
        self._fold = None
        self._run = None
        super().__init__()

    def on_run_end(self):
        #################### Initialize Evaluations Aggregator ####################
        if len(self.stat_aggregates.setdefault("evaluations", {}).keys()) == 0:
            for dataset_key, metric_results in self.last_evaluation_results.items():
                if metric_results is not None:
                    self.stat_aggregates["evaluations"].update(
                        {
                            "{}_{}".format(dataset_key, metric_key): dict(
                                runs=[], folds=[], reps=[], final=None
                            )
                            for metric_key in metric_results.keys()
                        }
                    )

        #################### Update Evaluations for Run ####################
        for agg_key, agg_val in self.stat_aggregates["evaluations"].items():
            agg_val["runs"].append(self.__loop_helper(agg_key))

        super().on_run_end()

    def on_fold_end(self):
        for agg_key, agg_val in self.stat_aggregates["evaluations"].items():
            agg_val["folds"].append(self.__loop_helper(agg_key))
        super().on_fold_end()

    def on_repetition_end(self):
        for agg_key, agg_val in self.stat_aggregates["evaluations"].items():
            agg_val["reps"].append(self.__loop_helper(agg_key))
        super().on_repetition_end()

    def on_experiment_end(self):
        for agg_key, agg_val in self.stat_aggregates["evaluations"].items():
            agg_val["final"] = self.__loop_helper(agg_key)

            #################### Reshape Aggregates to be of Proper Dimensions ####################
            runs_shape = (self._rep + 1, self._fold + 1, self._run + 1)
            agg_val["runs"] = np.reshape(agg_val["runs"], runs_shape).tolist()
            agg_val["folds"] = np.reshape(agg_val["folds"], runs_shape[:-1]).tolist()

        super().on_experiment_end()

    def __loop_helper(self, agg_key):
        # TODO: Add documentation
        for dataset_key, metric_results in self.last_evaluation_results.items():
            if metric_results is not None:
                for metric_key, metric_value in metric_results.items():
                    if agg_key == "{}_{}".format(dataset_key, metric_key):
                        return metric_value
        return None


class AggregatorOOF(BaseAggregatorCallback):
    pass  # TODO: Record "full_oof_predictions"


class AggregatorHoldout(BaseAggregatorCallback):
    pass  # TODO: Record "full_holdout_predictions"


class AggregatorTest(BaseAggregatorCallback):
    pass  # TODO: Record "full_test_predictions"


class AggregatorLosses(BaseAggregatorCallback):
    pass
