"""This module handles recording and properly formatting the various result files requested for a
completed Experiment. Coincidentally, if a particular result file was blacklisted by the active
Environment, that is also handled here

Related
-------
:mod:`hyperparameter_hunter.experiments`
    This is the intended user of the contents of :mod:`hyperparameter_hunter.recorders`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import EnvironmentInactiveError, EnvironmentInvalidError
from hyperparameter_hunter.leaderboards import GlobalLeaderboard
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.file_utils import write_json, add_to_json

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import os
from platform import node
import shutil
from sys import exc_info


class BaseRecorder(metaclass=ABCMeta):
    def __init__(self):
        """Base class for other classes that record various Experiment result files. Critical
        attributes of the descendants of :class`recorders.BaseRecorder` are set here, enabling them
        to function properly

        Returns
        -------
        None
            If :attr:`result_path` is None, which means the present result file was blacklisted by
            the active Environment

        Raises
        ------
        EnvironmentInactiveError
            If :attr:`settings.G.Env` is None
        EnvironmentInvalidError
            If any of the following occur: 1) :attr:`settings.G.Env` does not have an attribute
            named 'result_paths', 2) :attr:`settings.G.Env.result_paths` does not contain the
            current `result_path_key`, 3) :attr:`settings.G.Env.current_task` is None"""
        self.result_path = None
        self.result = None

        ##################################################
        # Get Result Path for Record, or Exit Early
        ##################################################
        try:
            self.result_path = G.Env.result_paths[self.result_path_key]
        except AttributeError as _ex:
            if G.Env is None:
                raise EnvironmentInactiveError(str(_ex)).with_traceback(exc_info()[2])
            if not hasattr(G.Env, "result_paths"):
                _err_message = f"{_ex!s}\nG.Env missing 'result_paths' attr"
                raise EnvironmentInvalidError(_err_message).with_traceback(exc_info()[2])
        except KeyError as _ex:
            _err_message = f"{_ex!s}\nG.Env.result_paths missing the key: '{self.result_path_key}'"
            raise EnvironmentInvalidError(_err_message).with_traceback(exc_info()[2])

        if self.result_path is None:
            return  # Result file blacklisted and should not be recorded. Kill recording process now

        ##################################################
        # Gather Attributes Required for Record
        ##################################################
        for required_attribute in self.required_attributes:
            try:
                setattr(self, required_attribute, getattr(G.Env.current_task, required_attribute))
            except AttributeError as _ex:
                if G.Env.current_task is None:
                    _err_message = f"{_ex!s}\nNo active experiment found"
                    raise EnvironmentInvalidError(_err_message).with_traceback(exc_info()[2])
                raise EnvironmentInvalidError(str(_ex)).with_traceback(exc_info()[2])

    @property
    @abstractmethod
    def result_path_key(self) -> str:
        """Return key from :attr:`environment.Environment.result_paths`, corresponding to the
        target record"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def required_attributes(self) -> list:
        """Return attributes from :class:`environment.Environment` that are necessary to properly
        record result"""
        raise NotImplementedError()

    @abstractmethod
    def format_result(self):
        """Set :attr:`BaseRecorder.result` to the final result object to be saved by
        :meth:`BaseRecorder.save_result`"""
        raise NotImplementedError()

    @abstractmethod
    def save_result(self):
        """Save :attr:`BaseRecorder.result` to :attr:`BaseRecorder.result_path`, or elsewhere if
        special case"""
        raise NotImplementedError()


class RecorderList(object):
    def __init__(self, file_blacklist=None):
        """Collection of :class:`BaseRecorder` subclasses to facilitate executing group methods

        Parameters
        ----------
        file_blacklist: List, or None, default=None
            If list, used to reject any elements of :attr:`RecorderList.recorders` whose
            :attr:`BaseRecorder.result_path_key` is in file_blacklist"""
        # WARNING: Take care if modifying the order/contents of :attr:`recorders`. See :meth:`save_result` documentation for info
        self.recorders = [
            TestedKeyRecorder,
            LeaderboardEntryRecorder,
            DescriptionRecorder,
            # PredictionsInFoldRecorder,
            PredictionsOOFRecorder,
            PredictionsHoldoutRecorder,
            PredictionsTestRecorder,
            HeartbeatRecorder,
        ]

        if file_blacklist is not None:
            if file_blacklist == "ALL":
                self.recorders = []
            self.recorders = [_ for _ in self.recorders if _.result_path_key not in file_blacklist]

        self.recorders = [_() for _ in self.recorders]

    def format_result(self):
        """Execute :meth:`format_result` for all classes in :attr:`recorders`"""
        for recorder in self.recorders:
            recorder.format_result()

    def save_result(self):
        """Execute :meth:`save_result` for all classes in :attr:`recorders`

        Notes
        -----
        When iterating through :attr:`recorders` and calling :meth:`save_result`, a check is
        performed for `exit_code`. Children classes of :class:`BaseRecorder` are NOT expected to
        explicitly return a value in their :meth:`save_result`. However, if a value is returned and
        `exit_code` == 'break', the result-saving loop will be broken, and no further results will
        be saved. In practice, this is only performed for the sake of
        :meth:`DescriptionRecorder.save_result`, which has the additional quality of being able to
        prevent any other result files from being saved if the result of
        :func:`DescriptionRecorder.do_full_save` returns False when given the formatted
        :attr:`DescriptionRecorder.result`. This can be useful when there are storage constraints,
        because it ensures that essential data - including keys and the results of the experiment -
        are saved (to ensure the experiment is not duplicated, and to enable optimization protocol
        learning), while extra results like Predictions are not saved"""
        for recorder in self.recorders:
            G.log(f'Saving result file for "{type(recorder).__name__}"')
            exit_code = recorder.save_result()

            if exit_code and exit_code == "break":
                break


##################################################
# Description
##################################################
class DescriptionRecorder(BaseRecorder):
    result_path_key = "description"
    required_attributes = [
        "experiment_id",
        "hyperparameter_key",
        "cross_experiment_key",
        "last_evaluation_results",
        "stat_aggregates",
        # 'train_features',
        "source_script",
        "notes",
        "model_initializer",
        "do_full_save",
        "model",
        "algorithm_name",
        "module_name",
    ]

    def format_result(self):
        """Format an OrderedDict containing the Experiment's identifying attributes, results,
        hyperparameters used, and other stats or information that may be useful"""
        self.result = OrderedDict(
            [
                ("experiment_id", self.experiment_id),
                ("algorithm_name", self.algorithm_name),
                ("module_name", self.module_name),
                ("hyperparameter_key", self.hyperparameter_key.key),
                ("cross_experiment_key", self.cross_experiment_key.key),
                ("final_evaluations", self.last_evaluation_results),
                ("hyperparameters", self.hyperparameter_key.parameters),
                ("cross_experiment_parameters", self.cross_experiment_key.parameters),
                ("train_features", None),  # TODO: Record the column features in train df
                ("platform", node()),
                ("source_script", self.source_script),
                ("notes", self.notes or ""),
                ("aggregates", self.stat_aggregates),
            ]
        )

        #################### Filter Hyperparameters' model_init_params ####################
        bad_keys = {"random_state", "seed"}

        self.result["hyperparameters"]["model_init_params"] = {
            _k: _v
            for _k, _v in self.result["hyperparameters"]["model_init_params"].items()
            if _k not in bad_keys
        }

    def save_result(self):
        """Save the Experiment description as a .json file, named after :attr:`experiment_id`. If
        :attr:`do_full_save` is a callable and returns False when given the description object, the
        result recording loop will be broken, and the remaining result files will not be saved

        Returns
        -------
        'break'
            This string will be returned if :attr:`do_full_save` is a callable and returns False
            when given the description object. This is the signal for
            :class:`recorders.RecorderList` to stop recording result files"""
        try:
            write_json(f"{self.result_path}/{self.experiment_id}.json", self.result, do_clear=False)
        except FileNotFoundError:
            os.makedirs(self.result_path, exist_ok=False)
            write_json(f"{self.result_path}/{self.experiment_id}.json", self.result, do_clear=False)

        if (self.do_full_save is not None) and (not self.do_full_save(self.result)):
            G.warn("Breaking result-saving loop early! Remaining result files will not be saved")
            return "break"


##################################################
# Heartbeat
##################################################
class HeartbeatRecorder(BaseRecorder):
    result_path_key = "heartbeat"
    required_attributes = ["experiment_id"]

    def format_result(self):
        """Do nothing"""
        pass

    def save_result(self):
        """Copy global Heartbeat log to results dir as .log file named for :attr:`experiment_id`"""
        try:
            self._copy_heartbeat()
        except FileNotFoundError:
            os.makedirs(self.result_path, exist_ok=False)
            self._copy_heartbeat()

    def _copy_heartbeat(self):
        """Helper method to copy the global Heartbeat log to a file named for the Experiment"""
        shutil.copyfile(
            f"{G.Env.result_paths['root']}/Heartbeat.log",
            f"{self.result_path}/{self.experiment_id}.log",
        )


##################################################
# Predictions
##################################################
prediction_requirements = [
    "experiment_id",
    "prediction_formatter",
    "target_column",
    "id_column",
    "to_csv_params",
]


class PredictionsHoldoutRecorder(BaseRecorder):
    result_path_key = "predictions_holdout"
    required_attributes = ["final_holdout_predictions", "holdout_dataset"] + prediction_requirements

    def format_result(self):
        """Format predictions according to the callable :attr:`prediction_formatter`"""
        self.result = self.prediction_formatter(
            self.final_holdout_predictions, self.holdout_dataset, self.target_column, self.id_column
        )

    def save_result(self):
        """Save holdout predictions to a .csv file, named after :attr:`experiment_id`"""
        try:
            self.result.to_csv(f"{self.result_path}/{self.experiment_id}.csv", **self.to_csv_params)
        except FileNotFoundError:
            os.makedirs(self.result_path, exist_ok=False)
            self.result.to_csv(f"{self.result_path}/{self.experiment_id}.csv", **self.to_csv_params)


class PredictionsOOFRecorder(BaseRecorder):
    result_path_key = "predictions_oof"
    required_attributes = ["final_oof_predictions", "train_dataset"] + prediction_requirements

    def format_result(self):
        """Format predictions according to the callable :attr:`prediction_formatter`"""
        self.result = self.prediction_formatter(
            self.final_oof_predictions, self.train_dataset, self.target_column, self.id_column
        )

    def save_result(self):
        """Save out-of-fold predictions to a .csv file, named after :attr:`experiment_id`"""
        try:
            self.result.to_csv(f"{self.result_path}/{self.experiment_id}.csv", **self.to_csv_params)
        except FileNotFoundError:
            os.makedirs(self.result_path, exist_ok=False)
            self.result.to_csv(f"{self.result_path}/{self.experiment_id}.csv", **self.to_csv_params)


class PredictionsTestRecorder(BaseRecorder):
    result_path_key = "predictions_test"
    required_attributes = ["final_test_predictions", "test_dataset"] + prediction_requirements

    def format_result(self):
        """Format predictions according to the callable :attr:`prediction_formatter`"""
        self.result = self.prediction_formatter(
            self.final_test_predictions, self.test_dataset, self.target_column, self.id_column
        )

    def save_result(self):
        """Save test predictions to a .csv file, named after :attr:`experiment_id`"""
        try:
            self.result.to_csv(f"{self.result_path}/{self.experiment_id}.csv", **self.to_csv_params)
        except FileNotFoundError:
            os.makedirs(self.result_path, exist_ok=False)
            self.result.to_csv(f"{self.result_path}/{self.experiment_id}.csv", **self.to_csv_params)


# class PredictionsInFoldRecorder(BaseRecorder):
#     result_path_key = 'predictions_in_fold'
#     required_attributes = ['final_in_fold_predictions', 'train_dataset'] + prediction_requirements


##################################################
# Keys (Cross-Experiment, Hyperparameter), and IDs
##################################################
class TestedKeyRecorder(BaseRecorder):
    result_path_key = "tested_keys"
    required_attributes = ["experiment_id", "hyperparameter_key", "cross_experiment_key"]

    def format_result(self):
        """Do nothing"""
        pass

    def save_result(self):
        """Save cross-experiment, and hyperparameter keys, and update their tested keys entries"""
        self.cross_experiment_key.save_key()
        self.hyperparameter_key.save_key()
        add_to_json(
            file_path=f"{self.hyperparameter_key.tested_keys_dir}/{self.cross_experiment_key.key}.json",
            data_to_add=self.experiment_id,
            key=self.hyperparameter_key.key,
            condition=lambda _: self.hyperparameter_key.key in _.keys(),
            append_value=True,
        )


##################################################
# Leaderboard
##################################################
class LeaderboardEntryRecorder(BaseRecorder):
    result_path_key = "tested_keys"
    required_attributes = ["result_paths", "current_task", "target_metric", "metrics_map"]

    def format_result(self):
        """Read existing global leaderboard, add current entry, then sort the updated leaderboard"""
        self.result = GlobalLeaderboard.from_path(path=self.result_paths["global_leaderboard"])
        self.result.add_entry(self.current_task)
        self.result.sort(
            by=list(self.result.data.columns),
            ascending=(self.metrics_map[self.target_metric[-1]].direction == "min"),
        )

    def save_result(self):
        """Save the updated leaderboard file"""
        try:
            self.result.save(path=self.result_paths["global_leaderboard"])
        except FileNotFoundError:
            os.makedirs(self.result_paths["leaderboards"], exist_ok=False)
            self.result.save(path=self.result_paths["global_leaderboard"])


if __name__ == "__main__":
    pass
