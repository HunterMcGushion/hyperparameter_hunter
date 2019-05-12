"""This module is central to the proper functioning of the entire library. It defines
:class:`Environment`, which (when activated) is used by the vast majority of the other
operation-critical modules in the library. :class:`Environment` can be viewed as a simple storage
container that defines settings that characterize the Experiments/OptimizationProtocols to be
conducted, and influence how those processes are carried out

Related
-------
:mod:`hyperparameter_hunter.settings`
    This module is the doorway for other modules to access the settings defined by
    :class:`environment.Environment`, which sets :attr:`hyperparameter_hunter.settings.G.Env` to
    itself as its first action. This allows other modules to access any information they need from
    the active :class:`environment.Environment` via :attr:`hyperparameter_hunter.settings.G.Env`.
    :class:`hyperparameter_hunter.settings.G` also provides other modules with access to the
    logging methods that are initialized by :class:`hyperparameter_hunter.environment.Environment`

Notes
-----
Despite the fact that :mod:`hyperparameter_hunter.settings` is the only module listed as being
"related", pretty much all the other modules in the library are related to
:class:`hyperparameter_hunter.environment.Environment` by way of this relation"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseCallback
from hyperparameter_hunter.metrics import format_metrics
from hyperparameter_hunter.sentinels import DatasetSentinel
from hyperparameter_hunter.settings import G, ASSETS_DIRNAME, RESULT_FILE_SUB_DIR_PATHS
from hyperparameter_hunter.reporting import ReportingHandler
from hyperparameter_hunter.keys.makers import CrossExperimentKeyMaker
from hyperparameter_hunter.utils.boltons_utils import remap
from hyperparameter_hunter.utils.file_utils import make_dirs, ParametersFromFile
from hyperparameter_hunter.utils.general_utils import Alias
from hyperparameter_hunter.utils.result_utils import format_predictions, default_do_full_save

##################################################
# Import Miscellaneous Assets
##################################################
from inspect import signature, isclass
import numpy as np
import os.path
import pandas as pd
from typing import List, Optional, Tuple, Union

##################################################
# Import Learning Assets
##################################################
# noinspection PyProtectedMember
from sklearn.model_selection import _split as sk_cv


class Environment:
    DEFAULT_PARAMS = dict(
        environment_params_path=None,
        results_path=None,
        target_column="target",
        id_column=None,
        do_predict_proba=False,
        prediction_formatter=format_predictions,
        metrics=None,
        metrics_params=dict(),
        cv_type="KFold",
        runs=1,
        global_random_seed=32,
        random_seeds=None,
        random_seed_bounds=[0, 100_000],
        cv_params=dict(),
        verbose=3,
        file_blacklist=None,
        reporting_params=dict(
            heartbeat_path=None, float_format="{:.5f}", console_params=None, heartbeat_params=None
        ),
        to_csv_params=dict(),
        do_full_save=default_do_full_save,
    )

    @ParametersFromFile(key="environment_params_path", verbose=True)
    @Alias("cv_type", ["cross_validation_type"])
    @Alias("cv_params", ["cross_validation_params"])
    @Alias("metrics", ["metrics_map"])
    @Alias("reporting_params", ["reporting_handler_params"])
    @Alias("results_path", ["root_results_path"])
    def __init__(
        self,
        train_dataset,  # TODO: Allow providing separate (train_input, train_target) dfs
        environment_params_path=None,
        *,
        results_path=None,
        metrics=None,
        holdout_dataset=None,  # TODO: Allow providing separate (holdout_input, holdout_target) dfs
        test_dataset=None,  # TODO: Allow providing separate (test_input, test_target) dfs
        target_column=None,
        id_column=None,
        do_predict_proba=None,
        prediction_formatter=None,
        metrics_params=None,
        cv_type=None,
        runs=None,
        global_random_seed=None,
        random_seeds=None,
        random_seed_bounds=None,
        cv_params=None,
        verbose=None,
        file_blacklist=None,
        reporting_params=None,
        to_csv_params=None,
        do_full_save=None,
        experiment_callbacks=None,
        experiment_recorders=None,
    ):
        """Class to organize the parameters that allow Experiments to be fairly compared

        Parameters
        ----------
        train_dataset: Pandas.DataFrame, or str path
            The training data for the experiment. Will be split into train/holdout data, if
            applicable, and train/validation data if cross-validation is to be performed. If str,
            will attempt to read file at path via :func:`pandas.read_csv`. For more information on
            which columns will be used during fitting/predicting, see the "Dataset columns" note
            in the "Notes" section below
        environment_params_path: String path, or None, default=None
            If not None and is valid .json filepath containing an object (dict), the file's contents
            are treated as the default values for all keys that match any of the below kwargs used
            to initialize :class:`Environment`
        results_path: String path, or None, default=None
            If valid directory path and the results directory has not yet been created, it will be
            created here. If this does not end with <ASSETS_DIRNAME>, it will be appended. If
            <ASSETS_DIRNAME> already exists at this path, new results will also be stored here. If
            None or invalid, results will not be stored
        metrics: Dict, List, or None, default=None
            Iterable describing the metrics to be recorded, along with a means to compute the value of
            each metric. Should be of one of the two following forms:

            List Form:

            * ["<metric name>", "<metric name>", ...]:
              Where each value is a string that names an attribute in :mod:`sklearn.metrics`
            * [`Metric`, `Metric`, ...]:
              Where each value of the list is an instance of :class:`metrics.Metric`
            * [(<name>, <metric_function>, [<direction>]), (<\*args>), ...]:
              Where each value of the list is a tuple of arguments that will be used to instantiate
              a :class:`metrics.Metric`. Arguments given in tuples must be in order expected by
              :class:`metrics.Metric`: (`name`, `metric_function`, `direction`)

            Dict Form:

            * {"<metric name>": <metric_function>, ...}:
              Where each key is a name for the corresponding metric callable, which is used to
              compute the value of the metric
            * {"<metric name>": (<metric_function>, <direction>), ...}:
              Where each key is a name for the corresponding metric callable and direction, all of
              which are used to instantiate a :class:`metrics.Metric`
            * {"<metric name>": "<sklearn metric name>", ...}:
              Where each key is a name for the metric, and each value is the name of the attribute
              in :mod:`sklearn.metrics` for which the corresponding key is an alias
            * {"<metric name>": None, ...}:
              Where each key is the name of the attribute in :mod:`sklearn.metrics`
            * {"<metric name>": `Metric`, ...}:
              Where each key names an instance of :class:`metrics.Metric`. This is the
              internally-used format to which all other formats will be converted

            Metric callable functions should expect inputs of form (target, prediction), and should
            return floats. See the documentation of :class:`metrics.Metric` for information
            regarding expected parameters and types
        holdout_dataset: Pandas.DataFrame, callable, str path, or None, default=None
            If pd.DataFrame, this is the holdout dataset. If callable, expects a function that takes
            (self.train: DataFrame, self.target_column: str) as input and returns the new
            (self.train: DataFrame, self.holdout: DataFrame). If str, will attempt to read file at
            path via :func:`pandas.read_csv`. Else, there is no holdout set. For more information on
            which columns will be used during fitting/predicting, see the "Dataset columns" note
            in the "Notes" section below
        test_dataset: Pandas.DataFrame, str path, or None, default=None
            The testing data for the experiment. Structure should be identical to that of
            `train_dataset`, except its `target_column` column can be empty or non-existent, because
            `test_dataset` predictions will never be evaluated. If str, will attempt to read file at
            path via :func:`pandas.read_csv`. For more information on which columns will be used
            during fitting/predicting, see the "Dataset columns" note in the "Notes" section below
        target_column: Str, or list, default='target'
            If str, denotes the column name in all provided datasets (except test) that contains the
            target output. If list, should be a list of strs designating multiple target columns.
            For example, in a multi-class classification dataset like UCI's hand-written digits,
            `target_column` would be a list containing ten strings. In this example, the
            `target_column` data would be sparse, with a 1 to signify that a sample is a written
            example of a digit (0-9). For a working example, see
            'hyperparameter_hunter/examples/lib_keras_multi_classification_example.py'
        id_column: Str, or None, default=None
            If not None, str denoting the column name in all provided datasets containing sample IDs
        do_predict_proba: Boolean, or int, default=False
            * If False, :meth:`.models.Model.fit` will call :meth:`models.Model.model.predict`
            * If True, it will call :meth:`models.Model.model.predict_proba`, and the values in all
              columns will be used as the actual prediction values
            * If `do_predict_proba` is an int, :meth:`.models.Model.fit` will call
              :meth:`models.Model.model.predict_proba`, as is the case when `do_predict_proba` is
              True, but the int supplied as `do_predict_proba` declares the column index to use as
              the actual prediction values
            * For example, for a model to call the `predict` method, `do_predict_proba=False`
              (default). For a model to call the `predict_proba` method, and use all of the class
              probabilities, `do_predict_proba=True`. To call the `predict_proba` method, and use
              the class probabilities in the first column, `do_predict_proba=0`. To use the second
              column (index 1) of the result, `do_predict_proba=1` - This often corresponds to the
              positive class's probabilities in binary classification problems. To use the third
              column `do_predict_proba=2`, and so on
        prediction_formatter: Callable, or None, default=None
            If callable, expected to have same signature as
            :func:`.utils.result_utils.format_predictions`. That is, the callable will receive
            (raw_predictions: np.array, dataset_df: pd.DataFrame, target_column: str,
            id_column: str or None) as input and should return a properly formatted prediction
            DataFrame. The callable uses raw_predictions as the content, dataset_df to provide any
            id column, and target_column to identify the column in which to place raw_predictions
        metrics_params: Dict, or None, default=dict()
            Dictionary of extra parameters to provide to :meth:`.metrics.ScoringMixIn.__init__`.
            `metrics` must be provided either 1) as an input kwarg to
            :meth:`Environment.__init__` (see `metrics`), or 2) as a key in `metrics_params`,
            but not both. An Exception will be raised if both are given, or if neither is given
        cv_type: Class or str, default='KFold'
            The class to define cross-validation splits. If str, it must be an attribute of
            `sklearn.model_selection._split`, and it must be a cross-validation class that inherits
            one of the following `sklearn` classes: `BaseCrossValidator`, or `_RepeatedSplits`.
            Valid str values include 'KFold', and 'RepeatedKFold', although there are many more. It
            must implement the following methods: [`__init__`, `split`]. If using a custom class,
            see the following tested `sklearn` classes for proper implementations:
            [`KFold`, `StratifiedKFold`, `RepeatedKFold`, `RepeatedStratifiedKFold`]. The arguments
            provided to :meth:`cv_type.__init__` will be :attr:`Environment.cv_params`, which should
            include the following: ['n_splits' <int>, 'n_repeats' <int> (if applicable)].
            :meth:`cv_type.split` will receive the following arguments:
            [:attr:`BaseExperiment.train_input_data`, :attr:`BaseExperiment.train_target_data`]
        runs: Int, default=1
            The number of times to fit a model within each fold to perform multiple-run-averaging
            with different random seeds
        global_random_seed: Int, default=32
            The initial random seed used just before generating an Experiment's random_seeds. This
            ensures consistency for `random_seeds` between Experiments, without having to explicitly
            provide it here
        random_seeds: None, or List, default=None
            If None, `random_seeds` of the appropriate shape will be created automatically. Else,
            must be a list of ints of shape (`cv_params['n_repeats']`, `cv_params['n_splits']`,
            `runs`). If `cv_params` does not have the key `n_repeats` (because standard
            cross-validation is being used), the value will default to 1. See
            :meth:`.experiments.BaseExperiment._random_seed_initializer` for info on expected shape
        random_seed_bounds: List, default=[0, 100000]
            A list containing two integers: the lower and upper bounds, respectively, for generating
            an Experiment's random seeds in
            :meth:`.experiments.BaseExperiment._random_seed_initializer`. Generally, leave this
            kwarg alone
        cv_params: dict, or None, default=dict()
            Parameters provided upon initialization of cv_type. Keys may be any args accepted by
            :meth:`cv_type.__init__`. Number of fold splits must be provided via "n_splits", and
            number of repeats (if applicable for `cv_type`) must be provided via "n_repeats"
        verbose: Int, boolean, default=3
            Verbosity of printing for any experiments performed while this Environment is active

            Higher values indicate more frequent logging. Logs are still recorded in the heartbeat
            file regardless of verbosity level. `verbose` only dictates which logs are visible in
            the console. The following table illustrates which types of logging messages will be
            visible with each verbosity level::

                | Verbosity | Keys/IDs | Final Score | Repetitions* | Folds | Runs* | Run Starts* | Result Files | Other |
                |:---------:|:--------:|:-----------:|:------------:|:-----:|:-----:|:-----------:|:------------:|:-----:|
                |     0     |          |             |              |       |       |             |              |       |
                |     1     |    Yes   |     Yes     |              |       |       |             |              |       |
                |     2     |    Yes   |     Yes     |      Yes     |  Yes  |       |             |              |       |
                |     3     |    Yes   |     Yes     |      Yes     |  Yes  |  Yes  |             |              |       |
                |     4     |    Yes   |     Yes     |      Yes     |  Yes  |  Yes  |     Yes     |      Yes     |  Yes  |

            *\*: If such logging is deemed appropriate with the given cross-validation parameters. In
            other words, repetition/run logging will only be verbose if Environment was given more
            than one repetition/run, respectively*
        file_blacklist: List of str, or None, or 'ALL', default=None
            If list of str, the result files named within are not saved to their respective
            directory in "<ASSETS_DIRNAME>/Experiments". If None, all result files are saved.
            If 'ALL', nothing at all will be saved for the Experiments. If the path of the file that
            initializes an Experiment does not end with a ".py" extension, the Experiment proceeds
            as if "script_backup" had been added to `file_blacklist`. This means that backup files
            will not be created for Jupyter notebooks (or any other non-".py" files). For info on
            acceptable values, see :func:`validate_file_blacklist`
        reporting_params: Dict, default=dict()
            Parameters passed to initialize :class:`.reporting.ReportingHandler`
        to_csv_params: Dict, default=dict()
            Parameters passed to the calls to :meth:`pandas.frame.DataFrame.to_csv` in
            :mod:`recorders`. In particular, this is where an Experiment's final prediction files
            are saved, so the values here will affect the format of the .csv prediction files.
            Warning: If `to_csv_params` contains the key "path_or_buf", it will be removed.
            Otherwise, all items are supplied directly to :meth:`to_csv`, including kwargs it might
            not be expecting if they are given
        do_full_save: None, or callable, default=:func:`utils.result_utils.default_do_full_save`
            If callable, expected to take an Experiment's result description dict as input and
            return a boolean. If None, treated as a callable that returns True. This parameter is
            used by :class:`recorders.DescriptionRecorder` to determine whether the Experiment
            result files following the description should also be created. If `do_full_save` returns
            False, result file-saving is stopped early, and only the description is saved. If
            `do_full_save` returns True, all files not in `file_blacklist` are saved normally. This
            allows you to skip creation of an Experiment's predictions, logs, and heartbeats if its
            score does not meet some threshold you set, for example. `do_full_save` receives the
            Experiment description dict as input, so for help setting `do_full_save`, just look into
            one of your Experiment descriptions
        experiment_callbacks: :class:`LambdaCallback`, list of :class:`LambdaCallback`, default=None
            If not None, should be a :class:`LambdaCallback` produced by
            :func:`.callbacks.bases.lambda_callback`, or a list of such classes. The contents will
            be added to the MRO of the executed Experiment class by
            :class:`.experiment_core.ExperimentMeta` at `__call__` time, making
            `experiment_callbacks` new base classes of the Experiment. See
            :func:`.callbacks.bases.lambda_callback` for more information
        experiment_recorders: List, None, default=None
            If not None, may be a list whose values are tuples of
            (<:class:`recorders.BaseRecorder` descendant>, <str result_path>). The result_path str
            should be a path relative to `results_path` that specifies the directory/file in
            which the product of the custom recorder should be saved. The contents of
            `experiment_recorders` will be provided to `recorders.RecorderList` upon completion of
            an Experiment, and, if the subclassing documentation in `recorders` is followed
            properly, will create or update a result file for the just-executed Experiment

        cross_validation_type: ...
            * Alias for `cv_type` *
        cross_validation_params: ...
            * Alias for `cv_params` *
        metrics_map: ...
            * Alias for `metrics` *
        reporting_handler_params: ...
            * Alias for `reporting_params` *
        root_results_path: ...
            * Alias for `results_path` *

        Notes
        -----
        Dataset columns: In order to specify the columns to be used by the three dataset kwargs
        (`train_dataset`, `holdout_dataset`, `test_dataset`) during fitting and predicting, a few
        attributes can be used. On `Environment` initialization, the columns specified by the
        following kwargs will be separated from the rest of the dataset during training/predicting:
        1) `target_column`, which names the column containing the target output labels for the input
        data; and 2) `id_column`, which (if given) represents the name of the column that contains
        identifying information for each data sample, and should otherwise have no relation to the
        actual data. Additionally, the `feature_selector` kwarg of the descendants of
        :class:`hyperparameter_hunter.experiments.BaseExperiment` (like
        :class:`hyperparameter_hunter.experiments.CVExperiment`) is used to filter out
        columns of the given datasets prior to fitting. See its documentation for more information,
        but it can effectively be used to remove any columns from the datasets

        Overriding default kwargs at `environment_params_path`: If you have any of the above kwargs
        specified in the .json file at environment_params_path (except environment_params_path,
        which will be ignored), you can override its value by passing it as a kwarg when
        initializing :class:`Environment`. The contents at environment_params_path are only used
        when the matching kwarg supplied at initialization is None. See
        "/examples/environment_params_path_example.py" for details

        The order of precedence for determining the value of each parameter is as follows, with
        items at the top having the highest priority, and deferring only to the items below if
        their own value is None:

        * 1)kwargs passed directly to :meth:`.Environment.__init__` on initialization,
        * 2)keys of the file at environment_params_path (if valid .json object),
        * 3)keys of :attr:`hyperparameter_hunter.environment.Environment.DEFAULT_PARAMS`

        do_predict_proba: Because this parameter can be either a boolean or an integer, it is
        important to explicitly pass booleans rather than truthy or falsey values. Similarly, only
        pass integers if you intend for the value to be used as a column index. Do not pass `0` to
        mean `False`, or `1` to mean `True`
        """
        G.Env = self
        self.environment_params_path = environment_params_path
        self.results_path = results_path

        #################### Attributes Used by Experiments ####################
        self.target_column = target_column
        self.id_column = id_column

        self.train_dataset = train_dataset
        self.holdout_dataset = holdout_dataset
        self.test_dataset = test_dataset

        self.do_predict_proba = do_predict_proba
        self.prediction_formatter = prediction_formatter
        self.metrics = metrics
        self.metrics_params = metrics_params

        self.cv_type = cv_type
        self.runs = runs
        self.global_random_seed = global_random_seed
        self.random_seeds = random_seeds
        self.random_seed_bounds = random_seed_bounds
        self.cv_params = cv_params

        #################### Ancillary Environment Settings ####################
        self.verbose = verbose
        self.file_blacklist = file_blacklist
        self.reporting_params = reporting_params or {}
        self.to_csv_params = to_csv_params or {}
        self.do_full_save = do_full_save
        self.experiment_callbacks = experiment_callbacks or []
        self.experiment_recorders = experiment_recorders or []

        self.result_paths = {
            "root": self.results_path,
            "checkpoint": None,
            "description": None,
            "heartbeat": None,
            "predictions_holdout": None,
            "predictions_in_fold": None,
            "predictions_oof": None,
            "predictions_test": None,
            "script_backup": None,
            "tested_keys": None,
            "key_attribute_lookup": None,
            "leaderboards": None,
            "global_leaderboard": None,
            "current_heartbeat": None,
        }
        self.current_task = None
        self.cross_experiment_key = None

        self.environment_workflow()

    def __repr__(self):
        return f"{self.__class__.__name__}(cross_experiment_key={self.cross_experiment_key!s})"

    def __eq__(self, other):
        return self.cross_experiment_key == other

    # def __enter__(self):
    #     pass

    # def __exit__(self):
    #     G.reset_attributes()

    ##################################################
    # Properties
    ##################################################
    #################### `results_path` ####################
    @property
    def results_path(self) -> Optional[str]:
        return self._results_path

    @results_path.setter
    def results_path(self, value):
        self._results_path = value
        if self._results_path is None:
            G.warn("Received results_path=None. Results will not be stored at all.")
        elif isinstance(self._results_path, str):
            if not self._results_path.endswith(ASSETS_DIRNAME):
                self._results_path = os.path.join(self._results_path, ASSETS_DIRNAME)
                # self.result_paths["root"] = self.results_path
            if not os.path.exists(self._results_path):
                make_dirs(self._results_path, exist_ok=True)
        else:
            raise TypeError(f"results_path must be None or str, not {value}")

    #################### `target_column` ####################
    @property
    def target_column(self) -> list:
        return self._target_column

    @target_column.setter
    def target_column(self, value):
        self._target_column = [value] if isinstance(value, str) else value

    #################### `train_dataset` ####################
    @property
    def train_dataset(self) -> pd.DataFrame:
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = pd.read_csv(value) if isinstance(value, str) else value

    #################### `test_dataset` ####################
    @property
    def test_dataset(self) -> Optional[pd.DataFrame]:
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = pd.read_csv(value) if isinstance(value, str) else value

    #################### `holdout_dataset` ####################
    @property
    def holdout_dataset(self) -> Optional[pd.DataFrame]:
        return self._holdout_dataset

    @holdout_dataset.setter
    def holdout_dataset(self, value):
        self._train_dataset, self._holdout_dataset = define_holdout_set(
            self.train_dataset, value, self.target_column
        )

    #################### `file_blacklist` ####################
    @property
    def file_blacklist(self) -> Union[list, str]:
        return self._file_blacklist

    @file_blacklist.setter
    def file_blacklist(self, value):
        self._file_blacklist = validate_file_blacklist(value)

        if self.results_path is None:
            self._file_blacklist = "ALL"

    #################### `metrics_params`/`metrics` ####################
    # TODO: Move their validations to properties here

    #################### `cv_type` ####################
    @property
    def cv_type(self) -> type:
        return self._cv_type

    @cv_type.setter
    def cv_type(self, value):
        if isinstance(value, str):
            try:
                self._cv_type = sk_cv.__getattribute__(value)
            except AttributeError:
                raise AttributeError(f"'{value}' not in `sklearn.model_selection._split`")
        else:  # Assumed to be a valid CV class
            self._cv_type = value

    #################### `to_csv_params` ####################
    @property
    def to_csv_params(self) -> dict:
        return self._to_csv_params

    @to_csv_params.setter
    def to_csv_params(self, value):
        self._to_csv_params = {k: v for k, v in value.items() if k != "path_or_buf"}

    #################### `cross_experiment_params` ####################
    @property
    def cross_experiment_params(self) -> dict:
        return dict(
            cv_type=self.cv_type,
            runs=self.runs,
            global_random_seed=self.global_random_seed,
            random_seeds=self.random_seeds,
            random_seed_bounds=self.random_seed_bounds,
        )

    #################### `experiment_callbacks` ####################
    @property
    def experiment_callbacks(self) -> list:
        return self._experiment_callbacks

    @experiment_callbacks.setter
    def experiment_callbacks(self, value):
        if not isinstance(value, list):
            self._experiment_callbacks = [value]
        else:
            self._experiment_callbacks = value
        for cb in self._experiment_callbacks:
            if not isclass(cb):
                raise TypeError(f"experiment_callbacks must be classes, not {type(cb)}: {cb}")
            if issubclass(cb, BaseCallback):
                continue
            if cb.__name__ != "LambdaCallback":
                raise ValueError(f"experiment_callbacks must be LambdaCallback instances, not {cb}")

    ##################################################
    # Core Methods
    ##################################################
    def environment_workflow(self):
        """Execute all methods required to validate the environment and run Experiments"""
        self.update_custom_environment_params()
        self.validate_parameters()
        self.format_result_paths()
        self.generate_cross_experiment_key()
        G.log("Cross-Experiment Key:   '{!s}'".format(self.cross_experiment_key))

    def validate_parameters(self):
        """Ensure the provided parameters are valid and properly formatted"""
        #################### metrics_params/metrics ####################
        if (self.metrics is not None) and ("metrics" in self.metrics_params.keys()):
            raise ValueError(
                "`metrics` may be provided as a kwarg, or as a `metrics_params` key, but NOT BOTH. Received: "
                + f"\n `metrics`={self.metrics}\n `metrics_params`={self.metrics_params}"
            )
        else:
            _metrics_alias = "metrics"
            if self.metrics is None:
                try:
                    self.metrics = self.metrics_params["metrics"]
                except KeyError:
                    self.metrics = self.metrics_params["metrics_map"]
                    _metrics_alias = "metrics_map"
            self.metrics = format_metrics(self.metrics)
            self.metrics_params = {**{_metrics_alias: self.metrics}, **self.metrics_params}

    def format_result_paths(self):
        """Remove paths contained in file_blacklist, and format others to prepare for saving results"""
        if self.file_blacklist == "ALL" or self.results_path is None:
            return

        # Blacklist the prediction files for any datasets that were not given
        if self.holdout_dataset is None:
            self.file_blacklist.append("predictions_holdout")
        if self.test_dataset is None:
            self.file_blacklist.append("predictions_test")

        # Add given `experiment_recorders` to `result_paths`
        for recorder in self.experiment_recorders:
            try:
                recorder, result_path = recorder
            except IndexError:
                raise IndexError(f"Expected `recorder` to be tuple of (class, str), not {recorder}")

            self.result_paths[recorder.result_path_key] = result_path

        # Set full filepath for result files relative to `results_path`, or to None (blacklist)
        for k in self.result_paths.keys():
            if k == "root":
                continue
            elif k not in self.file_blacklist:
                # If `k` not in `RESULT_FILE_SUB_DIR_PATHS`, then added via `experiment_recorders`
                self.result_paths[k] = os.path.join(
                    self.results_path, RESULT_FILE_SUB_DIR_PATHS.get(k, self.result_paths[k])
                )
            else:
                self.result_paths[k] = None
                # G.debug('Result file "{}" has been blacklisted'.format(k))

    def update_custom_environment_params(self):
        """Try to update null parameters from environment_params_path, or DEFAULT_PARAMS"""
        allowed_parameter_keys = [
            k for k, v in signature(Environment).parameters.items() if v.kind == v.KEYWORD_ONLY
        ]

        for k in allowed_parameter_keys:
            if getattr(self, k) is None:
                setattr(self, k, self.DEFAULT_PARAMS.get(k, None))

    def generate_cross_experiment_key(self):
        """Generate a key to describe the current Environment's cross-experiment parameters"""
        parameters = dict(
            metrics_params=self.metrics_params,
            cv_params=self.cv_params,
            target_column=self.target_column,
            id_column=self.id_column,
            do_predict_proba=self.do_predict_proba,
            prediction_formatter=self.prediction_formatter,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            holdout_dataset=self.holdout_dataset,
            cross_experiment_params=self.cross_experiment_params.copy(),
            to_csv_params=self.to_csv_params,
        )

        #################### Revert Aliases for Compatibility ####################
        # If any aliases were used during call to `Environment.__init__`, replace the default names
        # in `parameters` with the alias used. This ensures compatibility with Environment keys
        # made in earlier versions
        aliases_used = getattr(self, "__hh_aliases_used", {})

        # noinspection PyUnusedLocal
        def _visit(path, key, value):
            if key in aliases_used:
                key = aliases_used.pop(key)
            return (key, value)

        if aliases_used:
            parameters = remap(parameters, visit=_visit)

        #################### Make `cross_experiment_key` ####################
        self.cross_experiment_key = CrossExperimentKeyMaker(parameters)

    def initialize_reporting(self):
        """Initialize reporting for the Environment and Experiments conducted during its lifetime"""
        reporting_params = self.reporting_params
        reporting_params["heartbeat_path"] = self.result_paths["current_heartbeat"]
        reporting_handler = ReportingHandler(**reporting_params)

        #################### Make Unified Logging Globally Available ####################
        G.log = reporting_handler.log
        G.debug = reporting_handler.debug
        G.warn = reporting_handler.warn

    # @property
    # def initialized_model(self):
    #     """Sentinel for use with meta-estimators, such as those in SKLearn's `multioutput` module
    #
    #     When declaring `model_init_params` for the meta-estimator during `CVExperiment`
    #     or optimization protocol setup, provide this property as input to the meta-estimator's
    #     `estimator`/`base_estimator` parameter.
    #
    #     This property is actually a placeholder for the initialized model created by whatever model
    #     is at the index following the current index. For example, assuming a properly initialized
    #     `Environment`, to use a Support Vector Regression in a multi-regression problem...
    #     >>> from hyperparameter_hunter import CVExperiment
    #     >>> from sklearn.multioutput import MultiOutputRegressor
    #     >>> from sklearn.svm import SVR
    #     >>> env = Environment(...)
    #     >>> experiment = CVExperiment(
    #     ...     model_initializer=(MultiOutputRegressor, SVR),
    #     ...     model_init_params=(  # Dict of parameters for each `model_initializer`
    #     ...         dict(estimator=env.initialized_model),  # References model following current model (`SVR` at index 1)
    #     ...         dict(kernel="linear", C=10.0),
    #     ...     )
    #     ... )
    #     What happens behind the scenes is that because the first set of `model_init_params` contains
    #     the sentinel for another initialized model, its initialization is delayed until the others
    #     have been initialized.
    #     So the `SVR` is initialized with `dict(kernel="linear", C=10.0)`. Then the initialized SVR
    #     is provided as input to the model that preceded it and declared the initialized SVR as
    #     input: the `MultiOutputRegressor`. The end result is a model that can be parameterized by
    #     HyperparameterHunter that mirrors the following:
    #     >>> MultiOutputRegressor(estimator=SVR(kernel="linear", C=10.0))
    #     """

    ##################################################
    # Dataset Sentinels for Use as Extra Parameters
    ##################################################
    @property
    def train_input(self):
        """Get a `DatasetSentinel` representing an Experiment's `fold_train_input`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.fold_train_input` upon
            `Model` initialization"""
        return DatasetSentinel("train_input", **self._dataset_sentinel_helper())

    @property
    def train_target(self):
        """Get a `DatasetSentinel` representing an Experiment's `fold_train_target`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.fold_train_target` upon
            `Model` initialization"""
        return DatasetSentinel("train_target", **self._dataset_sentinel_helper())

    @property
    def validation_input(self):
        """Get a `DatasetSentinel` representing an Experiment's `fold_validation_input`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.fold_validation_input`
            upon `Model` initialization"""
        return DatasetSentinel("oof_input", **self._dataset_sentinel_helper())

    @property
    def validation_target(self):
        """Get a `DatasetSentinel` representing an Experiment's `fold_validation_target`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.fold_validation_target`
            upon `Model` initialization"""
        return DatasetSentinel("oof_target", **self._dataset_sentinel_helper())

    @property
    def holdout_input(self):
        """Get a `DatasetSentinel` representing an Experiment's `holdout_input_data`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.holdout_input_data`
            upon `Model` initialization"""
        return DatasetSentinel(
            "holdout_input", self.cross_experiment_key.parameters["holdout_dataset"]
        )

    @property
    def holdout_target(self):
        """Get a `DatasetSentinel` representing an Experiment's `holdout_target_data`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.holdout_target_data`
            upon `Model` initialization"""
        return DatasetSentinel(
            "holdout_target", self.cross_experiment_key.parameters["holdout_dataset"]
        )

    def _dataset_sentinel_helper(self):
        """Helper method for retrieving train/validation sentinel parameters"""
        params = self.cross_experiment_key.parameters
        return dict(
            dataset_hash=params["train_dataset"],
            cv_type=params["cross_experiment_params"].get(
                "cv_type", params["cross_experiment_params"].get("cross_validation_type", None)
            ),
            global_random_seed=params["cross_experiment_params"]["global_random_seed"],
            random_seeds=params["cross_experiment_params"]["random_seeds"],
        )


def define_holdout_set(
    train_set: pd.DataFrame,
    holdout_set: Union[pd.DataFrame, callable, str, None],
    target_column: Union[str, List[str]],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Create `holdout_set` (if necessary) by loading a DataFrame from a .csv file, or by separating
    `train_set`, and return the updated (`train_set`, `holdout_set`) pair

    Parameters
    ----------
    train_set: Pandas.DataFrame
        Training DataFrame. Will be split into train/holdout data, if `holdout_set` is callable
    holdout_set: Pandas.DataFrame, callable, str, or None
        If pd.DataFrame, this is the holdout dataset. If callable, expects a function that takes
        (`train_set`, `target_column`) as input and returns the new (`train_set`, `holdout_set`). If
        str, will attempt to read file at path via :func:`pandas.read_csv`. Else, no holdout set
    target_column: Str, or list
        If str, denotes the column name in provided datasets that contains the target output. If
        list, should be a list of strs designating multiple target columns

    Returns
    -------
    train_set: Pandas.DataFrame
        `train_set` if `holdout_set` is not callable. Else `train_set` modified by `holdout_set`
    holdout_set: Pandas.DataFrame, or None
        Original DataFrame, or DataFrame read from str filepath, or a portion of `train_set` if
        `holdout_set` is callable, or None"""
    #################### Update `holdout_set` ####################
    if callable(holdout_set):
        train_set, holdout_set = holdout_set(train_set, target_column)
    elif isinstance(holdout_set, str):
        holdout_set = pd.read_csv(holdout_set)
    #################### Validate `holdout_set` ####################
    try:
        if holdout_set is None or np.array_equal(train_set.columns, holdout_set.columns):
            return train_set, holdout_set
    except AttributeError:
        raise TypeError(f"holdout_set must be None, DataFrame, callable, or str, not {holdout_set}")
    raise ValueError(f"Mismatched columns\n{train_set.columns}\n!=\n{holdout_set.columns}")


##################################################
# File Blacklist Utilities
##################################################
def validate_file_blacklist(blacklist):
    """Validate contents of blacklist. For most values, the corresponding file is saved upon
    completion of the experiment. See the "Notes" section below for details on some special cases

    Parameters
    ----------
    blacklist: List of strings, or None
        The result files that should not be saved

    Returns
    -------
    blacklist: List
        If not empty, acceptable list of result file types to blacklist

    Notes
    -----
    'heartbeat': If the heartbeat file is saved, a new file is not generated and saved to the
    "Experiments/Heartbeats" directory as is the case with most other files. Instead, the general
    "Heartbeat.log" file is copied and renamed to the current experiment id, then saved to the
    appropriate dir. This is because the general "Heartbeat.log" file represents the heartbeat
    for whatever experiment is currently in progress.

    'script_backup': This file is saved as quickly as possible after starting a new experiment,
    rather than waiting for the experiment to end. There are two reasons for this behavior: 1) to
    avoid saving any changes that may have been made to a file after it has been executed, and 2)
    to have the offending file in the event of a catastrophic failure that results in no other
    files being saved. As stated in the documentation of the `file_blacklist` parameter of
    `Environment`, if the path of the file that initializes an Experiment does not end with a ".py"
    extension, the Experiment proceeds as if "script_backup" had been added to `blacklist`. This
    means that backup files will not be created for Jupyter notebooks (or any other non-".py" files)

    'description' and 'tested_keys': These two results types constitute a bare minimum of sorts for
    experiment recording. If either of these two are blacklisted, then as far as the library is
    concerned, the experiment never took place.

    'tested_keys' (continued): If this string is included in the blacklist, then the contents of the
    "KeyAttributeLookup" directory will also be excluded from the list of files to update

    'current_heartbeat': The general heartbeat file that should be stored at
    'HyperparameterHunterAssets/Heartbeat.log'. If this value is blacklisted, then 'heartbeat' is
    also added to `blacklist` automatically out of necessity. This is done because the heartbeat
    file for the current experiment cannot be created as a copy of the general heartbeat file if the
    general heartbeat file is never created in the first place"""
    valid_values = [
        # 'checkpoint',
        "description",
        "heartbeat",
        "predictions_holdout",
        "predictions_in_fold",
        "predictions_oof",
        "predictions_test",
        "script_backup",
        "tested_keys",
        "current_heartbeat",
    ]
    if blacklist == "ALL":
        G.warn('WARNING: Received `blacklist`="ALL". Nothing will be saved')
        return blacklist

    if not blacklist:
        return []
    elif not isinstance(blacklist, list):
        raise TypeError("Expected blacklist to be a list, not: {}".format(blacklist))
    elif not all([isinstance(_, str) for _ in blacklist]):
        invalid_files = [(type(_).__name__, _) for _ in blacklist if not isinstance(_, str)]
        raise TypeError("Expected blacklist contents to be strings, not: {}".format(invalid_files))

    for a_file in blacklist:
        if a_file not in valid_values:
            raise ValueError(f"Invalid blacklist value: {a_file}.\nExpected one of: {valid_values}")
        if a_file in ["description", "tested_keys"]:
            G.warn(f"Including {a_file!r} in blacklist will severely impede library functionality")

    # Blacklist experiment-specific heartbeat if general (current) heartbeat is blacklisted
    if ("current_heartbeat" in blacklist) and ("heartbeat" not in blacklist):
        blacklist.append("heartbeat")

    return blacklist
