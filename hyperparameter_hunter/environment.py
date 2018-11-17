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
from hyperparameter_hunter.metrics import format_metrics_map
from hyperparameter_hunter.sentinels import DatasetSentinel
from hyperparameter_hunter.settings import G, ASSETS_DIRNAME, RESULT_FILE_SUB_DIR_PATHS
from hyperparameter_hunter.reporting import ReportingHandler
from hyperparameter_hunter.key_handler import CrossExperimentKeyMaker
from hyperparameter_hunter.utils.file_utils import read_json
from hyperparameter_hunter.utils.general_utils import type_val
from hyperparameter_hunter.utils.result_utils import format_predictions, default_do_full_save

##################################################
# Import Miscellaneous Assets
##################################################
from inspect import signature, isclass
import numpy as np
import os
import os.path
import pandas as pd

##################################################
# Import Learning Assets
##################################################
# noinspection PyProtectedMember
from sklearn.model_selection import _split as sk_cv


class Environment:
    DEFAULT_PARAMS = dict(
        environment_params_path=None,
        root_results_path=None,
        target_column="target",
        id_column=None,
        do_predict_proba=False,
        prediction_formatter=format_predictions,
        metrics_map=None,
        metrics_params=dict(),
        cross_validation_type="KFold",
        runs=1,
        global_random_seed=32,
        random_seeds=None,
        random_seed_bounds=[0, 100000],
        cross_validation_params=dict(),
        verbose=True,
        file_blacklist=None,
        reporting_handler_params=dict(
            # reporting_type='logging',
            heartbeat_path=None,
            float_format="{:.5f}",
            console_params=None,
            heartbeat_params=None,
        ),
        to_csv_params=dict(),
        do_full_save=default_do_full_save,
    )

    def __init__(
        self,
        train_dataset,  # TODO: Allow providing separate (train_input, train_target) dfs
        environment_params_path=None,
        *,
        root_results_path=None,
        metrics_map=None,
        holdout_dataset=None,  # TODO: Allow providing separate (holdout_input, holdout_target) dfs
        test_dataset=None,  # TODO: Allow providing separate (test_input, test_target) dfs
        target_column=None,
        id_column=None,
        do_predict_proba=None,
        prediction_formatter=None,
        metrics_params=None,
        cross_validation_type=None,
        runs=None,
        global_random_seed=None,
        random_seeds=None,
        random_seed_bounds=None,
        cross_validation_params=None,
        verbose=None,
        file_blacklist=None,
        reporting_handler_params=None,
        to_csv_params=None,
        do_full_save=None,
        experiment_callbacks=None,
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
        root_results_path: String path, or None, default=None
            If valid directory path and the results directory has not yet been created, it will be
            created here. If this does not end with <ASSETS_DIRNAME>, it will be appended. If
            <ASSETS_DIRNAME> already exists at this path, new results will also be stored here. If
            None or invalid, results will not be stored
        metrics_map: Dict, List, or None, default=None
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
            * If True, it will call :meth:`models.Model.model.predict_proba`, and the values in the
              first column (index 0) will be used as the actual prediction values
            * If `do_predict_proba` is an int, :meth:`.models.Model.fit` will call
              :meth:`models.Model.model.predict_proba`, as is the case when `do_predict_proba` is
              True, but the int supplied as `do_predict_proba` declares the column index to use as
              the actual prediction values
            * For example, for a model to call the `predict` method, `do_predict_proba=False`
              (default). For a model to call the `predict_proba` method, and use the class
              probabilities in the first column, `do_predict_proba=0` or `do_predict_proba=True`. To
              use the second column (index 1) of the result, `do_predict_proba=1` - This
              often corresponds to the positive class's probabilities in binary classification
              problems. To use the third column `do_predict_proba=2`, and so on
        prediction_formatter: Callable, or None, default=None
            If callable, expected to have same signature as
            :func:`.utils.result_utils.format_predictions`. That is, the callable will receive
            (raw_predictions: np.array, dataset_df: pd.DataFrame, target_column: str,
            id_column: str or None) as input and should return a properly formatted prediction
            DataFrame. The callable uses raw_predictions as the content, dataset_df to provide any
            id column, and target_column to identify the column in which to place raw_predictions
        metrics_params: Dict, or None, default=dict()
            Dictionary of extra parameters to provide to :meth:`.metrics.ScoringMixIn.__init__`.
            `metrics_map` must be provided either 1) as an input kwarg to
            :meth:`Environment.__init__` (see `metrics_map`), or 2) as a key in `metrics_params`,
            but not both. An Exception will be raised if both are given, or if neither is given
        cross_validation_type: Class or str, default='KFold'
            The class to define cross-validation splits. If str, it must be an attribute of
            `sklearn.model_selection._split`, and it must be a cross-validation class that inherits
            one of the following `sklearn` classes: `BaseCrossValidator`, or `_RepeatedSplits`.
            Valid str values include 'KFold', and 'RepeatedKFold', although there are many more. It
            must implement the following methods: [`__init__`, `split`]. If using a custom class,
            see the following tested `sklearn` classes for proper implementations:
            [`KFold`, `StratifiedKFold`, `RepeatedKFold`, `RepeatedStratifiedKFold`]. The arguments
            provided to :meth:`cross_validation_type.__init__` will be
            :attr:`Environment.cross_validation_params`, which should include the following:
            ['n_splits' <int>, 'n_repeats' <int> (if applicable)].
            :meth:`cross_validation_type.split` will receive the following arguments:
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
            must be a list of ints of shape (`cross_validation_params['n_repeats']`,
            `cross_validation_params['n_splits']`, `runs`). If `cross_validation_params` does not
            have the key `n_repeats` (because standard cross-validation is being used), the value
            will default to 1. See :meth:`.experiments.BaseExperiment._random_seed_initializer` for
            more info on the expected shape
        random_seed_bounds: List, default=[0, 100000]
            A list containing two integers: the lower and upper bounds, respectively, for generating
            an Experiment's random seeds in
            :meth:`.experiments.BaseExperiment._random_seed_initializer`. Generally, leave this
            kwarg alone
        cross_validation_params: dict, or None, default=dict()
            Dict of parameters provided upon initialization of cross_validation_type. Keys may be
            any args accepted by :meth:`cross_validation_type.__init__`. Number of fold splits must
            be provided here via "n_splits", and number of repeats (if applicable according to
            `cross_validation_type`) must be provided via "n_repeats"
        verbose: Boolean, default=True
            Verbosity of printing for any experiments performed while this Environment is active
        file_blacklist: List of str, or None, or 'ALL', default=None
            If list of str, the result files named within are not saved to their respective
            directory in "<ASSETS_DIRNAME>/Experiments". If None, all result files are saved.
            If 'ALL', nothing at all will be saved for the Experiments. If the path of the file that
            initializes an Experiment does not end with a ".py" extension, the Experiment proceeds
            as if "script_backup" had been added to `file_blacklist`. This means that backup files
            will not be created for Jupyter notebooks (or any other non-".py" files). For info on
            acceptable values, see :func:`validate_file_blacklist`
        reporting_handler_params: Dict, default=dict()
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
        :class:`hyperparameter_hunter.experiments.CrossValidationExperiment`) is used to filter out
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
        self.root_results_path = root_results_path

        #################### Attributes Used by Experiments ####################
        self.train_dataset = train_dataset
        self.holdout_dataset = holdout_dataset
        self.test_dataset = test_dataset

        self.target_column = target_column
        self.id_column = id_column
        self.do_predict_proba = do_predict_proba
        self.prediction_formatter = prediction_formatter
        self.metrics_map = metrics_map
        self.metrics_params = metrics_params

        self.cross_experiment_params = dict()
        self.cross_validation_type = cross_validation_type
        self.runs = runs
        self.global_random_seed = global_random_seed
        self.random_seeds = random_seeds
        self.random_seed_bounds = random_seed_bounds
        self.cross_validation_params = cross_validation_params

        #################### Ancillary Environment Settings ####################
        self.verbose = verbose
        self.file_blacklist = file_blacklist
        self.reporting_handler_params = reporting_handler_params or {}
        self.to_csv_params = to_csv_params or {}
        self.do_full_save = do_full_save
        self.experiment_callbacks = experiment_callbacks or []

        self.result_paths = {
            "root": self.root_results_path,
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
    # Core Methods
    ##################################################
    def environment_workflow(self):
        """Execute all methods required to validate the environment and run Experiments"""
        self.update_custom_environment_params()
        self.validate_parameters()
        self.define_holdout_set()
        self.format_result_paths()
        self.generate_cross_experiment_key()
        G.log("Cross-Experiment Key: {!s}".format(self.cross_experiment_key))

    def validate_parameters(self):
        """Ensure the provided parameters are valid and properly formatted"""
        #################### root_results_path ####################
        if self.root_results_path is None:
            G.warn("Received root_results_path=None. Results will not be stored at all.")
        elif isinstance(self.root_results_path, str):
            if not self.root_results_path.endswith(ASSETS_DIRNAME):
                self.root_results_path = os.path.join(self.root_results_path, ASSETS_DIRNAME)
                self.result_paths["root"] = self.root_results_path
            if not os.path.exists(self.root_results_path):
                os.makedirs(self.root_results_path, exist_ok=True)
        else:
            raise TypeError(f"root_results_path must be None or str, not {self.root_results_path}")

        #################### target_column ####################
        if isinstance(self.target_column, str):
            self.target_column = [self.target_column]

        #################### verbose ####################
        if not isinstance(self.verbose, bool):
            raise TypeError("`verbose` must be bool, not {}: {}".format(*type_val(self.verbose)))

        #################### file_blacklist ####################
        self.file_blacklist = validate_file_blacklist(self.file_blacklist)

        #################### Train/Test Datasets ####################
        if isinstance(self.train_dataset, str):
            self.train_dataset = pd.read_csv(self.train_dataset)
        if isinstance(self.test_dataset, str):
            self.test_dataset = pd.read_csv(self.test_dataset)

        #################### metrics_params/metrics_map ####################
        if (self.metrics_map is not None) and ("metrics_map" in self.metrics_params.keys()):
            raise ValueError(
                "`metrics_map` may be provided as a kwarg, or as a `metrics_params` key, but NOT BOTH. Received: "
                + f"\n `metrics_map`={self.metrics_map}\n `metrics_params`={self.metrics_params}"
            )
        else:
            if self.metrics_map is None:
                self.metrics_map = self.metrics_params["metrics_map"]
            self.metrics_map = format_metrics_map(self.metrics_map)
            self.metrics_params = {**dict(metrics_map=self.metrics_map), **self.metrics_params}

        #################### cross_validation_type ####################
        if isinstance(self.cross_validation_type, str):
            try:
                self.cross_validation_type = sk_cv.__getattribute__(self.cross_validation_type)
            except AttributeError:
                raise AttributeError(
                    f"'{self.cross_validation_type}' not in `sklearn.model_selection._split`"
                )

        #################### to_csv_params ####################
        self.to_csv_params = {
            _k: _v for _k, _v in self.to_csv_params.items() if _k != "path_or_buf"
        }

        #################### cross_experiment_params ####################
        self.cross_experiment_params = dict(
            cross_validation_type=self.cross_validation_type,
            runs=self.runs,
            global_random_seed=self.global_random_seed,
            random_seeds=self.random_seeds,
            random_seed_bounds=self.random_seed_bounds,
        )

        #################### experiment_callbacks ####################
        if not isinstance(self.experiment_callbacks, list):
            self.experiment_callbacks = [self.experiment_callbacks]
        for callback in self.experiment_callbacks:
            if not isclass(callback):
                raise TypeError(
                    f"experiment_callbacks must be classes. Received {type(callback)}: {callback}"
                )
            if callback.__name__ != "LambdaCallback":
                raise ValueError(
                    f"experiment_callbacks must be LambdaCallback instances, not {callback.__name__}: {callback}"
                )

    def define_holdout_set(self):
        """Define :attr:`Environment.holdout_dataset`, and (if holdout_dataset is callable), also
        modifies train_dataset"""
        if callable(self.holdout_dataset):
            self.train_dataset, self.holdout_dataset = self.holdout_dataset(
                self.train_dataset, self.target_column
            )
        elif isinstance(self.holdout_dataset, str):
            try:
                self.holdout_dataset = pd.read_csv(self.holdout_dataset)
            except FileNotFoundError:
                raise
        elif self.holdout_dataset is not None and (
            not isinstance(self.holdout_dataset, pd.DataFrame)
        ):
            raise TypeError(
                f"holdout_dataset must be one of: [None, DataFrame, callable, str], not {type(self.holdout_dataset)}"
            )

        if (self.holdout_dataset is not None) and (
            not np.array_equal(self.train_dataset.columns, self.holdout_dataset.columns)
        ):
            raise ValueError(
                "\n".join(
                    [
                        "train_dataset and holdout_dataset must have the same columns. Instead, "
                        f"train_dataset had {len(self.train_dataset.columns)} columns: {self.train_dataset.columns}",
                        f"holdout_dataset had {len(self.holdout_dataset.columns)} columns: {self.holdout_dataset.columns}",
                    ]
                )
            )

    def format_result_paths(self):
        """Remove paths contained in file_blacklist, and format others to prepare for saving results"""
        if self.file_blacklist == "ALL":
            return

        if self.root_results_path is not None:
            # Blacklist prediction files for datasets not given
            if self.holdout_dataset is None:
                self.file_blacklist.append("predictions_holdout")
            if self.test_dataset is None:
                self.file_blacklist.append("predictions_test")

            for k in self.result_paths.keys():
                if k == "root":
                    continue
                elif k not in self.file_blacklist:
                    self.result_paths[k] = os.path.join(
                        self.root_results_path, RESULT_FILE_SUB_DIR_PATHS[k]
                    )
                else:
                    self.result_paths[k] = None
                    # G.debug('Result file "{}" has been blacklisted'.format(k))

    def update_custom_environment_params(self):
        """Try to update null parameters from environment_params_path, or DEFAULT_PARAMS"""
        allowed_parameter_keys = [
            k for k, v in signature(Environment).parameters.items() if v.kind == v.KEYWORD_ONLY
        ]
        user_defaults = {}

        if (not isinstance(self.environment_params_path, str)) and (
            self.environment_params_path is not None
        ):
            raise TypeError(
                "environment_params_path must be a str, not {}: {}".format(
                    *type_val(self.environment_params_path)
                )
            )

        try:
            user_defaults = read_json(self.environment_params_path)
        except TypeError:
            if self.environment_params_path is not None:
                raise
        except FileNotFoundError:
            raise

        if not isinstance(user_defaults, dict):
            raise TypeError(
                "environment_params_path must contain a dict. Received {}: {}".format(
                    *type_val(user_defaults)
                )
            )

        #################### Check user_defaults ####################
        for k, v in user_defaults.items():
            if k not in allowed_parameter_keys:
                G.warn(
                    "\n\t".join(
                        [
                            'Invalid key ({}) in user-defined default Environment parameter file at "{}". If expected to do something,',
                            "it really won't, so it should be removed or fixed. The following are valid default keys: {}",
                        ]
                    ).format(k, self.environment_params_path, allowed_parameter_keys)
                )
            elif getattr(self, k) is None:
                setattr(self, k, v)
                G.debug(
                    'Environment kwarg "{}" was set to user default at "{}"'.format(
                        k, self.environment_params_path
                    )
                )

        #################### Check Module Default Environment Arguments ####################
        for k in allowed_parameter_keys:
            if getattr(self, k) is None:
                setattr(self, k, self.DEFAULT_PARAMS.get(k, None))

    def generate_cross_experiment_key(self):
        """Generate a key to describe the current Environment's cross-experiment parameters"""
        parameters = dict(
            metrics_params=self.metrics_params,
            cross_validation_params=self.cross_validation_params,
            target_column=self.target_column,
            id_column=self.id_column,
            do_predict_proba=self.do_predict_proba,
            prediction_formatter=self.prediction_formatter,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            holdout_dataset=self.holdout_dataset,
            cross_experiment_params=self.cross_experiment_params,
            to_csv_params=self.to_csv_params,
        )
        self.cross_experiment_key = CrossExperimentKeyMaker(parameters)

    def initialize_reporting(self):
        """Initialize reporting for the Environment and Experiments conducted during its lifetime"""
        reporting_handler_params = self.reporting_handler_params
        reporting_handler_params["heartbeat_path"] = "{}/Heartbeat.log".format(
            self.root_results_path
        )
        reporting_handler = ReportingHandler(**reporting_handler_params)

        #################### Make Unified Logging Globally Available ####################
        G.log = reporting_handler.log
        G.debug = reporting_handler.debug
        G.warn = reporting_handler.warn

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
        return DatasetSentinel("validation_input", **self._dataset_sentinel_helper())

    @property
    def validation_target(self):
        """Get a `DatasetSentinel` representing an Experiment's `fold_validation_target`

        Returns
        -------
        DatasetSentinel:
            A `Sentinel` that will be converted to :attr:`hyperparameter_hunter.experiments.BaseExperiment.fold_validation_target`
            upon `Model` initialization"""
        return DatasetSentinel("validation_target", **self._dataset_sentinel_helper())

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
        return dict(
            dataset_hash=self.cross_experiment_key.parameters["train_dataset"],
            cross_validation_type=self.cross_experiment_key.parameters["cross_experiment_params"][
                "cross_validation_type"
            ],
            global_random_seed=self.cross_experiment_key.parameters["cross_experiment_params"][
                "global_random_seed"
            ],
            random_seeds=self.cross_experiment_key.parameters["cross_experiment_params"][
                "random_seeds"
            ],
        )


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
    "KeyAttributeLookup" directory will also be excluded from the list of files to update"""
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
    ]
    if blacklist == "ALL":
        G.warn('WARNING: Received `blacklist`="ALL". Nothing will be saved')
        return blacklist

    if not blacklist:
        return []
    elif not isinstance(blacklist, list):
        raise TypeError(
            "Expected blacklist to be a list, but received {}: {}".format(
                type(blacklist), blacklist
            )
        )
    elif not all([isinstance(_, str) for _ in blacklist]):
        invalid_files = [(type(_).__name__, _) for _ in blacklist if not isinstance(_, str)]
        raise TypeError(
            "Expected contents of blacklist to be strings, but received {}".format(invalid_files)
        )

    for a_file in blacklist:
        if a_file not in valid_values:
            raise ValueError(
                "Received invalid blacklist value: {}.\nExpected one of: [{}]".format(
                    a_file, valid_values
                )
            )
        if a_file in ["description", "tested_keys"]:
            G.warn(
                f"Including {a_file!r} in file_blacklist will severely impede the functionality of this library"
            )

    return blacklist
