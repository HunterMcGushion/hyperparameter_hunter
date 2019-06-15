"""This module contains the classes used for constructing and conducting an Experiment (most
notably, :class:`CVExperiment`). Any class contained herein whose name starts with "Base" should not
be used directly. :class:`CVExperiment` is the preferred means of conducting one-off experimentation

Related
-------
:mod:`hyperparameter_hunter.experiment_core`
    Defines :class:`ExperimentMeta`, an understanding of which is critical to being able to
    understand :mod:`experiments`
:mod:`hyperparameter_hunter.metrics`
    Defines :class:`ScoringMixIn`, a parent of :class:`experiments.BaseExperiment` that enables
    scoring and evaluating models
:mod:`hyperparameter_hunter.models`
    Used to instantiate the actual learning models, which are a single part of the entire
    experimentation workflow, albeit the most significant part

Notes
-----
As mentioned above, the inner workings of :mod:`experiments` will be very confusing without a grasp
on what's going on in :mod:`experiment_core`, and its related modules"""
##################################################
# Import Own Assets
##################################################
# noinspection PyProtectedMember
from hyperparameter_hunter import __version__
from hyperparameter_hunter.algorithm_handlers import (
    identify_algorithm,
    identify_algorithm_hyperparameters,
)
from hyperparameter_hunter.data import TrainDataset, OOFDataset, HoldoutDataset, TestDataset
from hyperparameter_hunter.exceptions import (
    EnvironmentInactiveError,
    EnvironmentInvalidError,
    RepeatedExperimentError,
)
from hyperparameter_hunter.experiment_core import ExperimentMeta
from hyperparameter_hunter.feature_engineering import FeatureEngineer
from hyperparameter_hunter.keys.makers import HyperparameterKeyMaker
from hyperparameter_hunter.metrics import ScoringMixIn, get_formatted_target_metric
from hyperparameter_hunter.models import model_selector
from hyperparameter_hunter.recorders import RecorderList
from hyperparameter_hunter.settings import G

# from hyperparameter_hunter.tracers import TranslateTrace  # TODO: Add when tested with `Mirror`
from hyperparameter_hunter.utils.file_utils import RetryMakeDirs
from hyperparameter_hunter.utils.general_utils import Deprecated

##################################################
# Import Miscellaneous Assets
##################################################
from abc import abstractmethod
from copy import deepcopy
from inspect import isclass
import numpy as np
import pandas as pd
import random
import shutil
from sys import exc_info
from uuid import uuid4 as uuid
import warnings

##################################################
# Import Learning Assets
##################################################
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
import sklearn.utils as sklearn_utils

pd.set_option("display.expand_frame_repr", False)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=sklearn_utils.DataConversionWarning)
np.random.seed(32)


class BaseExperiment(ScoringMixIn):
    # @TranslateTrace("model", ("model_initializer", "model_init_params"))  # TODO: Add when tested with `Mirror`
    def __init__(
        # TODO: Make `model_init_params` an optional kwarg - If not given, algorithm defaults used
        self,
        # model=None,
        # model_initializer=None,
        # model_init_params=None,
        # TODO: Convert below 2 to above 3 lines for `TranslateTrace`
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_engineer=None,
        feature_selector=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
        # TODO: When `TranslateTrace` added document `model` below with expectation that if `model`
        # TODO: ... given, (`model_initializer`, `model_init_params`) should not be, and vice versa
        # TODO: `model` (Class instance, default=None);
        # TODO: `model_initializer`/`model_init_params` docstring types += "default=None"
        """Base class for :class:`BaseCVExperiment`

        Parameters
        ----------
        model_initializer: Class, or functools.partial, or class instance
            The algorithm class being used to initialize a model
        model_init_params: Dict, or object
            The dictionary of arguments given when creating a model instance with
            `model_initializer` via the `__init__` method of :class:`models.Model`. Any kwargs that
            are considered valid by the `__init__` method of `model_initializer` are valid in
            `model_init_params`
        model_extra_params: Dict, or None, default=None
            A dictionary of extra parameters passed to :class:`models.Model`. This is used to
            provide parameters to models' non-initialization methods (like `fit`, `predict`,
            `predict_proba`, etc.), and for neural networks
        feature_engineer: `FeatureEngineer`, or None, default=None
            ...  # TODO: Add documentation
        feature_selector: List of str, callable, list of booleans, default=None
            Column names to include as input data for all provided DataFrames. If None,
            `feature_selector` is set to all columns in :attr:`train_dataset`, less
            :attr:`target_column`, and :attr:`id_column`. `feature_selector` is provided as the
            second argument for calls to `pandas.DataFrame.loc` when constructing datasets
        notes: String, or None, default=None
            Additional information about the Experiment that will be saved with the Experiment's
            description result file. This serves no purpose other than to facilitate saving
            Experiment details in a more readable format
        do_raise_repeated: Boolean, default=False
            If True and this Experiment locates a previous Experiment's results with matching
            Environment and Hyperparameter Keys, a RepeatedExperimentError will be raised. Else, a
            warning will be logged
        auto_start: Boolean, default=True
            If True, after the Experiment is initialized, it will automatically call
            :meth:`BaseExperiment.preparation_workflow`, followed by
            :meth:`BaseExperiment.experiment_workflow`, effectively completing all essential tasks
            without requiring additional method calls
        target_metric: Tuple, str, default=('oof', <:attr:`environment.Environment.metrics`[0]>)
            A path denoting the metric to be used to compare completed Experiments or to use for
            certain early stopping procedures in some model classes. The first value should be one
            of ['oof', 'holdout', 'in_fold']. The second value should be the name of a metric being
            recorded according to the values supplied in
            :attr:`environment.Environment.metrics_params`. See the documentation for
            :func:`metrics.get_formatted_target_metric` for more info. Any values returned by, or
            used as the `target_metric` input to this function are acceptable values for
            :attr:`BaseExperiment.target_metric`"""
        # self._model_original = model  # TODO: Add for `TranslateTrace`
        self.model_initializer = model_initializer
        self.model_init_params = identify_algorithm_hyperparameters(self.model_initializer)
        try:
            self.model_init_params.update(model_init_params)
        except TypeError:
            self.model_init_params.update(dict(build_fn=model_init_params))

        self.model_extra_params = model_extra_params if model_extra_params is not None else {}
        self.feature_engineer = (
            feature_engineer if callable(feature_engineer) else FeatureEngineer()
        )
        self.feature_selector = feature_selector if feature_selector is not None else []

        self.notes = notes
        self.do_raise_repeated = do_raise_repeated
        self.auto_start = auto_start
        self.target_metric = target_metric

        #################### Attributes From Active Environment ####################
        G.Env.initialize_reporting()
        self._validate_environment()

        self.train_dataset = G.Env.train_dataset.copy()
        try:
            self.holdout_dataset = G.Env.holdout_dataset.copy()
        except AttributeError:
            self.holdout_dataset = G.Env.holdout_dataset
        try:
            self.test_dataset = G.Env.test_dataset.copy()
        except AttributeError:
            self.test_dataset = G.Env.test_dataset

        self.target_column = G.Env.target_column
        self.id_column = G.Env.id_column
        self.do_predict_proba = G.Env.do_predict_proba
        self.prediction_formatter = G.Env.prediction_formatter
        self.metrics_params = G.Env.metrics_params
        self.experiment_params = G.Env.cross_experiment_params
        self.cv_params = G.Env.cv_params
        self.result_paths = G.Env.result_paths
        self.cross_experiment_key = G.Env.cross_experiment_key

        #################### Dataset Attributes ####################
        self.data_train = None
        self.data_oof = None
        self.data_holdout = None
        self.data_test = None

        #################### Other Attributes ####################
        self.model = None
        self.metrics = None  # Set by :class:`metrics.ScoringMixIn`
        self.stat_aggregates = dict()
        self.result_description = None

        #################### Experiment Identification Attributes ####################
        self.experiment_id = None
        self.hyperparameter_key = None
        self.algorithm_name, self.module_name = identify_algorithm(self.model_initializer)

        ScoringMixIn.__init__(self, **self.metrics_params if self.metrics_params else {})

        if self.auto_start is True:
            self.preparation_workflow()
            self.experiment_workflow()

    def __repr__(self):
        return '{}("{}", cross_experiment_key="{}", hyperparameter_key="{}")'.format(
            type(self).__name__,
            self.experiment_id,
            self.cross_experiment_key,
            self.hyperparameter_key,
        )

    def __getattr__(self, attr):
        """If AttributeError thrown, check :attr:`settings.G.Env` for target attribute"""
        try:
            return getattr(G.Env, attr)
        except AttributeError:
            raise AttributeError(
                "Could not find '{}' in 'G.Env', or any of the following locations: {}".format(
                    attr, [_.__name__ for _ in type(self).__mro__]
                )
            ).with_traceback(exc_info()[2]) from None

    def experiment_workflow(self):
        """Define the actual experiment process, including execution, result saving, and cleanup"""
        if self.hyperparameter_key.exists is True:
            _ex = f"{self!r} has already been run"
            if self.do_raise_repeated is True:
                self._clean_up()
                raise RepeatedExperimentError(_ex)
            G.debug(_ex)
            G.warn("WARNING: Duplicate experiment!")

        self._initialize_random_seeds()
        self.execute()

        #################### Save Experiment Results ####################
        recorders = RecorderList(
            file_blacklist=G.Env.file_blacklist, extra_recorders=G.Env.experiment_recorders
        )
        recorders.format_result()
        G.log(f"Saving results for Experiment: '{self.experiment_id}'")
        recorders.save_result()
        self._clean_up()

    def preparation_workflow(self):
        """Execute all tasks that must take place before the experiment is actually started. Such
        tasks include (but are not limited to): Creating experiment IDs and hyperparameter keys,
        creating script backups, and validating parameters"""
        G.debug("Starting preparation_workflow...")
        self._generate_experiment_id()
        self._create_script_backup()
        self._validate_parameters()
        self._generate_hyperparameter_key()
        self._additional_preparation_steps()
        G.debug("Completed preparation_workflow")

    @abstractmethod
    def _additional_preparation_steps(self):
        """Perform extra preparation tasks prior to initializing random seeds and preprocessing"""

    @abstractmethod
    def execute(self):
        """Execute the fitting protocol for the Experiment, comprising the following: instantiation
        of learners for each run, preprocessing of data as appropriate, training learners, making
        predictions, and evaluating and aggregating those predictions and other stats/metrics for
        later use"""

    ##################################################
    # Data Preprocessing Methods:
    ##################################################
    def on_experiment_start(self):
        """Prepare data prior to executing fitting protocol (cross-validation), by 1) Initializing
        formal :mod:`~hyperparameter_hunter.data.datasets` attributes, 2) Invoking
        `feature_engineer` to perform "pre_cv"-stage preprocessing, and 3) Updating datasets to
        include their (transformed) counterparts in `feature_engineer`"""
        #################### Build Datasets ####################
        data_kwargs = dict(feature_selector=self.feature_selector, target_column=self.target_column)
        self.data_train = TrainDataset(self.train_dataset, require_data=True, **data_kwargs)
        # TODO: Might be better to initialize `data_oof` with same data as `data_train`
        self.data_oof = OOFDataset(None, **data_kwargs)
        self.data_holdout = HoldoutDataset(self.holdout_dataset, **data_kwargs)
        self.data_test = TestDataset(self.test_dataset, feature_selector=self.feature_selector)

        #################### Perform Pre-CV Feature Engineering ####################
        self.feature_engineer(
            "pre_cv",
            train_inputs=deepcopy(self.data_train.input.d),
            train_targets=deepcopy(self.data_train.target.d),
            holdout_inputs=deepcopy(self.data_holdout.input.d),
            holdout_targets=deepcopy(self.data_holdout.target.d),
            test_inputs=deepcopy(self.data_test.input.d),
        )
        self.data_train.input.T.d = self.feature_engineer.datasets["train_inputs"]
        self.data_train.target.T.d = self.feature_engineer.datasets["train_targets"]
        self.data_holdout.input.T.d = self.feature_engineer.datasets["holdout_inputs"]
        self.data_holdout.target.T.d = self.feature_engineer.datasets["holdout_targets"]
        self.data_test.input.T.d = self.feature_engineer.datasets["test_inputs"]

        G.log("Initial preprocessing stage complete", 4)
        super().on_experiment_start()

    ##################################################
    # Supporting Methods:
    ##################################################
    def _validate_parameters(self):
        """Ensure provided input parameters are properly formatted"""
        #################### target_metric ####################
        self.target_metric = get_formatted_target_metric(self.target_metric, self.metrics)

        #################### feature_selector ####################
        self.feature_selector = self.feature_selector or self.train_dataset.columns.values
        restricted_cols = [_ for _ in self.target_column + [self.id_column] if _ is not None]
        self.feature_selector = [_ for _ in self.feature_selector if _ not in restricted_cols]

        G.debug("Experiment parameters have been validated")

    def _validate_environment(self):
        """Ensure there is a currently active Environment instance that is not already occupied"""
        if G.Env is None:
            raise EnvironmentInactiveError("")
        if G.Env.current_task is None:
            G.Env.current_task = self
            G.log(f"Validated Environment:  '{self.cross_experiment_key}'")
        else:
            raise EnvironmentInvalidError("Current experiment must finish before starting another")

    @staticmethod
    def _clean_up():
        """Clean up after experiment to prepare for next experiment"""
        G.Env.current_task = None

    ##################################################
    # Key/ID Methods:
    ##################################################
    def _generate_experiment_id(self):
        """Set :attr:`experiment_id` to a UUID"""
        self.experiment_id = str(uuid())
        G.log("Initialized Experiment: '{}'".format(self.experiment_id))

    def _generate_hyperparameter_key(self):
        """Set :attr:`hyperparameter_key` to a key to describe the experiment's hyperparameters"""
        parameters = dict(
            model_initializer=self.model_initializer,
            model_init_params=self.model_init_params,
            model_extra_params=self.model_extra_params,
            feature_engineer=self.feature_engineer,
            feature_selector=self.feature_selector,
            # FLAG: Should probably add :attr:`target_metric` to key - With option to ignore it?
        )

        self.hyperparameter_key = HyperparameterKeyMaker(parameters, self.cross_experiment_key)
        G.log("Hyperparameter Key:     '{}'".format(self.hyperparameter_key))
        G.debug("Raw hyperparameters...")
        G.debug(self.hyperparameter_key.parameters)

    def _create_script_backup(self):
        """Create and save a copy of the script that initialized the Experiment if allowed to, and
        if :attr:`source_script` ends with a ".py" extension"""
        #################### Attempt to Copy Source Script if Allowed ####################
        try:
            if not self.source_script.endswith(".py"):
                G.Env.result_paths["script_backup"] = None

            if G.Env.result_paths["script_backup"] is not None:
                self._source_copy_helper()
                G.log("Created source backup:  '{}'".format(self.source_script), 4)
            else:
                G.log("Skipped source backup:  '{}'".format(self.source_script), 4)
        #################### Exception Handling ####################
        except AttributeError as _ex:
            if G.Env is None:
                raise EnvironmentInactiveError(extra="\n{!s}".format(_ex))
            if not hasattr(G.Env, "result_paths"):
                raise EnvironmentInvalidError(extra=f"G.Env lacks 'result_paths' attr\n{_ex!s}")
            raise
        except KeyError as _ex:
            if "script_backup" not in G.Env.result_paths:
                raise EnvironmentInvalidError(
                    extra=f"G.Env.result_paths lacks 'script_backup' key\n{_ex!s}"
                )
            raise

    @RetryMakeDirs()
    def _source_copy_helper(self):
        """Helper method to handle attempting to copy source script to backup file"""
        shutil.copyfile(
            self.source_script, f"{self.result_paths['script_backup']}/{self.experiment_id}.py"
        )

    ##################################################
    # Utility Methods:
    ##################################################
    def _initialize_random_seeds(self):
        """Initialize global random seed, and generate random seeds for stages if not provided"""
        np.random.seed(self.experiment_params["global_random_seed"])
        random.seed(self.experiment_params["global_random_seed"])
        self._random_seed_initializer()
        G.debug("Initialized random seeds for experiment")

    def _random_seed_initializer(self):
        """Generate set of random seeds for each repetition/fold/run if not provided"""
        if self.experiment_params["random_seeds"] is None:
            self.experiment_params["random_seeds"] = np.random.randint(
                *self.experiment_params["random_seed_bounds"],
                size=(
                    self.cv_params.get("n_repeats", 1),
                    self.cv_params["n_splits"],
                    self.experiment_params["runs"],
                ),
            ).tolist()
        G.debug("BaseExperiment._random_seed_initializer() done")

    def _update_model_params(self):
        """Update random state of :attr:`model_init_params` according to :attr:`current_seed`"""
        # TODO: Add this to some workflow in Experiment class. For now it is never used, unless the subclass decides to...
        # `model_init_params` initialized to all algorithm hyperparameters - Works even if 'random_state' not explicitly given
        try:
            if "random_state" in self.model_init_params:
                self.model_init_params["random_state"] = self.current_seed
            elif "seed" in self.model_init_params:
                self.model_init_params["seed"] = self.current_seed
            else:
                G.debug("WARNING: Model has no random_state/seed parameter to update")
                # FLAG: HIGH PRIORITY BELOW
                # TODO: BELOW IS NOT THE CASE IF MODEL IS NN - SETTING THE GLOBAL RANDOM SEED DOES SOMETHING
                # TODO: If this is logged, there is no reason to execute multiple-run-averaging, so don't
                # TODO: ... Either 1) Set `runs` = 1 (this would mess with the environment key), or...
                # TODO: ... 2) Set the results of all subsequent runs to the results of the first run (this could be difficult)
                # FLAG: HIGH PRIORITY ABOVE
        except Exception as _ex:
            G.log("WARNING: Failed to update model's random_state     {}".format(_ex.__repr__()))

    def _empty_output_like(self, like: pd.DataFrame, index=None, target_column=None):
        """Make an empty DataFrame of the same shape and with the same index as `like`, intended for
        use with output :mod:`~hyperparameter_hunter.data.data_chunks`, like descendants of
        :class:`~hyperparameter_hunter.data.data_chunks.prediction_chunks.BasePredictionChunk` and
        :class:`~hyperparameter_hunter.data.data_chunks.target_chunks.BaseTargetChunk`

        Parameters
        ----------
        like: pd.DataFrame
            DataFrame to use as the basis for the shape and index of the returned DataFrame. `like`
            has no bearing on the column names of the returned DataFrame
        index: Array-like, or None, default=None
            If None, defaults to `like.index`. Else, defines the index of the returned DataFrame
        target_column: List[str], or None, default=None
            If None, defaults to the experiment's :attr:`target_column`. Else, defines the column
            names of the returned DataFrame

        Returns
        -------
        pd.DataFrame
            Zero-filled DataFrame with index of `index` or `like.index` and column names of
            `target_column` or :attr:`target_column`"""
        index = like.index.copy() if index is None else index
        target_column = self.target_column if target_column is None else target_column
        return pd.DataFrame(0, index=index, columns=target_column)


class BaseCVExperiment(BaseExperiment):
    def __init__(
        self,
        # model=None,
        # model_initializer=None,
        # model_init_params=None,
        # TODO: Convert below 2 to above 3 lines for `TranslateTrace`
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_engineer=None,
        feature_selector=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
        self._rep = 0
        self._fold = 0
        self._run = 0
        self.current_seed = None
        self.train_index = None
        self.validation_index = None
        self.folds = None

        #################### Initialize Result Placeholders ####################
        # self.full_oof_predictions = None  # (n_repeats * runs) intermediate columns
        # self.full_test_predictions = 0  # (n_splits * n_repeats * runs) intermediate columns
        # self.full_holdout_predictions = 0  # (n_splits * n_repeats * runs) intermediate columns

        BaseExperiment.__init__(
            self,
            # model=model,
            # model_initializer=model_initializer,
            # model_init_params=model_init_params,
            # TODO: Convert below 2 to above 3 lines for `TranslateTrace`
            model_initializer,
            model_init_params,
            model_extra_params=model_extra_params,
            feature_engineer=feature_engineer,
            feature_selector=feature_selector,
            notes=notes,
            do_raise_repeated=do_raise_repeated,
            auto_start=auto_start,
            target_metric=target_metric,
        )

    def _additional_preparation_steps(self):
        """Perform extra preparation tasks prior to initializing random seeds and preprocessing"""
        self._initialize_folds()

    @abstractmethod
    def _initialize_folds(self):
        """"""

    def execute(self):
        self.cross_validation_workflow()

    def cross_validation_workflow(self):
        """Execute workflow for cross-validation process, consisting of the following tasks:
        1) Create train and validation split indices for all folds, 2) Iterate through folds,
        performing `cv_fold_workflow` for each, 3) Average accumulated predictions over fold
        splits, 4) Evaluate final predictions, 5) Format final predictions to prepare for saving"""
        self.on_experiment_start()

        reshaped_indices = get_cv_indices(
            self.folds, self.cv_params, self.data_train.input.d, self.data_train.target.d.iloc[:, 0]
        )

        for self._rep, rep_indices in enumerate(reshaped_indices):
            self.on_repetition_start()

            for self._fold, (self.train_index, self.validation_index) in enumerate(rep_indices):
                self.cv_fold_workflow()

            self.on_repetition_end()
        self.on_experiment_end()

        G.log("")

    ##################################################
    # Fold Workflow Methods:
    ##################################################
    def on_fold_start(self):
        """Override :meth:`on_fold_start` tasks set by :class:`experiment_core.ExperimentMeta`,
        consisting of: 1) Split train/validation data, 2) Make copies of holdout/test data for
        current fold (for feature engineering), 3) Log start, 4) Execute original tasks"""
        #################### Split Train and Validation Data ####################
        self.data_train.input.fold = self.data_train.input.d.iloc[self.train_index, :].copy()
        self.data_oof.input.fold = self.data_train.input.d.iloc[self.validation_index, :].copy()

        self.data_train.input.T.fold = self.data_train.input.T.d.iloc[self.train_index, :].copy()
        self.data_oof.input.T.fold = self.data_train.input.T.d.iloc[self.validation_index, :].copy()

        self.data_train.target.fold = self.data_train.target.d.iloc[self.train_index].copy()
        self.data_oof.target.fold = self.data_train.target.d.iloc[self.validation_index].copy()

        self.data_train.target.T.fold = self.data_train.target.T.d.iloc[self.train_index].copy()
        self.data_oof.target.T.fold = self.data_train.target.T.d.iloc[self.validation_index].copy()

        #################### Set Fold Copies of Holdout/Test Data ####################
        for data_chunk in [self.data_holdout.input, self.data_holdout.target, self.data_test.input]:
            if data_chunk.d is not None:
                data_chunk.fold = data_chunk.d.copy()
                data_chunk.T.fold = data_chunk.T.d.copy()

        #################### Perform Intra-CV Feature Engineering ####################
        self.feature_engineer(
            "intra_cv",
            train_inputs=self.data_train.input.T.fold,
            train_targets=self.data_train.target.T.fold,
            validation_inputs=self.data_oof.input.T.fold,
            validation_targets=self.data_oof.target.T.fold,
            holdout_inputs=self.data_holdout.input.T.fold,
            holdout_targets=self.data_holdout.target.T.fold,
            test_inputs=self.data_test.input.T.fold,
        )
        self.data_train.input.T.fold = self.feature_engineer.datasets["train_inputs"]
        self.data_train.target.T.fold = self.feature_engineer.datasets["train_targets"]
        self.data_oof.input.T.fold = self.feature_engineer.datasets["validation_inputs"]
        self.data_oof.target.T.fold = self.feature_engineer.datasets["validation_targets"]
        self.data_holdout.input.T.fold = self.feature_engineer.datasets["holdout_inputs"]
        self.data_holdout.target.T.fold = self.feature_engineer.datasets["holdout_targets"]
        self.data_test.input.T.fold = self.feature_engineer.datasets["test_inputs"]

        super().on_fold_start()

    def cv_fold_workflow(self):
        """Execute workflow for individual fold, consisting of the following tasks: Execute
        overridden :meth:`on_fold_start` tasks, 2) Perform cv_run_workflow for each run, 3) Execute
        overridden :meth:`on_fold_end` tasks"""
        self.on_fold_start()
        G.log("Intra-CV preprocessing stage complete", 4)

        for self._run in range(self.experiment_params.get("runs", 1)):
            self.cv_run_workflow()
        self.on_fold_end()

    ##################################################
    # Run Workflow Methods:
    ##################################################
    def on_run_start(self):
        """Override :meth:`on_run_start` tasks organized by :class:`experiment_core.ExperimentMeta`,
        consisting of: 1) Set random seed and update model parameters according to current seed,
        2) Log run start, 3) Execute original tasks"""
        self.current_seed = self.experiment_params["random_seeds"][self._rep][self._fold][self._run]
        np.random.seed(self.current_seed)
        self._update_model_params()
        super().on_run_start()

    def cv_run_workflow(self):
        """Execute run workflow, consisting of: 1) Execute overridden :meth:`on_run_start` tasks,
        2) Initialize and fit Model, 3) Execute overridden :meth:`on_run_end` tasks"""
        self.on_run_start()
        self.model = model_selector(self.model_initializer)(
            self.model_initializer,
            self.model_init_params,
            self.model_extra_params,
            train_input=self.data_train.input.T.fold,
            train_target=self.data_train.target.T.fold,
            validation_input=self.data_oof.input.T.fold,
            validation_target=self.data_oof.target.T.fold,
            do_predict_proba=self.do_predict_proba,
            target_metric=self.target_metric,
            metrics=self.metrics,
        )
        self.model.fit()
        self.on_run_end()


##################################################
# Core CV Experiment Classes:
##################################################
class CVExperiment(BaseCVExperiment, metaclass=ExperimentMeta):
    def __init__(
        self,
        # model=None,
        # model_initializer=None,
        # model_init_params=None,
        # TODO: Convert below 2 to above 3 lines for `TranslateTrace`
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_engineer=None,
        feature_selector=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
        BaseCVExperiment.__init__(
            self,
            # model=model,
            # model_initializer=model_initializer,
            # model_init_params=model_init_params,
            # TODO: Convert below 2 to above 3 lines for `TranslateTrace`
            model_initializer,
            model_init_params,
            model_extra_params=model_extra_params,
            feature_engineer=feature_engineer,
            feature_selector=feature_selector,
            notes=notes,
            do_raise_repeated=do_raise_repeated,
            auto_start=auto_start,
            target_metric=target_metric,
        )

    def _initialize_folds(self):
        """Set :attr:`folds` per `cv_type` and :attr:`cv_params`"""
        cv_type = self.experiment_params["cv_type"]  # Allow failure
        if not isclass(cv_type):
            raise TypeError(f"Expected a cross-validation class, not {type(cv_type)}")

        try:
            _split_method = getattr(cv_type, "split")
            if not callable(_split_method):
                raise TypeError("`cv_type` must implement a callable :meth:`split`")
        except AttributeError:
            raise AttributeError("`cv_type` must be class with :meth:`split`")

        self.folds = cv_type(**self.cv_params)


##################################################
# Experiment Helpers
##################################################
def get_cv_indices(folds, cv_params, input_data, target_data):
    """Produce iterables of cross validation indices in the shape of (n_repeats, n_folds)

    Parameters
    ----------
    folds: Instance of `cv_type`
        Cross validation folds object, whose :meth:`split` receives `input_data` and `target_data`
    cv_params: Dict
        Parameters given to instantiate `folds`. Must contain `n_splits`. May contain `n_repeats`
    input_data: pandas.DataFrame
        Input data to be split by `folds`, to which yielded indices will correspond
    target_data: pandas.DataFrame
        Target data to be split by `folds`, to which yielded indices will correspond

    Yields
    ------
    Generator
        Cross validation indices in shape of (<n_repeats or 1>, <n_splits>)"""
    indices = folds.split(input_data, target_data)

    for rep in range(cv_params.get("n_repeats", 1)):
        yield (next(indices) for _ in range(cv_params["n_splits"]))


# class NoValidationExperiment(BaseExperiment):
#     pass


if __name__ == "__main__":
    pass
