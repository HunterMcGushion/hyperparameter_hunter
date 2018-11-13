"""This module contains the classes used for constructing and conducting an Experiment (most
notably, :class:`CrossValidationExperiment`). Any class contained herein whose name starts with
'Base' should not be used directly. :class:`CrossValidationExperiment` is the preferred means of
conducting one-off experimentation

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
on whats going on in :mod:`experiment_core`, and its related modules"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.algorithm_handlers import (
    identify_algorithm,
    identify_algorithm_hyperparameters,
)
from hyperparameter_hunter.exceptions import (
    EnvironmentInactiveError,
    EnvironmentInvalidError,
    RepeatedExperimentError,
)
from hyperparameter_hunter.experiment_core import ExperimentMeta
from hyperparameter_hunter.key_handler import HyperparameterKeyMaker
from hyperparameter_hunter.metrics import ScoringMixIn, get_formatted_target_metric
from hyperparameter_hunter.models import model_selector
from hyperparameter_hunter.recorders import RecorderList
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
from abc import abstractmethod
from inspect import isclass
import numpy as np
import os
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
    def __init__(
        # TODO: Make `model_init_params` an optional kwarg - If not given, algorithm defaults used
        self,
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_selector=None,
        preprocessing_pipeline=None,
        preprocessing_params=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
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
        feature_selector: List of str, callable, list of booleans, default=None
            The value provided when splitting apart the input data for all provided DataFrames.
            `feature_selector` is provided as the second argument for calls to
            `pandas.DataFrame.loc` in :meth:`BaseExperiment._initial_preprocessing`. If None,
            `feature_selector` is set to all columns in :attr:`train_dataset`, less
            :attr:`target_column`, and :attr:`id_column`
        preprocessing_pipeline: ...
            ... Experimental...
        preprocessing_params: ...
            ... Experimental...
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
        target_metric: Tuple, str, default=('oof', <:attr:`environment.Environment.metrics_map`[0]>)
            A path denoting the metric to be used to compare completed Experiments or to use for
            certain early stopping procedures in some model classes. The first value should be one
            of ['oof', 'holdout', 'in_fold']. The second value should be the name of a metric being
            recorded according to the values supplied in
            :attr:`environment.Environment.metrics_params`. See the documentation for
            :func:`metrics.get_formatted_target_metric` for more info. Any values returned by, or
            used as the `target_metric` input to this function are acceptable values for
            :attr:`BaseExperiment.target_metric`"""
        self.model_initializer = model_initializer
        self.model_init_params = identify_algorithm_hyperparameters(self.model_initializer)
        try:
            self.model_init_params.update(model_init_params)
        except TypeError:
            self.model_init_params.update(dict(build_fn=model_init_params))

        self.model_extra_params = model_extra_params
        self.feature_selector = feature_selector
        self.preprocessing_pipeline = preprocessing_pipeline
        self.preprocessing_params = preprocessing_params
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
        self.cross_validation_params = G.Env.cross_validation_params
        self.result_paths = G.Env.result_paths
        self.cross_experiment_key = G.Env.cross_experiment_key

        #################### Instantiate Other Attributes ####################
        self.train_input_data = None
        self.train_target_data = None
        self.holdout_input_data = None
        self.holdout_target_data = None
        self.test_input_data = None

        self.model = None
        self.metrics_map = None  # Set by :class:`metrics.ScoringMixIn`
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
            G.warn(_ex)

        self._initialize_random_seeds()
        self._initial_preprocessing()
        self.execute()

        recorders = RecorderList(file_blacklist=G.Env.file_blacklist)
        recorders.format_result()
        G.log(f'Saving results for Experiment: "{self.experiment_id}"')
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
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        """Execute the fitting protocol for the Experiment, comprising the following: instantiation
        of learners for each run, preprocessing of data as appropriate, training learners, making
        predictions, and evaluating and aggregating those predictions and other stats/metrics for
        later use"""
        raise NotImplementedError()

    ##################################################
    # Data Preprocessing Methods:
    ##################################################
    def _initial_preprocessing(self):
        """Perform preprocessing steps prior to executing fitting protocol (usually
        cross-validation), consisting of: 1) Split train/holdout data into respective train/holdout
        input and target data attributes, 2) Feature selection on input data sets, 3) Set target
        datasets to target_column contents, 4) Initialize PreprocessingPipeline to perform core
        preprocessing, 5) Set datasets to their (modified) counterparts in PreprocessingPipeline,
        6) Log whether datasets changed"""
        #################### Preprocessing ####################
        # preprocessor = PreprocessingPipelineMixIn(
        #     pipeline=[], preprocessing_params=dict(apply_standard_scale=True), features=self.features,
        #     target_column=self.target_column, train_input_data=self.train_input_data,
        #     train_target_data=self.train_target_data, holdout_input_data=self.holdout_input_data,
        #     holdout_target_data=self.holdout_target_data, test_input_data=self.test_input_data,
        #     fitting_guide=None, fail_gracefully=False, preprocessing_stage='infer'
        # )
        #
        # # TODO: Switch from below direct calls to preprocessor.execute_pipeline() call
        # # TODO: After calling execute_pipeline(), set data attributes to their counterparts in preprocessor class
        # preprocessor.data_imputation()
        # preprocessor.target_data_transformation()
        # preprocessor.data_scaling()
        #
        # for dataset_name in preprocessor.all_input_sets + preprocessor.all_target_sets:
        #     old_val, new_val = getattr(self, dataset_name), getattr(preprocessor, dataset_name)
        #     G.log('Dataset: "{}" {} updated'.format(dataset_name, 'was not' if old_val.equals(new_val) else 'was'))
        #     setattr(self, dataset_name, new_val)

        self.train_input_data = self.train_dataset.copy().loc[:, self.feature_selector]
        self.train_target_data = self.train_dataset.copy().loc[:, self.target_column]

        if isinstance(self.holdout_dataset, pd.DataFrame):
            self.holdout_input_data = self.holdout_dataset.copy().loc[:, self.feature_selector]
            self.holdout_target_data = self.holdout_dataset.copy().loc[:, self.target_column]

        if isinstance(self.test_dataset, pd.DataFrame):
            self.test_input_data = self.test_dataset.copy().loc[:, self.feature_selector]

        G.log("Initial preprocessing stage complete")

    ##################################################
    # Supporting Methods:
    ##################################################
    def _validate_parameters(self):
        """Ensure provided input parameters are properly formatted"""
        #################### target_metric ####################
        self.target_metric = get_formatted_target_metric(self.target_metric, self.metrics_map)

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
            G.log(f"Validated Environment with key: '{self.cross_experiment_key}'")
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
        G.log("")
        G.log("Initialized new Experiment with ID: {}".format(self.experiment_id))

    def _generate_hyperparameter_key(self):
        """Set :attr:`hyperparameter_key` to a key to describe the experiment's hyperparameters"""
        parameters = dict(
            model_initializer=self.model_initializer,
            model_init_params=self.model_init_params,
            model_extra_params=self.model_extra_params,
            preprocessing_pipeline=self.preprocessing_pipeline,
            preprocessing_params=self.preprocessing_params,
            feature_selector=self.feature_selector,
            # FLAG: Should probably add :attr:`target_metric` to key - With option to ignore it?
        )

        self.hyperparameter_key = HyperparameterKeyMaker(parameters, self.cross_experiment_key)
        G.log("Generated hyperparameter key: {}".format(self.hyperparameter_key))

    def _create_script_backup(self):
        """Create and save a copy of the script that initialized the Experiment if allowed to, and
        if :attr:`source_script` ends with a ".py" extension"""
        #################### Attempt to Copy Source Script if Allowed ####################
        try:
            if not self.source_script.endswith(".py"):
                G.Env.result_paths["script_backup"] = None

            if G.Env.result_paths["script_backup"] is not None:
                try:
                    self._source_copy_helper()
                except FileNotFoundError:
                    os.makedirs(self.result_paths["script_backup"], exist_ok=False)
                    self._source_copy_helper()
                G.log("Created backup of file: '{}'".format(self.source_script))
            else:
                G.log("Skipped creating backup of file: '{}'".format(self.source_script))
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
                    self.cross_validation_params.get("n_repeats", 1),
                    self.cross_validation_params["n_splits"],
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
                G.log("Model has no random_state/seed parameter to update")
                # FLAG: HIGH PRIORITY BELOW
                # TODO: BELOW IS NOT THE CASE IF MODEL IS NN - SETTING THE GLOBAL RANDOM SEED DOES SOMETHING
                # TODO: If this is logged, there is no reason to execute multiple-run-averaging, so don't
                # TODO: ... Either 1) Set `runs` = 1 (this would mess with the environment key), or...
                # TODO: ... 2) Set the results of all subsequent runs to the results of the first run (this could be difficult)
                # FLAG: HIGH PRIORITY ABOVE
        except Exception as _ex:
            G.log("Failed to update model's random_state     {}".format(_ex.__repr__()))


class BaseCVExperiment(BaseExperiment):
    def __init__(
        self,
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_selector=None,
        preprocessing_pipeline=None,
        preprocessing_params=None,
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

        self.fold_train_input = None
        self.fold_validation_input = None
        self.fold_train_target = None
        self.fold_validation_target = None

        self.repetition_oof_predictions = None
        self.repetition_holdout_predictions = None
        self.repetition_test_predictions = None

        self.fold_holdout_predictions = None
        self.fold_test_predictions = None

        self.run_validation_predictions = None
        self.run_holdout_predictions = None
        self.run_test_predictions = None

        #################### Initialize Result Placeholders ####################
        # self.full_oof_predictions = None  # (n_repeats * runs) intermediate columns
        # self.full_test_predictions = 0  # (n_splits * n_repeats * runs) intermediate columns
        # self.full_holdout_predictions = 0  # (n_splits * n_repeats * runs) intermediate columns

        self.final_oof_predictions = None
        self.final_test_predictions = 0
        self.final_holdout_predictions = 0

        BaseExperiment.__init__(
            self,
            model_initializer,
            model_init_params,
            model_extra_params=model_extra_params,
            feature_selector=feature_selector,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_params=preprocessing_params,
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
        raise NotImplementedError()

    def execute(self):
        self.cross_validation_workflow()

    def cross_validation_workflow(self):
        """Execute workflow for cross-validation process, consisting of the following tasks:
        1) Create train and validation split indices for all folds, 2) Iterate through folds,
        performing `cv_fold_workflow` for each, 3) Average accumulated predictions over fold
        splits, 4) Evaluate final predictions, 5) Format final predictions to prepare for saving"""
        self.on_experiment_start()

        cv_indices = self.folds.split(self.train_input_data, self.train_target_data.iloc[:, 0])
        new_shape = (
            self.cross_validation_params.get("n_repeats", 1),
            self.cross_validation_params["n_splits"],
            2,
        )
        reshaped_indices = np.reshape(np.array(list(cv_indices)), new_shape)

        for self._rep, rep_indices in enumerate(reshaped_indices.tolist()):
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
        consisting of: 1) Log start, 2) Execute original tasks, 3) Split train/validation data"""
        super().on_fold_start()

        #################### Split Train and Validation Data ####################
        self.fold_train_input = self.train_input_data.iloc[self.train_index, :].copy()
        self.fold_validation_input = self.train_input_data.iloc[self.validation_index, :].copy()

        self.fold_train_target = self.train_target_data.iloc[self.train_index].copy()
        self.fold_validation_target = self.train_target_data.iloc[self.validation_index].copy()

    def cv_fold_workflow(self):
        """Execute workflow for individual fold, consisting of the following tasks: Execute
        overridden :meth:`on_fold_start` tasks, 2) Perform cv_run_workflow for each run, 3) Execute
        overridden :meth:`on_fold_end` tasks"""
        self.on_fold_start()
        # TODO: Call self.intra_cv_preprocessing() - Ensure the 4 fold input/target attributes (from on_fold_start) are changed

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
            train_input=self.fold_train_input,
            train_target=self.fold_train_target,
            validation_input=self.fold_validation_input,
            validation_target=self.fold_validation_target,
            do_predict_proba=self.do_predict_proba,
            target_metric=self.target_metric,
            metrics_map=self.metrics_map,
        )
        self.model.fit()
        self.on_run_end()


##################################################
# Core CV Experiment Classes:
##################################################
class CrossValidationExperiment(BaseCVExperiment, metaclass=ExperimentMeta):
    def __init__(
        self,
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_selector=None,
        preprocessing_pipeline=None,
        preprocessing_params=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
        BaseCVExperiment.__init__(
            self,
            model_initializer,
            model_init_params,
            model_extra_params=model_extra_params,
            feature_selector=feature_selector,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_params=preprocessing_params,
            notes=notes,
            do_raise_repeated=do_raise_repeated,
            auto_start=auto_start,
            target_metric=target_metric,
        )

    def _initialize_folds(self):
        """Set :attr:`folds` per cross_validation_type and :attr:`cross_validation_params`"""
        cross_validation_type = self.experiment_params["cross_validation_type"]  # Allow failure
        if not isclass(cross_validation_type):
            raise TypeError(f"Expected a cross-validation class, not {type(cross_validation_type)}")

        try:
            _split_method = getattr(cross_validation_type, "split")
            if not callable(_split_method):
                raise TypeError("`cross_validation_type` must implement a callable :meth:`split`")
        except AttributeError:
            raise AttributeError("`cross_validation_type` must be class with :meth:`split`")

        self.folds = cross_validation_type(**self.cross_validation_params)


##################################################
# Other Experiment Classes:
##################################################
class RepeatedCVExperiment(BaseCVExperiment, metaclass=ExperimentMeta):
    def __init__(
        self,
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_selector=None,
        preprocessing_pipeline=None,
        preprocessing_params=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
        BaseCVExperiment.__init__(
            self,
            model_initializer,
            model_init_params,
            model_extra_params=model_extra_params,
            feature_selector=feature_selector,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_params=preprocessing_params,
            notes=notes,
            do_raise_repeated=do_raise_repeated,
            auto_start=auto_start,
            target_metric=target_metric,
        )

    def _initialize_folds(self):
        """Initialize :attr:`folds` per cross_validation_type and :attr:`cross_validation_params`"""
        cross_validation_type = self.experiment_params.get(
            "cross_validation_type", "repeatedkfold"
        ).lower()
        if cross_validation_type in ("stratifiedkfold", "repeatedstratifiedkfold"):
            self.folds = RepeatedStratifiedKFold(**self.cross_validation_params)
        else:
            self.folds = RepeatedKFold(**self.cross_validation_params)


class StandardCVExperiment(BaseCVExperiment, metaclass=ExperimentMeta):
    def __init__(
        self,
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_selector=None,
        preprocessing_pipeline=None,
        preprocessing_params=None,
        notes=None,
        do_raise_repeated=False,
        auto_start=True,
        target_metric=None,
    ):
        BaseCVExperiment.__init__(
            self,
            model_initializer,
            model_init_params,
            model_extra_params=model_extra_params,
            feature_selector=feature_selector,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_params=preprocessing_params,
            notes=notes,
            do_raise_repeated=do_raise_repeated,
            auto_start=auto_start,
            target_metric=target_metric,
        )

    def _initialize_folds(self):
        """Initialize :attr:`folds` per cross_validation_type and :attr:`cross_validation_params`"""
        cross_validation_type = self.experiment_params.get("cross_validation_type", "kfold").lower()
        if cross_validation_type == "stratifiedkfold":
            self.folds = StratifiedKFold(**self.cross_validation_params)
        else:
            self.folds = KFold(**self.cross_validation_params)


# class NoValidationExperiment(BaseExperiment):
#     pass


if __name__ == "__main__":
    pass
