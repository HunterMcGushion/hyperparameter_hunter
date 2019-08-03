"""This module defines the base Optimization Protocol classes. The classes defined herein are not
intended for direct use, but are rather parent classes to those defined in
:mod:`hyperparameter_hunter.optimization.backends.skopt.protocols`

Related
-------
:mod:`hyperparameter_hunter.optimization.backends.skopt.protocols`
    Defines the optimization classes that are intended for direct use. All classes defined in
    :mod:`hyperparameter_hunter.optimization.backends.skopt.protocols` should be descendants of
    :class:`~hyperparameter_hunter.optimization.protocol_core.BaseOptPro`
:mod:`hyperparameter_hunter.result_reader`
    Used to locate result files for Experiments that are similar to the current optimization
    constraints, and produce data to learn from in the case of :class:`SKOptPro`
:mod:`hyperparameter_hunter.space`
    Defines the child classes of `hyperparameter_hunter.space.Dimension`, which are used to define
    the hyperparameters to optimize
:mod:`hyperparameter_hunter.utils.optimization_utils`:
    Provides utility functions for locating saved Experiments that fit within the constraints
    currently being optimized"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import __version__
from hyperparameter_hunter.algorithm_handlers import (
    identify_algorithm,
    identify_algorithm_hyperparameters,
)
from hyperparameter_hunter.exceptions import (
    EnvironmentInactiveError,
    EnvironmentInvalidError,
    RepeatedExperimentError,
    DeprecatedWarning,
)
from hyperparameter_hunter.experiments import CVExperiment
from hyperparameter_hunter.feature_engineering import FeatureEngineer
from hyperparameter_hunter.library_helpers.keras_helper import reinitialize_callbacks
from hyperparameter_hunter.library_helpers.keras_optimization_helper import (
    keras_prep_workflow,
    link_choice_ids,
)
from hyperparameter_hunter.metrics import get_formatted_target_metric
from hyperparameter_hunter.optimization.backends.skopt.engine import Optimizer, cook_estimator
from hyperparameter_hunter.reporting import OptimizationReporter
from hyperparameter_hunter.result_reader import finder_selector
from hyperparameter_hunter.settings import G, TEMP_MODULES_DIR_PATH
from hyperparameter_hunter.space.dimensions import RejectedOptional
from hyperparameter_hunter.space.space_core import Space
from hyperparameter_hunter.utils.boltons_utils import get_path
from hyperparameter_hunter.utils.general_utils import deep_restricted_update, subdict
from hyperparameter_hunter.utils.optimization_utils import get_choice_dimensions, dimension_subset
from hyperparameter_hunter.utils.version_utils import Deprecated

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
from datetime import datetime
from inspect import currentframe, getframeinfo
from os import walk, remove, rmdir
from os.path import abspath
from typing import Any, Dict
from warnings import warn

##################################################
# Import Learning Assets
##################################################
from skopt.callbacks import check_callback

# noinspection PyProtectedMember
from skopt.utils import eval_callbacks

try:
    from keras import backend as K
except ImportError:
    K = None


class OptProMeta(type):
    """Metaclass to accurately set :attr:`source_script` for its descendants even if the original
    call was the product of scripts calling other scripts that eventually instantiated an
    optimization protocol"""

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        """Prepare `namespace` to include :attr:`source_script`"""
        namespace = dict(source_script=None)
        return namespace

    def __call__(cls, *args, **kwargs):
        """Set the instance's :attr:`source_script` to the absolute path of the file that
        instantiated the OptimizationProtocol"""
        setattr(cls, "source_script", abspath(getframeinfo(currentframe().f_back)[0]))
        return super().__call__(*args, **kwargs)


class MergedOptProMeta(OptProMeta, ABCMeta):
    """Metaclass to combine :class:`OptProMeta`, and `ABCMeta`"""

    pass


class BaseOptPro(metaclass=MergedOptProMeta):
    source_script: str

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
    ):
        """Base class for intermediate base optimization protocol classes

        There are two important methods for all :class:`BaseOptPro` descendants that should be
        invoked after initialization:

        1. :meth:`~hyperparameter_hunter.optimization.protocol_core.BaseOptPro.forge_experiment`
        2. :meth:`~hyperparameter_hunter.optimization.protocol_core.BaseOptPro.go`

        Parameters
        ----------
        target_metric: Tuple, default=("oof", <:attr:`environment.Environment.metrics`[0]>)
            Rarely necessary to explicitly provide this, as the default is usually sufficient. Path
            denoting the metric to be used to compare Experiment performance. The first value
            should be one of ["oof", "holdout", "in_fold"]. The second value should be the name of
            a metric being recorded according to :attr:`environment.Environment.metrics_params`.
            See the documentation for :func:`metrics.get_formatted_target_metric` for more info.
            Any values returned by, or given as the `target_metric` input to,
            :func:`~hyperparameter_hunter.metrics.get_formatted_target_metric` are acceptable
            values for :attr:`BaseOptPro.target_metric`
        iterations: Int, default=1
            Number of Experiments to conduct during optimization upon invoking :meth:`BaseOptPro.go`
        verbose: {0, 1, 2}, default=1
            Verbosity mode for console logging. 0: Silent. 1: Show only logs from the Optimization
            Protocol. 2: In addition to logs shown when verbose=1, also show the logs from
            individual Experiments
        read_experiments: Boolean, default=True
            If True, all Experiment records that fit in the current :attr:`space` and guidelines,
            and match :attr:`algorithm_name`, will be read in and used to fit any optimizers
        reporter_parameters: Dict, or None, default={}
            Additional parameters passed to :meth:`reporting.OptimizationReporter.__init__`. Note:
            Unless provided explicitly, the key "do_maximize" will be added by default to
            `reporter_params`, with a value inferred from the `direction` of :attr:`target_metric`
            in `G.Env.metrics`. In nearly all cases, the "do_maximize" key should be ignored,
            as there are very few reasons to explicitly include it
        warn_on_re_ask: Boolean, default=False
            If True, and the internal `optimizer` recommends a point that has already been evaluated
            on invocation of `ask`, a warning is logged before recommending a random point. Either
            way, a random point is used instead of already-evaluated recommendations. However,
            logging the fact that this has taken place can be useful to indicate that the optimizer
            may be stalling, especially if it repeatedly recommends the same point. In these cases,
            if the suggested point is not optimal, it can be helpful to switch a different OptPro
            (especially `DummyOptPro`), which will suggest points using different criteria

        Methods
        -------
        forge_experiment
            Define constraints on Experiments conducted by OptPro (like hyperparameter search space)
        go
            Start optimization

        Notes
        -----
        By default, 'script_backup' for Experiments is blacklisted when executed within
        :class:`BaseOptPro` since it would just repeatedly create copies of the same, unchanged
        file. So don't expect any script_backup files for Experiments executed by OptPros"""
        #################### Optimization Protocol Parameters ####################
        self.target_metric = target_metric
        self.iterations = iterations
        self.verbose = verbose
        self.read_experiments = read_experiments
        self.reporter_parameters = reporter_parameters or {}
        self.warn_on_re_ask = warn_on_re_ask

        #################### Experiment Guidelines ####################
        self.model_initializer = None
        self.model_init_params = None
        self.model_extra_params = None
        self.feature_engineer = None
        self.feature_selector = None
        self.notes = None
        self.do_raise_repeated = True

        #################### Search Parameters ####################
        self.dimensions = []
        self.search_bounds = dict()

        self.space = None
        self.similar_experiments = []
        self.best_experiment = None
        self.best_score = None
        self.successful_iterations = 0
        self.skipped_iterations = 0
        self.tested_keys = []
        self._search_space_size = None

        #################### Incumbent Hyperparameters ####################
        self.current_init_params = None
        self.current_extra_params = None
        self.current_feature_engineer = None

        #################### Identification Attributes ####################
        self.algorithm_name = None
        self.module_name = None
        self.current_experiment = None
        self.current_score = None

        #################### Keras-Specific Attributes ####################
        self.dummy_layers = []
        self.dummy_compile_params = dict()

        #################### Secret Sauce ####################
        self.init_iter_attrs = []
        self.extra_iter_attrs = []
        self.fe_iter_attrs = [lambda p, k, v: isinstance(v, FeatureEngineer)]

        self.logger = None
        self._preparation_workflow()
        self.do_maximize = G.Env.metrics[self.target_metric[-1]].direction == "max"

    ##################################################
    # Core Methods:
    ##################################################
    def forge_experiment(
        self,
        model_initializer,
        model_init_params=None,
        model_extra_params=None,
        feature_engineer=None,
        feature_selector=None,
        notes=None,
        do_raise_repeated=True,
    ):
        """Define hyperparameter search scaffold for building Experiments during optimization

        OptPros use this method to guide Experiment construction behind the scenes, which is why it
        looks just like :meth:`hyperparameter_hunter.experiments.BaseExperiment.__init__`.
        `forge_experiment` offers one major upgrade to standard Experiment initialization: it
        accepts hyperparameters not only as concrete values, but also as space choices -- using
        :class:`~hyperparameter_hunter.space.dimensions.Real`,
        :class:`~hyperparameter_hunter.space.dimensions.Integer`, and
        :class:`~hyperparameter_hunter.space.dimensions.Categorical`. This functionality applies to
        the `model_init_params`, `model_extra_params` and `feature_engineer` kwargs. Any Dimensions
        provided to `forge_experiment` are detected by the OptPro and used to define the
        hyperparameter search space to be optimized

        Parameters
        ----------
        model_initializer: Class, or functools.partial, or class instance
            Algorithm class used to initialize a model, such as XGBoost's `XGBRegressor`, or
            SKLearn's `KNeighborsClassifier`; although, there are hundreds of possibilities across
            many different ML libraries. `model_initializer` is expected to define at least `fit`
            and `predict` methods. `model_initializer` will be initialized with `model_init_params`,
            and its extra methods (`fit`, `predict`, etc.) will be invoked with parameters in
            `model_extra_params`
        model_init_params: Dict, or object (optional)
            Dictionary of arguments given to create an instance of `model_initializer`. Any kwargs
            that are considered valid by the `__init__` method of `model_initializer` are valid in
            `model_init_params`.

            In addition to providing concrete values, hyperparameters can be expressed as choices
            (dimensions to optimize) by using instances of
            :class:`~hyperparameter_hunter.space.dimensions.Real`,
            :class:`~hyperparameter_hunter.space.dimensions.Integer`, or
            :class:`~hyperparameter_hunter.space.dimensions.Categorical`. Furthermore,
            hyperparameter choices and concrete values can be used together in `model_init_params`.

            Using XGBoost's `XGBClassifier` to illustrate, the `model_init_params` kwarg of
            :class:`~hyperparameter_hunter.experiments.CVExperiment` is limited to using concrete
            values, such as ``dict(max_depth=10, learning_rate=0.1, booster="gbtree")``. This is
            still valid for :meth:`.forge_experiment`. However, :meth:`.forge_experiment` also
            allows `model_init_params` to consist entirely of space choices, such as
            ``dict(max_depth=Integer(2, 20), learning_rate=Real(0.001, 0.5),
            booster=Categorical(["gbtree", "dart"]))``, or as any combination of concrete values
            and choices, for instance, ``dict(max_depth=10, learning_rate=Real(0.001, 0.5),
            booster="gbtree")``.

            One of the key features that makes HyperparameterHunter so magical is that **ALL**
            hyperparameters in the signature of `model_initializer` (and their default values) are
            discovered -- whether or not they are explicitly given in `model_init_params`. Not only
            does this make Experiment result descriptions incredibly thorough, it also makes
            optimization smoother, more effective, and far less work for the user. For example, take
            LightGBM's `LGBMRegressor`, with `model_init_params`=`dict(learning_rate=0.2)`.
            HyperparameterHunter recognizes that this differs from the default of 0.1. It also
            recognizes that `LGBMRegressor` is actually initialized with more than a dozen other
            hyperparameters we didn't bother mentioning, and it records their values, too. So if we
            want to optimize `num_leaves` tomorrow, the OptPro doesn't start from scratch. It knows
            that we ran an Experiment that didn't explicitly mention `num_leaves`, but its default
            value was 31, and it uses this information to fuel optimization -- all without us having
            to manually keep track of tons of janky collections of hyperparameters. In fact, we
            really don't need to go out of our way at all. HyperparameterHunter just acts as our
            faithful lab assistant, keeping track of all the stuff we'd rather not worry about
        model_extra_params: Dict (optional)
            Dictionary of extra parameters for models' non-initialization methods (like `fit`,
            `predict`, `predict_proba`, etc.), and for neural networks. To specify parameters for
            an extra method, place them in a dict named for the extra method to which the
            parameters should be given. For example, to call `fit` with `early_stopping_rounds`=5,
            use `model_extra_params`=`dict(fit=dict(early_stopping_rounds=5))`.

            Declaring hyperparameter space choices works identically to `model_init_params`, meaning
            that in addition to concrete values, extra parameters can be given as instances of
            :class:`~hyperparameter_hunter.space.dimensions.Real`,
            :class:`~hyperparameter_hunter.space.dimensions.Integer`, or
            :class:`~hyperparameter_hunter.space.dimensions.Categorical`. To optimize over a space
            in which `early_stopping_rounds` is between 3 and 9, use
            `model_extra_params`=`dict(fit=dict(early_stopping_rounds=Real(3, 9)))`.

            For models whose `fit` methods have a kwarg like `eval_set` (such as XGBoost's), one can
            use the `DatasetSentinel` attributes of the current active
            :class:`~hyperparameter_hunter.environment.Environment`, documented under its
            "Attributes" section and under
            :attr:`~hyperparameter_hunter.environment.Environment.train_input`. An example using
            several DatasetSentinels can be found in HyperparameterHunter's
            [XGBoost Classification Example](https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples/xgboost_examples/classification.py)
        feature_engineer: `FeatureEngineer`, or list (optional)
            Feature engineering/transformation/pre-processing steps to apply to datasets defined in
            :class:`~hyperparameter_hunter.environment.Environment`. If list, will be used to
            initialize :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`, and can
            contain any of the following values:

                1. :class:`~hyperparameter_hunter.feature_engineering.EngineerStep` instance
                2. Function input to :class:~hyperparameter_hunter.feature_engineering.EngineerStep`
                3. :class:`~hyperparameter_hunter.space.dimensions.Categorical`, with `categories`
                   comprising a selection of the previous two values (optimization only)

            For important information on properly formatting `EngineerStep` functions, please see
            the documentation of :class:`~hyperparameter_hunter.feature_engineering.EngineerStep`.

            To search a space optionally including an `EngineerStep`, use the `optional` kwarg of
            :class:`~hyperparameter_hunter.space.dimensions.Categorical`. This functionality is
            illustrated in :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`. If
            using a `FeatureEngineer` instance to optimize `feature_engineer`, this instance cannot
            be used with `CVExperiment` because Experiments can't handle space choices
        feature_selector: List of str, callable, or list of booleans (optional)
            Column names to include as input data for all provided DataFrames. If None,
            `feature_selector` is set to all columns in :attr:`train_dataset`, less
            :attr:`target_column`, and :attr:`id_column`. `feature_selector` is provided as the
            second argument for calls to `pandas.DataFrame.loc` when constructing datasets
        notes: String (optional)
            Additional information about the Experiment that will be saved with the Experiment's
            description result file. This serves no purpose other than to facilitate saving
            Experiment details in a more readable format
        do_raise_repeated: Boolean, default=False
            If True and this Experiment locates a previous Experiment's results with matching
            Environment and Hyperparameter Keys, a RepeatedExperimentError will be raised. Else, a
            warning will be logged

        Notes
        -----
        The `auto_start` kwarg is not available here because :meth:`._execute_experiment` sets it
        to False in order to check for duplicated keys before running the whole Experiment. This
        and `target_metric` being moved to :meth:`.__init__` are the most notable differences
        between calling :meth:`forge_experiment` and instantiating
        :class:`~hyperparameter_hunter.experiments.CVExperiment`

        A more accurate name for this method might be something like "build_experiment_forge", since
        `forge_experiment` itself does not actually execute any Experiments. However,
        `forge_experiment` sounds cooler and much less clunky

        See Also
        --------
        :class:`hyperparameter_hunter.experiments.BaseExperiment`
            One-off experimentation counterpart to an OptPro's :meth:`.forge_experiment`.
            Internally, OptPros feed the processed arguments from `forge_experiment` to initialize
            Experiments. This hand-off to Experiments takes place in :meth:`._execute_experiment`
        """
        self.model_initializer = model_initializer
        self.model_init_params = identify_algorithm_hyperparameters(self.model_initializer)
        model_init_params = model_init_params if model_init_params is not None else {}
        try:
            self.model_init_params.update(model_init_params)
        except TypeError:
            self.model_init_params.update(dict(build_fn=model_init_params))

        self.model_extra_params = model_extra_params if model_extra_params is not None else {}

        self.feature_engineer = feature_engineer
        if not isinstance(self.feature_engineer, FeatureEngineer):
            self.feature_engineer = FeatureEngineer(self.feature_engineer)

        self.feature_selector = feature_selector if feature_selector is not None else []

        self.notes = notes
        self.do_raise_repeated = do_raise_repeated

        if self.do_raise_repeated is False:
            G.warn_("WARNING: Setting `do_raise_repeated`=False allows duplicated Experiments")

        self.algorithm_name, self.module_name = identify_algorithm(self.model_initializer)
        self._validate_guidelines()

        #################### Deal with Keras ####################
        if self.module_name == "keras":
            reusable_build_fn, reusable_wrapper_params, dummy_layers, dummy_compile_params = keras_prep_workflow(
                self.model_initializer,
                self.model_init_params["build_fn"],
                self.model_extra_params,
                self.source_script,
            )
            self.model_init_params = dict(build_fn=reusable_build_fn)
            self.model_extra_params = reusable_wrapper_params
            self.dummy_layers = dummy_layers
            self.dummy_compile_params = dummy_compile_params
            # FLAG: Deal with capitalization conflicts when comparing similar experiments: `optimizer`='Adam' vs 'adam'

        self.set_dimensions()

    @Deprecated(
        v_deprecate="3.0.0a2",
        v_remove="3.2.0",
        v_current=__version__,
        details="Renamed to `forge_experiment`",
    )
    def set_experiment_guidelines(self, *args, **kwargs):
        self.forge_experiment(*args, **kwargs)

    def set_dimensions(self):
        """Locate given hyperparameters that are `space` choice declarations and add them to
        :attr:`dimensions`"""
        all_dimension_choices = []

        #################### Remap Extra Objects ####################
        if self.module_name == "keras":
            from keras.initializers import Initializer as KerasInitializer
            from keras.callbacks import Callback as KerasCB

            self.init_iter_attrs.append(lambda _p, _k, _v: isinstance(_v, KerasInitializer))
            self.extra_iter_attrs.append(lambda _p, _k, _v: isinstance(_v, KerasCB))

        #################### Collect Choice Dimensions ####################
        init_dim_choices = get_choice_dimensions(self.model_init_params, self.init_iter_attrs)
        extra_dim_choices = get_choice_dimensions(self.model_extra_params, self.extra_iter_attrs)
        fe_dim_choices = get_choice_dimensions(self.feature_engineer, self.fe_iter_attrs)

        for (path, choice) in init_dim_choices:
            choice._name = ("model_init_params",) + path
            all_dimension_choices.append(choice)

        for (path, choice) in extra_dim_choices:
            choice._name = ("model_extra_params",) + path
            all_dimension_choices.append(choice)

        for (path, choice) in fe_dim_choices:
            choice._name = ("feature_engineer",) + path
            all_dimension_choices.append(choice)

        self.dimensions = all_dimension_choices

        if self.module_name == "keras":
            self.model_extra_params = link_choice_ids(
                self.dummy_layers,
                self.dummy_compile_params,
                self.model_extra_params,
                self.dimensions,
            )

    def get_ready(self):
        """Prepare for optimization by finalizing hyperparameter space and identifying similar
        Experiments. This method is automatically invoked when :meth:`go` is called if necessary"""
        if self.model_initializer is None:
            raise ValueError("Must invoke `forge_experiment` before starting optimization")

        _reporter_params = dict(dict(do_maximize=self.do_maximize), **self.reporter_parameters)
        self.logger = OptimizationReporter(self.dimensions, **_reporter_params)

        self.tested_keys = []
        self._set_hyperparameter_space()
        self._find_similar_experiments()

    def go(self, force_ready=True):
        """Execute hyperparameter optimization, building an Experiment for each iteration

        This method may only be invoked after invoking :meth:`.forge_experiment`, which defines
        experiment guidelines and search dimensions. `go` performs a few important tasks: 1)
        Formally setting the hyperparameter space; 2) Locating similar experiments to be used as
        learning material (for OptPros that suggest incumbent search points by estimating utilities
        using surrogate models); and 3) Actually setting off the optimization process, via
        :meth:`._optimization_loop`

        Parameters
        ----------
        force_ready: Boolean, default=False
            If True, :meth:`get_ready` will be invoked even if it has already been called. This will
            re-initialize the hyperparameter `space` and `similar_experiments`. Standard behavior is
            for :meth:`go` to invoke :meth:`get_ready`, so `force_ready` is ignored unless
            :meth:`get_ready` has been manually invoked"""
        if force_ready or self.space is None:
            self.get_ready()

        loop_start_time = datetime.now()
        self._optimization_loop()
        loop_end_time = datetime.now()
        G.log_(f"Optimization loop completed in {loop_end_time - loop_start_time}")
        G.log_(f'Best score was {self.best_score} from Experiment "{self.best_experiment}"')
        self._clean_up_optimization()

    ##################################################
    # Helper Methods:
    ##################################################
    def _optimization_loop(self, iteration=0):
        """Perform Experiment execution loop while `iteration` < `iterations`. At each iteration, an
        Experiment will be executed, its results will be logged, and it will be compared to the
        current best experiment

        Parameters
        ----------
        iteration: Int, default=0
            The current iteration in the optimization loop"""
        self.logger.print_optimization_header()

        while iteration < self.iterations:
            try:
                self._execute_experiment()
            except RepeatedExperimentError:
                # G.debug_(F'Skipping repeated Experiment: {_ex!s}\n')
                if len(self.similar_experiments) + len(self.tested_keys) >= self.search_space_size:
                    G.log_(f"Hyperparameter search space has been exhausted")
                    break
                self.skipped_iterations += 1
                continue
            # NOTE: If reimplementing grid search, like `UninformedOptimizationProtocol`, add
            #   `except StopIteration` and see this commit, and 9b7ca73 / e2c3b73 (October 25, 2018)

            self.logger.print_result(
                self.current_hyperparameters_list,
                self.current_score,
                experiment_id=self.current_experiment.experiment_id,
            )

            #################### Update Best Experiment ####################
            if (
                (self.best_experiment is None)  # First evaluation
                or (self.do_maximize and (self.best_score < self.current_score))  # New best max
                or (not self.do_maximize and (self.best_score > self.current_score))  # New best min
            ):
                self.best_experiment = self.current_experiment.experiment_id
                self.best_score = self.current_score

            iteration += 1

    def _execute_experiment(self):
        """Instantiate and run a :class:`experiments.CVExperiment` after checking for duplicate keys

        Notes
        -----
        As described in the Notes of :meth:`BaseOptPro.forge_experiment`, the
        `auto_start` kwarg of :meth:`experiments.CVExperiment.__init__` is set to False in order to
        check for duplicated keys"""
        self._update_current_hyperparameters()

        #################### Initialize Experiment (Without Running) ####################
        self.current_experiment = CVExperiment(
            model_initializer=self.model_initializer,
            model_init_params=self.current_init_params,
            model_extra_params=self.current_extra_params,
            feature_engineer=self.current_feature_engineer,
            feature_selector=self.feature_selector,  # TODO: Add `current_feature_selector`
            notes=self.notes,
            do_raise_repeated=self.do_raise_repeated,
            auto_start=False,
        )
        # Fix `current_experiment.source_script` - Probably saying "protocol_core", which is a lie
        self.current_experiment.source_script = self.source_script

        #################### Run Experiment ####################
        self.current_experiment.preparation_workflow()
        self.current_experiment.experiment_workflow()
        # If above raised `RepeatedExperimentError`, it is caught by :meth:`_optimization_loop`,
        #   stopping this method before it can incorrectly update `tested_keys` below

        # Future Hunter, if multi-cross_experiment_keys ever supported, this will be a problem. Should've fixed it earlier, dummy
        if self.current_experiment.hyperparameter_key.key not in self.tested_keys:
            self.tested_keys.append(self.current_experiment.hyperparameter_key.key)

        self.current_score = get_path(
            self.current_experiment.last_evaluation_results, self.target_metric
        )
        self.successful_iterations += 1
        self._clean_up_experiment()

    @staticmethod
    def _clean_up_optimization():
        """Perform any cleanup necessary after completion of the optimization loop. Most notably,
        this handles removal of temporary model files created for Keras optimization"""
        for (root, dirs, files) in walk(TEMP_MODULES_DIR_PATH, topdown=False):
            for file in files:
                if file.startswith("__temp_"):
                    remove(f"{root}/{file}")
            try:
                rmdir(root)
            except OSError:
                G.warn_(f"Unidentified file found in temporary directory: {root}")

    def _clean_up_experiment(self):
        """Perform any cleanup necessary after completion of an Experiment"""
        if self.module_name == "keras":
            K.clear_session()

    @staticmethod
    def _select_params(param_prefix: str, current_params: Dict[tuple, Any]):
        """Retrieve sub-dict of `current_params` whose keys start with `param_prefix`

        Parameters
        ----------
        param_prefix: {"model_init_params", "model_extra_params", "feature_engineer"}
            Target to filter keys of `current_params`. Only key/value pairs in `current_params`
            whose keys start with `param_prefix` are returned. `param_prefix` is dropped from the
            keys in the returned dict
        current_params: Dict[tuple, Any]
            Dict from which to return the subset whose keys start with `param_prefix`

        Returns
        -------
        Dict[tuple, Any]
            Contents of `current_params`, whose keys started with `param_prefix` - with
            `param_prefix` dropped from the resulting keys"""
        return {k[1:]: v for k, v in current_params if k[0] == param_prefix}

    def _update_current_hyperparameters(self):
        """Update :attr:`current_init_params`, and :attr:`current_extra_params` according to the
        upcoming set of hyperparameters to be searched"""
        current_hyperparameters = self._get_current_hyperparameters().items()

        init_params = self._select_params("model_init_params", current_hyperparameters)
        extra_params = self._select_params("model_extra_params", current_hyperparameters)
        fe_params = self._select_params("feature_engineer", current_hyperparameters)

        # TODO: Add `fs_params` for `current_feature_selector`

        # FLAG: At this point, `dummy_layers` shows "kernel_initializer" as `orthogonal` instance with "__hh" attrs
        # FLAG: HOWEVER, the `orthogonal` instance does have `gain` set to the correct dummy value, ...
        # FLAG: ... so it might be ok, as long as experiment matching can still work with that

        self.current_init_params = deep_restricted_update(
            self.model_init_params, init_params, iter_attrs=self.init_iter_attrs
        )
        self.current_extra_params = deep_restricted_update(
            self.model_extra_params, extra_params, iter_attrs=self.extra_iter_attrs
        )
        self.current_feature_engineer = deep_restricted_update(
            self.feature_engineer, fe_params, iter_attrs=self.fe_iter_attrs
        )
        # TODO: Add `current_feature_selector`

        #################### Initialize `current_feature_engineer` ####################
        current_fe = subdict(self.current_feature_engineer, keep=["steps", "do_validate"])
        current_fe["steps"] = [_ for _ in current_fe["steps"] if _ != RejectedOptional()]
        self.current_feature_engineer = FeatureEngineer(**current_fe)

        #################### Deal with Keras ####################
        if (self.module_name == "keras") and ("callbacks" in self.current_extra_params):
            self.current_extra_params["callbacks"] = reinitialize_callbacks(
                self.current_extra_params["callbacks"]
            )

        # No need to reinitialize Keras `initializers` - Their values are passed to `build_fn` via extra `params`

    ##################################################
    # Abstract Methods:
    ##################################################
    @abstractmethod
    def _set_hyperparameter_space(self):
        """Initialize :attr:`space` according to the provided hyperparameter search dimensions"""

    @abstractmethod
    def _get_current_hyperparameters(self):
        """Retrieve the upcoming set of hyperparameters to be searched

        Returns
        -------
        current_hyperparameters: Dict
            The next set of hyperparameters that will be searched"""

    @property
    @abstractmethod
    def search_space_size(self):
        """The number of different hyperparameter permutations possible given the current
        hyperparameter search space"""

    ##################################################
    # Utility Methods:
    ##################################################
    def _preparation_workflow(self):
        """Perform housekeeping tasks to prepare for core functionality like validating the
        `Environment` and parameters, and updating the verbosity of individual Experiments"""
        self._validate_environment()
        self._validate_parameters()
        self._update_verbosity()

    @staticmethod
    def _validate_environment():
        """Check that there is a currently active and unoccupied Environment instance"""
        if G.Env is None:
            raise EnvironmentInactiveError()
        if G.Env.current_task is None:
            G.log_(f'Validated Environment with key: "{G.Env.cross_experiment_key}"')
        else:
            raise EnvironmentInvalidError("Must finish current task before starting a new one")

    def _validate_parameters(self):
        """Ensure provided input parameters are properly formatted"""
        self.target_metric = get_formatted_target_metric(
            self.target_metric, G.Env.metrics, default_dataset="oof"
        )

    def _validate_guidelines(self):
        """Ensure provided Experiment guideline parameters are properly formatted"""
        target_column = G.Env.target_column
        id_column = G.Env.id_column
        train_dataset = G.Env.train_dataset.copy()

        self.feature_selector = self.feature_selector or train_dataset.columns.values
        restricted_cols = [_ for _ in target_column + [id_column] if _ is not None]
        self.feature_selector = [_ for _ in self.feature_selector if _ not in restricted_cols]

    def _find_similar_experiments(self):
        """Look for Experiments that were performed under similar conditions (algorithm and
        cross-experiment parameters)"""
        if self.read_experiments is False:
            return

        model_params = dict(
            model_init_params=self.model_init_params,
            model_extra_params=self.model_extra_params,
            feature_engineer=self.feature_engineer,
            feature_selector=self.feature_selector,
        )

        if self.module_name == "keras":
            model_params["model_init_params"]["layers"] = self.dummy_layers
            model_params["model_init_params"]["compile_params"] = self.dummy_compile_params

        experiment_finder = finder_selector(self.module_name)(
            self.algorithm_name,
            self.module_name,
            G.Env.cross_experiment_key,
            self.target_metric,
            self.space,
            G.Env.result_paths["global_leaderboard"],
            G.Env.result_paths["description"],
            model_params,
        )
        experiment_finder.find()
        self.similar_experiments = experiment_finder.similar_experiments
        self.logger.print_saved_results_header()

    def _update_verbosity(self):
        """Update :attr:`environment.Environment.reporting_params` if required by :attr:`verbose`"""
        #################### Mute non-critical console logging for Experiments ####################
        if self.verbose in [0, 1]:
            G.Env.reporting_params.setdefault("console_params", {})["level"] = "CRITICAL"

        #################### Blacklist 'script_backup' ####################
        G.Env.result_paths["script_backup"] = None


class SKOptPro(BaseOptPro, metaclass=ABCMeta):
    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
        #################### Optimizer Class Parameters ####################
        base_estimator="GP",
        n_initial_points=10,
        acquisition_function="gp_hedge",
        acquisition_optimizer="auto",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        #################### Minimizer Parameters ####################
        n_random_starts="DEPRECATED",
        callbacks=None,
        #################### Other Parameters ####################
        base_estimator_kwargs=None,
    ):
        """Base class for SKOpt-based Optimization Protocols

        There are two important methods for all :class:`SKOptPro` descendants that should be
        invoked after initialization:

        1. :meth:`~hyperparameter_hunter.optimization.protocol_core.BaseOptPro.forge_experiment`
        2. :meth:`~hyperparameter_hunter.optimization.protocol_core.BaseOptPro.go`

        Parameters
        ----------
        target_metric: Tuple, default=("oof", <:attr:`environment.Environment.metrics`[0]>)
            Rarely necessary to explicitly provide this, as the default is usually sufficient. Path
            denoting the metric to be used to compare Experiment performance. The first value
            should be one of ["oof", "holdout", "in_fold"]. The second value should be the name of
            a metric being recorded according to :attr:`environment.Environment.metrics_params`.
            See the documentation for :func:`metrics.get_formatted_target_metric` for more info.
            Any values returned by, or given as the `target_metric` input to,
            :func:`~hyperparameter_hunter.metrics.get_formatted_target_metric` are acceptable
            values for :attr:`BaseOptPro.target_metric`
        iterations: Int, default=1
            Number of Experiments to conduct during optimization upon invoking :meth:`BaseOptPro.go`
        verbose: {0, 1, 2}, default=1
            Verbosity mode for console logging. 0: Silent. 1: Show only logs from the Optimization
            Protocol. 2: In addition to logs shown when verbose=1, also show the logs from
            individual Experiments
        read_experiments: Boolean, default=True
            If True, all Experiment records that fit in the current :attr:`space` and guidelines,
            and match :attr:`algorithm_name`, will be read in and used to fit any optimizers
        reporter_parameters: Dict, or None, default=None
            Additional parameters passed to :meth:`reporting.OptimizationReporter.__init__`. Note:
            Unless provided explicitly, the key "do_maximize" will be added by default to
            `reporter_params`, with a value inferred from the `direction` of :attr:`target_metric`
            in `G.Env.metrics`. In nearly all cases, the "do_maximize" key should be ignored,
            as there are very few reasons to explicitly include it
        warn_on_re_ask: Boolean, default=False
            If True, and the internal `optimizer` recommends a point that has already been evaluated
            on invocation of `ask`, a warning is logged before recommending a random point. Either
            way, a random point is used instead of already-evaluated recommendations. However,
            logging the fact that this has taken place can be useful to indicate that the optimizer
            may be stalling, especially if it repeatedly recommends the same point. In these cases,
            if the suggested point is not optimal, it can be helpful to switch a different OptPro
            (especially `DummyOptPro`), which will suggest points using different criteria

        Other Parameters
        ----------------
        base_estimator: {SKLearn Regressor, "GP", "RF", "ET", "GBRT", "DUMMY"}, default="GP"
            If not string, should inherit from `sklearn.base.RegressorMixin`. In addition, the
            `predict` method should have an optional `return_std` argument, which returns
            `std(Y | x)`, along with `E[Y | x]`.

            If `base_estimator` is a string in {"GP", "RF", "ET", "GBRT", "DUMMY"}, a surrogate
            model corresponding to the relevant `X_minimize` function is created
        n_initial_points: Int, default=10
            Number of complete evaluation points necessary before allowing Experiments to be
            approximated with `base_estimator`. Any valid Experiment records found will count as
            initialization points. If enough Experiment records are not found, additional points
            will be randomly sampled
        acquisition_function:{"LCB", "EI", "PI", "gp_hedge"}, default="gp_hedge"
            Function to minimize over the posterior distribution. Can be any of the following:

            * "LCB": Lower confidence bound
            * "EI": Negative expected improvement
            * "PI": Negative probability of improvement
            * "gp_hedge": Probabilistically choose one of the above three acquisition functions at
              every iteration

                * The gains `g_i` are initialized to zero
                * At every iteration,

                    * Each acquisition function is optimised independently to propose a candidate
                      point `X_i`
                    * Out of all these candidate points, the next point `X_best` is chosen by
                      `softmax(eta g_i)`
                    * After fitting the surrogate model with `(X_best, y_best)`, the gains are
                      updated such that `g_i -= mu(X_i)`
        acquisition_optimizer: {"sampling", "lbfgs", "auto"}, default="auto"
            Method to minimize the acquisition function. The fit model is updated with the optimal
            value obtained by optimizing `acq_func` with `acq_optimizer`

            * "sampling": `acq_func` is optimized by computing `acq_func` at `n_initial_points`
              randomly sampled points.
            * "lbfgs": `acq_func` is optimized by

                  * Randomly sampling `n_restarts_optimizer` (from `acq_optimizer_kwargs`) points
                  * "lbfgs" is run for 20 iterations with these initial points to find local minima
                  * The optimal of these local minima is used to update the prior

            * "auto": `acq_optimizer` is configured on the basis of the `base_estimator` and the
              search space. If the space is `Categorical` or if the provided estimator is based on
              tree-models, then this is set to "sampling"
        random_state: Int, `RandomState` instance, or None, default=None
            Set to something other than None for reproducible results
        acquisition_function_kwargs: Dict, or None, default=dict(xi=0.01, kappa=1.96)
            Additional arguments passed to the acquisition function
        acquisition_optimizer_kwargs: Dict, or None, default=dict(n_points=10000, n_restarts_optimizer=5, n_jobs=1)
            Additional arguments passed to the acquisition optimizer
        n_random_starts: ...
            .. deprecated:: 3.0.0
                Use `n_initial_points`, instead. Will be removed in 3.2.0
        callbacks: Callable, list of callables, or None, default=[]
            If callable, then `callbacks(self.optimizer_result)` is called after each update to
            :attr:`optimizer`. If list, then each callable is called
        base_estimator_kwargs: Dict, or None, default={}
            Additional arguments passed to `base_estimator` when it is initialized

        Methods
        -------
        forge_experiment
            Define constraints on Experiments conducted by OptPro (like hyperparameter search space)
        go
            Start optimization

        Notes
        -----
        To provide initial input points for evaluation, individual Experiments can be executed prior
        to instantiating an Optimization Protocol. The results of these Experiments will
        automatically be detected and cherished by the optimizer.

        :class:`.SKOptPro` and its children in :mod:`.optimization` rely heavily
        on the utilities provided by the `Scikit-Optimize` library, so thank you to the creators and
        contributors for their excellent work."""
        # TODO: Add 'EIps', and 'PIps' to the allowable `acquisition_function` values - Will need to return execution times

        #################### Optimizer Parameters ####################
        self.base_estimator = base_estimator
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.acquisition_optimizer = acquisition_optimizer
        self.random_state = random_state

        self.acquisition_function_kwargs = dict(xi=0.01, kappa=1.96)
        self.acquisition_function_kwargs.update(acquisition_function_kwargs or {})
        self.acquisition_optimizer_kwargs = dict(n_points=10000, n_restarts_optimizer=5, n_jobs=1)
        self.acquisition_optimizer_kwargs.update(acquisition_optimizer_kwargs or {})

        self.callbacks = callbacks or []

        if n_random_starts != "DEPRECATED":
            self.n_initial_points = n_random_starts
            warn(DeprecatedWarning("n_random_starts", "3.0.0", "3.2.0", "Use `n_initial_points`"))

        #################### Other Parameters ####################
        self.base_estimator_kwargs = base_estimator_kwargs or {}

        #################### Placeholder Attributes ####################
        self.optimizer = None
        self.optimizer_result = None
        self.current_hyperparameters_list = None

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
            warn_on_re_ask=warn_on_re_ask,
        )

    def _set_hyperparameter_space(self):
        """Initialize :attr:`space` according to the provided hyperparameter search dimensions, and
        :attr:`base_estimator`, and :attr:`optimizer`"""
        self.space = Space(dimensions=self.dimensions)
        self._prepare_estimator()
        self._build_optimizer()

    def _prepare_estimator(self):
        """Initialize :attr:`base_estimator` with :attr:`space` via `skopt.utils.cook_estimator`"""
        self.base_estimator = cook_estimator(
            self.base_estimator, space=self.space, **self.base_estimator_kwargs
        )

    def _build_optimizer(self):
        """Set :attr:`optimizer` to the optimizing class used to both estimate the utility of sets
        of hyperparameters by learning from executed Experiments, and suggest points at which the
        objective should be evaluated"""
        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            acq_optimizer=self.acquisition_optimizer,
            random_state=self.random_state,
            acq_func_kwargs=self.acquisition_function_kwargs,
            acq_optimizer_kwargs=self.acquisition_optimizer_kwargs,
            warn_on_re_ask=self.warn_on_re_ask,
        )

    def _update_optimizer(self, hyperparameters, score, fit=True):
        """Record an observation (or set of observations) of the objective function

        To add observations without fitting a new model, `fit`=False. To add multiple observations
        in a batch, pass a list-of-lists for `hyperparameters` and a list of scalars for `score`

        Parameters
        ----------
        hyperparameters: List:
            Hyperparameter values in the search space at which the objective function was evaluated
        score: Number, list
            Value of the objective function at `hyperparameters` in the hyperparameter space
        fit: Boolean, default=True
            Fit a model to observed evaluations of the objective. Regardless of `fit`, a model will
            only be fitted after telling :attr:`n_initial_points` points to :attr:`optimizer`"""
        if self.do_maximize:
            score = -score
        self.optimizer_result = self.optimizer.tell(hyperparameters, score, fit=fit)

    def _execute_experiment(self):
        """After executing parent's :meth:`_execute_experiment`, fit :attr:`optimizer` with the set
        of hyperparameters that were used, and the utility of those hyperparameters"""
        super()._execute_experiment()
        self._update_optimizer(self.current_hyperparameters_list, self.current_score)
        if eval_callbacks(self.callbacks, self.optimizer_result):
            return

    def _get_current_hyperparameters(self):
        """Ask :attr:`optimizer` for the upcoming set of hyperparameters that should be searched,
        then format them to be used in the next Experiment

        Returns
        -------
        current_hyperparameters: Dict
            The next set of hyperparameters that will be searched"""
        self.current_hyperparameters_list = self.optimizer.ask()

        current_hyperparameters = dict(
            zip(self.space.names(use_location=False), self.current_hyperparameters_list)
        )

        return current_hyperparameters

    def _find_similar_experiments(self):
        """After locating similar experiments by way of the parent's
        :meth:`_find_similar_experiments`, fit :attr:`optimizer` with the hyperparameters and
        results of each located experiment"""
        super()._find_similar_experiments()

        # TODO: Remove below reversal of `similar_experiments` when `result_reader.ResultFinder.sort` finished
        for _i, _experiment in enumerate(self.similar_experiments[::-1]):
            _hyperparameters = dimension_subset(_experiment[0], self.space.names())
            _evaluation = _experiment[1]
            _experiment_id = _experiment[2] if len(_experiment) > 2 else None
            self.logger.print_result(_hyperparameters, _evaluation, experiment_id=_experiment_id)
            self._update_optimizer(_hyperparameters, _evaluation)

            # self.optimizer_result = self.optimizer.tell(
            #     _hyperparameters, _evaluation, fit=(_i == len(self.similar_experiments) - 1))

            if eval_callbacks(self.callbacks, self.optimizer_result):
                return self.optimizer_result
            # FLAG: Could wrap above `tell` call in try/except, then attempt `_tell` with improper dimensions

    def _validate_parameters(self):
        """Ensure provided input parameters are properly formatted"""
        super()._validate_parameters()

        #################### callbacks ####################
        self.callbacks = check_callback(self.callbacks)

    @property
    def search_space_size(self):
        """The number of different hyperparameter permutations possible given the current
        hyperparameter search dimensions

        Returns
        -------
        :attr:`_search_space_size`: Int, or `numpy.inf`
            Infinity returned if any of the following constraints are met: 1) the hyperparameter
            dimensions include any real-valued boundaries, 2) the boundaries include values that are
            neither categorical nor integer, or 3) search space size is otherwise incalculable"""
        if self._search_space_size is None:
            self._search_space_size = len(self.space)
        return self._search_space_size


if __name__ == "__main__":
    pass
