"""This module defines the base Optimization Protocol classes. The classes defined herein are not
intended for direct use, but are rather parent classes to those defined in
:mod:`hyperparameter_hunter.optimization`

Related
-------
:mod:`hyperparameter_hunter.optimization`
    Defines the optimization classes that are intended for direct use. All classes defined in
    :mod:`hyperparameter_hunter.optimization` should be descendants of
    :class:`optimization_core.BaseOptimizationProtocol`
:mod:`hyperparameter_hunter.result_reader`
    Used to locate result files for Experiments that are similar to the current optimization
    constraints, and produce data to learn from in the case of :class:`SKOptimizationProtocol`
:mod:`hyperparameter_hunter.space`
    Defines the child classes of `hyperparameter_hunter.space.Dimension`, which are used to define
    the hyperparameters to optimize
:mod:`hyperparameter_hunter.utils.optimization_utils`:
    Provides utility functions for locating saved Experiments that fit within the constraints
    currently being optimized, as well as :class:`AskingOptimizer`, which guides the search of
    :class:`optimization_core.SKOptimizationProtocol`"""
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
from hyperparameter_hunter.experiments import CrossValidationExperiment
from hyperparameter_hunter.library_helpers.keras_helper import reinitialize_callbacks
from hyperparameter_hunter.library_helpers.keras_optimization_helper import (
    keras_prep_workflow,
    link_choice_ids,
)
from hyperparameter_hunter.metrics import get_formatted_target_metric
from hyperparameter_hunter.reporting import OptimizationReporter
from hyperparameter_hunter.result_reader import finder_selector
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.space import Space, dimension_subset
from hyperparameter_hunter.utils.boltons_utils import get_path
from hyperparameter_hunter.utils.general_utils import deep_restricted_update
from hyperparameter_hunter.utils.optimization_utils import AskingOptimizer, get_choice_dimensions

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
from datetime import datetime
import inspect
import os

##################################################
# Import Learning Assets
##################################################
from skopt.callbacks import check_callback

# noinspection PyProtectedMember
from skopt.utils import cook_estimator, eval_callbacks

try:
    from keras import backend as K
except ImportError:
    K = None


class OptimizationProtocolMeta(type):
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
        setattr(
            cls,
            "source_script",
            os.path.abspath(inspect.getframeinfo(inspect.currentframe().f_back)[0]),
        )
        return super().__call__(*args, **kwargs)


class MergedOptimizationMeta(OptimizationProtocolMeta, ABCMeta):
    """Metaclass to combine :class:`OptimizationProtocolMeta`, and `ABCMeta`"""

    pass


class BaseOptimizationProtocol(metaclass=MergedOptimizationMeta):
    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
    ):
        """Base class for intermediate base optimization protocol classes

        Parameters
        ----------
        target_metric: Tuple, default=('oof', <first key in :attr:`environment.Environment.metrics_map`>)
            A path denoting the metric to be used to compare completed Experiments within the
            Optimization Protocol. The first value should be one of ['oof', 'holdout', 'in_fold'].
            The second value should be the name of a metric being recorded according to the values
            supplied in :attr:`environment.Environment.metrics_params`. See the documentation for
            :func:`metrics.get_formatted_target_metric` for more info. Any values returned by, or
            given as the `target_metric` input to, :func:`metrics.get_formatted_target_metric` are
            acceptable values for :attr:`BaseOptimizationProtocol.target_metric`
        iterations: Int, default=1
            The number of distinct experiments to execute
        verbose: Int 0, 1, or 2, default=1
            Verbosity mode for console logging. 0: Silent. 1: Show only logs from the Optimization
            Protocol. 2: In addition to logs shown when verbose=1, also show the logs from individual
            Experiments
        read_experiments: Boolean, default=True
            If True, all Experiment records that fit within the current
            :attr:`hyperparameter_space`, and are for the same :attr:`algorithm_name`, and match the
            current guidelines, will be read in and used to fit any optimizers
        reporter_parameters: Dict, or None, default={}
            Additional parameters passed to :meth:`reporting.OptimizationReporter.__init__`. Note:
            Unless provided explicitly, the key "highlight_max" will be added by default to
            `reporter_params`, with a value inferred from the `direction` of :attr:`target_metric`
            in `G.Env.metrics_map`. In nearly all cases, the "highlight_max" key should be ignored,
            as there are very few reasons to explicitly include it

        Notes
        -----
        By default, 'script_backup' for Experiments is blacklisted when executed within
        :class:`BaseOptimizationProtocol` since it would just repeatedly create copies of the same,
        unchanged file (this file). So don't expect any script_backup files for Experiments executed
        during optimization rounds"""
        #################### Optimization Protocol Parameters ####################
        self.target_metric = target_metric
        self.iterations = iterations
        self.verbose = verbose
        self.read_experiments = read_experiments
        self.reporter_parameters = reporter_parameters or {}

        #################### Experiment Guidelines ####################
        self.model_initializer = None
        self.model_init_params = None
        self.model_extra_params = None
        self.feature_selector = None
        self.preprocessing_pipeline = None
        self.preprocessing_params = None
        self.notes = None
        self.do_raise_repeated = True

        #################### Search Parameters ####################
        self.dimensions = []
        self.search_bounds = dict()

        self.hyperparameter_space = None
        self.similar_experiments = []
        self.best_experiment = None
        self.best_score = None
        self.successful_iterations = 0
        self.skipped_iterations = 0
        self.tested_keys = []
        self._search_space_size = None

        self.current_init_params = None
        self.current_extra_params = None

        #################### Identification Attributes ####################
        self.algorithm_name = None
        self.module_name = None
        self.current_experiment = None
        self.current_score = None

        #################### Keras-Specific Attributes ####################
        self.dummy_layers = []
        self.dummy_compile_params = dict()
        self.init_iter_attrs = []
        self.extra_iter_attrs = []

        self.logger = None
        self._preparation_workflow()

    ##################################################
    # Core Methods:
    ##################################################
    def set_experiment_guidelines(
        self,
        model_initializer,
        model_init_params,
        model_extra_params=None,
        feature_selector=None,
        preprocessing_pipeline=None,
        preprocessing_params=None,
        notes=None,
        do_raise_repeated=True,
    ):
        """Provide the arguments necessary to instantiate
        :class:`experiments.CrossValidationExperiment`. This method has the same signature as
        :meth:`experiments.BaseExperiment.__init__` except where noted

        Parameters
        ----------
        model_initializer: Class, or functools.partial, or class instance
            The algorithm class being used to initialize a model
        model_init_params: Dict, or object
            The dictionary of arguments given when creating a model instance with
            `model_initializer` via the `__init__` method of :class:`models.Model`. Any kwargs that
            are considered valid by the `__init__` method of `model_initializer` are
            valid in `model_init_params`
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

        Notes
        -----
        The `auto_start` kwarg is not available here because
        :meth:`BaseOptimizationProtocol._execute_experiment` sets it to False in order to check for
        duplicated keys before running the whole Experiment. This is the most notable difference
        between calling :meth:`set_experiment_guidelines` and instantiating
        :class:`experiments.CrossValidationExperiment`"""
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

        if self.do_raise_repeated is False:
            G.warn_(
                "WARNING: Setting `do_raise_repeated`=False will allow Experiments to be unnecessarily duplicated"
            )

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

    def set_dimensions(self):
        """Locate given hyperparameters that are `space` choice declarations and add them to
        :attr:`dimensions`"""
        all_dimension_choices = []

        #################### Remap Extra Objects ####################
        if self.module_name == "keras":
            from keras.callbacks import Callback as BaseKerasCallback

            self.extra_iter_attrs.append(
                lambda _path, _key, _value: isinstance(_value, BaseKerasCallback)
            )

        #################### Collect Choice Dimensions ####################
        init_dimension_choices = get_choice_dimensions(
            self.model_init_params, iter_attrs=self.init_iter_attrs
        )
        extra_dimension_choices = get_choice_dimensions(
            self.model_extra_params, iter_attrs=self.extra_iter_attrs
        )

        for (path, choice) in init_dimension_choices:
            choice._name = ("model_init_params",) + path
            all_dimension_choices.append(choice)

        for (path, choice) in extra_dimension_choices:
            choice._name = ("model_extra_params",) + path
            all_dimension_choices.append(choice)

        self.dimensions = all_dimension_choices

        if self.module_name == "keras":
            self.model_extra_params = link_choice_ids(
                self.dummy_layers,
                self.dummy_compile_params,
                self.model_extra_params,
                self.dimensions,
            )

    def go(self):
        """Begin hyperparameter optimization process after experiment guidelines have been set and
        search dimensions are in place. This process includes the following: setting the
        hyperparameter space; locating similar experiments to be used as learning material for
        :class:`SKOptimizationProtocol` s; and executing :meth:`_optimization_loop`, which
        actually sets off the Experiment execution process"""
        if self.model_initializer is None:
            raise ValueError("Experiment guidelines must be set before starting optimization")

        _reporter_params = dict(
            dict(highlight_max=G.Env.metrics_map[self.target_metric[-1]].direction == "max"),
            **self.reporter_parameters,
        )
        self.logger = OptimizationReporter([_.name for _ in self.dimensions], **_reporter_params)

        self.tested_keys = []
        self._set_hyperparameter_space()
        self._find_similar_experiments()

        loop_start_time = datetime.now()
        self._optimization_loop()
        loop_end_time = datetime.now()
        G.log_(f"Optimization loop completed in {loop_end_time - loop_start_time}")
        G.log_(f'Best score was {self.best_score} from Experiment "{self.best_experiment}"')

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
            except StopIteration:
                if len(self.similar_experiments) + len(self.tested_keys) >= self.search_space_size:
                    G.log_(f"Hyperparameter search space has been exhausted")
                    break
                # G.debug_(f'Re-initializing hyperparameter grid after testing {len(self.tested_keys)} keys')
                self._set_hyperparameter_space()
                continue

            self.logger.print_result(
                self.current_hyperparameters_list,
                self.current_score,
                experiment_id=self.current_experiment.experiment_id,
            )

            if (self.best_experiment is None) or (self.current_score > self.best_score):
                self.best_experiment = self.current_experiment.experiment_id
                self.best_score = self.current_score

            iteration += 1

    def _execute_experiment(self):
        """Instantiate and run a :class:`experiments.CrossValidationExperiment` after checking for
        duplicated keys

        Notes
        -----
        As described in the Notes of :meth:`BaseOptimizationProtocol.set_experiment_guidelines`, the
        `auto_start` kwarg of :meth:`experiments.CrossValidationExperiment.__init__` is set to False
        in order to check for duplicated keys"""
        self._update_current_hyperparameters()

        self.current_experiment = CrossValidationExperiment(
            model_initializer=self.model_initializer,
            model_init_params=self.current_init_params,
            model_extra_params=self.current_extra_params,
            feature_selector=self.feature_selector,
            preprocessing_pipeline=self.preprocessing_pipeline,
            preprocessing_params=self.preprocessing_params,
            notes=self.notes,
            do_raise_repeated=self.do_raise_repeated,
            auto_start=False,
        )

        self.current_experiment.preparation_workflow()

        # Future Hunter, if multi-cross_experiment_keys ever supported, this will be a problem. Should've fixed it earlier, dummy
        if self.current_experiment.hyperparameter_key.key not in self.tested_keys:
            self.tested_keys.append(self.current_experiment.hyperparameter_key.key)

        self.current_experiment.experiment_workflow()
        self.current_score = get_path(
            self.current_experiment.last_evaluation_results, self.target_metric
        )
        self.successful_iterations += 1
        self._clean_up_experiment()

    def _clean_up_experiment(self):
        """Perform any cleanup necessary after completion of an Experiment"""
        if self.module_name == "keras":
            K.clear_session()

    def _update_current_hyperparameters(self):
        """Update :attr:`current_init_params`, and :attr:`current_extra_params` according to the
        upcoming set of hyperparameters to be searched"""
        current_hyperparameters = self._get_current_hyperparameters().items()

        init_params = {
            _k[1:]: _v for _k, _v in current_hyperparameters if _k[0] == "model_init_params"
        }
        extra_params = {
            _k[1:]: _v for _k, _v in current_hyperparameters if _k[0] == "model_extra_params"
        }

        self.current_init_params = deep_restricted_update(
            self.model_init_params, init_params, iter_attrs=self.init_iter_attrs
        )
        self.current_extra_params = deep_restricted_update(
            self.model_extra_params, extra_params, iter_attrs=self.extra_iter_attrs
        )

        if (self.module_name == "keras") and ("callbacks" in self.current_extra_params):
            self.current_extra_params["callbacks"] = reinitialize_callbacks(
                self.current_extra_params["callbacks"]
            )

    ##################################################
    # Abstract Methods:
    ##################################################
    @abstractmethod
    def _set_hyperparameter_space(self):
        """Initialize :attr:`hyperparameter_space` according to the provided hyperparameter search
        dimensions"""
        raise NotImplementedError()

    @abstractmethod
    def _get_current_hyperparameters(self):
        """Retrieve the upcoming set of hyperparameters to be searched

        Returns
        -------
        current_hyperparameters: Dict
            The next set of hyperparameters that will be searched"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def search_space_size(self):
        """The number of different hyperparameter permutations possible given the current
        hyperparameter search space"""
        raise NotImplementedError()

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
            self.target_metric, G.Env.metrics_map, default_dataset="oof"
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

        self.logger.print_saved_results_header()

        model_params = dict(
            model_init_params=self.model_init_params,
            model_extra_params=self.model_extra_params,
            preprocessing_pipeline=self.preprocessing_pipeline,
            preprocessing_params=self.preprocessing_params,
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
            self.hyperparameter_space,
            G.Env.result_paths["global_leaderboard"],
            G.Env.result_paths["description"],
            model_params,
        )
        experiment_finder.find()
        self.similar_experiments = experiment_finder.similar_experiments

    def _update_verbosity(self):
        """Update the contents of :attr:`environment.Environment.reporting_handler_params` if
        required by :attr:`verbose`"""
        #################### Mute non-critical console logging for Experiments ####################
        if self.verbose in [0, 1]:
            G.Env.reporting_handler_params.setdefault("console_params", {})["level"] = "CRITICAL"

        #################### Blacklist 'script_backup' ####################
        G.Env.result_paths["script_backup"] = None


class SKOptimizationProtocol(BaseOptimizationProtocol, metaclass=ABCMeta):
    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        #################### Optimizer Class Parameters ####################
        base_estimator="GP",
        n_initial_points=10,
        acquisition_function="gp_hedge",
        acquisition_optimizer="auto",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        #################### Minimizer Parameters ####################
        n_random_starts=10,
        callbacks=None,
        #################### Other Parameters ####################
        base_estimator_kwargs=None,
    ):
        """Base class for SKOpt-based Optimization Protocols

        Parameters
        ----------
        target_metric: Tuple, default=('oof', <first key in :attr:`environment.Environment.metrics_map`>)
            A path denoting the metric to be used to compare completed Experiments within the
            Optimization Protocol. The first value should be one of ['oof', 'holdout', 'in_fold'].
            The second value should be the name of a metric being recorded according to the values
            supplied in :attr:`environment.Environment.metrics_params`. See the documentation for
            :func:`metrics.get_formatted_target_metric` for more info; any values returned by, or
            used as the `target_metric` input to this function are acceptable values for
            :attr:`BaseOptimizationProtocol.target_metric`
        iterations: Int, default=1
            The number of distinct experiments to execute
        verbose: Int 0, 1, or 2, default=1
            Verbosity mode for console logging. 0: Silent. 1: Show only logs from the Optimization
            Protocol. 2: In addition to logs shown when verbose=1, also show the logs from
            individual Experiments
        read_experiments: Boolean, default=True
            If True, all Experiment records that fit within the current
            :attr:`hyperparameter_space`, and are for the same :attr:`algorithm_name`, and match the
            current guidelines, will be read in and used to fit any optimizers
        reporter_parameters: Dict, or None, default=None
            Additional parameters passed to :meth:`reporting.OptimizationReporter.__init__`
        base_estimator: String in ['GP', 'GBRT', 'RF', 'ET', 'DUMMY'], or an `sklearn` regressor, default='GP'
            If one of the above strings, a default model of that type will be used. Else, should
            inherit from :class:`sklearn.base.RegressorMixin`, and its :meth:`predict` should have
            an optional `return_std` argument, which returns `std(Y | x)`, along with `E[Y | x]`
        n_initial_points: Int, default=10
            The number of complete evaluation points necessary before allowing Experiments to be
            approximated with `base_estimator`. Any valid Experiment records found will count as
            initialization points. If enough Experiment records are not found, additional points
            will be randomly sampled
        acquisition_function: String in ['LCB', 'EI', 'PI', 'gp_hedge'], default='gp_hedge'
            Function to minimize over the posterior distribution. 'LCB': lower confidence bound.
            'EI': negative expected improvement. 'PI': negative probability of improvement.
            'gp_hedge': Probabilistically choose one of the preceding three acquisition functions at
            each iteration
        acquisition_optimizer: String in ['sampling', 'lbfgs', 'auto'], default='auto'
            Method to minimize the acquisition function. The fit model is updated with the optimal
            value obtained by optimizing `acquisition_function` with `acquisition_optimizer`.
            'sampling': optimize by computing `acquisition_function` at
            `acquisition_optimizer_kwargs['n_points']` randomly sampled points. 'lbfgs': optimize by
            sampling `n_restarts_optimizer` random points, then run 'lbfgs' for 20 iterations with
            those points to find local minima, the optimal of which is used to update the prior.
            'auto': configure on the basis of `base_estimator` and `dimensions`
        random_state: Int, `RandomState` instance, or None, default=None
            Set to something other than None for reproducible results
        acquisition_function_kwargs: Dict, or None, default=dict(xi=0.01, kappa=1.96)
            Additional arguments passed to the acquisition function
        acquisition_optimizer_kwargs: Dict, or None, default=dict(n_points=10000, n_restarts_optimizer=5, n_jobs=1)
            Additional arguments passed to the acquisition optimizer
        n_random_starts: Int, default=10
            The number of Experiments to execute with random points before checking that
            `n_initial_points` have been evaluated
        callbacks: Callable, list of callables, or None, default=[]
            If callable, then `callbacks(self.optimizer_result)` is called after each update to
            :attr:`optimizer`. If list, then each callable is called
        base_estimator_kwargs: Dict, or None, default={}
            Additional arguments passed to `base_estimator` when it is initialized

        Notes
        -----
        To provide initial input points for evaluation, individual Experiments can be executed prior
        to instantiating an Optimization Protocol. The results of these Experiments will
        automatically be detected and cherished by the optimizer.

        :class:`.SKOptimizationProtocol` and its children in :mod:`.optimization` rely heavily
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
        self.acquisition_optimizer_kwargs = dict(n_points=10000, n_restarts_optimizer=5, n_jobs=1)

        self.acquisition_function_kwargs.update(acquisition_function_kwargs or {})
        self.acquisition_optimizer_kwargs.update(acquisition_optimizer_kwargs or {})

        #################### Minimizer Parameters ####################
        # TODO: n_random_starts does nothing currently - Fix that
        self.n_random_starts = n_random_starts
        self.callbacks = callbacks or []

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
        )

    def _set_hyperparameter_space(self):
        """Initialize :attr:`hyperparameter_space` according to the provided hyperparameter search
        dimensions, and :attr:`base_estimator` and :attr:`optimizer`"""
        self.hyperparameter_space = Space(dimensions=self.dimensions)
        self._prepare_estimator()
        self._build_optimizer()

    def _prepare_estimator(self):
        """Initialize :attr:`base_estimator` with :attr:`hyperparameter_space` and any other kwargs,
        using `skopt.utils.cook_estimator`"""
        self.base_estimator = cook_estimator(
            self.base_estimator, space=self.hyperparameter_space, **self.base_estimator_kwargs
        )

    def _build_optimizer(self):
        """Set :attr:`optimizer` to the optimizing class used to both estimate the utility of sets
        of hyperparameters by learning from executed Experiments, and suggest points at which the
        objective should be evaluated"""
        self.optimizer = AskingOptimizer(
            dimensions=self.hyperparameter_space,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            acq_optimizer=self.acquisition_optimizer,
            random_state=self.random_state,
            acq_func_kwargs=self.acquisition_function_kwargs,
            acq_optimizer_kwargs=self.acquisition_optimizer_kwargs,
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
        if G.Env.metrics_map[self.target_metric[-1]].direction == "max":
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
        _current_hyperparameters = self.optimizer.ask()

        if _current_hyperparameters == self.current_hyperparameters_list:
            new_parameters = self.hyperparameter_space.rvs(random_state=None)[0]
            G.debug_("REPEATED  asked={}  new={}".format(_current_hyperparameters, new_parameters))
            _current_hyperparameters = new_parameters

        self.current_hyperparameters_list = _current_hyperparameters

        current_hyperparameters = dict(
            zip(
                self.hyperparameter_space.names(use_location=False),
                self.current_hyperparameters_list,
            )
        )

        return current_hyperparameters

    def _find_similar_experiments(self):
        """After locating similar experiments by way of the parent's
        :meth:`_find_similar_experiments`, fit :attr:`optimizer` with the hyperparameters and
        results of each located experiment"""
        super()._find_similar_experiments()

        # TODO: Remove below reversal of `similar_experiments` when `result_reader.ResultFinder.sort` finished
        for _i, _experiment in enumerate(self.similar_experiments[::-1]):
            _hyperparameters = dimension_subset(_experiment[0], self.hyperparameter_space.names())
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
            Infinity will be returned if any of the following constraints are met: 1) the
            hyperparameter dimensions include any real-valued boundaries, 2) the boundaries include
            values that are neither categorical nor integer, or 3) the search space size is
            otherwise incalculable"""
        if self._search_space_size is None:
            self._search_space_size = len(self.hyperparameter_space)
        return self._search_space_size


if __name__ == "__main__":
    pass
