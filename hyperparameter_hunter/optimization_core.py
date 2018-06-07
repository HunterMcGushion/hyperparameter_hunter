##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.algorithm_handlers import identify_algorithm, identify_algorithm_hyperparameters
from hyperparameter_hunter.exception_handler import EnvironmentInactiveError, EnvironmentInvalidError, RepeatedExperimentError
from hyperparameter_hunter.experiments import CrossValidationExperiment
from hyperparameter_hunter.metrics import get_formatted_target_metric
from hyperparameter_hunter.reporting import OptimizationReporter
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.space import Space, dimension_subset
from hyperparameter_hunter.utils.boltons_utils import get_path, PathAccessError
from hyperparameter_hunter.utils.general_utils import deep_restricted_update
from hyperparameter_hunter.utils.optimization_utils import get_ids_by, get_scored_params, filter_by_space, filter_by_guidelines
from hyperparameter_hunter.utils.optimization_utils import AskingOptimizer

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd

##################################################
# Import Learning Assets
##################################################
from sklearn.model_selection import StratifiedKFold

from skopt.callbacks import check_callback
# noinspection PyProtectedMember
from skopt.utils import cook_estimator, eval_callbacks


class BaseOptimizationProtocol(metaclass=ABCMeta):
    def __init__(self, target_metric=None, iterations=1, verbose=1, read_experiments=True, reporter_parameters=None):
        """Base class for :class:`InformedOptimizationProtocol`, and :class:`UninformedOptimizationProtocol`

        Parameters
        ----------
        target_metric: Tuple, default=('oof', <first key in :attr:`environment.Environment.metrics_map`>)
            A path denoting the metric to be used to compare completed Experiments within the Optimization Protocol. The first
            value should be one of ['oof', 'holdout', 'in_fold']. The second value should be the name of a metric being recorded
            according to the values supplied in :attr:`environment.Environment.metrics_params`. See the documentation for
            :func:`metrics.get_formatted_target_metric` for more info; any values returned by, or used as the `target_metric`
            input to this function are acceptable values for :attr:`BaseOptimizationProtocol.target_metric`
        iterations: Int, default=1
            The number of distinct experiments to execute
        verbose: Int 0, 1, or 2, default=1
            Verbosity mode for console logging. 0: Silent. 1: Show only logs from the Optimization Protocol. 2: In addition to
            logs shown when verbose=1, also show the logs from individual Experiments
        read_experiments: Boolean, default=True
            If True, all Experiment records that fit within the current :attr:`hyperparameter_space`, and are for the same
            :attr:`algorithm_name`, and match the current guidelines, will be read in and used to fit any optimizers
        reporter_parameters: Dict, or None, default=None
            Additional parameters passed to :meth:`reporting.OptimizationReporter.__init__`

        Examples
        --------
        >>> from hyperparameter_hunter.environment import Environment
        >>> from xgboost import XGBClassifier
        >>> env = Environment(
        ...     train_dataset=pd.DataFrame(),
        ...     root_results_path='./HyperparameterHunterAssets',
        ...     metrics_map=dict(roc='roc_auc_score'),
        ...     cross_validation_type=StratifiedKFold,
        ...     cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
        ... )
        >>> optimizer = RandomizedGridSearch(target_metric=('oof', 'roc'))
        >>> optimizer.set_experiment_guidelines(
        ...     model_initializer=XGBClassifier,
        ...     model_init_params=dict(objective='reg:linear', learning_rate=0.1, subsample=0.5)
        ... )
        >>> optimizer.add_init_selection('max_depth', [2, 3, 4, 5])
        >>> optimizer.add_init_selection(('n_estimators', ), [90, 100, 110])
        >>> optimizer.go()

        Notes
        -----
        By default, 'script_backup' for Experiments is blacklisted when executed within :class:`BaseOptimizationProtocol` since
        it would just repeatedly create copies of the same, unchanged file (this file). So don't expect any script_backup files
        for Experiments executed during optimization rounds"""
        #################### Optimization Protocol Parameters ####################
        self.target_metric = target_metric
        self.iterations = iterations
        self.verbose = verbose
        self.read_experiments = read_experiments
        self.reporter_parameters = reporter_parameters or {}  # FLAG: PROBABLY DOESN'T NEED TO BE AN ATTRIBUTE

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
        self.search_bounds = dict()
        self.search_rules = []

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

        self._preparation_workflow()

        self.logger = OptimizationReporter(
            [_.name for _ in self.dimensions] if hasattr(self, 'dimensions') else [],
            **self.reporter_parameters
            # verbose=1
        )

    ##################################################
    # Core Methods:
    ##################################################
    def set_experiment_guidelines(
            self, model_initializer, model_init_params, model_extra_params=None, feature_selector=None,
            preprocessing_pipeline=None, preprocessing_params=None, notes=None, do_raise_repeated=True,
    ):
        """Provide the arguments necessary to instantiate :class:`experiments.CrossValidationExperiment`. This method has the same
        signature as :meth:`experiments.BaseExperiment.__init__` except where noted

        Parameters
        ----------
        model_initializer: See :class:`experiments.BaseExperiment`
        model_init_params: See :class:`experiments.BaseExperiment`
        model_extra_params: See :class:`experiments.BaseExperiment`
        feature_selector: See :class:`experiments.BaseExperiment`
        preprocessing_pipeline: See :class:`experiments.BaseExperiment`
        preprocessing_params: See :class:`experiments.BaseExperiment`
        notes: See :class:`experiments.BaseExperiment`
        do_raise_repeated: See :class:`experiments.BaseExperiment`

        Notes
        -----
        The `auto_start` kwarg is not available here because :meth:`BaseOptimizationProtocol._execute_experiment` sets it to False
        in order to check for duplicated keys before running the whole Experiment. This is the most notable difference between
        calling :meth:`set_experiment_guidelines` and instantiating :class:`experiments.CrossValidationExperiment`"""
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
            G.warn_('WARNING: Setting `do_raise_repeated`=False will allow Experiments to be unnecessarily duplicated')

        self.algorithm_name, self.module_name = identify_algorithm(self.model_initializer)

        if self.module_name == 'keras':
            raise ValueError('Sorry, Hyperparameter Optimization algorithms are not yet equipped to work with Keras. Stay tuned')

    def go(self):
        """Begin hyperparameter optimization process after experiment guidelines have been set and search dimensions are in place.
        This process includes the following: setting the hyperparameter space; locating similar experiments to be used as
        learning material for :class:`InformedOptimizationProtocol` s; and executing :meth:`_optimization_loop`, which actually
        sets off the Experiment execution process"""
        if self.model_initializer is None:
            raise ValueError('Experiment guidelines and options must be set before hyperparameter optimization can be started')

        self.tested_keys = []
        self._set_hyperparameter_space()
        self._find_similar_experiments()

        loop_start_time = datetime.now()
        self._optimization_loop()
        loop_end_time = datetime.now()
        G.log_(F'Optimization loop completed in {loop_end_time - loop_start_time}')
        G.log_(F'Best score was {self.best_score} from Experiment "{self.best_experiment}"')

    ##################################################
    # Helper Methods:
    ##################################################
    def _optimization_loop(self, iteration=0):
        self.logger.print_optimization_header()

        while iteration < self.iterations:
            # print(iteration)
            try:
                # print(F'try         {iteration}')
                self._execute_experiment()
                # print(F'success     {iteration}')
            except RepeatedExperimentError:
                # print(F'repeated    {iteration}')
                # G.debug_(F'Skipping repeated Experiment: {_ex!s}\n')
                self.skipped_iterations += 1
                continue
            except StopIteration:
                # print(F'stopped     {iteration}')
                if len(self.tested_keys) >= self.search_space_size:
                    G.log_(F'Hyperparameter search space has been exhausted after testing {len(self.tested_keys)} keys')
                    break
                # G.debug_(F'Re-initializing hyperparameter grid after testing {len(self.tested_keys)} keys')
                # print(F'resetting     {iteration}')
                self._set_hyperparameter_space()
                continue

            # print(F'after     {iteration}')
            # FLAG: TEST BELOW - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            # FLAG: TEST BELOW - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            # FLAG: TEST BELOW - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            # FLAG: TEST BELOW - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            self.logger.print_result(self.current_hyperparameters_list, self.current_score)  # FLAG: TEST
            # G.log_(F'Iteration {iteration}, Experiment "{self.current_experiment.experiment_id}": {self.current_score}')  # FLAG: ORIGINAL
            # FLAG: TEST ABOVE - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            # FLAG: TEST ABOVE - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            # FLAG: TEST ABOVE - :attr:`current_hyperparameters_list` only exists in Informed Protocols
            # FLAG: TEST ABOVE - :attr:`current_hyperparameters_list` only exists in Informed Protocols

            if (self.best_experiment is None) or (self.current_score > self.best_score):
                self.best_experiment = self.current_experiment.experiment_id
                self.best_score = self.current_score

            iteration += 1

    def _execute_experiment(self):
        """Instantiate and run a :class:`experiments.CrossValidationExperiment` after checking for duplicated keys

        Notes
        -----
        As described in the Notes of :meth:`BaseOptimizationProtocol.set_experiment_guidelines`, the `auto_start` kwarg of
        :meth:`experiments.CrossValidationExperiment.__init__` is set to False in order to check for duplicated keys"""
        self._update_current_hyperparameters()

        self.current_experiment = CrossValidationExperiment(
            model_initializer=self.model_initializer, model_init_params=self.current_init_params,
            model_extra_params=self.current_extra_params, feature_selector=self.feature_selector,
            preprocessing_pipeline=self.preprocessing_pipeline, preprocessing_params=self.preprocessing_params, notes=self.notes,
            do_raise_repeated=self.do_raise_repeated, auto_start=False
        )

        self.current_experiment.preparation_workflow()

        if self.current_experiment.hyperparameter_key.key not in self.tested_keys:
            # If support for multiple cross_experiment_keys is ever added, this will be a problem
            self.tested_keys.append(self.current_experiment.hyperparameter_key.key)

        self.current_experiment.experiment_workflow()
        self.current_score = get_path(self.current_experiment.last_evaluation_results, self.target_metric)
        self.successful_iterations += 1

    def _update_current_hyperparameters(self):
        current_hyperparameters = self._get_current_hyperparameters()

        init_params = {_k[1:]: _v for _k, _v in current_hyperparameters.items() if _k[0] == 'model_init_params'}
        extra_params = {_k[1:]: _v for _k, _v in current_hyperparameters.items() if _k[0] == 'model_extra_params'}

        self.current_init_params = deep_restricted_update(self.model_init_params, init_params)
        self.current_extra_params = deep_restricted_update(self.model_extra_params, extra_params)

    ##################################################
    # Abstract Methods:
    ##################################################
    @abstractmethod
    def _set_hyperparameter_space(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_current_hyperparameters(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def search_space_size(self):
        raise NotImplementedError()

    ##################################################
    # Hyperparameter Search Declaration Methods:
    ##################################################
    def add_init_range(self, hyperparameter, bounds):
        hyperparameter = (hyperparameter,) if not isinstance(hyperparameter, tuple) else hyperparameter
        self._add_option((('model_init_params',) + hyperparameter), 'range', bounds)

    def add_init_selection(self, hyperparameter, selection):
        hyperparameter = (hyperparameter,) if not isinstance(hyperparameter, tuple) else hyperparameter
        self._add_option((('model_init_params',) + hyperparameter), 'selection', selection)

    def add_extra_range(self, hyperparameter, bounds):
        hyperparameter = (hyperparameter,) if not isinstance(hyperparameter, tuple) else hyperparameter
        self._add_option((('model_extra_params',) + hyperparameter), 'range', bounds)

    def add_extra_selection(self, hyperparameter, selection):
        hyperparameter = (hyperparameter,) if not isinstance(hyperparameter, tuple) else hyperparameter
        self._add_option((('model_extra_params',) + hyperparameter), 'selection', selection)

    def add_default_options(self, hyperparameter):
        # TODO: Add default acceptable selections/ranges for algorithms' hyperparameters and add "default" kwargs to "add" methods
        # TODO: Check :attr:`module_name`'s library_helper for :attr:`model_initializer` for a default `hyperparameter` list
        # FLAG: If above fails, raise below ValueError
        raise ValueError(F'Could not find default options in {self.algorithm_name} for the hyperparameter "{hyperparameter}"')

    def _add_option(self, hyperparameter, bound_type, choices):
        """Declare a hyperparameter as a variable that is allowed to be changed during the optimization process

        Parameters
        ----------
        hyperparameter: Tuple
            Path location of the hyperparameter for which choices are being provided. Should start with the containing attribute's
            name ('model_init_params', or 'model_extra_params'). Subsequent items will be looked up in order within the attribute
        bound_type: String in: ['range', 'selection']
            How to interpret the contents of `choices`. If 'range', `choices` should contain two numbers: a lower bound, and an
            upper bound. If 'selection', `choices` should contain all the permitted values for `hyperparameter`
        choices: List
            The permitted values for `hyperparameter`. If `bound_type` == 'range', list of two numbers: a lower bound, and an
            upper bound. If `bound_type` == 'selection', a list of all permitted values for `hyperparameter`"""
        if self.model_initializer is None:
            raise ValueError('Experiment guidelines must be set before adding hyperparameter options')

        #################### Validate Hyperparameter Path ####################
        if not self._is_valid_hyperparameter_path(hyperparameter):
            _err_message = F'Received nonexistent hyperparameter path: {hyperparameter}. '
            if not hasattr(self, hyperparameter[0]):
                _err_message += F'"{hyperparameter[0]}" is not an attribute of :class:`BaseOptimizationProtocol`'
            raise KeyError(_err_message)

        if hyperparameter[0] not in ['model_init_params', 'model_extra_params']:
            raise KeyError(F'"{hyperparameter}" is not tunable. Please review guidelines on tunable hyperparameters')

        #################### Validate Bound Type and Choices ####################
        if not isinstance(choices, (list, tuple)):
            raise TypeError(F'`choices` must be a list. Received {type(choices)}: {choices}')

        valid_bound_types = ['range', 'selection']
        if bound_type not in valid_bound_types:
            raise ValueError(F'Received invalid `bound_type`: "{bound_type}". Expected string in: {valid_bound_types}')

        if bound_type == 'range':
            if not all([isinstance(_, (int, float)) for _ in choices]):
                raise TypeError(F'Expected `choices` to contain numbers given `bound_type` == "range". Received: {choices}')
            if len(choices) != 2:
                raise ValueError(F'Expected `choices` to contain two numbers: an upper and lower bound. Received: {choices}')
            if choices[0] >= choices[1]:
                raise ValueError(F'Expected the first element of `choices` to be less than the second. Received: {choices}')

        # TODO: If `bound_type` == 'range', create the range and set `choices` to it

        # self.search_bounds[hyperparameter] = (bound_type, choices)  # FLAG: ORIGINAL
        self.search_bounds[hyperparameter] = choices  # FLAG: TEST

    # def _add_rule(self):
    #     # TODO: Supply callables declaring valid/invalid relationships between hyperparameters to filter the total choices
    #     pass

    ##################################################
    # Utility Methods:
    ##################################################
    def _preparation_workflow(self):
        """Perform housekeeping tasks to prepare for core functionality like validating the `Environment` and parameters,
        and updating the verbosity of individual Experiments"""
        self._validate_environment()
        self._validate_parameters()
        self._update_verbosity()

    @staticmethod
    def _validate_environment():
        """Check that there is a currently active Environment instance that is not already occupied"""
        if G.Env is None:
            raise EnvironmentInactiveError()
        if G.Env.current_task is None:
            G.log_(F'Validated Environment with key: "{G.Env.cross_experiment_key}"')
        else:
            raise EnvironmentInvalidError('A task is in progress. It must finish before a new one can be started')

    def _validate_parameters(self):
        """Ensure provided input parameters are properly formatted"""
        self.target_metric = get_formatted_target_metric(self.target_metric, G.Env.metrics_map, default_dataset='oof')

    def _find_similar_experiments(self):
        """Look for Experiments that were performed under similar conditions (algorithm and cross-experiment parameters)"""
        if self.read_experiments is False:
            return

        self.logger.print_saved_results_header()

        experiment_ids = get_ids_by(
            G.Env.result_paths['global_leaderboard'], algorithm_name=self.algorithm_name,
            cross_experiment_key=G.Env.cross_experiment_key, hyperparameter_key=None,
        )

        G.debug_(F'Experiments found with matching cross-experiment key and algorithm:   {len(experiment_ids)}')

        hyperparameters_and_scores = [get_scored_params(
            F'{G.Env.result_paths["description"]}/{_}.json', self.target_metric
        ) for _ in experiment_ids]

        hyperparameters_and_scores = filter_by_space(
            hyperparameters_and_scores, self.hyperparameter_space, self.model_init_params, self.model_extra_params,
            self.preprocessing_pipeline, self.preprocessing_params, self.feature_selector,
        )
        G.debug_(F'Experiments whose hyperparameters fit in the currently defined space:   {len(hyperparameters_and_scores)}')

        hyperparameters_and_scores = filter_by_guidelines(
            hyperparameters_and_scores, self.hyperparameter_space, self.model_init_params, self.model_extra_params,
            self.preprocessing_pipeline, self.preprocessing_params, self.feature_selector,
        )
        G.debug_(F'Experiments whose hyperparameters matched the current guidelines:   {len(hyperparameters_and_scores)}')

        self.similar_experiments = hyperparameters_and_scores

    def _is_valid_hyperparameter_path(self, hyperparameter):
        """Determine whether the given hyperparameter path is valid

        Parameters
        ----------
        hyperparameter: Tuple
            Path location of a hyperparameter. Should start with the containing attribute's name ('model_init_params', or
            'model_extra_params'). Subsequent items will be looked up in order within the attribute

        Returns
        -------
        Boolean
            True if the hyperparameter path exists. Else False"""
        try:
            _ = get_path(getattr(self, hyperparameter[0]), hyperparameter[1:])
            return True
        except (AttributeError, PathAccessError):
            return False

    def _update_verbosity(self):
        """Update the contents of :attr:`environment.Environment.reporting_handler_params` if required by :attr:`verbose`"""
        #################### Mute non-critical console logging for Experiments ####################
        if self.verbose in [0, 1]:
            G.Env.reporting_handler_params.setdefault('console_params', {})['level'] = 'CRITICAL'

        #################### Blacklist 'script_backup' ####################
        G.Env.result_paths['script_backup'] = None


class InformedOptimizationProtocol(BaseOptimizationProtocol, metaclass=ABCMeta):
    # TODO: Reorganize kwargs to start with `target_metric`, `dimensions`, `iterations` - The only really important ones
    def __init__(
            self, target_metric=None, iterations=1, verbose=1, read_experiments=True, reporter_parameters=None,

            #################### Optimizer Class Parameters ####################
            dimensions=None,
            base_estimator='GP',
            n_initial_points=10,
            acquisition_function='gp_hedge',
            acquisition_optimizer='auto',
            random_state=32,  # FLAG: ORIGINAL
            acquisition_function_kwargs=None,
            acquisition_optimizer_kwargs=None,

            #################### Minimizer Parameters ####################
            n_random_starts=10,
            callbacks=None,

            #################### Other Parameters ####################
            base_estimator_kwargs=None,
    ):
        """Base class for Informed Optimization Protocols

        Parameters
        ----------
        target_metric: See :class:`optimization_core.BaseOptimizationProtocol`
        iterations: See :class:`optimization_core.BaseOptimizationProtocol`
        verbose: See :class:`optimization_core.BaseOptimizationProtocol`
        read_experiments: See :class:`optimization_core.BaseOptimizationProtocol`
        reporter_parameters: See :class:`optimization_core.BaseOptimizationProtocol`
        dimensions: List
            List of hyperparameter search space dimensions, in which each dimension is an instance of :class:`space.Real`, or
            :class:`space.Integer`, or :class:`space.Categorical`. Additionally, each of the `Dimension` classes MUST be given a
            valid `name` kwarg that corresponds to the hyperparameter name for which dimensions are being provided
        base_estimator: String in ['GP', 'GBRT', 'RF', 'ET', 'DUMMY'], or an `sklearn` regressor, default='GP'
            If one of the above strings, a default model of that type will be used. Else, should inherit from
            :class:`sklearn.base.RegressorMixin`, and its :meth:`predict` should have an optional `return_std` argument, which
            returns `std(Y | x)`, along with `E[Y | x]`
        n_initial_points: Int, default=10
            The number of complete evaluation points necessary before allowing Experiments to be approximated with
            `base_estimator`. Any valid Experiment records found will count as initialization points. If enough Experiment records
            are not found, additional points will be randomly sampled
        acquisition_function: String in ['LCB', 'EI', 'PI', 'gp_hedge'], default='gp_hedge'
            Function to minimize over the posterior distribution. 'LCB': lower confidence bound. 'EI': negative expected
            improvement. 'PI': negative probability of improvement. 'gp_hedge': Probabilistically choose one of the preceding
            three acquisition functions at each iteration
        acquisition_optimizer: String in ['sampling', 'lbfgs', 'auto'], default='auto'
            Method to minimize the acquisition function. The fit model is updated with the optimal value obtained by optimizing
            `acquisition_function` with `acquisition_optimizer`. 'sampling': optimize by computing `acquisition_function` at
            `acquisition_optimizer_kwargs['n_points']` randomly sampled points. 'lbfgs': optimize by sampling
            `n_restarts_optimizer` random points, then run 'lbfgs' for 20 iterations with those points to find local minima, the
            optimal of which is used to update the prior. 'auto': configure on the basis of `base_estimator` and `dimensions`
        random_state: Int, `RandomState` instance, or None, default=None
            Set to something other than None for reproducible results
        acquisition_function_kwargs: Dict, or None, default=dict(xi=0.01, kappa=1.96)
            Additional arguments passed to the acquisition function
        acquisition_optimizer_kwargs: Dict, or None, default=dict(n_points=10000, n_restarts_optimizer=5, n_jobs=1)
            Additional arguments passed to the acquisition optimizer
        n_random_starts: Int, default=10
            The number of Experiments to execute with random points before checking that `n_initial_points` have been evaluated
        callbacks: Callable, list of callables, or None, default=[]
            If callable, then `callbacks(self.optimizer_result)` is called after each update to :attr:`optimizer`. If list, then
            each callable is called
        base_estimator_kwargs: Dict, or None, default={}
            Additional arguments passed to `base_estimator` when it is initialized

        Notes
        -----
        To provide initial input points for evaluation, individual Experiments can be executed prior to instantiating an
        Optimization Protocol. The results of these Experiments will automatically be detected and cherished by the optimizer.

        :class:`.InformedOptimizationProtocol` and its children in :mod:`.optimization` rely heavily on the utilities provided by
        the `Scikit-Optimize` library, so thank you to the creators and contributors for their excellent work."""
        # TODO: Add 'EIps', and 'PIps' to the allowable `acquisition_function` values - Will need to return execution times

        #################### Optimizer Parameters ####################
        self.dimensions = dimensions  # FLAG: Technically, this is just represented differently than uninformed bounds
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
        # TODO: This does nothing currently - Fix that
        self.n_random_starts = n_random_starts  # TODO: This does nothing currently - Fix that
        # TODO: This does nothing currently - Fix that
        self.callbacks = callbacks or []

        #################### Other Parameters ####################
        self.base_estimator_kwargs = base_estimator_kwargs or {}

        #################### Placeholder Attributes ####################
        self.optimizer = None
        self.optimizer_result = None
        self.current_hyperparameters_list = None

        super().__init__(
            target_metric=target_metric, iterations=iterations, verbose=verbose, read_experiments=read_experiments,
            reporter_parameters=reporter_parameters
        )

    def _set_hyperparameter_space(self):
        self.hyperparameter_space = Space(dimensions=self.dimensions)
        self._prepare_estimator()
        self._build_optimizer()

    def _prepare_estimator(self):
        self.base_estimator = cook_estimator(self.base_estimator, space=self.hyperparameter_space, **self.base_estimator_kwargs)

    def _build_optimizer(self):
        """Set :attr:`optimizer` to the optimizing class used to both estimate the utility of sets of hyperparameters by learning
        from executed Experiments, and suggest points at which the objective should be evaluated"""
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

    def _execute_experiment(self):
        super()._execute_experiment()

        # FLAG: BIG BREAKING TEST BELOW
        # self.optimizer_result = self.optimizer.tell(self.current_hyperparameters_list, self.current_score, fit=True)  # FLAG: ORIGINAL
        self.optimizer_result = self.optimizer.tell(self.current_hyperparameters_list, -self.current_score, fit=True)  # FLAG: TEST
        # FLAG: BIG BREAKING TEST ABOVE

        if eval_callbacks(self.callbacks, self.optimizer_result):
            return

    def _get_current_hyperparameters(self):
        _current_hyperparameters = self.optimizer.ask()

        if _current_hyperparameters == self.current_hyperparameters_list:
            # G.debug_('Repeated hyperparameters selected. Switching to random selection for this iteration')
            new_parameters = self.hyperparameter_space.rvs(random_state=None)[0]
            G.debug_('REPEATED     asked={}     new={}'.format(_current_hyperparameters, new_parameters))
            _current_hyperparameters = new_parameters

        self.current_hyperparameters_list = _current_hyperparameters
        current_hyperparameters = zip([_.name for _ in self.hyperparameter_space.dimensions], self.current_hyperparameters_list)

        current_hyperparameters = {
            (('model_init_params' if _k in self.model_init_params else 'model_extra_params'), _k): _v
            for _k, _v in dict(current_hyperparameters).items()
        }
        return current_hyperparameters

    def _find_similar_experiments(self):
        super()._find_similar_experiments()

        for _i, _experiment in enumerate(self.similar_experiments[::-1]):
            _hyperparameters = dimension_subset(_experiment[0], [_.name for _ in self.hyperparameter_space.dimensions])
            _evaluation = _experiment[1]

            self.logger.print_result(_hyperparameters, _evaluation)

            # FLAG: BIG BREAKING TEST BELOW
            # self.optimizer_result = self.optimizer.tell(_hyperparameters, _evaluation)  # FLAG: ORIGINAL
            self.optimizer_result = self.optimizer.tell(_hyperparameters, -_evaluation)  # FLAG: TEST
            # FLAG: BIG BREAKING TEST ABOVE

            # self.optimizer_result = self.optimizer.tell(
            #     _hyperparameters, _evaluation, fit=(_i == len(self.similar_experiments) - 1)
            # )

            if eval_callbacks(self.callbacks, self.optimizer_result):
                return self.optimizer_result
            # FLAG: Could wrap above `tell` call in try/except, then attempt `_tell` with improper dimensions

    def _validate_parameters(self):
        super()._validate_parameters()

        #################### callbacks ####################
        self.callbacks = check_callback(self.callbacks)

    @property
    def search_space_size(self):
        """The number of different hyperparameter permutations possible given the current hyperparameter search dimensions.

        Returns
        -------
        :attr:`_search_space_size`: Int, or `numpy.inf`
            Infinity will be returned if any of the following constraints are met: 1) the hyperparameter dimensions include any
            real-valued boundaries, 2) the boundaries include values that are neither categorical nor integer, or 3) the search
            space size is otherwise incalculable"""
        if self._search_space_size is None:
            self._search_space_size = len(self.hyperparameter_space)
        return self._search_space_size


class UninformedOptimizationProtocol(BaseOptimizationProtocol, metaclass=ABCMeta):
    def __init__(self, target_metric=None, iterations=1, verbose=1, read_experiments=True, reporter_parameters=None):
        super().__init__(
            target_metric=target_metric, iterations=iterations, verbose=verbose, read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
        )

    def _get_current_hyperparameters(self):
        current_hyperparameters = next(self.hyperparameter_space)
        return current_hyperparameters


if __name__ == '__main__':
    pass
