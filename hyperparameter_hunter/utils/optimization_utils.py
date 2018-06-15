##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.space import dimension_subset, Space
from hyperparameter_hunter.utils.boltons_utils import get_path, remap
from hyperparameter_hunter.utils.file_utils import read_json

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd

##################################################
# Import Learning Assets
##################################################
from skopt.optimizer.optimizer import Optimizer

# FLAG: TEMP IMPORTS BELOW
import sys
import warnings
from math import log
from numbers import Number

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone, is_regressor
from sklearn.externals.joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state
# noinspection PyProtectedMember
from skopt.acquisition import _gaussian_acquisition, gaussian_acquisition_1D
from skopt.learning import GaussianProcessRegressor
# from skopt.space import Space, Categorical
from hyperparameter_hunter.space import Space, Categorical
# from skopt.utils import normalize_dimensions
from hyperparameter_hunter.space import normalize_dimensions
# noinspection PyProtectedMember
from skopt.utils import check_x_in_space, cook_estimator, create_result, has_gradients, is_listlike, is_2Dlistlike
# FLAG: TEMP IMPORTS ABOVE


class AskingOptimizer(Optimizer):
    # FLAG: ORIGINAL BELOW
    # def __init__(
    #         self, dimensions, base_estimator="gp", n_random_starts=None, n_initial_points=10, acq_func="gp_hedge",
    #         acq_optimizer="auto", random_state=None, acq_func_kwargs=None, acq_optimizer_kwargs=None,
    #         repeated_ask_kwargs=None
    # ):
    #     self.__repeated_ask_kwargs = repeated_ask_kwargs or {}
    #
    #     super().__init__(
    #         dimensions, base_estimator=base_estimator, n_random_starts=n_random_starts, n_initial_points=n_initial_points,
    #         acq_func=acq_func, acq_optimizer=acq_optimizer, random_state=random_state, acq_func_kwargs=acq_func_kwargs,
    #         acq_optimizer_kwargs=acq_optimizer_kwargs,
    #     )
    # FLAG: ORIGINAL ABOVE

    # FLAG: TEST BELOW
    # noinspection PyMissingConstructor
    def __init__(
            self, dimensions, base_estimator='gp',
            n_random_starts=None, n_initial_points=10,
            acq_func='gp_hedge',
            acq_optimizer='auto',
            random_state=None, acq_func_kwargs=None,
            acq_optimizer_kwargs=None
    ):
        """This is nearly identical to :meth:`skopt.optimizer.optimizer.Optimizer.__init__`. It is recreated here to use the
        modified :class:`hyperparameter_hunter.space.Space`, rather than the original `skopt` version. This is not an ideal
        solution, and other options are being considered

        Parameters
        ----------
        dimensions: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        base_estimator: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        n_random_starts: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        n_initial_points: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        acq_func: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        acq_optimizer: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        random_state: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        acq_func_kwargs: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`
        acq_optimizer_kwargs: See :class:`skopt.optimizer.optimizer.Optimizer.__init__`"""
        # TODO: Figure out way to override skopt Optimizer's use of skopt Space without having to rewrite __init__
        self.__repeated_ask_kwargs = {}
        self.rng = check_random_state(random_state)

        # Configure acquisition function - Store and create acquisition function set
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ['gp_hedge', 'EI', 'LCB', 'PI', 'EIps', 'PIps']
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError(F'Expected `acq_func` to be in {allowed_acq_funcs}, got {self.acq_func}')

        # Treat hedging method separately
        if self.acq_func == 'gp_hedge':
            self.cand_acq_funcs_ = ['EI', 'LCB', 'PI']
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get('eta', 1.0)

        # Configure counters of points - Check `n_random_starts` deprecation first
        if n_random_starts is not None:
            warnings.warn(('`n_random_starts` will be removed in favour of `n_initial_points`'), DeprecationWarning)
            n_initial_points = n_random_starts
        if n_initial_points < 0:
            raise ValueError(F'Expected `n_initial_points` >= 0, got {n_initial_points}')
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # Configure estimator - Build `base_estimator` if doesn't exist
        if isinstance(base_estimator, str):
            base_estimator = cook_estimator(
                base_estimator, space=dimensions, random_state=self.rng.randint(0, np.iinfo(np.int32).max)
            )

        # Check if regressor
        if not is_regressor(base_estimator) and base_estimator is not None:
            raise ValueError(F'`base_estimator`={base_estimator} must be a regressor')

        # Treat per second acquisition function specially
        is_multi_regressor = isinstance(base_estimator, MultiOutputRegressor)
        if 'ps' in self.acq_func and not is_multi_regressor:
            self.base_estimator_ = MultiOutputRegressor(base_estimator)
        else:
            self.base_estimator_ = base_estimator

        # Configure optimizer - Decide optimizer based on gradient information
        if acq_optimizer == 'auto':
            if has_gradients(self.base_estimator_):
                acq_optimizer = 'lbfgs'
            else:
                acq_optimizer = 'sampling'

        if acq_optimizer not in ['lbfgs', 'sampling']:
            raise ValueError('Expected `acq_optimizer` to be "lbfgs" or "sampling", got {}'.format(acq_optimizer))
        if (not has_gradients(self.base_estimator_) and acq_optimizer != 'sampling'):
            raise ValueError('The regressor {} should run with `acq_optimizer`="sampling"'.format(type(base_estimator)))
        self.acq_optimizer = acq_optimizer

        # Record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get('n_points', 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get('n_restarts_optimizer', 5)
        n_jobs = acq_optimizer_kwargs.get('n_jobs', 1)
        self.n_jobs = n_jobs
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # Configure search space - Normalize space if GP regressor
        if isinstance(self.base_estimator_, GaussianProcessRegressor):
            dimensions = normalize_dimensions(dimensions)
        self.space = Space(dimensions)

        # Record categorical and non-categorical indices
        self._cat_inds = []
        self._non_cat_inds = []
        for ind, dim in enumerate(self.space.dimensions):
            if isinstance(dim, Categorical):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)

        # Initialize storage for optimization
        self.models = []
        self.Xi = []
        self.yi = []

        # Initialize cache for `ask` method responses
        # This ensures that multiple calls to `ask` with n_points set return same sets of points. Reset to {} at call to `tell`
        self.cache_ = {}
    # FLAG: TEST ABOVE

    def _ask(self):
        # TODO: Add documentation
        ask_result = super()._ask()

        do_retell = self.__repeated_ask_kwargs.get('do_retell', True)
        return_val = self.__repeated_ask_kwargs.get('return_val', 'ask')
        persistent_check = self.__repeated_ask_kwargs.get('persistent_check', True)

        if persistent_check is True:
            counter = 100
            while (ask_result in self.Xi) and (counter > 0):
                ask_result = self.__ask_helper(ask_result, do_retell, return_val)
                # print(F'{counter}     {ask_result}')
                counter -= 1

        return ask_result

    def __ask_helper(self, ask_result, do_retell=True, return_val='ask'):
        """

        Parameters
        ----------
        ask_result: Iterable of hyperparameters
            The result of :meth:`skopt.optimizer.optimizer.Optimizer._ask`
        do_retell: Boolean, default=True
            If True and `ask_result` has already been tested, the optimizer will be re-`tell`ed the hyperparameters and their
            original score
        return_val: String in ['ask', 'random'], default='ask'
            If 'ask', :meth:`skopt.optimizer.optimizer.Optimizer._ask` will be repeatedly called for a new result. If 'random',
            :meth:`space.Space.rvs` will be used to retrieve the next set of hyperparameters

        Returns
        -------
        ask_result"""
        # TODO: Fill in documentation description
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            ask_result = self.space.rvs(random_state=self.rng)[0]
        else:
            min_delta_x = min([self.space.distance(ask_result, _) for _ in self.Xi])
            if abs(min_delta_x) <= 1e-8:
                # G.debug_(F'Received repeated point:   {ask_result}')

                if do_retell is True:
                    self.tell(ask_result, self.yi[self.Xi.index(ask_result)])
                    # G.debug_(F'Optimizer was re-`tell`ed point:   {ask_result}   ->   {self.yi[self.Xi.index(ask_result)]}')

                if return_val == 'ask':
                    ask_result = super()._ask()
                    # G.debug_(F'Re-`ask`ed, and received point:   {ask_result}')
                elif return_val == 'random':
                    ask_result = self.space.rvs(random_state=self.rng)[0]
                    # G.debug_(F'Set repeated point to random:   {ask_result}')
                else:
                    raise ValueError(F'Received invalid value for `return_val`: {return_val}')

        return ask_result


def get_ids_by(leaderboard_path, algorithm_name=None, cross_experiment_key=None, hyperparameter_key=None, drop_duplicates=True):
    """Get a list of experiment_ids that match the provided criteria

    Parameters
    ----------
    leaderboard_path: String
        The path to a leaderboard .csv file, which has at least the following columns: 'experiment_id', 'hyperparameter_key',
        'cross_experiment_key', 'algorithm_name'
    algorithm_name: String, or None, default=None
        If string, expects the name of an algorithm that may exist on the leaderboard, such as the following: 'LGBMRegressor',
        'XGBClassifier', 'KerasClassifier', 'KMeans', 'BayesianRidge', 'RGFClassifier', etc.
    cross_experiment_key: String, or None, default=None
        If string, expects a cross-experiment key hash produced during initialization of :class:`environment.Environment`
    hyperparameter_key: String, or None, default=None
        If string, expects a hyperparameter key hash produced by a child of :class:`experiments.BaseExperiment`
    drop_duplicates: Boolean, default=True
        If True, only a single entry for every unique triple of ('algorithm_name', 'cross_experiment_key', 'hyperparameter_key')
        will be returned

    Returns
    -------
    matching_ids: List
        A list of experiment_id strings"""
    leaderboard = pd.read_csv(leaderboard_path, index_col=None)

    if algorithm_name is not None:
        leaderboard = leaderboard.loc[leaderboard['algorithm_name'] == algorithm_name]
    if cross_experiment_key is not None:
        leaderboard = leaderboard.loc[leaderboard['cross_experiment_key'] == cross_experiment_key]
    if hyperparameter_key is not None:
        leaderboard = leaderboard.loc[leaderboard['hyperparameter_key'] == hyperparameter_key]

    if drop_duplicates is True:
        leaderboard.drop_duplicates(subset=['algorithm_name', 'cross_experiment_key', 'hyperparameter_key'], inplace=True)

    matching_ids = leaderboard['experiment_id'].values.tolist()
    return matching_ids


def get_scored_params(experiment_description_path, target_metric):
    """Retrieve the hyperparameters of a completed Experiment, along with an evaluation of its performance

    Parameters
    ----------
    experiment_description_path: String
        The path to an Experiment's description .json file
    target_metric: Tuple
        A path denoting the metric to be used. If tuple, the first value should be one of ['oof', 'holdout', 'in_fold'], and the
        second value should be the name of a metric supplied in :attr:`environment.Environment.metrics_params`

    Returns
    -------
    Tuple double, containing the following: 1) a dict of the hyperparameters used, and 2) the value of the Experiment's
    `target_metric`"""
    description = read_json(file_path=experiment_description_path)
    evaluation = get_path(description['final_evaluations'], target_metric)
    all_hyperparameters = description['hyperparameters']
    return (all_hyperparameters, evaluation)


def filter_by_space(
        hyperparameters_and_scores, hyperparameter_space, model_init_params, model_extra_params,
        preprocessing_pipeline, preprocessing_params, feature_selector,
):
    """Reject any `hyperparameters_and_scores` tuples whose hyperparameters do not fit within `hyperparameter_space`

    Parameters
    ----------
    hyperparameters_and_scores: List of tuples
        Each tuple in list should be a pair of form (hyperparameters <dict>, evaluation <float>), where the hyperparameter dict
        should contain at least the following keys: ['model_init_params', 'model_extra_params', 'preprocessing_pipeline',
        'preprocessing_params', 'feature_selector']
    hyperparameter_space: instance of :class:`space.Space`
        The boundaries of the hyperparameters to be searched
    model_init_params: Dict
    model_extra_params: Dict, or None
    preprocessing_pipeline: Dict, or None
    preprocessing_params: Dict, or None
    feature_selector: List of column names, callable, list of booleans, or None

    Returns
    -------
    hyperparameters_and_scores: List of tuples
        Filtered to include only those whose hyperparameters fit within the `hyperparameter_space`"""
    dimension_names = [_.name for _ in hyperparameter_space.dimensions]

    for dimension_name in dimension_names:
        if dimension_name not in list(model_init_params.keys()):
            raise ValueError(F'Tuning is only supported for model-initializing hyperparameters. "{dimension_name}" is invalid')

    hyperparameters_and_scores = list(filter(
        lambda _: dimension_subset(_[0], dimension_names) in hyperparameter_space, hyperparameters_and_scores
    ))

    return hyperparameters_and_scores


def filter_by_guidelines(
        hyperparameters_and_scores, hyperparameter_space, model_init_params, model_extra_params,
        preprocessing_pipeline, preprocessing_params, feature_selector
):
    """Reject any `hyperparameters_and_scores` tuples whose hyperparameters do not match the guideline hyperparameters (all
    hyperparameters not in `hyperparameter_space`), after ignoring unimportant hyperparameters

    Parameters
    ----------
    hyperparameters_and_scores: List of tuples
        Each tuple in list should be a pair of form (hyperparameters <dict>, evaluation <float>), where the hyperparameter dict
        should contain at least the following keys: ['model_init_params', 'model_extra_params', 'preprocessing_pipeline',
        'preprocessing_params', 'feature_selector']
    hyperparameter_space: instance of :class:`space.Space`
        The boundaries of the hyperparameters to be searched
    model_init_params: Dict
    model_extra_params: Dict, or None
    preprocessing_pipeline: Dict, or None
    preprocessing_params: Dict, or None
    feature_selector: List of column names, callable, list of booleans, or None

    Returns
    -------
    hyperparameters_and_scores: List of tuples
        Filtered to include only those whose hyperparameters matched the guideline hyperparameters (all hyperparameters that are
        not in `hyperparameter_space`), after ignoring unimportant hyperparameters"""
    dimensions = [_.name for _ in hyperparameter_space.dimensions]
    dimensions = [('model_init_params', _) if isinstance(_, str) else _ for _ in dimensions]
    # `dimensions` represents the hyperparameters to be ignored. Filter by all remaining

    dimensions_to_ignore = [
        ('model_initializer',),
        ('model_init_params', 'verbose'),
        ('model_init_params', 'silent'),
        ('model_init_params', 'random_state'),
        ('model_init_params', 'seed'),
        ('model_init_params', 'n_jobs'),
        ('model_init_params', 'nthread'),
        ('model_extra_params', 'fit', 'verbose'),
        ('model_extra_params', 'fit', 'silent')
    ]

    current_guidelines = dict(
        model_init_params=model_init_params, model_extra_params=model_extra_params, preprocessing_pipeline=preprocessing_pipeline,
        preprocessing_params=preprocessing_params, feature_selector=feature_selector,
    )

    # noinspection PyUnusedLocal
    def _visit(path, key, value):
        for dimension in dimensions + dimensions_to_ignore:
            if path + (key,) == dimension:
                return False
        return True

    filtered_guidelines = remap(current_guidelines, visit=_visit)

    hyperparameters_and_scores = list(filter(
        lambda _: remap(_[0], visit=_visit) == filtered_guidelines, hyperparameters_and_scores
    ))

    return hyperparameters_and_scores


if __name__ == '__main__':
    pass
