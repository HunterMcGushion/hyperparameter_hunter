"""This module defines utility functions used to organize hyperparameter optimization, specifically
the gathering of saved Experiment files in order to identify similar Experiments that can be used as
learning material for the current OptimizationProtocol. Additionally, :class:`AskingOptimizer` is
defined here, which is used to direct OptimizationProtocols' searches through hyperparameter space

Related
-------
:mod:`hyperparameter_hunter.optimization_core`
    The primary user of the utilities defined in
    :mod:`hyperparameter_hunter.utils.optimization_utils`

Notes
-----
:class:`AskingOptimizer` is a blatant adaptation of Scikit-Optimize's
`optimizer.optimizer.Optimizer` class. This situation is far from ideal, but the creators and
contributors of SKOpt deserve all the credit for their excellent work"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.space import dimension_subset, Space, Real, Integer, Categorical
from hyperparameter_hunter.utils.boltons_utils import get_path, remap, default_enter
from hyperparameter_hunter.utils.file_utils import read_json

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd

##################################################
# Import Learning Assets
##################################################
from skopt.optimizer.optimizer import Optimizer

try:
    from hyperparameter_hunter.library_helpers.keras_optimization_helper import consolidate_layers
except ImportError:
    pass

# FLAG: TEMP IMPORTS BELOW
import warnings
import numpy as np

from sklearn.base import is_regressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state

# noinspection PyProtectedMember
from skopt.acquisition import gaussian_acquisition_1D
from skopt.learning import GaussianProcessRegressor
from hyperparameter_hunter.space import Space, Categorical
from hyperparameter_hunter.space import normalize_dimensions

# noinspection PyProtectedMember
from skopt.utils import cook_estimator, create_result, has_gradients, is_listlike, is_2Dlistlike


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
        self,
        dimensions,
        base_estimator="gp",
        n_random_starts=None,
        n_initial_points=10,
        acq_func="gp_hedge",
        acq_optimizer="auto",
        random_state=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
    ):
        """This is nearly identical to :meth:`skopt.optimizer.optimizer.Optimizer.__init__`. It is
        recreated here to use the modified :class:`hyperparameter_hunter.space.Space`, rather than
        the original `skopt` version. This is not an ideal solution, and other options are being
        considered

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

        allowed_acq_funcs = ["gp_hedge", "EI", "LCB", "PI", "EIps", "PIps"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError(
                f"Expected `acq_func` to be in {allowed_acq_funcs}, got {self.acq_func}"
            )

        # Treat hedging method separately
        if self.acq_func == "gp_hedge":
            self.cand_acq_funcs_ = ["EI", "LCB", "PI"]
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        # Configure counters of points - Check `n_random_starts` deprecation first
        if n_random_starts is not None:
            warnings.warn(
                ("`n_random_starts` will be removed in favour of `n_initial_points`"),
                DeprecationWarning,
            )
            n_initial_points = n_random_starts
        if n_initial_points < 0:
            raise ValueError(f"Expected `n_initial_points` >= 0, got {n_initial_points}")
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # Configure estimator - Build `base_estimator` if doesn't exist
        if isinstance(base_estimator, str):
            base_estimator = cook_estimator(
                base_estimator,
                space=dimensions,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
            )

        # Check if regressor
        if not is_regressor(base_estimator) and base_estimator is not None:
            raise ValueError(f"`base_estimator`={base_estimator} must be a regressor")

        # Treat per second acquisition function specially
        is_multi_regressor = isinstance(base_estimator, MultiOutputRegressor)
        if "ps" in self.acq_func and not is_multi_regressor:
            self.base_estimator_ = MultiOutputRegressor(base_estimator)
        else:
            self.base_estimator_ = base_estimator

        # Configure optimizer - Decide optimizer based on gradient information
        if acq_optimizer == "auto":
            if has_gradients(self.base_estimator_):
                acq_optimizer = "lbfgs"
            else:
                acq_optimizer = "sampling"

        if acq_optimizer not in ["lbfgs", "sampling"]:
            raise ValueError(
                'Expected `acq_optimizer` to be "lbfgs" or "sampling", got {}'.format(acq_optimizer)
            )
        if not has_gradients(self.base_estimator_) and acq_optimizer != "sampling":
            raise ValueError(
                'The regressor {} should run with `acq_optimizer`="sampling"'.format(
                    type(base_estimator)
                )
            )
        self.acq_optimizer = acq_optimizer

        # Record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get("n_restarts_optimizer", 5)
        n_jobs = acq_optimizer_kwargs.get("n_jobs", 1)
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

        do_retell = self.__repeated_ask_kwargs.get("do_retell", True)
        return_val = self.__repeated_ask_kwargs.get("return_val", "ask")
        persistent_check = self.__repeated_ask_kwargs.get("persistent_check", True)

        if persistent_check is True:
            counter = 100
            while (ask_result in self.Xi) and (counter > 0):
                ask_result = self.__ask_helper(ask_result, do_retell, return_val)
                # print(F'{counter}     {ask_result}')
                counter -= 1

        return ask_result

    def __ask_helper(self, ask_result, do_retell=True, return_val="ask"):
        """

        Parameters
        ----------
        ask_result: Iterable of hyperparameters
            The result of :meth:`skopt.optimizer.optimizer.Optimizer._ask`
        do_retell: Boolean, default=True
            If True and `ask_result` has already been tested, the optimizer will be re-`tell`ed the
            hyperparameters and their original score
        return_val: String in ['ask', 'random'], default='ask'
            If 'ask', :meth:`skopt.optimizer.optimizer.Optimizer._ask` will be repeatedly called
            for a new result. If 'random', :meth:`space.Space.rvs` will be used to retrieve the
            next set of hyperparameters

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

                if return_val == "ask":
                    ask_result = super()._ask()
                    # G.debug_(F'Re-`ask`ed, and received point:   {ask_result}')
                elif return_val == "random":
                    ask_result = self.space.rvs(random_state=self.rng)[0]
                    # G.debug_(F'Set repeated point to random:   {ask_result}')
                else:
                    raise ValueError(f"Received invalid value for `return_val`: {return_val}")

        return ask_result


##################################################
# Optimization Utility Functions
##################################################
def get_ids_by(
    leaderboard_path,
    algorithm_name=None,
    cross_experiment_key=None,
    hyperparameter_key=None,
    drop_duplicates=True,
):
    """Get a list of experiment_ids that match the provided criteria

    Parameters
    ----------
    leaderboard_path: String
        The path to a leaderboard .csv file, which has at least the following columns:
        'experiment_id', 'hyperparameter_key', 'cross_experiment_key', 'algorithm_name'
    algorithm_name: String, or None, default=None
        If string, expects the name of an algorithm that may exist on the leaderboard, such as the
        following: 'LGBMRegressor', 'XGBClassifier', 'KerasClassifier', 'KMeans', 'BayesianRidge',
        'RGFClassifier', etc.
    cross_experiment_key: String, or None, default=None
        If string, expects a cross-experiment key hash produced during initialization of
        :class:`environment.Environment`
    hyperparameter_key: String, or None, default=None
        If string, expects a hyperparameter key hash produced by a child of
        :class:`experiments.BaseExperiment`
    drop_duplicates: Boolean, default=True
        If True, only a single entry for every unique triple of ('algorithm_name',
        'cross_experiment_key', 'hyperparameter_key') will be returned

    Returns
    -------
    matching_ids: List
        A list of experiment_id strings"""
    try:
        leaderboard = pd.read_csv(leaderboard_path, index_col=None)
    except FileNotFoundError:
        return []

    if algorithm_name is not None:
        leaderboard = leaderboard.loc[leaderboard["algorithm_name"] == algorithm_name]
    if cross_experiment_key is not None:
        leaderboard = leaderboard.loc[leaderboard["cross_experiment_key"] == cross_experiment_key]
    if hyperparameter_key is not None:
        leaderboard = leaderboard.loc[leaderboard["hyperparameter_key"] == hyperparameter_key]

    if drop_duplicates is True:
        leaderboard.drop_duplicates(
            subset=["algorithm_name", "cross_experiment_key", "hyperparameter_key"], inplace=True
        )

    matching_ids = leaderboard["experiment_id"].values.tolist()
    return matching_ids


def get_scored_params(experiment_description_path, target_metric, get_description=False):
    """Retrieve the hyperparameters of a completed Experiment, along with its performance evaluation

    Parameters
    ----------
    experiment_description_path: String
        The path to an Experiment's description .json file
    target_metric: Tuple
        A path denoting the metric to be used. If tuple, the first value should be one of ['oof',
        'holdout', 'in_fold'], and the second value should be the name of a metric supplied in
        :attr:`environment.Environment.metrics_params`
    get_description: Boolean, default=False
        If True, return a tuple of: ((`all_hyperparameters`, `evaluation`), `description`), in which
        `description` is the original description dict for the experiment. Else, return a tuple of:
        (`all_hyperparameters`, `evaluation`)

    Returns
    -------
    all_hyperparameters: Dict
        A dict of the hyperparameters used by the Experiment
    evaluation: Float
        Value of the Experiment's `target_metric`"""
    description = read_json(file_path=experiment_description_path)
    evaluation = get_path(description["final_evaluations"], target_metric)
    all_hyperparameters = description["hyperparameters"]

    if description["module_name"].lower() == "keras":
        all_hyperparameters["model_init_params"]["layers"] = consolidate_layers(
            all_hyperparameters["model_init_params"]["layers"],
            class_name_key=False,
            separate_args=False,
        )

    if get_description:
        return ((all_hyperparameters, evaluation), description)
    return (all_hyperparameters, evaluation)


def filter_by_space(hyperparameters_and_scores, hyperparameter_space):
    """Reject any `hyperparameters_and_scores` tuples whose hyperparameters do not fit within
    `hyperparameter_space`

    Parameters
    ----------
    hyperparameters_and_scores: List of tuples
        Each tuple in list should be a pair of form (hyperparameters <dict>, evaluation <float>),
        where the hyperparameter dict should contain at least the following keys:
        ['model_init_params', 'model_extra_params', 'preprocessing_pipeline',
        'preprocessing_params', 'feature_selector']
    hyperparameter_space: instance of :class:`space.Space`
        The boundaries of the hyperparameters to be searched

    Returns
    -------
    hyperparameters_and_scores: List of tuples
        Filtered to include only those whose hyperparameters fit within `hyperparameter_space`"""
    dimension_names = hyperparameter_space.names()
    hyperparameters_and_scores = list(
        filter(
            lambda _: dimension_subset(_[0], dimension_names) in hyperparameter_space,
            hyperparameters_and_scores,
        )
    )

    return hyperparameters_and_scores


def filter_by_guidelines(
    hyperparameters_and_scores,
    hyperparameter_space,
    model_init_params,
    model_extra_params,
    preprocessing_pipeline,
    preprocessing_params,
    feature_selector,
    **kwargs,
):
    """Reject any `hyperparameters_and_scores` tuples whose hyperparameters do not match the
    guideline hyperparameters (all hyperparameters not in `hyperparameter_space`), after ignoring
    unimportant hyperparameters

    Parameters
    ----------
    hyperparameters_and_scores: List of tuples
        Each tuple should be of form (hyperparameters <dict>, evaluation <float>), in which
        hyperparameters contains at least the keys: ['model_init_params', 'model_extra_params',
        'preprocessing_pipeline', 'preprocessing_params', 'feature_selector']
    hyperparameter_space: instance of :class:`space.Space`
        The boundaries of the hyperparameters to be searched
    model_init_params: Dict
    model_extra_params: Dict, or None
    preprocessing_pipeline: Dict, or None
    preprocessing_params: Dict, or None
    feature_selector: List of column names, callable, list of booleans, or None
    **kwargs: Dict
        Extra parameter dicts to include in `guidelines`. For example, if filtering the
        hyperparameters of a Keras neural network, this should contain the following keys:
        'layers', 'compile_params'

    Returns
    -------
    hyperparameters_and_scores: List of tuples
        Filtered to include only those whose hyperparameters matched guideline hyperparameters"""
    dimensions = [
        ("model_init_params", _) if isinstance(_, str) else _ for _ in hyperparameter_space.names()
    ]
    # `dimensions` = hyperparameters to be ignored. Filter by all remaining

    dimensions_to_ignore = [
        ("model_initializer",),
        ("model_init_params", "build_fn"),
        (None, "verbose"),
        (None, "silent"),
        (None, "random_state"),
        (None, "seed"),
        ("model_init_params", "n_jobs"),
        ("model_init_params", "nthread"),
        # TODO: Remove below once loss_functions are hashed in description files
        ("model_init_params", "compile_params", "loss_functions"),
    ]

    temp_guidelines = dict(
        model_init_params=model_init_params,
        model_extra_params=model_extra_params,
        preprocessing_pipeline=preprocessing_pipeline,
        preprocessing_params=preprocessing_params,
        feature_selector=feature_selector,
        **kwargs,
    )

    # noinspection PyUnusedLocal
    def _visit(path, key, value):
        """Return False if element in hyperparameter_space dimensions, or in dimensions being
        ignored. Else, return True. If `value` is of type tuple or set, it will be converted to a
        list in order to simplify comparisons to the JSON-formatted `hyperparameters_and_scores`"""
        for dimension in dimensions + dimensions_to_ignore:
            if (path + (key,) == dimension) or (dimension[0] is None and dimension[1] == key):
                return False
        if isinstance(value, (tuple, set)):
            return key, list(value)
        return True

    guidelines = remap(temp_guidelines, visit=_visit)
    # `guidelines` = `temp_guidelines` that are neither `hyperparameter_space` choices, nor in `dimensions_to_ignore`

    hyperparameters_and_scores = list(
        filter(lambda _: remap(_[0], visit=_visit) == guidelines, hyperparameters_and_scores)
    )

    return hyperparameters_and_scores


def get_choice_dimensions(params, iter_attrs=None):
    """List all elements in the nested structure `params` that are hyperparameter space choices

    Parameters
    ----------
    params: Dict
        Parameters that may be nested and that may contain hyperparameter space choices to collect
    iter_attrs: Callable, list of callables, or None, default=None
        If callable, must evaluate to True or False when given three inputs: (path, key, value).
        Callable should return True if the current value should be entered by `remap`. If callable
        returns False, `default_enter` will be called. If `iter_attrs` is a list of callables, the
        value will be entered if any evaluates to True. If None, `default_enter` will be called

    Returns
    -------
    choices: List
        A list of tuple pairs, in which `choices[<index>][0]` is a tuple path specifying the
        location of the hyperparameter given a choice, and `choices[<index>][1]` is the space
        choice instance for that hyperparameter"""
    choices = []
    iter_attrs = iter_attrs or [lambda *_args: False]
    iter_attrs = [iter_attrs] if not isinstance(iter_attrs, list) else iter_attrs

    def _visit(path, key, value):
        """If `value` is a descendant of :class:`space.Dimension`, collect inputs, and return True.
        Else, return False"""
        if isinstance(value, (Real, Integer, Categorical)):
            choices.append(((path + (key,)), value))
            return True
        return False

    def _enter(path, key, value):
        """If any in `iter_attrs` is True, enter `value` as a dict, iterating over non-magic
        attributes. Else, `default_enter`"""
        if any([_(path, key, value) for _ in iter_attrs]):
            included_attrs = [_ for _ in dir(value) if not _.startswith("__")]
            return dict(), [(_, getattr(value, _)) for _ in included_attrs]
        return default_enter(path, key, value)

    _ = remap(params, visit=_visit, enter=_enter)
    return choices


if __name__ == "__main__":
    pass
