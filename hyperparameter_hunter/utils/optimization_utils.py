"""This module defines utility functions used to organize hyperparameter optimization, specifically
the gathering of saved Experiment files in order to identify similar Experiments that can be used as
learning material for the current OptimizationProtocol

Related
-------
:mod:`hyperparameter_hunter.optimization.protocol_core`
    The primary user of the utilities defined in
    :mod:`hyperparameter_hunter.utils.optimization_utils`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import ContinueRemap
from hyperparameter_hunter.keys.hashing import make_hash_sha256
from hyperparameter_hunter.space.dimensions import Real, Integer, Categorical, RejectedOptional
from hyperparameter_hunter.utils.boltons_utils import get_path, remap
from hyperparameter_hunter.utils.file_utils import read_json
from hyperparameter_hunter.utils.general_utils import extra_enter_attrs

try:
    from hyperparameter_hunter.library_helpers.keras_optimization_helper import consolidate_layers
except ImportError:
    pass

##################################################
# Import Miscellaneous Assets
##################################################
from contextlib import suppress
import pandas as pd


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
        # TODO: Above should be `leaderboards.Leaderboard.from_path(leaderboard_path)`, instead
        # TODO: Keep current enclosing try/except
    except FileNotFoundError:
        return []

    if algorithm_name is not None:
        leaderboard = leaderboard.loc[leaderboard["algorithm_name"] == algorithm_name]
    if cross_experiment_key is not None:
        leaderboard = leaderboard.loc[leaderboard["cross_experiment_key"] == cross_experiment_key]
    if hyperparameter_key is not None:
        leaderboard = leaderboard.loc[leaderboard["hyperparameter_key"] == hyperparameter_key]

    if drop_duplicates is True:
        # TODO: `drop_duplicates`' `keep` kwarg may be helpful if lb rows are sorted chronologically
        # TODO: ... https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop_duplicates.html
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
            all_hyperparameters["model_init_params"]["layers"], class_name_key=False
        )

    if get_description:
        return ((all_hyperparameters, evaluation), description)
    return (all_hyperparameters, evaluation)


def filter_by_space(hyperparameters_and_scores, space):
    """Reject any `hyperparameters_and_scores` tuples whose hyperparameters do not fit in `space`

    Parameters
    ----------
    hyperparameters_and_scores: List of tuples
        Each tuple in list should be a pair of form (hyperparameters <dict>, evaluation <float>),
        where the hyperparameter dict should contain at least the following keys:
        ['model_init_params', 'model_extra_params', 'feature_engineer', 'feature_selector']
    space: `space.space_core.Space`
        The boundaries of the hyperparameters to be searched

    Returns
    -------
    hyperparameters_and_scores: List of tuples
        Filtered to include only those whose hyperparameters fit within `space`"""
    return [_ for _ in hyperparameters_and_scores if does_fit_in_space(_[0], space)]


def does_fit_in_space(root, space):
    """Determine if the subset of `root` identified by `space` fits within dimensions of `space`

    Parameters
    ----------
    root: Object
        Iterable, whose values at the locations specified in `space` will be checked. For each
        dimension in `space`, the dimension's `location`/`name` is looked up in `root`, and the
        value is tested to see if it falls within the dimension's range of allowed values
    space: `space.space_core.Space`
        Instance of :class:`space.space_core.Space` that defines dimension choices for select
        hyperparameters. Each dimension in `space` should have an appropriate `name`
        (or `location`, if necessary) attribute to match `root`

    Returns
    -------
    Boolean
        True if `root` subset (at `space` locations) fits in `space` dimensions. Else, False"""
    return dimension_subset(root, space.names()) in space


def visit_feature_engineer(path, key, value):
    """Helper to be used within a `visit` function intended for a `remap`-like function

    Parameters
    ----------
    path: Tuple
        The path of keys that leads to `key`
    key: String
        The parameter name
    value: Object
        The value of the parameter `key`

    Returns
    -------
    False if the value represents a dataset, or tuple of (`key`, <hash of `value`>). If neither of
    these are returned, a `ContinueRemap` exception is raised

    Raises
    ------
    ContinueRemap
        If a value is not returned by `visit_function_engineer`. For proper functioning, this raised
        `ContinueRemap` is assumed to be handled by the calling `visit` function. Usually, the
        `except` block for `ContinueRemap` will simply continue execution of `visit`

    Examples
    --------
    >>> visit_feature_engineer(("feature_engineer",), "datasets", dict())
    False
    >>> visit_feature_engineer(("feature_engineer", "steps"), "f", lambda _: _)  # pytest: +ELLIPSIS
    ('f', '...')
    >>> visit_feature_engineer(("feature_engineer", "steps"), "foo", lambda _: _)
    Traceback (most recent call last):
        File "optimization_utils.py", line ?, in visit_feature_engineer
    hyperparameter_hunter.exceptions.ContinueRemap: Just keep doing what you were doing
    >>> visit_feature_engineer(("feature_engineer",), "foo", dict())
    Traceback (most recent call last):
        File "optimization_utils.py", line ?, in visit_feature_engineer
    hyperparameter_hunter.exceptions.ContinueRemap: Just keep doing what you were doing
    >>> visit_feature_engineer(("foo",), "bar", dict())
    Traceback (most recent call last):
        File "optimization_utils.py", line ?, in visit_feature_engineer
    hyperparameter_hunter.exceptions.ContinueRemap: Just keep doing what you were doing"""
    if path and path[0] == "feature_engineer":
        # Drop dataset hashes
        if key in ("datasets", "original_hashes", "updated_hashes") and isinstance(value, dict):
            return False
        # Ensure `EngineerStep.f` is hashed
        with suppress(IndexError):
            if path[1] == "steps" and key == "f" and callable(value):
                return key, make_hash_sha256(value)
    raise ContinueRemap


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

    def _visit(path, key, value):
        """If `value` is a descendant of :class:`space.Dimension`, collect inputs, and return True.
        Else, return False"""
        if isinstance(value, (Real, Integer, Categorical)):
            choices.append(((path + (key,)), value))
            return True
        return False

    _ = remap(params, visit=_visit, enter=extra_enter_attrs(iter_attrs))
    return choices


def dimension_subset(hyperparameters, dimensions):
    """Return only the values of `hyperparameters` specified by `dimensions`, in the same order as
    `dimensions`

    Parameters
    ----------
    hyperparameters: Dict
        Dict of hyperparameters containing at least the following keys: ['model_init_params',
        'model_extra_params', 'feature_engineer', 'feature_selector']
    dimensions: List of: (strings, or tuples)
        Locations and order of the values to return from `hyperparameters`. If a value is a string,
        it is assumed to belong to `model_init_params`, and its path will be adjusted accordingly

    Returns
    -------
    List of hyperparameter values"""
    dimensions = [("model_init_params", _) if isinstance(_, str) else _ for _ in dimensions]
    values = [get_path(hyperparameters, _, default=RejectedOptional()) for _ in dimensions]
    return values


if __name__ == "__main__":
    pass
