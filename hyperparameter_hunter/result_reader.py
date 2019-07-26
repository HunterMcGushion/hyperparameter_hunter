##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import IncompatibleCandidateError
from hyperparameter_hunter.feature_engineering import EngineerStep, FeatureEngineer
from hyperparameter_hunter.library_helpers.keras_helper import (
    keras_callback_to_dict,
    keras_initializer_to_dict,
)
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.space.dimensions import Categorical, RejectedOptional
from hyperparameter_hunter.space.space_core import Space
from hyperparameter_hunter.utils.boltons_utils import remap, get_path
from hyperparameter_hunter.utils.general_utils import multi_visit
from hyperparameter_hunter.utils.optimization_utils import (
    does_fit_in_space,
    get_ids_by,
    get_scored_params,
)

##################################################
# Import Miscellaneous Assets
##################################################
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Dict, List, Tuple, Union
import wrapt


def finder_selector(module_name):
    """Selects the appropriate :class:`ResultFinder` to use for `module_name`

    Parameters
    ----------
    module_name: String
        Module from whence the algorithm being used came

    Returns
    -------
    Uninitialized :class:`ResultFinder`, or one of its descendants

    Examples
    --------
    >>> assert finder_selector("Keras") == KerasResultFinder
    >>> assert finder_selector("xgboost") == ResultFinder
    >>> assert finder_selector("lightgbm") == ResultFinder
    """
    if module_name.lower() == "keras":
        return KerasResultFinder
    else:
        return ResultFinder


def update_match_status(target_attr="match_status") -> callable:
    """Build a decorator to apply to class instance methods to record inputs/outputs

    Parameters
    ----------
    target_attr: String, default="match_status"
        Name of dict attribute in the class instance of the decorated method, in which the decorated
        method's inputs and outputs should be recorded. This attribute should be predefined and
        documented by the class whose method is being decorated

    Returns
    -------
    Callable
        Decorator that will save the decorated method's inputs and outputs to the attribute dict
        named by `target_attr`. Decorator assumes that the method will receive at least three
        positional arguments: "exp_id", "params", and "score". "exp_id" is used as the key added to
        `target_attr`, with a dict value, which includes the latter two positional arguments. Each
        time the decorator is invoked with an "exp_id", an additional key is added to its dict that
        is the name of the decorated method, and whose value is the decorated method's output

    See Also
    --------
    :class:`ResultFinder`
        Decorates "does_match..." methods using `update_match_status` in order to keep a detailed
        record of the full pool of candidate Experiments in :attr:`ResultFinder.match_status`

    Examples
    --------
    >>> class X:
    ...     def __init__(self):
    ...         self.match_status = dict()
    ...     @update_match_status()
    ...     def method_a(self, exp_id, params, score):
    ...         return True
    ...     @update_match_status()
    ...     def method_b(self, exp_id, params, score):
    ...         return False
    >>> x = X()
    >>> x.match_status
    {}
    >>> assert x.method_a("foo", None, 0.8) is True
    >>> x.match_status  # doctest: +NORMALIZE_WHITESPACE
    {'foo': {'params': None,
             'score': 0.8,
             'method_a': True}}
    >>> assert x.method_b("foo", None, 0.8) is False
    >>> x.match_status  # doctest: +NORMALIZE_WHITESPACE
    {'foo': {'params': None,
             'score': 0.8,
             'method_a': True,
             'method_b': False}}
    >>> assert x.method_b("bar", "some stuff", 0.5) is False
    >>> x.match_status  # doctest: +NORMALIZE_WHITESPACE
    {'foo': {'params': None,
             'score': 0.8,
             'method_a': True,
             'method_b': False},
     'bar': {'params': 'some stuff',
             'score': 0.5,
             'method_b': False}}
    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)

        getattr(instance, target_attr).setdefault(args[0], dict(params=args[1], score=args[2]))[
            wrapped.__name__
        ] = result

        return result

    return wrapper


def does_match_guidelines(
    candidate_params: dict,
    space: Space,
    template_params: dict,
    visitors=(),
    dims_to_ignore: List[tuple] = None,
) -> bool:
    """Check candidate compatibility with template guideline hyperparameters

    Parameters
    ----------
    candidate_params: Dict
        Candidate Experiment hyperparameters to be compared to `template_params` after processing
    space: Space
        Hyperparameter search space constraints for the current template
    template_params: Dict
        Template hyperparameters to which `candidate_params` will be compared after processing.
        Although the name of the function implies that these will all be guideline hyperparameters,
        this is not a requirement, as any non-guideline hyperparameters will be removed during
        processing by checking `space.names`
    visitors: Callable, or Tuple[callable] (optional)
        Extra `visit` function(s) invoked when
        :func:`~hyperparameter_hunter.utils.boltons_utils.remap`-ing both `template_params` and
        `candidate_params`. Can be used to filter out unwanted values, or to pre-process selected
        values (and more) prior to performing the final compatibility check between the processed
        `candidate_params` and guidelines in `template_params`
    dims_to_ignore: List[tuple] (optional)
        Paths to hyperparameter(s) that should be ignored when comparing `candidate_params` and
        `template_params`. By default, hyperparameters pertaining to verbosity and random states
        are ignored. Paths should be tuples of the form expected by
        :func:`~hyperparameter_hunter.utils.boltons_utils.get_path`. Additionally a path may start
        with None, which acts as a wildcard, matching any hyperparameters whose terminal path nodes
        match the value following None. For example, ``(None, "verbose")`` would match paths such as
        ``("model_init_params", "a", "verbose")`` and ``("model_extra_params", "b", 2, "verbose")``

    Returns
    -------
    Boolean
        True if the processed version of `candidate_params` is equal to the extracted and processed
        guidelines from `template_params`. Else, False"""
    dimensions_to_ignore = [
        (None, "verbose"),
        (None, "silent"),
        (None, "random_state"),
        (None, "seed"),
    ]
    if isinstance(dims_to_ignore, list):
        dimensions_to_ignore.extend(dims_to_ignore)

    # `dimensions_to_ignore` = hyperparameters to be ignored. Filter by all remaining (less
    #   dimensions in `space`, which are also ignored)

    def _visit(path, key, value):
        """Return False if element in space dimensions, or in dimensions being ignored. Else, return
        True. If `value` is of type tuple or set, it will be converted to a list in order to
        simplify comparisons to the JSON-formatted `candidate_params`"""
        # Remove elements whose full paths are in `dimensions_to_ignore` or `space.names()`
        for dim in space.names() + dimensions_to_ignore:
            if (path + (key,) == dim) or (dim[0] is None and dim[-1] == key):
                return False
        # Convert tuples/sets to lists
        if isinstance(value, (tuple, set)):
            return key, list(value)
        return True

    #################### Chain Together Visit Functions ####################
    if callable(visitors):
        visitors = (visitors,)
    visit = multi_visit(*visitors, _visit)
    # Extra `visitors` will be called first, with `_visit` acting as the default visit, called last

    guidelines = remap(template_params, visit=visit)
    # `guidelines` = `template_params` that are neither `space` choices, nor `dimensions_to_ignore`
    return remap(candidate_params, visit=visit) == guidelines


##################################################
# `FeatureEngineer` Matching Utilities
##################################################
def validate_feature_engineer(
    candidate: Union[dict, FeatureEngineer], template: FeatureEngineer
) -> Union[bool, dict, FeatureEngineer]:
    """Check `candidate` "feature_engineer" compatibility with `template` and sanitize `candidate`.
    This is mostly a wrapper around :func:`validate_fe_steps` to ensure different inputs are
    handled properly and to return False, rather than raising `IncompatibleCandidateError`

    Parameters
    ----------
    candidate: Dict, or FeatureEngineer
        Candidate "feature_engineer" to compare to `template`. If compatible with `template`, a
        sanitized version of `candidate` will be returned (described below)
    template: FeatureEngineer
        Template "feature_engineer" to which `candidate` will be compared after processing

    Returns
    -------
    Boolean, dict, or FeatureEngineer
        False if `candidate` is deemed incompatible with `template`. Else, a sanitized `candidate`
        with reinitialized :class:`~hyperparameter_hunter.feature_engineering.EngineerStep` steps
        and with :class:`~hyperparameter_hunter.spaces.dimensions.RejectedOptional` filling in
        missing :class:`~hyperparameter_hunter.spaces.dimensions.Categorical` steps that were
        declared as :attr:`~hyperparameter_hunter.spaces.dimensions.Categorical.optional` by the
        `template`"""
    # Extract `steps` from `candidate`
    if isinstance(candidate, FeatureEngineer):
        steps = candidate.steps
    else:  # `candidate` must be a dict
        steps = candidate["steps"]

    # Dataset hashes in `feature_engineer` and candidates can be ignored, since it is assumed
    #   that candidates here had matching `Environment`s

    try:
        steps = validate_fe_steps(steps, template)

        if isinstance(candidate, FeatureEngineer):
            candidate.steps = steps
        else:  # `candidate` must be a dict
            candidate["steps"] = steps

        return candidate
    except IncompatibleCandidateError:
        return False


def validate_fe_steps(
    candidate: Union[list, FeatureEngineer], template: Union[list, FeatureEngineer]
) -> list:
    """Check `candidate` "feature_engineer" `steps` compatibility with `template` and sanitize
    `candidate`

    Parameters
    ----------
    candidate: List, or FeatureEngineer
        Candidate "feature_engineer" `steps` to compare to `template`. If compatible with
        `template`, a sanitized version of `candidate` will be returned (described below)
    template: List, or FeatureEngineer
        Template "feature_engineer" `steps` to which `candidate` will be compared. `template` is
        also used to sanitize `candidate` (described below)

    Returns
    -------
    List
        If `candidate` is compatible with `template`, returns a list resembling `candidate`, with
        the following changes: 1) all step dicts in `candidate` are reinitialized to proper
        `EngineerStep` instances; and 2) wherever `candidate` was missing a step that was tagged as
        `optional` in `template`, `RejectedOptional` is added. In the end, if a list is returned, it
        is built from `candidate`, guaranteed to be the same length as `template` and guaranteed
        to contain only `EngineerStep` and `RejectedOptional` instances

    Raises
    ------
    IncompatibleCandidateError
        If `candidate` is incompatible with `template`. `candidate` may be incompatible with
        `template` for any of the following reasons:

        1. `candidate` has more steps than `template`
        2. `candidate` has a step that differs from a concrete (non-`Categorical`) `template` step
        2. `candidate` has a step that differs from a concrete (non-`Categorical`) `template` step
        3. `candidate` has a step that does not fit in a `Categorical` `template` step
        4. `candidate` is missing a concrete step in `template`
        5. `candidate` is missing a non-`optional` `Categorical` step in `template`"""
    # Extract `steps` if given `FeatureEngineer`
    if isinstance(candidate, FeatureEngineer):
        candidate = candidate.steps
    if isinstance(template, FeatureEngineer):
        template = template.steps

    if len(template) == 0:
        if len(candidate) == 0:
            # All steps have been exhausted and passed, so `candidate` was a match
            return []
        # `candidate` steps remain while `template` is empty, so `candidate` is incompatible
        raise IncompatibleCandidateError(candidate, template)

    #################### Categorical Template Step ####################
    if isinstance(template[-1], Categorical):
        #################### Optional Categorical Template Step ####################
        if template[-1].optional is True:
            if len(candidate) == 0 or (candidate[-1] not in template[-1]):
                # `candidate` is either empty or doesn't match an `optional` template step
                candidate.append(RejectedOptional())

        #################### Non-Optional Categorical Template Step ####################
        elif len(candidate) == 0:
            # `candidate` is empty while non-`optional` `template` steps remain - Incompatible
            raise IncompatibleCandidateError(candidate, template)
        elif candidate[-1] not in template[-1]:
            raise IncompatibleCandidateError(candidate, template)

    elif len(candidate) == 0:
        raise IncompatibleCandidateError(candidate, template)

    #################### Concrete Template Step ####################
    elif candidate[-1] != template[-1]:
        raise IncompatibleCandidateError(candidate, template)

    #################### Reinitialize EngineerStep Dict ####################
    if isinstance(candidate[-1], dict):
        if isinstance(template[-1], Categorical):
            candidate[-1] = EngineerStep.honorary_step_from_dict(candidate[-1], template[-1])
        elif isinstance(template[-1], EngineerStep) and template[-1] == candidate[-1]:
            candidate[-1] = template[-1]  # Adopt template value if `EngineerStep` equivalent

    return validate_fe_steps(candidate[:-1], template[:-1]) + [candidate[-1]]


##################################################
# ResultFinder Classes
##################################################
class ResultFinder:
    def __init__(
        self,
        algorithm_name,
        module_name,
        cross_experiment_key,
        target_metric,
        space,
        leaderboard_path,
        descriptions_dir,
        model_params,
        sort=None,  # TODO: Unfinished - To be used in `get_scored_params`/`experiment_ids`
    ):
        """Locate saved Experiments that are compatible with the given constraints

        Parameters
        ----------
        algorithm_name: String
            Name of the algorithm whose hyperparameters are being optimized
        module_name: String
            Name of the module from whence the algorithm being used came
        cross_experiment_key: String
            :attr:`hyperparameter_hunter.environment.Environment.cross_experiment_key` produced by
            the current `Environment`
        target_metric: Tuple
            Path denoting the metric to be used. The first value should be one of {"oof",
            "holdout", "in_fold"}, and the second value should be the name of a metric supplied in
            :attr:`hyperparameter_hunter.environment.Environment.metrics_params`
        space: Space
            Instance of :class:`~hyperparameter_hunter.space.space_core.Space`, defining
            hyperparameter search space constraints
        leaderboard_path: String
            Path to a leaderboard file, whose listed Experiments will be tested for compatibility
        descriptions_dir: String
            Path to a directory containing the description files of saved Experiments
        model_params: Dict
            All hyperparameters for the model, both concrete and choice. Common keys include
            "model_init_params" and "model_extra_params", both of which can be pointers to dicts of
            hyperparameters. Additionally, "feature_engineer" may be included with an instance of
            :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`
        sort: {"target_asc", "target_desc", "chronological", "reverse_chronological"}, or int
            ... Experimental...
            How to sort the experiment results that fit within the given constraints

            * "target_asc": Sort from experiments with the lowest value for `target_metric` to
              those with the greatest
            * "target_desc": Sort from experiments with the highest value for `target_metric` to
              those with the lowest
            * "chronological": Sort from oldest experiments to newest
            * "reverse_chronological": Sort from newest experiments to oldest
            * int: Random seed with which to shuffle experiments

        Attributes
        ----------
        similar_experiments: List[Tuple[dict, Number, str]]
            Candidate saved Experiment results that are fully compatible with the template
            hyperparameters. Each value is a tuple triple of
            (<hyperparameters>, <`target_metric` value>, <candidate `experiment_id`>).
            `similar_experiments` is composed of the "finalists" from :attr:`match_status`
        match_status: Dict[str, dict]
            Record of the hyperparameters and `target_metric` values for all discovered Experiments,
            keyed by values of :attr:`experiment_ids`. Each value is a dict containing two keys:
            "params" (hyperparameter dict), and "score" (`target_metric` value number). In addition
            to these two keys, a key may be added for every `ResultFinder` method decorated by
            :func:`update_match_status`. The exact key will be the name of the method invoked (which
            will always start with "does_match", followed by the name of the hyperparameter group
            being checked). The value for each "does_match..." key is the value returned by that
            method, whose truthiness dictates whether the candidate Experiment was a successful
            match to the template hyperparameters for that group. For example, a `match_status`
            entry for one Experiment could look like this::

                {
                    "params": <dict of hyperparameters for candidate>,
                    "score": 0.42,  # `target_metric` value for candidate Experiment
                    "does_match_init_params_space": True,
                    "does_match_init_params_guidelines": False,
                    "does_match_extra_params_space": False,
                    "does_match_extra_params_guidelines": True,
                    "does_match_feature_engineer": <`FeatureEngineer`>,  # Still truthy
                }

            Note that "model_init_params" and "model_extra_params" both check the compatibility of
            "space" choices and concrete "guidelines" separately. Conversely, "feature_engineer" is
            checked in its entirety by the single :meth:`does_match_feature_engineer`. Also note
            that "does_match..." values are not restricted to booleans. For instance,
            "does_match_feature_engineer" may be set to a reinitialized `FeatureEngineer`, which is
            still truthy, even though it's not True. If all of the "does_match..." keys have truthy
            values, the candidate is a full match and is added to :attr:`similar_experiments`

        Methods
        -------
        find
            Performs actual matching work by populating both :attr:`similar_experiments` and
            :attr:`match_status`

        See Also
        --------
        :func:`update_match_status`
            Used to decorate "does_match..." methods in order to keep a detailed record of the full
            pool of candidate Experiments in :attr:`match_status`. Aside from being used to compile
            the list of finalist :attr:`similar_experiments`, :attr:`match_status` is helpful for
            debugging purposes, specifically figuring out which aspects of a candidate are
            incompatible with the template"""
        self.algorithm_name = algorithm_name
        self.module_name = module_name
        self.cross_experiment_key = cross_experiment_key
        self.target_metric = target_metric
        self.space = space
        self.leaderboard_path = leaderboard_path
        self.descriptions_dir = descriptions_dir
        self.model_params = model_params
        self.sort = sort

        self._experiment_ids = None
        self._mini_spaces = None

        self.match_status = {}
        self.similar_experiments = []

    @property
    def experiment_ids(self) -> List[str]:
        """Experiment IDs in the target Leaderboard that match :attr:`algorithm_name` and
        :attr:`cross_experiment_key`

        Returns
        -------
        List[str]
            All saved Experiment IDs listed in the Leaderboard at :attr:`leaderboard_path` that
            match the :attr:`algorithm_name` and :attr:`cross_experiment_key` of the template"""
        if self._experiment_ids is None:
            # TODO: If `sort`-ing chronologically, can use the "experiment_#" column in leaderboard
            self._experiment_ids = get_ids_by(
                leaderboard_path=self.leaderboard_path,
                algorithm_name=self.algorithm_name,
                cross_experiment_key=self.cross_experiment_key,
                hyperparameter_key=None,
            )

        return self._experiment_ids

    @property
    def mini_spaces(self) -> Dict[str, Space]:
        """Separate :attr:`space` into subspaces based on :attr:`model_params` keys

        Returns
        -------
        Dict[str, Space]
            Dict of subspaces, wherein keys are all keys of :attr:`model_params`. Each key's
            corresponding value is a filtered subspace, containing all the dimensions in
            :attr:`space` whose name tuples start with that key. Keys will usually be one of the
            core hyperparameter group names ("model_init_params", "model_extra_params",
            "feature_engineer", "feature_selector")

        Examples
        --------
        >>> from hyperparameter_hunter import Integer
        >>> def es_0(all_inputs):
        ...     return all_inputs
        >>> def es_1(all_inputs):
        ...     return all_inputs
        >>> def es_2(all_inputs):
        ...     return all_inputs
        >>> s = Space([
        ...     Integer(900, 1500, name=("model_init_params", "max_iter")),
        ...     Categorical(["svd", "cholesky", "lsgr"], name=("model_init_params", "solver")),
        ...     Categorical([es_1, es_2], name=("feature_engineer", "steps", 1)),
        ... ])
        >>> rf = ResultFinder(
        ...     "a", "b", "c", ("oof", "d"), space=s, leaderboard_path="e", descriptions_dir="f",
        ...     model_params=dict(
        ...         model_init_params=dict(
        ...             max_iter=s.dimensions[0], normalize=True, solver=s.dimensions[1],
        ...         ),
        ...         feature_engineer=FeatureEngineer([es_0, s.dimensions[2]]),
        ...     ),
        ... )
        >>> rf.mini_spaces  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'model_init_params': Space([Integer(low=900, high=1500),
                                     Categorical(categories=('svd', 'cholesky', 'lsgr'))]),
         'feature_engineer': Space([Categorical(categories=(<function es_1 at ...>,
                                                            <function es_2 at ...>))])}
        """
        if self._mini_spaces is None:
            self._mini_spaces = {}

            # Need to use `space.names` and `get_by_name` because the `location` attribute should
            #   be preferred to `name` if it exists, which is the case for Keras
            names = self.space.names()

            for param_group_name in self.model_params.keys():
                # Use `space.get_by_name` to respect `location` (see comment above)
                self._mini_spaces[param_group_name] = Space(
                    [self.space.get_by_name(n) for n in names if n[0] == param_group_name]
                )

        return self._mini_spaces

    def find(self):
        """Execute full result-finding workflow, populating :attr:`similar_experiments`

        See Also
        --------
        :func:`update_match_status`
            Used to decorate "does_match..." methods in order to keep a detailed record of the full
            pool of candidate Experiments in :attr:`match_status`. Aside from being used to compile
            the list of finalist :attr:`similar_experiments`, :attr:`match_status` is helpful for
            debugging purposes, specifically figuring out which aspects of a candidate are
            incompatible with the template
        :meth:`does_match_feature_engineer`
            Performs special functionality beyond that of the other "does_match..." methods, namely
            providing an updated "feature_engineer" value for compatible candidates to use.
            Specifics are documented in :meth:`does_match_feature_engineer`"""
        for exp_id in self.experiment_ids:
            description_path = f"{self.descriptions_dir}/{exp_id}.json"
            # TODO: Get `description` from `get_scored_params` - Take whatever value `sort` needs
            params, score = get_scored_params(description_path, self.target_metric)

            #################### Match Init Params ####################
            self.does_match_init_params_space(exp_id, params["model_init_params"], score)

            multi_targets = [("model_init_params", "compile_params", "optimizer")]
            if self.module_name == "keras" and multi_targets[0] in self.space.names():
                self.does_match_init_params_guidelines_multi(
                    exp_id, params["model_init_params"], score, multi_targets[0][1:]
                )
            else:
                self.does_match_init_params_guidelines(exp_id, params["model_init_params"], score)

            #################### Match Extra Params ####################
            self.does_match_extra_params_space(exp_id, params["model_extra_params"], score)
            self.does_match_extra_params_guidelines(exp_id, params["model_extra_params"], score)

            #################### Match Feature Engineer ####################
            # NOTE: Matching "feature_engineer" is critically different from the other "does_match"
            #   methods. `does_match_feature_engineer` builds on the straight-forward compatibility
            #   checks of the others by returning an updated "feature_engineer" for the candidate
            #   if compatible. See :meth:`does_match_feature_engineer` for details
            params["feature_engineer"] = self.does_match_feature_engineer(
                exp_id, params["feature_engineer"], score
            )

            # Since updated "feature_engineer" is saved in `params`, clean up `match_status` entry
            if self.match_status[exp_id]["does_match_feature_engineer"] is not False:
                self.match_status[exp_id]["does_match_feature_engineer"] = True

            #################### Determine Overall Match ####################
            if all(v for k, v in self.match_status[exp_id].items() if k.startswith("does_match")):
                self.similar_experiments.append((params, score, exp_id))

        G.debug_(
            "Matching Experiments:  {}  (Candidates:  {})".format(
                len(self.similar_experiments), len(self.experiment_ids)
            )
        )
        # TODO: Add option to print table of all candidates and their `match_status` values

    ##################################################
    # Match Helpers: Feature Engineering
    ##################################################
    # noinspection PyUnusedLocal
    @update_match_status(target_attr="match_status")
    def does_match_feature_engineer(
        self, exp_id, params, score
    ) -> Union[bool, dict, FeatureEngineer]:
        """Check candidate compatibility with `feature_engineer` template guidelines and space
        choices. This method is different from the other "does_match..." methods in two important
        aspects:

        1. It checks both guidelines and choices in a single method
        2. It returns an updated `feature_engineer` for compatible candidates, rather than True

        Parameters
        ----------
        exp_id: String
            Candidate Experiment ID
        params: Dict
            Candidate "feature_engineer" to compare to the template in :attr:`model_params`. This
            should always be a dict, not an instance of `FeatureEngineer`, which is not the case
            for the template "feature_engineer" in :attr:`model_params`
        score: Number
            Value of the candidate Experiment's target metric

        Returns
        -------
        Boolean, dict, or FeatureEngineer
            Expanding on the second difference noted in the description, False will still be
            returned if the candidate is deemed incompatible with the template (as is the case with
            the other "does_match..." methods). The return value differs with compatible candidates
            in order to provide a `feature_engineer` with reinitialized
            :class:`~hyperparameter_hunter.feature_engineering.EngineerStep` steps and to fill in
            missing :class:`~hyperparameter_hunter.spaces.dimensions.Categorical` steps that were
            declared as :attr:`~hyperparameter_hunter.spaces.dimensions.Categorical.optional` by the
            template. This updated `feature_engineer` is the value that then gets included in the
            candidate's :attr:`similar_experiments` entry (assuming candidate is a full match)"""
        return validate_feature_engineer(params, self.model_params["feature_engineer"])

    ##################################################
    # Match Helpers: Model Init Params
    ##################################################
    # noinspection PyUnusedLocal
    @update_match_status(target_attr="match_status")
    def does_match_init_params_space(self, exp_id, params, score) -> bool:
        """Check candidate compatibility with `model_init_params` template space choices

        Parameters
        ----------
        exp_id: String
            Candidate Experiment ID
        params: Dict
            Candidate "model_init_params" to compare to the template in :attr:`model_params`
        score: Number
            Value of the candidate Experiment's target metric

        Returns
        -------
        Boolean
            True if candidate `params` fit in `model_init_params` space choices. Else, False"""
        return does_fit_in_space(
            dict(model_init_params=params), self.mini_spaces["model_init_params"]
        )

    # noinspection PyUnusedLocal
    @update_match_status(target_attr="match_status")
    def does_match_init_params_guidelines(
        self, exp_id, params, score, template_params=None
    ) -> bool:
        """Check candidate compatibility with `model_init_params` template guidelines

        Parameters
        ----------
        exp_id: String
            Candidate Experiment ID
        params: Dict
            Candidate "model_init_params" to compare to the template in :attr:`model_params`
        score: Number
            Value of the candidate Experiment's target metric
        template_params: Dict (optional)
            If given, used as the template hyperparameters against which to compare candidate
            `params`, rather than the standard guideline template of the "model_init_params" in
            :attr:`model_params`. This is used by :meth:`does_match_init_params_guidelines_multi`

        Returns
        -------
        Boolean
            True if candidate `params` match `model_init_params` guidelines. Else, False

        Notes
        -----
        Template hyperparameters are generally considered "guidelines" if they are declared as
        concrete values, rather than space choices present in :attr:`space`"""
        _res = does_match_guidelines(
            dict(model_init_params=params),
            self.mini_spaces["model_init_params"],
            dict(model_init_params=(template_params or self.model_params["model_init_params"])),
            dims_to_ignore=[
                ("model_init_params", "build_fn"),
                ("model_init_params", "n_jobs"),
                ("model_init_params", "nthread"),
                # TODO: Remove below once loss_functions are hashed in description files
                ("model_init_params", "compile_params", "loss_functions"),
            ],
        )

        return _res

    def does_match_init_params_guidelines_multi(self, exp_id, params, score, location) -> bool:
        """Check candidate compatibility with `model_init_params` template guidelines when a
        guideline hyperparameter is directly affected by another hyperparameter that is given as
        a space choice

        Parameters
        ----------
        exp_id: String
            Candidate Experiment ID
        params: Dict
            Candidate "model_init_params" to compare to the template in :attr:`model_params`
        score: Number
            Value of the candidate Experiment's target metric
        location: Tuple
            Location of the hyperparameter space choice that affects the acceptable guideline values
            of a particular hyperparameter. In other words, this is the path of a hyperparameter,
            which, if changed, would change the expected default value of another hyperparameter

        Returns
        -------
        Boolean
            True if candidate `params` match `model_init_params` guidelines. Else, False

        Notes
        -----
        This is used for Keras Experiments when the `optimizer` value in a model's `compile_params`
        is given as a hyperparameter space choice. Each possible value of `optimizer` prescribes
        different default values for the `optimizer_params` argument, so special measures need to be
        taken to ensure the correct Experiments are declared to fit within the constraints"""
        _model_params = deepcopy(self.model_params["model_init_params"])

        if location == ("compile_params", "optimizer"):
            from keras.optimizers import get as k_opt_get

            update_location = ("compile_params", "optimizer_params")
            # `update_location` = Path to hyperparameter whose default value depends on `location`
            allowed_values = get_path(_model_params, location).bounds
            # `allowed_values` = Good `("model_init_params", "compile_params", "optimizer")` values

            #################### Handle First Value (Dummy) ####################
            is_match = self.does_match_init_params_guidelines(exp_id, params, score)
            # The first value gets handled separately from the rest because the value at
            #   `update_location` is set according to `allowed_values[0]`. For the remaining
            #   `allowed_values`, we need to manually set `update_location` for each

            # If the first value was a match, the below `while` loop will never be entered because
            #   `is_match` is already True

            #################### Handle Remaining Values ####################
            allowed_val_index = 1
            while is_match is not True and allowed_val_index < len(allowed_values):
                allowed_val = allowed_values[allowed_val_index]
                # Determine current default value for the dependent hyperparameter
                updated_val = k_opt_get(allowed_val).get_config()

                # Set value at `update_location` to `updated_val`, then check if params match
                def _visit(path, key, value):
                    """If `path` + `key` == `update_location`, return default for this choice. Else,
                    default_visit"""
                    if path + (key,) == update_location:
                        return (key, updated_val)
                    return (key, value)

                is_match = self.does_match_init_params_guidelines(
                    exp_id, params, score, template_params=remap(_model_params, visit=_visit)
                )
                # If `is_match` is True, the loop stops and :attr:`match_status`'s value at `exp_id`
                #   for `does_match_init_params_guidelines` remains truthy

                allowed_val_index += 1
            return is_match
        else:
            raise ValueError("Received unhandled location: {}".format(location))

    ##################################################
    # Match Helpers: Model Extra Params
    ##################################################
    # noinspection PyUnusedLocal
    @update_match_status(target_attr="match_status")
    def does_match_extra_params_space(self, exp_id, params, score) -> bool:
        """Check candidate compatibility with `model_extra_params` template space choices

        Parameters
        ----------
        exp_id: String
            Candidate Experiment ID
        params: Dict
            Candidate "model_extra_params" to compare to the template in :attr:`model_params`
        score: Number
            Value of the candidate Experiment's target metric

        Returns
        -------
        Boolean
            True if candidate `params` fit in `model_extra_params` space choices. Else, False"""
        return does_fit_in_space(
            dict(model_extra_params=params), self.mini_spaces["model_extra_params"]
        )

    # noinspection PyUnusedLocal
    @update_match_status(target_attr="match_status")
    def does_match_extra_params_guidelines(self, exp_id, params, score) -> bool:
        """Check candidate guideline compatibility with `model_extra_params` template

        Parameters
        ----------
        exp_id: String
            Candidate Experiment ID
        params: Dict
            Candidate "model_extra_params" to compare to the template in :attr:`model_params`
        score: Number
            Value of the candidate Experiment's target metric

        Returns
        -------
        Boolean
            True if candidate `params` match `model_extra_params` guidelines. Else, False"""

        # noinspection PyUnusedLocal
        def visit_empty_dicts(path, key, value):
            """Remove empty dicts in ("model_extra_params"). Simplify comparison between experiments
            with no `model_extra_params` and, for example, `dict(fit=dict(verbose=True))`"""
            if path and path[0] == "model_extra_params" and value == {}:
                return False

        _res = does_match_guidelines(
            dict(model_extra_params=params),
            self.mini_spaces["model_extra_params"],
            dict(model_extra_params=self.model_params["model_extra_params"]),
            visitors=(visit_empty_dicts,),
        )

        return _res


class KerasResultFinder(ResultFinder):
    def __init__(
        self,
        algorithm_name,
        module_name,
        cross_experiment_key,
        target_metric,
        space,
        leaderboard_path,
        descriptions_dir,
        model_params,
        sort=None,  # TODO: Unfinished - To be used in `get_scored_params`/`experiment_ids`
    ):
        """ResultFinder for locating saved Keras Experiments compatible with the given constraints

        Parameters
        ----------
        algorithm_name: String
            Name of the algorithm whose hyperparameters are being optimized
        module_name: String
            Name of the module from whence the algorithm being used came
        cross_experiment_key: String
            :attr:`hyperparameter_hunter.environment.Environment.cross_experiment_key` produced by
            the current `Environment`
        target_metric: Tuple
            Path denoting the metric to be used. The first value should be one of {"oof",
            "holdout", "in_fold"}, and the second value should be the name of a metric supplied in
            :attr:`hyperparameter_hunter.environment.Environment.metrics_params`
        space: Space
            Instance of :class:`~hyperparameter_hunter.space.space_core.Space`, defining
            hyperparameter search space constraints
        leaderboard_path: String
            Path to a leaderboard file, whose listed Experiments will be tested for compatibility
        descriptions_dir: String
            Path to a directory containing the description files of saved Experiments
        model_params: Dict
            Concrete hyperparameters for the model. Common keys include "model_init_params" and
            "model_extra_params", both of which can be pointers to dicts of hyperparameters.
            Additionally, "feature_engineer" may be included with an instance of
            :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`
        sort: {"target_asc", "target_desc", "chronological", "reverse_chronological"}, or int
            ... Experimental...
            How to sort the experiment results that fit within the given constraints

            * "target_asc": Sort from experiments with the lowest value for `target_metric` to
              those with the greatest
            * "target_desc": Sort from experiments with the highest value for `target_metric` to
              those with the lowest
            * "chronological": Sort from oldest experiments to newest
            * "reverse_chronological": Sort from newest experiments to oldest
            * int: Random seed with which to shuffle experiments"""
        super().__init__(
            algorithm_name=algorithm_name,
            module_name=module_name,
            cross_experiment_key=cross_experiment_key,
            target_metric=target_metric,
            space=space,
            leaderboard_path=leaderboard_path,
            descriptions_dir=descriptions_dir,
            model_params=model_params,
            sort=sort,
        )

        from keras.callbacks import Callback as BaseKerasCallback
        from keras.initializers import Initializer as BaseKerasInitializer

        # noinspection PyUnusedLocal
        def _visit(path, key, value):
            """If `value` is `BaseKerasCallback` or `BaseKerasInitializer`, return dict
            representation. Else default_visit"""
            if isinstance(value, BaseKerasCallback):
                return (key, keras_callback_to_dict(value))
            if isinstance(value, BaseKerasInitializer):
                return (key, keras_initializer_to_dict(value))
            return (key, value)

        self.model_params = remap(self.model_params, visit=_visit)

        # Below cleans out the temporary "params" dict built by `keras_optimization_helper`.
        #   It exists in order to pass concrete values for choices during optimization through the
        #   Keras model `build_fn`. However, at this stage, it just gets in the way since
        #   :attr:`space` defines the choices, and their `location`s point to where they are within
        #   :attr:`model_params`. Not deleting them would basically duplicate all choice Dimensions
        try:
            del self.model_params["model_extra_params"]["params"]
        except KeyError:
            pass


##################################################
# Utilities
##################################################
def has_experiment_result_file(results_dir, experiment_id, result_type=None):
    """Check if the specified result files exist in `results_dir` for Experiment `experiment_id`

    Parameters
    ----------
    results_dir: String
        HyperparameterHunterAssets directory in which to search for Experiment result files
    experiment_id: String, or BaseExperiment
        ID of the Experiment whose result files should be searched for in `results_dir`. If not
        string, should be an instance of a descendant of
        :class:`~hyperparameter_hunter.experiments.BaseExperiment` with an "experiment_id" attribute
    result_type: List, or string (optional)
        Result file types for which to check. Valid values include any subdirectory name that can be
        included in "HyperparameterHunterAssets/Experiments" by default: ["Descriptions",
        "Heartbeats", "PredictionsOOF", "PredictionsHoldout", "PredictionsTest", "ScriptBackups"].
        If string, should be one of the aforementioned strings, or "ALL" to use all of the results.
        If list, should be a subset of the aforementioned list of valid values. Else, default is
        ["Descriptions", "Heartbeats", "PredictionsOOF", "ScriptBackups"]. The returned boolean
        signifies whether ALL of the `result_type` files were found, not whether ANY were found

    Returns
    -------
    Boolean
        True if all result files specified by `result_type` exist in `results_dir` for the
        Experiment specified by `experiment_id`. Else, False"""
    experiment_id = experiment_id if isinstance(experiment_id, str) else experiment_id.experiment_id

    #################### Format `result_type` ####################
    if not result_type:
        result_type = ["Descriptions", "Heartbeats", "PredictionsOOF", "ScriptBackups"]
    elif result_type == "ALL":
        result_type = [
            "Descriptions",
            "Heartbeats",
            "PredictionsOOF",
            "PredictionsHoldout",
            "PredictionsTest",
            "ScriptBackups",
        ]
    if isinstance(result_type, str):
        result_type = [result_type]

    for subdir in result_type:
        #################### Select Result File Suffix ####################
        if subdir == "Descriptions":
            suffix = ".json"
        elif subdir == "Heartbeats":
            suffix = ".log"
        elif subdir == "ScriptBackups":
            suffix = ".py"
        elif subdir.startswith("Predictions"):
            suffix = ".csv"
        else:
            raise ValueError(f"Cannot resolve suffix for subdir `result_type`: {subdir}")

        #################### Check "Experiments" Directory ####################
        if results_dir.endswith("HyperparameterHunterAssets"):
            experiments_dir = Path(results_dir) / "Experiments"
        else:
            experiments_dir = Path(results_dir) / "HyperparameterHunterAssets" / "Experiments"

        if not (experiments_dir / subdir / f"{experiment_id}{suffix}").exists():
            return False

    return True
