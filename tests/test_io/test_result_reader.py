##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import FeatureEngineer, Categorical
from hyperparameter_hunter.io.exceptions import IncompatibleCandidateError
from hyperparameter_hunter.io.result_reader import validate_fe_steps
from hyperparameter_hunter.space.dimensions import RejectedOptional

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


def CAT(*categories, **kwargs):
    """Experimental `Categorical` syntax, wherein all parameters other than `categories` are
    keyword-only arguments and all non-keyword arguments comprise `categories` itself. Furthermore,
    Python's Ellipsis (`...`) may be provided as the last non-keyword argument as an alternative to
    the more verbose `optional=True`

    Parameters
    ----------
    *categories: Tuple
        Values of the returned `Categorical`'s `categories`. If the last element of `categories` is
        `...`, it will be removed and the value of "optional" in `kwargs` will be overridden to True
    **kwargs: Dict
        Keyword arguments provided to initialize `Categorical`. In other words, all `Categorical`
        arguments other than `categories`. If the last element of `categories` is `...`, then the
        value of "optional" in `kwargs` will be overridden to True, whether or not "optional" was
        already given in `kwargs`

    Returns
    -------
    Categorical
        Instantiated with `categories` and `kwargs`

    Examples
    --------
    >>> c0 = Categorical(["a", "b", "c"], optional=True)
    >>> c1 = CAT("a", "b", "c", ...)
    >>> assert c0 == c1
    >>> c2 = Categorical([5, 10, 15], transform="identity")
    >>> c3 = CAT(5, 10, 15, transform="identity")
    >>> assert c2 == c3
    """
    if categories[-1] is ...:
        categories = categories[:-1]
        kwargs["optional"] = True

    return Categorical(list(categories), **kwargs)


##################################################
# Dummy EngineerStep Functions
##################################################
def es_a(all_inputs):
    return all_inputs


def es_b(all_inputs):
    return all_inputs


def es_c(all_inputs):
    return all_inputs


def es_d(all_inputs):
    return all_inputs


def es_e(all_inputs):
    return all_inputs


##################################################
# Fixtures
##################################################
@pytest.fixture(
    params=[FeatureEngineer, lambda _: FeatureEngineer(_).get_key_data()["steps"]],
    ids=["EngineerSteps", "step_dicts"],
)
def candidate_step_cast(request):
    """Processing method applied to `candidate` to produce the candidate steps passed to
    :func:`~hyperparameter_hunter.result_reader.validate_fe_steps`. May be either 1) instantiation
    as a `FeatureEngineer` (which is how `template` is processed), or 2) result of invoking
    :meth:`~hyperparameter_hunter.feature_engineering.FeatureEngineer.get_key_data` on the
    former, then taking its "steps" value. The second method produces a list of
    `EngineerStep`-like dicts, which more closely resembles a candidate retrieved from a saved
    Experiment result description file"""
    return request.param


##################################################
# `validate_fe_steps` Match Tests
##################################################
@pytest.mark.parametrize(
    ["candidate", "template", "expected"],
    [
        #################### Non-Optional Scenarios ####################
        pytest.param([es_a], [es_a], [es_a], id="concrete_single"),
        pytest.param([es_a], [CAT(es_a, es_b)], [es_a], id="cat_single_0"),
        pytest.param([es_b], [CAT(es_a, es_b)], [es_b], id="cat_single_1"),
        pytest.param([es_a, es_c], [CAT(es_a, es_b), es_c], [es_a, es_c], id="cat+concrete"),
        pytest.param(
            [es_a, es_c, es_d],
            [CAT(es_a, es_b), es_c, CAT(es_d, es_e)],
            [es_a, es_c, es_d],
            id="cat+concrete+cat",
        ),
        pytest.param(
            [es_a, es_c, es_e],
            [CAT(es_a, es_b), CAT(es_c, es_d), es_e],
            [es_a, es_c, es_e],
            id="cat+cat+concrete",
        ),
        pytest.param(
            [es_a, es_c, es_e],
            [es_a, CAT(es_b, es_c), CAT(es_d, es_e)],
            [es_a, es_c, es_e],
            id="concrete+cat+cat",
        ),
        #################### Exclusively Optional Scenarios ####################
        pytest.param([es_a], [CAT(es_a, ...)], [es_a], id="opt_single_0"),
        pytest.param([], [CAT(es_a, ...)], [RejectedOptional()], id="opt_single_1"),
        pytest.param(
            [],
            [CAT(es_a, ...), CAT(es_b, ...), CAT(es_c, ...)],
            [RejectedOptional(), RejectedOptional(), RejectedOptional()],
            id="opt+opt+opt_0",
        ),
        pytest.param(
            [es_a],
            [CAT(es_a, ...), CAT(es_b, ...), CAT(es_c, ...)],
            [es_a, RejectedOptional(), RejectedOptional()],
            id="opt+opt+opt_1",
        ),
        pytest.param(
            [es_b],
            [CAT(es_a, ...), CAT(es_b, ...), CAT(es_c, ...)],
            [RejectedOptional(), es_b, RejectedOptional()],
            id="opt+opt+opt_2",
        ),
        pytest.param(
            [es_c],
            [CAT(es_a, ...), CAT(es_b, ...), CAT(es_c, ...)],
            [RejectedOptional(), RejectedOptional(), es_c],
            id="opt+opt+opt_3",
        ),
        #################### Mixed Optional Scenarios ####################
        pytest.param(
            [es_c, es_d],
            [CAT(es_a, es_b, ...), es_c, CAT(es_d, es_e)],
            [RejectedOptional(), es_c, es_d],
            id="opt+concrete+cat",
        ),
        pytest.param(
            [es_c],
            [CAT(es_a, es_b, ...), es_c, CAT(es_d, es_e, ...)],
            [RejectedOptional(), es_c, RejectedOptional()],
            id="opt+concrete+opt",
        ),
        pytest.param(
            [es_e],
            [CAT(es_a, es_b, ...), CAT(es_c, es_d, ...), es_e],
            [RejectedOptional(), RejectedOptional(), es_e],
            id="opt+opt+concrete",
        ),
        pytest.param(
            [es_e],
            [es_e, CAT(es_a, es_b, ...), CAT(es_c, es_d, ...)],
            [es_e, RejectedOptional(), RejectedOptional()],
            id="concrete+opt+opt",
        ),
        pytest.param(
            [es_e],
            [CAT(es_a, es_b, ...), CAT(es_c, es_d, ...), CAT(es_a, es_e)],
            [RejectedOptional(), RejectedOptional(), es_e],
            id="opt+opt+cat",
        ),
        pytest.param(
            [es_e],
            [CAT(es_a, es_e), CAT(es_a, es_b, ...), CAT(es_c, es_d, ...)],
            [es_e, RejectedOptional(), RejectedOptional()],
            id="cat+opt+opt",
        ),
    ],
)
def test_validate_fe_steps(candidate, template, expected, candidate_step_cast):
    """Test that `validate_fe_steps` produces the `expected` output

    Parameters
    ----------
    candidate: List
        `candidate` value given to :func:`~hyperparameter_hunter.result_reader.validate_fe_steps`
    template: List
        `template` value given to :func:`~hyperparameter_hunter.result_reader.validate_fe_steps`
    expected: List
        Output expected from invoking `validate_fe_steps` with `candidate` and `template`"""
    actual = validate_fe_steps(candidate_step_cast(candidate), FeatureEngineer(template))

    # Because `actual` is going to be a list of `EngineerStep`/`RejectedOptional`, `expected` must
    #   also be passed through a `FeatureEngineer` to convert each function to an `EngineerStep`
    assert actual == FeatureEngineer(expected).steps


##################################################
# `validate_fe_steps` `IncompatibleCandidateError` Tests
##################################################
@pytest.mark.parametrize(
    ["candidate", "template"],
    [
        ([es_a, es_b], [es_a]),
        ([es_a, es_b, es_d], [es_a, es_b]),
        ([es_a, es_b, es_d], [es_a, CAT(es_b)]),
        ([es_a, es_b, es_d], [es_a, CAT(es_b, es_c)]),
        ([es_a, es_b, es_d], [es_a, CAT(es_b, es_c, ...)]),
    ],
)
def test_validate_fe_steps_error_candidate_too_big(candidate, template, candidate_step_cast):
    """Test that `IncompatibleCandidateError` is raised by `validate_fe_steps` when `candidate`
    has more steps than `template`. See `test_validate_fe_steps` for parameter descriptions"""
    with pytest.raises(IncompatibleCandidateError):
        validate_fe_steps(candidate_step_cast(candidate), FeatureEngineer(template))


@pytest.mark.parametrize(
    ["candidate", "template"],
    [
        ([es_a], [es_b]),
        ([es_a], [es_b, CAT(es_c, ...)]),
        ([es_a, es_d], [CAT(es_a, es_b), es_c]),
        ([es_a, es_d], [CAT(es_a, es_b, ...), es_c]),
        ([es_a, es_d], [es_a, es_b, CAT(es_c, ...)]),
        ([es_a, es_d, es_c], [es_a, es_b, CAT(es_c, ...)]),
        ([es_a, es_d, es_c], [es_a, es_b, es_c]),
    ],
)
def test_validate_fe_steps_error_concrete_mismatch(candidate, template, candidate_step_cast):
    """Test that `IncompatibleCandidateError` is raised by `validate_fe_steps` when `candidate`
    has a step that differs from a concrete (non-`Categorical`) step in `template`. See
    `test_validate_fe_steps` for parameter descriptions"""
    with pytest.raises(IncompatibleCandidateError):
        validate_fe_steps(candidate_step_cast(candidate), FeatureEngineer(template))


@pytest.mark.parametrize(
    ["candidate", "template"],
    [
        ([es_a], [CAT(es_b, es_c)]),
        ([es_a], [CAT(es_b, es_c, ...)]),
        ([es_a, es_b], [CAT(es_d, es_e, ...), CAT(es_b, es_c)]),
        ([es_a, es_b], [CAT(es_d, es_e), CAT(es_b, es_c, ...)]),
        ([es_a, es_d], [CAT(es_a, es_e, ...), CAT(es_b, es_c)]),
        ([es_a, es_d], [CAT(es_a, es_e), CAT(es_b, es_c, ...)]),
    ],
)
def test_validate_fe_steps_error_categorical_mismatch(candidate, template, candidate_step_cast):
    """Test that `IncompatibleCandidateError` is raised by `validate_fe_steps` when `candidate`
    has a step that does not fit in a `Categorical` step in `template`. See
    `test_validate_fe_steps` for parameter descriptions"""
    with pytest.raises(IncompatibleCandidateError):
        validate_fe_steps(candidate_step_cast(candidate), FeatureEngineer(template))


@pytest.mark.parametrize(
    ["candidate", "template"],
    [
        ([], [es_a]),
        ([es_a], [CAT(es_a, es_b), es_c]),
        ([es_a], [es_c, CAT(es_a, es_b)]),
        ([es_a], [es_c, CAT(es_a, es_b)]),
        ([es_a, es_d], [es_a, es_c]),
        ([es_a, es_d], [es_b, es_d]),
        ([es_a, es_d], [CAT(es_a, ...), es_c]),
        ([es_a, es_d], [es_b, CAT(es_d, ...)]),
        ([es_a, es_d, es_e], [CAT(es_a, ...), es_c, es_e]),
        ([es_a, es_d, es_e], [es_b, CAT(es_d, ...), es_e]),
        ([es_a, es_d, es_e], [CAT(es_a, ...), es_c, CAT(es_e, es_a)]),
        ([es_a, es_d, es_e], [es_b, CAT(es_d, ...), CAT(es_e, es_a)]),
    ],
)
def test_validate_fe_steps_error_concrete_missing(candidate, template, candidate_step_cast):
    """Test that `IncompatibleCandidateError` is raised by `validate_fe_steps` when `candidate`
    is missing a concrete (non-`Categorical`) step in `template`. See `test_validate_fe_steps`
    for parameter descriptions"""
    with pytest.raises(IncompatibleCandidateError):
        validate_fe_steps(candidate_step_cast(candidate), FeatureEngineer(template))


@pytest.mark.parametrize(
    ["candidate", "template"],
    [
        ([], [CAT(es_a, es_b)]),
        ([es_a], [es_a, CAT(es_c, es_d)]),
        ([es_c], [CAT(es_a, es_b), es_c]),
        ([es_a], [CAT(es_a, es_b), CAT(es_c, es_d)]),
        ([es_c], [CAT(es_a, es_b), CAT(es_c, es_d)]),
        ([es_a], [CAT(es_a, es_b, ...), CAT(es_c, es_d)]),
        ([es_c], [CAT(es_a, es_b), CAT(es_c, es_d, ...)]),
    ],
)
@pytest.mark.parametrize(
    ["candidate_suffix", "template_suffix"],
    [([], []), ([es_e], [es_e]), ([es_e], [CAT(es_b, es_e)]), ([es_e], [CAT(es_b, es_e, ...)])],
)
def test_validate_fe_steps_error_categorical_missing(
    candidate, template, candidate_suffix, template_suffix, candidate_step_cast
):
    """Test that `IncompatibleCandidateError` is raised by `validate_fe_steps` when `candidate`
    is missing a non-`optional` `Categorical` step in `template`

    Parameters
    ----------
    candidate: List
        `candidate` value given to :func:`~hyperparameter_hunter.result_reader.validate_fe_steps`
    template: List
        `template` value given to :func:`~hyperparameter_hunter.result_reader.validate_fe_steps`
    candidate_suffix: List
        Additional steps to append to the end of `candidate` before invoking `validate_fe_steps`
    template_suffix: List
        Additional steps to append to the end of `template` before invoking `validate_fe_steps`"""
    with pytest.raises(IncompatibleCandidateError):
        validate_fe_steps(
            candidate_step_cast(candidate + candidate_suffix),
            FeatureEngineer(template + template_suffix),
        )
