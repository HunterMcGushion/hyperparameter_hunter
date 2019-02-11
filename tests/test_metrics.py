##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.metrics import ScoringMixIn, Metric, format_metrics
from hyperparameter_hunter.metrics import get_formatted_target_metric, get_clean_prediction

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import roc_auc_score, hamming_loss, r2_score, f1_score


##################################################
# Dummy Objects for Testing
##################################################
class EmptyClass(object):
    pass


def my_r2_score(foo, bar):
    return r2_score(foo, bar)


_metrics = dict(roc_auc_score=roc_auc_score)
_in_fold, _oof, _holdout = "all", "all", "all"
empty_class, empty_func = EmptyClass(), lambda _: _


def args_ids_for(scenarios):
    return dict(argvalues=scenarios, ids=[f"{_}" for _ in range(len(scenarios))])


def keyed_args_ids_for(scenarios):
    arg_values, scenario_ids = [], []

    for group_key, scenario_group in scenarios.items():
        arg_values.extend(scenario_group)
        scenario_ids.extend([f"{group_key}[{_}]" for _ in range(len(scenario_group))])

    return dict(argvalues=arg_values, ids=scenario_ids)


##################################################
# Metric Scenarios
##################################################


@pytest.fixture(scope="session")
def metric_init_params_lookup():
    """Lookup dictionary for `Metric` initialization parameters used in test scenarios. Keys
    correspond to those in `metric_init_final_attributes_lookup`"""
    return dict(
        m_0=("roc_auc_score",),
        m_1=("roc_auc_score", roc_auc_score),
        m_2=("my_f1_score", "f1_score"),
        m_3=("hamming_loss", hamming_loss),
        m_4=("r2_score", r2_score, "min"),
        m_5=("my_r2_score", my_r2_score),
    )


@pytest.fixture(scope="session")
def metric_init_final_attributes_lookup():
    """Lookup dictionary for the expected values of `Metric` attributes after an instance has been
    initialized with the value of the corresponding key in `metric_init_params_lookup`. The `Metric`
    attributes whose values are verified are as follows: `name`, `metric_function`, `direction`"""
    return dict(
        m_0=("roc_auc_score", roc_auc_score, "max"),
        m_1=("roc_auc_score", roc_auc_score, "max"),
        m_2=("my_f1_score", f1_score, "max"),
        m_3=("hamming_loss", hamming_loss, "min"),
        m_4=("r2_score", r2_score, "min"),
        m_5=("my_r2_score", my_r2_score, "max"),
    )


@pytest.fixture(scope="function", params=["m_0", "m_1", "m_2", "m_3", "m_4", "m_5"])
def metric_instance(metric_init_params_lookup, metric_init_final_attributes_lookup, request):
    """Instance of `metrics.Metric` initialized with the corresponding values in
    `metric_init_params_lookup`"""
    metric = Metric(*metric_init_params_lookup[request.param])

    #################### Ensure Attributes Properly Initialized ####################
    (_name, _metric_function, _direction) = metric_init_final_attributes_lookup[request.param]
    assert metric.metric_function == _metric_function
    assert metric.direction == _direction
    assert str(metric) == "Metric({}, {}, {})".format(_name, _metric_function.__name__, _direction)

    return metric


def test_metric_initialization_helpers(metric_instance):
    assert hasattr(metric_instance, "name")
    assert hasattr(metric_instance, "metric_function")
    assert hasattr(metric_instance, "direction")
    # TODO: Add test to verify `Metric.__call__` calls `Metric.metric_function` with expected inputs


@pytest.mark.parametrize("direction", ["foo", "MAX", "bar"])
def test_metric_initialization_invalid_direction(direction):
    with pytest.raises(ValueError, match="`direction` must be 'infer', 'max', or 'min', not .*"):
        Metric("some_metric", roc_auc_score, direction)


##################################################
# ScoringMixIn Initialization Scenarios
##################################################
scoring_mix_in_init_params = ["metrics", "in_fold", "oof", "holdout"]
scenarios_valid_metrics = [
    [_metrics],
    [{"1": roc_auc_score}],
    [dict(my_roc_auc=roc_auc_score, roc_auc_score=None)],
    [dict(foo=roc_auc_score, roc_auc_score=None)],
    [dict(foo=roc_auc_score, roc_auc_score=None, foo_2="roc_auc_score")],
    [["roc_auc_score"]],
    [["f1_score", "accuracy_score", "roc_auc_score"]],
]
scenarios_valid_metrics_lists = [
    (_metrics, _in_fold, None, None),
    (_metrics, None, None, None),
    (_metrics, ["roc_auc_score"], _oof, _holdout),
    (
        ["f1_score", "accuracy_score", "roc_auc_score"],
        ["f1_score"],
        ["accuracy_score"],
        ["roc_auc_score"],
    ),
    (["f1_score", "accuracy_score", "roc_auc_score"], ["f1_score"], _oof, _holdout),
]
scenarios_type_error = dict(
    metrics=[
        ("foo", _in_fold, _oof, _holdout),
        (1, _in_fold, _oof, _holdout),
        (None, _in_fold, _oof, _holdout),
        # (['f1_score', 'accuracy_score', 'roc_auc_score'], _in_fold, _oof, _holdout),  # This correctly fails
        (empty_class, _in_fold, _oof, _holdout),
        (empty_func, _in_fold, _oof, _holdout),
        (tuple(), _in_fold, _oof, _holdout),
        (list(), _in_fold, _oof, _holdout),
        (dict(), _in_fold, _oof, _holdout),
    ],
    metrics_key=[
        ({1: roc_auc_score}, _in_fold, _oof, _holdout),
        ({empty_class: roc_auc_score}, _in_fold, _oof, _holdout),
        ({empty_func: roc_auc_score}, _in_fold, _oof, _holdout),
        ({tuple(): roc_auc_score}, _in_fold, _oof, _holdout),
    ],
    metrics_value=[
        ({"roc_auc_score": 1}, _in_fold, _oof, _holdout),
        ({"roc_auc_score": 1.2}, _in_fold, _oof, _holdout),
        ({"roc_auc_score": ["a", "b"]}, _in_fold, _oof, _holdout),
        ({"roc_auc_score": dict(a=1, b=2)}, _in_fold, _oof, _holdout),
    ],
    metrics_lists=[
        (_metrics, "foo", _oof, _holdout),
        (_metrics, _in_fold, "foo", _holdout),
        (_metrics, _in_fold, _oof, "foo"),
        (_metrics, empty_class, _oof, _holdout),
        (_metrics, empty_func, _oof, _holdout),
        (_metrics, ("a", "b"), _oof, _holdout),
        (_metrics, 1, _oof, _holdout),
        (_metrics, 1.2, _oof, _holdout),
        (_metrics, 1.2, "foo", empty_func),
    ],
    metrics_lists_values=[
        (_metrics, [1], _oof, _holdout),
        (_metrics, _in_fold, [1.2], _holdout),
        (_metrics, _in_fold, _oof, [empty_func]),
        (_metrics, [empty_class], _oof, _holdout),
        (_metrics, [tuple()], _oof, _holdout),
        (_metrics, [["roc_auc"]], _oof, _holdout),
        (_metrics, [dict(a=1, b=2)], 1, 1),
        (_metrics, [None], _oof, _holdout),
    ],
)
scenarios_attribute_error = [
    (dict(roc_auc="foo"), _in_fold, _oof, _holdout),
    (dict(foo=None), _in_fold, _oof, _holdout),
    (["foo"], _in_fold, _oof, _holdout),
    (["roc_auc", "foo"], _in_fold, _oof, _holdout),
    ({"roc_auc_score": ("a", "b")}, _in_fold, _oof, _holdout),
]
scenarios_key_error = [
    (_metrics, ["foo"], _oof, _holdout),
    (_metrics, _in_fold, ["foo"], _holdout),
    (_metrics, _in_fold, _oof, ["foo"]),
    (_metrics, ["roc_auc", "foo"], _oof, _holdout),
]


@pytest.mark.parametrize(["metrics"], **args_ids_for(scenarios_valid_metrics))
def test_valid_scoring_mix_in_initialization_metrics(metrics):
    ScoringMixIn(metrics=metrics, in_fold=_in_fold, oof=_oof, holdout=_holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **args_ids_for(scenarios_valid_metrics_lists))
def test_valid_scoring_mix_in_initialization_metrics_lists(metrics, in_fold, oof, holdout):
    ScoringMixIn(metrics=metrics, in_fold=in_fold, oof=oof, holdout=holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **keyed_args_ids_for(scenarios_type_error))
def test_type_error_scoring_mix_in_initialization(metrics, in_fold, oof, holdout):
    with pytest.raises(TypeError):
        ScoringMixIn(metrics=metrics, in_fold=in_fold, oof=oof, holdout=holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **args_ids_for(scenarios_attribute_error))
def test_attribute_error_scoring_mix_in_initialization(metrics, in_fold, oof, holdout):
    with pytest.raises(AttributeError):
        ScoringMixIn(metrics=metrics, in_fold=in_fold, oof=oof, holdout=holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **args_ids_for(scenarios_key_error))
def test_key_error_scoring_mix_in_initialization(metrics, in_fold, oof, holdout):
    with pytest.raises(KeyError):
        ScoringMixIn(metrics=metrics, in_fold=in_fold, oof=oof, holdout=holdout)


##################################################
# get_formatted_target_metric Scenarios
##################################################
@pytest.mark.parametrize(
    "target_metric",
    argvalues=[[], {}, lambda: True, type("Foo", tuple(), {}), type("Foo", tuple(), {})(), 1, 3.14],
)
def test_get_formatted_target_metric_type_error(target_metric):
    with pytest.raises(TypeError):
        get_formatted_target_metric(target_metric, format_metrics(["roc_auc_score"]))


@pytest.mark.parametrize(
    "target_metric",
    argvalues=[("oof", "roc_auc_score", "foo"), ("foo", "roc_auc_score"), ("holdout", "foo")],
)
def test_get_formatted_target_metric_value_error(target_metric):
    with pytest.raises(ValueError):
        get_formatted_target_metric(target_metric, format_metrics(["roc_auc_score"]))


##################################################
# get_clean_prediction Scenarios
##################################################
@pytest.mark.parametrize(
    ["target", "prediction", "expected"],
    argvalues=[
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]),
        ([[3.1, 2.2], [4.1, 0.9]], [[3.2, 2.3], [3.9, 0.8]], [[3.2, 2.3], [3.9, 0.8]]),
        ([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], [1.0, 0.0, 1.0, 0.0]),
        ([1, 0, 1, 0], [2.3, -1.2, 1.9, 0.01], [1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_get_clean_prediction(target, prediction, expected):
    assert pd.DataFrame(
        get_clean_prediction(pd.DataFrame(target), pd.DataFrame(prediction))
    ).equals(pd.DataFrame(expected))
