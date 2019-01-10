##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import metrics

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.metrics import roc_auc_score


##################################################
# Dummy Objects for Testing
##################################################
class EmptyClass(object):
    pass


_metrics_map = dict(roc_auc_score=roc_auc_score)
_in_fold, _oof, _holdout = "all", "all", "all"
empty_class, empty_func, empty_tuple = EmptyClass(), lambda _: _, tuple()


def args_ids_for(scenarios):
    return dict(argvalues=scenarios, ids=[f"{_}" for _ in range(len(scenarios))])


def keyed_args_ids_for(scenarios):
    arg_values, scenario_ids = [], []

    for group_key, scenario_group in scenarios.items():
        arg_values.extend(scenario_group)
        scenario_ids.extend([f"{group_key}[{_}]" for _ in range(len(scenario_group))])

    return dict(argvalues=arg_values, ids=scenario_ids)


##################################################
# ScoringMixIn Initialization Scenarios
##################################################
scoring_mix_in_init_params = ["metrics_map", "in_fold", "oof", "holdout"]
scenarios_valid_metrics_map = [
    [_metrics_map],
    [{"1": roc_auc_score}],
    [dict(my_roc_auc=roc_auc_score, roc_auc_score=None)],
    [dict(foo=roc_auc_score, roc_auc_score=None)],
    [dict(foo=roc_auc_score, roc_auc_score=None, foo_2="roc_auc_score")],
    [["roc_auc_score"]],
    [["f1_score", "accuracy_score", "roc_auc_score"]],
]
scenarios_valid_metrics_lists = [
    (_metrics_map, _in_fold, None, None),
    (_metrics_map, None, None, None),
    (_metrics_map, ["roc_auc_score"], _oof, _holdout),
    (
        ["f1_score", "accuracy_score", "roc_auc_score"],
        ["f1_score"],
        ["accuracy_score"],
        ["roc_auc_score"],
    ),
    (["f1_score", "accuracy_score", "roc_auc_score"], ["f1_score"], _oof, _holdout),
    #################### Below cases result in no metrics being calculated at all ####################
    (dict(), None, None, None),
    ([], None, None, None),
]
scenarios_type_error = dict(
    metrics_map=[
        # ("foo", _in_fold, _oof, _holdout),  # TODO: ORIGINAL: Raises AttributeError (`f`) instead of TypeError
        (1, _in_fold, _oof, _holdout),
        (None, _in_fold, _oof, _holdout),
        # (['f1_score', 'accuracy_score', 'roc_auc_score'], _in_fold, _oof, _holdout),  # This correctly fails
        (empty_class, _in_fold, _oof, _holdout),
        (empty_func, _in_fold, _oof, _holdout),
        # (empty_tuple, _in_fold, _oof, _holdout),  # TODO: ORIGINAL: Might be valid now - No metrics
    ],
    metrics_map_key=[
        ({1: roc_auc_score}, _in_fold, _oof, _holdout),
        ({empty_class: roc_auc_score}, _in_fold, _oof, _holdout),
        ({empty_func: roc_auc_score}, _in_fold, _oof, _holdout),
        # ({empty_tuple: roc_auc_score}, _in_fold, _oof, _holdout),  # TODO: ORIGINAL: Raises nothing, but probably should
    ],
    metrics_map_value=[
        ({"roc_auc_score": 1}, _in_fold, _oof, _holdout),
        ({"roc_auc_score": 1.2}, _in_fold, _oof, _holdout),
        ({"roc_auc_score": ["a", "b"]}, _in_fold, _oof, _holdout),
        ({"roc_auc_score": dict(a=1, b=2)}, _in_fold, _oof, _holdout),
        # ({"roc_auc_score": ("a", "b")}, _in_fold, _oof, _holdout),  # TODO: ORIGINAL: Raises AttributeError (`a`) instead of TypeError
    ],
    metrics_lists=[
        (_metrics_map, "foo", _oof, _holdout),
        (_metrics_map, _in_fold, "foo", _holdout),
        (_metrics_map, _in_fold, _oof, "foo"),
        (_metrics_map, empty_class, _oof, _holdout),
        (_metrics_map, empty_func, _oof, _holdout),
        (_metrics_map, ("a", "b"), _oof, _holdout),
        (_metrics_map, 1, _oof, _holdout),
        (_metrics_map, 1.2, _oof, _holdout),
        (_metrics_map, 1.2, "foo", empty_func),
    ],
    metrics_lists_values=[
        (_metrics_map, [1], _oof, _holdout),
        (_metrics_map, _in_fold, [1.2], _holdout),
        (_metrics_map, _in_fold, _oof, [empty_func]),
        (_metrics_map, [empty_class], _oof, _holdout),
        (_metrics_map, [empty_tuple], _oof, _holdout),
        (_metrics_map, [["roc_auc"]], _oof, _holdout),
        (_metrics_map, [dict(a=1, b=2)], 1, 1),
        (_metrics_map, [None], _oof, _holdout),
    ],
)
scenarios_attribute_error = [
    (dict(roc_auc="foo"), _in_fold, _oof, _holdout),
    (dict(foo=None), _in_fold, _oof, _holdout),
    (["foo"], _in_fold, _oof, _holdout),
    (["roc_auc", "foo"], _in_fold, _oof, _holdout),
]
scenarios_key_error = [
    (_metrics_map, ["foo"], _oof, _holdout),
    (_metrics_map, _in_fold, ["foo"], _holdout),
    (_metrics_map, _in_fold, _oof, ["foo"]),
    (_metrics_map, ["roc_auc", "foo"], _oof, _holdout),
    (dict(), ["roc_auc"], _oof, _holdout),
    (dict(), _in_fold, ["roc_auc"], _holdout),
    ([], _in_fold, _oof, ["roc_auc"]),
]


@pytest.mark.parametrize(["metrics_map"], **args_ids_for(scenarios_valid_metrics_map))
def test_valid_scoring_mix_in_initialization_metrics_map(metrics_map):
    metrics.ScoringMixIn(metrics_map=metrics_map, in_fold=_in_fold, oof=_oof, holdout=_holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **args_ids_for(scenarios_valid_metrics_lists))
def test_valid_scoring_mix_in_initialization_metrics_lists(metrics_map, in_fold, oof, holdout):
    metrics.ScoringMixIn(metrics_map=metrics_map, in_fold=in_fold, oof=oof, holdout=holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **keyed_args_ids_for(scenarios_type_error))
def test_type_error_scoring_mix_in_initialization(metrics_map, in_fold, oof, holdout):
    with pytest.raises(TypeError):
        metrics.ScoringMixIn(metrics_map=metrics_map, in_fold=in_fold, oof=oof, holdout=holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **args_ids_for(scenarios_attribute_error))
def test_attribute_error_scoring_mix_in_initialization(metrics_map, in_fold, oof, holdout):
    with pytest.raises(AttributeError):
        metrics.ScoringMixIn(metrics_map=metrics_map, in_fold=in_fold, oof=oof, holdout=holdout)


@pytest.mark.parametrize(scoring_mix_in_init_params, **args_ids_for(scenarios_key_error))
def test_key_error_scoring_mix_in_initialization(metrics_map, in_fold, oof, holdout):
    with pytest.raises(KeyError):
        metrics.ScoringMixIn(metrics_map=metrics_map, in_fold=in_fold, oof=oof, holdout=holdout)
