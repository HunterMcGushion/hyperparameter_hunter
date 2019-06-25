##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Real, Categorical, Integer
from hyperparameter_hunter.optimization.backends.skopt.space import Space
from hyperparameter_hunter.utils.optimization_utils import (
    does_fit_in_space,
    filter_by_space,
    get_choice_dimensions,
    get_ids_by,
)

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Dummy Objects for Testing
##################################################
path_lb_0 = "tests/file_resources/optimization_utils_tests/GlobalLeaderboard0.csv"


def args_ids_for(scenarios):
    return dict(argvalues=scenarios, ids=[f"{_}" for _ in range(len(scenarios))])


##################################################
# get_ids_by Scenarios
##################################################
scenarios_get_ids_by_valid = [
    (dict(drop_duplicates=False), [1, 4, 0, 9, 2, 3, 6, 7, 5, 8]),
    (dict(algorithm_name="alg_b", drop_duplicates=False), [4, 2, 3]),
    (dict(algorithm_name="not_a_real_algorithm", drop_duplicates=False), []),
    (dict(cross_experiment_key="env_key_0", drop_duplicates=False), [0, 9, 6, 8]),
    (dict(hyperparameter_key="hyperparameter_key_2", drop_duplicates=False), [4, 2, 3]),
    (dict(), [1, 4, 0, 9, 2, 6, 7, 5, 8]),
    (dict(hyperparameter_key="hyperparameter_key_2"), [4, 2]),
    (dict(algorithm_name="alg_d", cross_experiment_key="env_key_5", drop_duplicates=False), [7]),
]


@pytest.mark.parametrize(["params", "expected"], **args_ids_for(scenarios_get_ids_by_valid))
def test_get_ids_by(params, expected):
    assert get_ids_by(leaderboard_path=path_lb_0, **params) == [f"id_{_}" for _ in expected]


##################################################
# get_choice_dimensions Scenarios
##################################################
choice_i = Integer(low=2, high=20)
choice_r = Real(low=0.0001, high=0.5)
choice_c = Categorical(["auc", "rmse", "mae"], transform="onehot")

scenarios_get_choice_dimensions = [
    (dict(a=200, b=choice_i, c=0.5, d=choice_r), [(("b",), choice_i), (("d",), choice_r)]),
    (dict(a=200, c=0.5), []),
    (dict(b=choice_i, d=choice_r), [(("b",), choice_i), (("d",), choice_r)]),
    (dict(fit=dict(e=choice_c, f=5), predict=dict(g=100)), [(("fit", "e"), choice_c)]),
    (
        dict(fit=dict(e=choice_c, f=5), predict=dict(g=100, h=dict(i=choice_r))),
        [(("fit", "e"), choice_c), (("predict", "h", "i"), choice_r)],
    ),
]


@pytest.mark.parametrize(["params", "expected"], **args_ids_for(scenarios_get_choice_dimensions))
def test_get_choice_dimensions(params, expected):
    assert get_choice_dimensions(params) == expected


##################################################
# `filter_by_space` Scenarios
##################################################
@pytest.fixture(scope="function")
def space_fixture():
    dimensions = [Real(0.1, 0.9), Categorical(["foo", "bar", "baz"]), Integer(12, 18)]
    locations = [
        ("model_init_params", "a"),
        ("model_init_params", "b", "c"),
        ("model_extra_params", "e"),
    ]

    for i in range(len(dimensions)):
        setattr(dimensions[i], "location", locations[i])

    return Space(dimensions)


#################### Sample `scored_hyperparameters` Tuples ####################
sh_0 = (
    dict(
        model_init_params=dict(a=0.4, b=dict(c="bar", d=9)),
        model_extra_params=dict(e=16, f="hello"),
    ),
    "score_0",
)
sh_1 = (
    dict(
        model_init_params=dict(a=3.14159, b=dict(c="bar", d=9)),
        model_extra_params=dict(e=16, f="hello"),
    ),
    "score_1",
)


@pytest.mark.parametrize(
    ["scored_hyperparameters", "expected"],
    [
        ([sh_0], [sh_0]),
        ([sh_0, sh_1], [sh_0]),
        ([sh_0, sh_1, sh_0], [sh_0, sh_0]),
        ([sh_1, sh_1], []),
    ],
)
def test_filter_by_space(space_fixture, scored_hyperparameters, expected):
    assert filter_by_space(scored_hyperparameters, space_fixture) == expected


# TODO: Add more tests dealing with Keras-specific issues like layers, callbacks and initializers
##################################################
# `does_fit_in_space` Scenarios
##################################################
@pytest.mark.parametrize(
    ["params", "does_fit"], [(sh_0[0], True), (sh_1[0], False), ([sh_0[0], sh_0[0]], False)]
)
def test_does_fit_in_space(space_fixture, params, does_fit):
    assert does_fit_in_space(params, space_fixture) is does_fit
