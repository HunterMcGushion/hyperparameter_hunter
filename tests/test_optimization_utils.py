##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.optimization_utils import get_ids_by, get_choice_dimensions
from hyperparameter_hunter.space import Real, Integer, Categorical

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
