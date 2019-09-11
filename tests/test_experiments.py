##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import settings
from hyperparameter_hunter.experiments import BaseExperiment, CVExperiment, get_cv_indices
from hyperparameter_hunter.i_o.exceptions import EnvironmentInactiveError, EnvironmentInvalidError

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
from numpy.testing import assert_equal
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RepeatedKFold

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# `BaseExperiment._validate_environment` Tests
##################################################
def test_inactive_environment(monkeypatch, env_fixture_0):
    """Test that initializing an Experiment without an active `Environment` raises
    `EnvironmentInactiveError`"""
    # Currently have a valid `settings.G.Env` (`env_fixture_0`), so set it to None
    monkeypatch.setattr(settings.G, "Env", None)
    with pytest.raises(EnvironmentInactiveError):
        BaseExperiment(LogisticRegression, dict())


def test_invalid_environment(monkeypatch, env_fixture_0):
    """Test that initializing an Experiment when there is an active `Environment` -- but
    :attr:`hyperparameter_hunter.environment.Environment.current_task` is not None -- raises
    `EnvironmentInvalidError`"""
    # Currently have a valid `settings.G.Env` (`env_fixture_0`), so give it a fake `current_task`
    monkeypatch.setattr(settings.G.Env, "current_task", "some other task")
    with pytest.raises(EnvironmentInvalidError, match="Current experiment must finish before .*"):
        CVExperiment(LogisticRegression, dict())


##################################################
# Dummy Data
##################################################
#################### Single-Target Binary Classification ####################
dummy_input_data_0 = np.array(
    [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"], ["i"], ["j"], ["k"], ["l"]]
)
dummy_target_data_0 = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])
#################### Single-Target Categorical Classification ####################
# dummy_input_data_1 = np.array([
#
# ])
# dummy_target_data_1 = np.array([
#     [0], [0], [0], [1], [1], [1], [2], [2], [2], [3], [3], [3]
# ])
#################### Multi-Target Categorical Classification ####################
# dummy_input_data_2 = np.array([
#
# ])
# mtcc_0, mtcc_1, mtcc_2, mtcc_3 = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
# dummy_target_data_2 = np.array([
#     mtcc_0, mtcc_0, mtcc_0, mtcc_1, mtcc_1, mtcc_1, mtcc_2, mtcc_2, mtcc_2, mtcc_3, mtcc_3, mtcc_3
# ])
#################### Single-Target Regression #0 ####################
# dummy_input_data_3 = np.array([
#
# ])
# dummy_target_data_3 = np.array([
#     [1], [2], [3], [5], [8], [13], [21], [34], [55], [89], [144], [233]
# ])
#################### Single-Target Regression #1 ####################
# dummy_input_data_4 = np.random.RandomState(32).random((12, 5))
# dummy_target_data_4 = np.random.RandomState(32).random((12, 1))

##################################################
# Dummy CV Parameters / Folds
##################################################
dummy_cv_params_0 = dict(n_splits=2, shuffle=True, random_state=32)
dummy_cv_params_1 = dict(n_splits=3, shuffle=True, random_state=32)
dummy_cv_params_2 = dict(n_splits=4, shuffle=True, random_state=32)

dummy_cv_params_3 = dict(n_repeats=1, n_splits=2, random_state=32)
dummy_cv_params_4 = dict(n_repeats=2, n_splits=2, random_state=32)
dummy_cv_params_5 = dict(n_repeats=3, n_splits=2, random_state=32)
dummy_cv_params_6 = dict(n_repeats=3, n_splits=4, random_state=32)
dummy_cv_params_7 = dict(n_repeats=1, n_splits=3, random_state=32)

dummy_folds_0 = KFold(**dummy_cv_params_0)
dummy_folds_1 = KFold(**dummy_cv_params_1)
dummy_folds_2 = KFold(**dummy_cv_params_2)

dummy_folds_3 = RepeatedKFold(**dummy_cv_params_3)
dummy_folds_4 = RepeatedKFold(**dummy_cv_params_4)
dummy_folds_5 = RepeatedKFold(**dummy_cv_params_5)
dummy_folds_6 = RepeatedKFold(**dummy_cv_params_6)
dummy_folds_7 = RepeatedKFold(**dummy_cv_params_7)

##################################################
# Expected Indices
##################################################
#################### dummy_folds_0, dummy_folds_3 ####################
exp_cv_0 = [
    [
        (np.array([3, 4, 5, 6, 7, 8]), np.array([0, 1, 2, 9, 10, 11])),
        (np.array([0, 1, 2, 9, 10, 11]), np.array([3, 4, 5, 6, 7, 8])),
    ]
]
#################### dummy_folds_1, dummy_folds_7 ####################
exp_cv_1 = [
    [
        (np.array([1, 3, 4, 5, 6, 7, 8, 11]), np.array([0, 2, 9, 10])),
        (np.array([0, 2, 5, 6, 7, 8, 9, 10]), np.array([1, 3, 4, 11])),
        (np.array([0, 1, 2, 3, 4, 9, 10, 11]), np.array([5, 6, 7, 8])),
    ]
]
#################### dummy_folds_2 ####################
exp_cv_2 = [
    [
        (np.array([1, 3, 4, 5, 6, 7, 8, 10, 11]), np.array([0, 2, 9])),
        (np.array([0, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 10, 11])),
        (np.array([0, 1, 2, 5, 6, 7, 9, 10, 11]), np.array([3, 4, 8])),
        (np.array([0, 1, 2, 3, 4, 8, 9, 10, 11]), np.array([5, 6, 7])),
    ]
]
#################### dummy_folds_4 ####################
exp_cv_3 = [
    [
        (np.array([3, 4, 5, 6, 7, 8]), np.array([0, 1, 2, 9, 10, 11])),
        (np.array([0, 1, 2, 9, 10, 11]), np.array([3, 4, 5, 6, 7, 8])),
    ],
    [
        (np.array([1, 2, 3, 4, 9, 10]), np.array([0, 5, 6, 7, 8, 11])),
        (np.array([0, 5, 6, 7, 8, 11]), np.array([1, 2, 3, 4, 9, 10])),
    ],
]
#################### dummy_folds_5 ####################
exp_cv_4 = [
    [
        (np.array([3, 4, 5, 6, 7, 8]), np.array([0, 1, 2, 9, 10, 11])),
        (np.array([0, 1, 2, 9, 10, 11]), np.array([3, 4, 5, 6, 7, 8])),
    ],
    [
        (np.array([1, 2, 3, 4, 9, 10]), np.array([0, 5, 6, 7, 8, 11])),
        (np.array([0, 5, 6, 7, 8, 11]), np.array([1, 2, 3, 4, 9, 10])),
    ],
    [
        (np.array([1, 2, 4, 5, 6, 10]), np.array([0, 3, 7, 8, 9, 11])),
        (np.array([0, 3, 7, 8, 9, 11]), np.array([1, 2, 4, 5, 6, 10])),
    ],
]

#################### dummy_folds_6 ####################
exp_cv_5 = [
    exp_cv_2[0],
    [
        (np.array([0, 1, 2, 3, 4, 7, 9, 10, 11]), np.array([5, 6, 8])),
        (np.array([1, 2, 3, 4, 5, 6, 8, 9, 10]), np.array([0, 7, 11])),
        (np.array([0, 1, 3, 4, 5, 6, 7, 8, 11]), np.array([2, 9, 10])),
        (np.array([0, 2, 5, 6, 7, 8, 9, 10, 11]), np.array([1, 3, 4])),
    ],
    [
        (np.array([1, 2, 3, 4, 5, 6, 8, 10, 11]), np.array([0, 7, 9])),
        (np.array([0, 1, 2, 4, 5, 6, 7, 9, 10]), np.array([3, 8, 11])),
        (np.array([0, 2, 3, 4, 7, 8, 9, 10, 11]), np.array([1, 5, 6])),
        (np.array([0, 1, 3, 5, 6, 7, 8, 9, 11]), np.array([2, 4, 10])),
    ],
]


##################################################
# `get_cv_indices` Tests
##################################################
@pytest.mark.parametrize(
    ["folds", "cv_params", "input_data", "target_data", "expected_indices"],
    [
        # (dummy_folds_0, dummy_cv_params_0, input_data_fixture, target_data_fixture, exp_cv_0),  # TODO: Upgrade to this, since all datasets should yield same indices (except stratified)
        (dummy_folds_0, dummy_cv_params_0, dummy_input_data_0, dummy_target_data_0, exp_cv_0),
        (dummy_folds_3, dummy_cv_params_3, dummy_input_data_0, dummy_target_data_0, exp_cv_0),
        (dummy_folds_1, dummy_cv_params_1, dummy_input_data_0, dummy_target_data_0, exp_cv_1),
        (dummy_folds_7, dummy_cv_params_7, dummy_input_data_0, dummy_target_data_0, exp_cv_1),
        (dummy_folds_2, dummy_cv_params_2, dummy_input_data_0, dummy_target_data_0, exp_cv_2),
        (dummy_folds_4, dummy_cv_params_4, dummy_input_data_0, dummy_target_data_0, exp_cv_3),
        (dummy_folds_5, dummy_cv_params_5, dummy_input_data_0, dummy_target_data_0, exp_cv_4),
        (dummy_folds_6, dummy_cv_params_6, dummy_input_data_0, dummy_target_data_0, exp_cv_5),
    ],
    ids=[
        "folds_0-exp_0",
        "folds_3-exp_0",
        "folds_1-exp_1",
        "folds_7-exp_1",
        "folds_2-exp_2",
        "folds_4-exp_3",
        "folds_5-exp_4",
        "folds_6-exp_5",
    ],
)
def test_get_cv_indices(folds, cv_params, input_data, target_data, expected_indices):
    # result = get_cv_indices(folds, cv_params, input_data, target_data)
    # assert np.array_equal(result, expected_indices)
    # assert_array_equal(result, expected_indices)
    result = list(list(_) for _ in get_cv_indices(folds, cv_params, input_data, target_data))
    assert_equal(result, expected_indices)
