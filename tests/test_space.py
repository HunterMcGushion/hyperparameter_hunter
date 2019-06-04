##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.feature_engineering import EngineerStep
from hyperparameter_hunter.space import Real, Categorical, Integer, Space

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
from sys import maxsize


##################################################
# Dimension Name Error
##################################################
def test_dimension_name_value_error():
    with pytest.raises(ValueError, match="Dimension's name must be one of: string, tuple, or .*"):
        Real(0.3, 0.9, name=14)


##################################################
# Dimension Contains Tests
##################################################
@pytest.mark.parametrize(
    ["value", "is_in"], [(1, True), (5, True), (10, True), (0, False), (11, False), ("x", False)]
)
def test_integer_contains(value, is_in):
    assert (value in Integer(1, 10)) is is_in


##################################################
# Space Size Tests
##################################################
@pytest.mark.parametrize(
    ["space", "size"],
    [
        (Space([Categorical(["a", "b"]), Real(0.1, 0.7)]), maxsize),
        (Space([Categorical(["a", "b"]), Integer(1, 5)]), 10),
    ],
)
def test_space_len(space, size):
    assert len(space) == size


##################################################
# Dimension `get_params` Tests
##################################################
#################### `Real.get_params` ####################
@pytest.mark.parametrize(
    ["given_params", "expected_params"],
    [
        (
            dict(low=0.1, high=0.9),
            dict(low=0.1, high=0.9, prior="uniform", transform="identity", name=None),
        ),
        (
            dict(low=0.1, high=0.9, transform="normalize", name="Reginald"),
            dict(low=0.1, high=0.9, prior="uniform", transform="normalize", name="Reginald"),
        ),
    ],
)
def test_real_get_params(given_params, expected_params):
    assert Real(**given_params).get_params() == expected_params


#################### `Integer.get_params` ####################
@pytest.mark.parametrize(
    ["given_params", "expected_params"],
    [
        (dict(low=17, high=32), dict(low=17, high=32, transform="identity", name=None)),
        (
            dict(low=32, high=117, transform="normalize", name="Isabella"),
            dict(low=32, high=117, transform="normalize", name="Isabella"),
        ),
    ],
)
def test_integer_get_params(given_params, expected_params):
    assert Integer(**given_params).get_params() == expected_params


#################### `Categorical.get_params` ####################
def dummy_engineer_a(train_inputs, train_targets):
    return train_inputs, train_targets


def dummy_engineer_b(train_inputs, non_train_inputs):
    return train_inputs, non_train_inputs


def dummy_engineer_c(train_targets, non_train_targets):
    return train_targets, non_train_targets


@pytest.mark.parametrize(
    ["given_params", "expected_params"],
    [
        (
            dict(categories=["a", "b", "c"]),
            dict(
                categories=("a", "b", "c"),
                prior=None,
                transform="onehot",
                optional=False,
                name=None,
            ),
        ),
        (
            dict(categories=["a"], name="Cornelius"),
            dict(
                categories=("a",), prior=None, transform="onehot", optional=False, name="Cornelius"
            ),
        ),
        (
            dict(categories=[5, 10, 15], prior=[0.6, 0.2, 0.2], transform="identity"),
            dict(
                categories=(5, 10, 15),
                prior=[0.6, 0.2, 0.2],
                transform="identity",
                optional=False,
                name=None,
            ),
        ),
        (
            dict(categories=[dummy_engineer_a, dummy_engineer_b]),
            dict(
                categories=(dummy_engineer_a, dummy_engineer_b),
                prior=None,
                transform="onehot",
                optional=False,
                name=None,
            ),
        ),
        (
            dict(
                categories=[
                    EngineerStep(dummy_engineer_a),
                    EngineerStep(dummy_engineer_b),
                    EngineerStep(dummy_engineer_c),
                ]
            ),
            dict(
                categories=(
                    EngineerStep(dummy_engineer_a),
                    EngineerStep(dummy_engineer_b),
                    EngineerStep(dummy_engineer_c),
                ),
                prior=None,
                transform="onehot",
                optional=False,
                name=None,
            ),
        ),
    ],
)
def test_categorical_get_params(given_params, expected_params):
    assert Categorical(**given_params).get_params() == expected_params


@pytest.mark.parametrize(
    ["given_params", "unexpected_params"],
    [
        (
            dict(categories=[EngineerStep(dummy_engineer_a), EngineerStep(dummy_engineer_b)]),
            dict(
                categories=(EngineerStep(dummy_engineer_a), EngineerStep(dummy_engineer_c)),
                prior=None,
                transform="onehot",
                name=None,
            ),
        ),
        (
            dict(categories=[EngineerStep(dummy_engineer_a, name="some_other_name")]),
            dict(
                categories=(EngineerStep(dummy_engineer_a),),
                prior=None,
                transform="onehot",
                name=None,
            ),
        ),
        (
            dict(categories=[EngineerStep(dummy_engineer_a, stage="intra_cv")]),
            dict(
                categories=(EngineerStep(dummy_engineer_a),),
                prior=None,
                transform="onehot",
                name=None,
            ),
        ),
        (
            dict(categories=[EngineerStep(dummy_engineer_a), EngineerStep(dummy_engineer_b)]),
            dict(
                categories=(EngineerStep(dummy_engineer_b), EngineerStep(dummy_engineer_a)),
                prior=None,
                transform="onehot",
                name=None,
            ),
        ),
    ],
    ids=["different_f", "different_name", "different_stage", "different_order"],
)
def test_categorical_not_get_params(given_params, unexpected_params):
    """Silly sanity tests ensuring `Categorical` doesn't think two similar things are the same"""
    assert Categorical(**given_params).get_params() != unexpected_params
