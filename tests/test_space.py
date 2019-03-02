##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.space import Real, Categorical, Integer, Space

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
from sys import maxsize


def test_dimension_name_value_error():
    with pytest.raises(ValueError, match="Dimension's name must be one of: string, tuple, or .*"):
        Real(0.3, 0.9, name=14)


@pytest.mark.parametrize(
    ["value", "is_in"], [(1, True), (5, True), (10, True), (0, False), (11, False), ("x", False)]
)
def test_integer_contains(value, is_in):
    assert (value in Integer(1, 10)) is is_in


@pytest.mark.parametrize(
    ["space", "size"],
    [
        (Space([Categorical(["a", "b"]), Real(0.1, 0.7)]), maxsize),
        (Space([Categorical(["a", "b"]), Integer(1, 5)]), 10),
    ],
)
def test_space_len(space, size):
    assert len(space) == size
