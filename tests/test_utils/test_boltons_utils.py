"""Tests for modified functionality in :mod:`hyperparameter_hunter.utils.boltons_utils`. Because
:mod:`bolton_utils` is a vendorization of the "iterutils" module from the excellent
[Boltons](https://github.com/mahmoud/boltons) library, it is excluded from test coverage
measurement. This module only tests added/modified functionality that diverges from the original
Boltons functionality. For all of the original tests that apply to :mod:`boltons_utils`, see
https://github.com/mahmoud/boltons/blob/master/tests/test_iterutils.py"""
##################################################
# Import Own Assets
##################################################
# noinspection PyProtectedMember
from hyperparameter_hunter.utils.boltons_utils import get_path, PathAccessError, _UNSET

##################################################
# Import Miscellaneous Assets
##################################################
import pytest


def boltons_id_builder(param) -> str:
    """Prettify IDs for Boltons tests"""
    if param is PathAccessError:
        return "PathAccessError"
    if param == _UNSET:
        return "UNSET"
    return repr(param)


##################################################
# Test `get_path`
##################################################
@pytest.mark.parametrize(
    ["root", "path", "expected"],
    [
        (["foo"], (0,), "foo"),
        (["foo"], ("0",), PathAccessError),  # Divergence from `expected`="foo"
        ({"a": "foo"}, ("a",), "foo"),
        ({"a": "foo"}, "a", "foo"),  # Original - Dotted string `path`
        ({"a": ["foo"]}, ("a", 0), "foo"),
        ({0: "foo", "0": "bar"}, (0,), "foo"),
        ({0: "foo", "0": "bar"}, ("0",), "bar"),
        ({0: "foo", "1": "bar"}, ("0",), PathAccessError),
        ({0: "foo", "1": "bar"}, "0", PathAccessError),  # Original - Dotted string `path`
    ],
    ids=boltons_id_builder,
)
def test_get_path(root, path, expected):
    """Test modified functionality of :func:`~hyperparameter_hunter.utils.boltons_utils.get_path`.
    Verify that invocation with (`root`, `path`) returns `expected`, or raises
    `PathAccessError` when it should break"""
    if expected is PathAccessError:
        with pytest.raises(PathAccessError, match=".*"):
            get_path(root, path, default=_UNSET)
    else:
        assert get_path(root, path, default=_UNSET) == expected
