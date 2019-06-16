##################################################
# Import Own Assets
##################################################
import hyperparameter_hunter as hh
from hyperparameter_hunter.utils.version_utils import HHVersion

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
import re


##################################################
# Current Version Tests
##################################################
def test_valid_hh_version():
    """Verify that the current HyperparameterHunter version is valid"""
    # TODO: Basically only enforcing correct main segment, since not using `re.fullmatch`
    # TODO: Probably want `re.fullmatch` here - Currently ignoring any potentially invalid suffix
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(|a[0-9]|b[0-9]|rc[0-9])"
    res = re.match(version_pattern, hh.__version__)
    assert res is not None


##################################################
# `HHVersion` Tests
##################################################
@pytest.mark.parametrize(["v_0", "v_1"], [("3.0.0a1", "3.0.0alpha1"), ("3.0.0b0", "3.0.0beta0")])
def test_pre_release_equality(v_0, v_1):
    assert HHVersion(v_0) == HHVersion(v_1)
    assert HHVersion(v_0) == v_1
    assert HHVersion(v_1) == v_0


def test_main_versions():
    assert HHVersion("1.8.0") == "1.8.0"
    for ver in ["1.9.0", "2.0.0", "1.8.1"]:
        assert HHVersion("1.8.0") < ver
        assert HHVersion(ver) > HHVersion("1.8.0")

    for ver in ["1.7.0", "1.7.1", "0.9.9"]:
        assert HHVersion("1.8.0") > ver
        assert HHVersion(ver) < HHVersion("1.8.0")


def test_version_1_point_10():
    assert HHVersion("1.9.0") < "1.10.0"
    assert HHVersion("1.11.0") < "1.11.1"
    assert HHVersion("1.11.0") == "1.11.0"
    assert HHVersion("1.99.11") < "1.99.12"


def test_alpha_beta_rc():
    assert HHVersion("1.8.0rc1") == "1.8.0rc1"
    for ver in ["1.8.0", "1.8.0rc2"]:
        assert HHVersion("1.8.0rc1") < ver
        assert HHVersion("1.8.0rc1") <= ver
        assert HHVersion(ver) > HHVersion("1.8.0rc1")

    for ver in ["1.8.0a2", "1.8.0b3", "1.7.2rc4"]:
        assert HHVersion("1.8.0rc1") > ver
        assert HHVersion("1.8.0rc1") >= ver
        assert HHVersion(ver) < HHVersion("1.8.0rc1")

    assert HHVersion("1.8.0b1") > "1.8.0a2"


def test_dev_version():
    assert HHVersion("1.9.0.dev-Unknown") < "1.9.0"
    assert HHVersion("1.9.0alpha0") > "1.9.0alpha0.dev-Unknown"
    for ver in ["1.9.0", "1.9.0a1", "1.9.0b2", "1.9.0b2.dev-ffffffff"]:
        assert HHVersion("1.9.0.dev-f16acvda") < ver

    assert HHVersion("1.9.0.dev-f16acvda") == "1.9.0.dev-11111111"


def test_dev_a_b_rc_mixed():
    assert HHVersion("1.9.0a2.dev-f16acvda") == "1.9.0a2.dev-11111111"
    assert HHVersion("1.9.0a2.dev-6acvda54") < "1.9.0a2"


def test_dev0_version():
    assert HHVersion("1.9.0.dev0+Unknown") < "1.9.0"
    for ver in ["1.9.0", "1.9.0a1", "1.9.0b2", "1.9.0b2.dev0+ffffffff"]:
        assert HHVersion("1.9.0.dev0+f16acvda") < ver

    assert HHVersion("1.9.0.dev0+f16acvda") == "1.9.0.dev0+11111111"


def test_dev0_a_b_rc_mixed():
    assert HHVersion("1.9.0a2.dev0+f16acvda") == "1.9.0a2.dev0+11111111"
    assert HHVersion("1.9.0a2.dev0+6acvda54") < "1.9.0a2"


@pytest.mark.parametrize(
    "v",
    [
        "1.9",
        "1,9.0",
        "1.7.x",
        pytest.param(
            "2.4.1$dev", marks=pytest.mark.xfail(reason="Should raise, but its kinda ridiculous")
        ),
    ],
)
def test_raises_init(v):
    with pytest.raises(ValueError):
        HHVersion(v)


def test_raises_compare():
    with pytest.raises(ValueError):
        _ = HHVersion("3.0.0") != 12


def test_version_repr():
    assert f"{HHVersion('3.0.0alpha2')!r}" == "HHVersion(3.0.0alpha2)"
