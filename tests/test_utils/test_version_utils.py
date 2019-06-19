##################################################
# Import Own Assets
##################################################
import hyperparameter_hunter as hh
from hyperparameter_hunter.utils.version_utils import HHVersion, Deprecated

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
import re
import warnings


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


##################################################
# `Deprecated` Tests
##################################################
def test_deprecated_type_error():
    with pytest.raises(TypeError, match="Cannot set `v_remove` without also setting `v_deprecate`"):
        Deprecated(v_remove="0.0.1", details="Nice try, buddy. Gotta deprecate before removal")


#################### Deprecated Class Tests ####################
class OKClass:
    def __init__(self, a="a", b="b"):
        """Some class to test deprecations

        Parameters
        ----------
        a: String
            Some parameter `a`
        b: String
            Some other parameter `b`

        Notes
        -----
        Look at me! I'm a note!"""
        self.a = a
        self.b = b


@Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", v_current="2.2.0", details="Renamed to `OKClass`"
)
class DeprecatedClass(OKClass):
    ...


@Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", v_current=hh.__version__, details="Renamed to `OKClass`"
)
class UnsupportedClass(OKClass):
    ...


@Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", v_current="2.0.0", details="Renamed to `OKClass`"
)
class NotDeprecatedClass(OKClass):
    ...


@pytest.mark.parametrize("cls", [DeprecatedClass, UnsupportedClass])
def test_deprecated_classes(cls):
    with pytest.deprecated_call():
        cls()
    doc = cls.__init__.__doc__
    assert doc.endswith(".. deprecated:: 2.0.1\n\tWill be removed in 2.3.0. Renamed to `OKClass`")


def test_not_deprecated_class(recwarn):
    warnings.simplefilter("always")
    NotDeprecatedClass()
    assert len(recwarn) == 0
    assert NotDeprecatedClass.__init__.__doc__ == OKClass.__init__.__doc__


#################### Deprecated Function Tests ####################
def ok_func(a="a", b="b"):
    """Some function to test deprecations

    Parameters
    ----------
    a: String
        Some parameter `a`
    b: String
        Some other parameter `b`

    Notes
    -----
    Look at me! I'm a note!"""
    return a + b


deprecated_func_0 = Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", v_current="2.2.0", details="Renamed to `ok_func`"
)(ok_func)

deprecated_func_1 = Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", details="Renamed to `ok_func`"
)(ok_func)

unsupported_func = Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", v_current=hh.__version__, details="Renamed to `ok_func`"
)(ok_func)

not_deprecated_func = Deprecated(
    v_deprecate="2.0.1", v_remove="2.3.0", v_current="2.0.0", details="Renamed to `ok_func`"
)(ok_func)


@pytest.mark.parametrize("func", [deprecated_func_0, deprecated_func_1, unsupported_func])
def test_deprecated_functions(func):
    with pytest.deprecated_call():
        func()
    doc = func.__doc__
    assert doc.endswith(".. deprecated:: 2.0.1\n\tWill be removed in 2.3.0. Renamed to `ok_func`")


def test_not_deprecated_func(recwarn):
    warnings.simplefilter("always")
    not_deprecated_func()
    assert len(recwarn) == 0
    assert not_deprecated_func.__doc__ == ok_func.__doc__
