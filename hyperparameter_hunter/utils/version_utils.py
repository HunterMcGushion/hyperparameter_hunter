"""This module defines utilities for comparing versions of the library (:class:`HHVersion`), as well
as deprecation utilities, namely :class:`Deprecated`

Related
-------
:mod:`hyperparameter_hunter.exceptions`
    Defines the deprecation warnings issued by :class:`Deprecated`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import DeprecatedWarning, UnsupportedWarning

##################################################
# Import Miscellaneous Assets
##################################################
from functools import wraps
import re
from six import string_types
from textwrap import dedent
from typing import Union
from warnings import warn, simplefilter


##################################################
# Version Utilities
##################################################
class HHVersion:
    def __init__(self, v_str: str):
        """Parse and compare HyperparameterHunter version strings

        Comparisons must be performed with a valid version string or another `HHVersion` instance.

        HyperparameterHunter follows the "<major>.<minor>.<micro>" versioning scheme, with a few
        other variants, but all start with a triplet of period-delimited numbers. Supported version
        schemes are as follows (numbers given in examples may be greater than 9 in practice):

        * Final Release Version: "1.0.2", "2.2.0", "3.0.0", etc.
        * Alpha: "3.0.0alpha0", "3.0.0a1", "3.0.0a2", etc.
        * Beta: "3.0.0beta0", "3.0.0b1", "3.0.0b2", etc.
        * Release Candidate: "3.0.0rc0", "3.0.0rc1", "3.0.0rc2", etc.
        * Development Version: "1.8.0.dev-f1234afa" (git commit hash appended)
        * Development Version (Pre-Release): "1.8.0a1.dev-f1234afa",
          "1.8.0b2.dev-f1234afa", "1.8.1rc1.dev-f1234afa", etc.
        * Development Version (no git hash available): "1.8.0.dev-Unknown"

        Parameters
        ----------
        v_str: String
            HyperparameterHunter version string, such as `hyperparameter_hunter.__version__`

        Attributes
        ----------
        v_str: String
            Original value provided as input on initialization
        version: String
            Main version segment string of "<major>.<minor>.<micro>"
        major: Integer
            Major version number
        minor: Integer
            Minor version number
        micro: Integer
            Micro version number
        pre_release: String
            Pre-release version segment of `v_str`, or "final" if there is no pre-release segment.
            String starting with "a"/"alpha", "b"/"beta", or "rc" for pre-release segments of alpha,
            beta, or release candidate, respectively. If `v_str` does not end with one of the
            aforementioned pre-release segments but is a development version, then `pre_release` is
            an empty string. "alpha" and "beta" will be shortened to "a" and "b", respectively
        is_dev_version: Boolean
            True if the final segment of `v_str` starts with ".dev", denoting a development version.
            All development versions of the same release or pre-release are considered equal

        Notes
        -----
        Thank you to the brilliant [Ralf Gommers](https://github.com/rgommers), author of SciPy's
        [`NumpyVersion` class](https://github.com/scipy/scipy/blob/master/scipy/_lib/_version.py).
        He generously gave his permission to adapt his code for use here. Ralf Gommers: a gentleman
        and a scholar, and a selfless contributor to NumPy, SciPy, and countless other libraries

        Examples
        --------
        >>> from hyperparameter_hunter import __version__
        >>> HHVersion(__version__) > "1.0.2"
        True
        >>> HHVersion(__version__) > "3.0.0a1"
        True
        >>> HHVersion(__version__) <= "999.999.999"
        True
        >>> HHVersion("2.1")  # Missing "micro" number
        Traceback (most recent call last):
            File "version_utils.py", line ?, in HHVersion
        ValueError: Not a valid HyperparameterHunter version string
        """
        self.v_str = v_str

        #################### Parse Major, Minor, Micro Numbers ####################
        ver_main = re.match(r"\d+[.]\d+[.]\d+", v_str)
        if not ver_main:
            raise ValueError("Not a valid HyperparameterHunter version string")

        self.version = ver_main.group()
        self.major, self.minor, self.micro = [int(x) for x in self.version.split(".")]

        #################### Parse Pre-Release Suffix ####################
        if len(v_str) == ver_main.end():
            self.pre_release = "final"
        else:
            alpha = re.match(r"(a)(?:lpha)?(\d)", v_str[ver_main.end() :])
            beta = re.match(r"(b)(?:eta)?(\d)", v_str[ver_main.end() :])
            rc = re.match(r"(rc\d)", v_str[ver_main.end() :])
            pre_rel = [m for m in [alpha, beta, rc] if m is not None]
            if pre_rel:
                self.pre_release = "".join(pre_rel[0].groups())
            else:
                self.pre_release = ""

        self.is_dev_version = bool(re.search(r"\.dev", v_str))

    def _compare_main_version(self, other: "HHVersion") -> int:
        """Compare major.minor.micro main version segment numbers

        Parameters
        ----------
        other: HHVersion
            HyperparameterHunter version object with which to compare self

        Returns
        -------
        Integer in {0, 1, -1}
            * 0 if `self` and `other` have equivalent main version segments
            * 1 if `self` main version is greater than that of `other`
            * -1 if `other` main version is greater than that of `self`"""
        if self.major == other.major:
            if self.minor == other.minor:
                if self.micro == other.micro:
                    cmp = 0
                elif self.micro > other.micro:
                    cmp = 1
                else:
                    cmp = -1
            elif self.minor > other.minor:
                cmp = 1
            else:
                cmp = -1
        elif self.major > other.major:
            cmp = 1
        else:
            cmp = -1

        return cmp

    def _compare_pre_release(self, other: "HHVersion") -> int:
        """Compare pre-release (alpha/beta/rc) version segments

        Parameters
        ----------
        other: HHVersion
            HyperparameterHunter version object with which to compare self

        Returns
        -------
        Integer in {0, 1, -1}
            * 0 if `self` and `other` have equivalent pre-release version segments
            * 1 if `self` pre-release version segment is greater than that of `other`
            * -1 if `other` pre-release version segment is greater than that of `self`"""
        if self.pre_release == other.pre_release:
            cmp = 0
        elif self.pre_release == "final":
            cmp = 1
        elif other.pre_release == "final":
            cmp = -1
        elif self.pre_release > other.pre_release:
            cmp = 1
        else:
            cmp = -1

        return cmp

    def _compare(self, other: Union[str, "HHVersion"]) -> int:
        """Compare full versions, comprising main release, pre-release, and development segments

        Parameters
        ----------
        other: String, or HHVersion
            Version with which to compare self

        Returns
        -------
        Integer in {0, 1, -1}
            * 0 if `self` and `other` are equivalent versions
            * 1 if `self` is greater than `other`
            * -1 if `other` is greater than `self`"""
        if not isinstance(other, (string_types, HHVersion)):
            raise ValueError("Invalid object to compare with HHVersion")

        if isinstance(other, string_types):
            other = HHVersion(other)

        cmp = self._compare_main_version(other)

        if cmp == 0:
            # Same x.y.z version, check for alpha/beta/rc
            cmp = self._compare_pre_release(other)
            if cmp == 0:
                # Same version and same pre-release, check if dev version
                if self.is_dev_version is other.is_dev_version:
                    cmp = 0
                elif self.is_dev_version:
                    cmp = -1
                else:
                    cmp = 1

        return cmp

    def __eq__(self, other):
        return self._compare(other) == 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __repr__(self):
        return f"HHVersion({self.v_str})"


##################################################
# Deprecation Utilities
##################################################
MESSAGE_LOCATION = "bottom"


# TODO: Add `parameter: str` kwarg, which, if given, signals that input parameter to the decorated
#   callable is deprecated - Might help to piggyback off of `general_utils.Alias`
class Deprecated(object):
    def __init__(self, v_deprecate=None, v_remove=None, v_current=None, details=""):
        """Decorator to mark a function or class as deprecated. Issue warning when the function is
        called or the class is instantiated, and add a warning to the docstring. The optional
        `details` argument will be appended to the deprecation message and the docstring

        Parameters
        ----------
        v_deprecate: String, default=None
            Version in which the decorated callable is considered deprecated. This will usually
            be the next version to be released when the decorator is added. If None, deprecation
            will be immediate, and the `v_remove` and `v_current` arguments are ignored
        v_remove: String, default=None
            Version in which the decorated callable will be removed. If None, the callable is not
            currently planned to be removed. Cannot be set if `v_deprecate` = None
        v_current: String, default=None
            Source of version information for currently running code. When `v_current` = None, the
            ability to determine whether the wrapped callable is actually in a period of deprecation
            or time for removal fails, raising a :class:`DeprecatedWarning` in all cases
        details: String, default=""
            Extra details added to callable docstring/warning, such as a suggested replacement

        Notes
        -----
        Thank you to the ingenious [Brian Curtin](https://github.com/briancurtin), author of the
        excellent [`deprecation` library](https://github.com/briancurtin/deprecation). He generously
        gave his permission to adapt his code for use here. Brian Curtin: his magnanimity is
        surpassed only by his intelligence and imitability"""
        self.v_deprecate = v_deprecate
        self.v_remove = v_remove
        self.v_current = v_current
        self.details = details

        if self.v_deprecate is None and self.v_remove is not None:
            raise TypeError("Cannot set `v_remove` without also setting `v_deprecate`")

        #################### Determine Deprecation Status ####################
        self.is_deprecated = False
        self.is_unsupported = False

        if self.v_current:
            self.v_current = HHVersion(self.v_current)

            if self.v_remove and self.v_current >= HHVersion(self.v_remove):
                self.is_unsupported = True
            elif self.v_deprecate and self.v_current >= HHVersion(self.v_deprecate):
                self.is_deprecated = True
        else:
            self.is_deprecated = True

        self.do_warn = any([self.is_deprecated, self.is_unsupported])
        self.warning = None

    def __call__(self, obj):
        """Call method on callable `obj` to deprecate

        Parameters
        ----------
        obj: Object
            The callable being decorated as deprecated

        Object
            Callable result of either: 1) :meth:`_decorate_class` if `obj` is a class, or
            2) :meth:`_decorate_function` if `obj` is a function/method"""
        #################### Log Deprecation Warning ####################
        if self.do_warn:
            warn_cls = UnsupportedWarning if self.is_unsupported else DeprecatedWarning
            self.warning = warn_cls(obj.__name__, self.v_deprecate, self.v_remove, self.details)
            warn(self.warning, category=DeprecationWarning, stacklevel=2)

        #################### Decorate and Return Callable ####################
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_function(obj)

    def _decorate_class(self, cls):
        """Helper method to handle wrapping of class callables

        Parameters
        ----------
        cls: Class
            The class to be wrapped with a deprecation warning, and an updated docstring

        Returns
        -------
        cls: Class
            Updated `cls` that raises a deprecation warning before being called, and contains
            an updated docstring"""
        init = cls.__init__

        def wrapped(*args, **kwargs):
            self._verbose_warning()
            return init(*args, **kwargs)

        cls.__init__ = wrapped
        wrapped.__name__ = "__init__"
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init
        return cls

    def _decorate_function(self, f):
        """Helper method to handle wrapping of function/method callables

        Parameters
        ----------
        f: Function
            The function to be wrapped with a deprecation warning, and an updated docstring

        Returns
        -------
        wrapped: Function
            Updated `f` that raises a deprecation warning before being called, and contains
            an updated docstring"""

        @wraps(f)
        def wrapped(*args, **kwargs):
            self._verbose_warning()
            return f(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(wrapped.__doc__)
        wrapped.__wrapped__ = f
        return wrapped

    def _update_doc(self, old_doc):
        """Create a docstring containing the old docstring, in addition to a deprecation warning

        Parameters
        ----------
        old_doc: String
            Original docstring for the callable being deprecated, to which a warning will be added

        Returns
        -------
        String
            A new docstring with both the original docstring and a deprecation warning"""
        if not self.do_warn:
            return old_doc

        old_doc = old_doc or ""

        parts = dict(
            v_deprecate=" {}".format(self.v_deprecate) if self.v_deprecate else "",
            v_remove="\n\tWill be removed in {}.".format(self.v_remove) if self.v_remove else "",
            details=" {}".format(self.details) if self.details else "",
        )

        deprecation_note = ".. deprecated::{v_deprecate}{v_remove}{details}".format(**parts)
        loc = 1
        string_list = old_doc.split("\n", 1)

        if len(string_list) > 1:
            string_list[1] = dedent(string_list[1])
            string_list.insert(loc, "\n")

            if MESSAGE_LOCATION != "top":
                loc = 3

        string_list.insert(loc, deprecation_note)
        string_list.insert(loc, "\n\n")

        return "".join(string_list)

    def _verbose_warning(self):
        """Issue :attr:`warning`, bypassing the standard filter that silences DeprecationWarnings"""
        if self.do_warn:
            simplefilter("always", DeprecatedWarning)
            simplefilter("always", UnsupportedWarning)
            warn(self.warning, category=DeprecationWarning, stacklevel=4)
            simplefilter("default", DeprecatedWarning)
            simplefilter("default", UnsupportedWarning)
