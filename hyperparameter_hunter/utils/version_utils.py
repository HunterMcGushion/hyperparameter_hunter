"""This module defines utilities for comparing versions of the library, as well as deprecation
utilities, namely :class:`Deprecated`

Related
-------
:mod:`hyperparameter_hunter.exceptions`
    Defines the deprecation warnings issued by :class:`Deprecated`"""
##################################################
# Import Miscellaneous Assets
##################################################
import re
from six import string_types
from typing import Union


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
# TODO: Move `Deprecated`, `split_version` and `MESSAGE_LOCATION` from `general_utils.py`
