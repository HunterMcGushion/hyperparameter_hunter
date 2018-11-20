"""This module defines assorted general-use utilities used throughout the library. The contents are
primarily small functions that perform oft-repeated tasks. This module also defines deprecation
utilities, namely :class:`Deprecated`, which is used to deprecate callables

Related
-------
:mod:`hyperparameter_hunter.exceptions`
    Defines the deprecation warnings issued by :class:`Deprecated`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import DeprecatedWarning, UnsupportedWarning
from hyperparameter_hunter.utils.boltons_utils import remap, default_enter

##################################################
# Import Miscellaneous Assets
##################################################
from collections import defaultdict
from datetime import datetime
from functools import wraps
import string
from textwrap import dedent
from warnings import warn, simplefilter


def deep_restricted_update(default_vals, new_vals, iter_attrs=None):
    """Return an updated dictionary that mirrors `default_vals`, except where the key in `new_vals`
    matches the path in `default_vals`, in which case the `new_vals` value is used

    Parameters
    ----------
    default_vals: Dict
        Dict containing the values to return if an alternative is not found in `new_vals`
    new_vals: Dict
        Dict whose keys are expected to be tuples corresponding to key paths in `default_vals`
    iter_attrs: Callable, list of callables, or None, default=None
        If callable, must evaluate to True or False when given three inputs: (path, key, value).
        Callable should return True if the current value should be entered by `remap`. If callable
        returns False, `default_enter` will be called. If `iter_attrs` is a list of callables, the
        value will be entered if any evaluates to True. If None, `default_enter` will be called

    Returns
    -------
    Dict, or None

    Examples
    --------
    >>> deep_restricted_update({'a': 1, 'b': 2}, {('b',): 'foo', ('c',): 'bar'})
    {'a': 1, 'b': 'foo'}
    >>> deep_restricted_update({'a': 1, 'b': {'b1': 2, 'b2': 3}}, {('b', 'b1'): 'foo', ('c', 'c1'): 'bar'})
    {'a': 1, 'b': {'b1': 'foo', 'b2': 3}}"""
    iter_attrs = iter_attrs or [lambda *_args: False]
    iter_attrs = [iter_attrs] if not isinstance(iter_attrs, list) else iter_attrs

    def _visit(path, key, value):
        """If (`path` + `key`) is a key in `new_vals`, return its value. Else, default return"""
        for _current_key, _current_val in new_vals.items():
            if path + (key,) == _current_key:
                return (key, _current_val)
        return (key, value)

    def _enter(path, key, value):
        """If any in `iter_attrs` is True, enter `value` as a dict, iterating over non-magic
        attributes. Else, `default_enter`"""
        if any([_(path, key, value) for _ in iter_attrs]):
            included_attrs = [_ for _ in dir(value) if not _.startswith("__")]
            return dict(), [(_, getattr(value, _)) for _ in included_attrs]
        return default_enter(path, key, value)

    return remap(default_vals, visit=_visit, enter=_enter) if default_vals else default_vals


def now_time():
    return datetime.now().time().strftime("%H:%M:%S")


def sec_to_hms(seconds, ms_places=5, as_str=False):
    t_hour, temp_sec = divmod(seconds, 3600)
    t_min, t_sec = divmod(temp_sec, 60)
    t_sec = round(t_sec, ms_places)

    if as_str is True:
        result = ""
        if t_hour != 0:
            result += "{} h, ".format(t_hour)
        if t_min != 0:
            result += "{} m, ".format(t_min)
        if t_sec != 0:
            result += "{} s".format(t_sec)
        return result
    else:
        return (t_hour, t_min, t_sec)


def flatten(l):
    return [item for sublist in l for item in sublist]


def type_val(val):
    return type(val), val


def to_standard_string(a_string):
    for to_replace in string.punctuation + " ":
        a_string = a_string.replace(to_replace, "")

    return a_string.lower()


def standard_equality(string_1, string_2):
    # assert (isinstance(string_1, str) and isinstance(string_2, str))
    return to_standard_string(string_1) == to_standard_string(string_2)


def to_even(value, append_char=" "):
    try:
        if len(value) % 2 != 0:
            return value + append_char
    except TypeError:
        if value % 2 != 0:
            return value + 1
    return value


def composed(*decorators):
    def _deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return _deco


##################################################
# Recursive Dictionary for Deep Setting
##################################################
recursive_dict = lambda: defaultdict(recursive_dict)


def deep_dict_set(rec_dict, keys, value):
    for key in keys[:-1]:
        rec_dict = rec_dict[key]

    rec_dict[keys[-1]] = value
    return rec_dict


##################################################
# Deprecation Utilities
##################################################
# Below utilities adapted from the `deprecation` library: https://github.com/briancurtin/deprecation
# Thank you to the creator and contributors of `deprecation` for their excellent work
MESSAGE_LOCATION = "bottom"


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
            Extra details added to callable docstring/warning, such as a suggested replacement"""
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
            self.v_current = split_version(self.v_current)

            if self.v_remove and self.v_current >= split_version(self.v_remove):
                self.is_unsupported = True
            elif self.v_deprecate and self.v_current >= split_version(self.v_deprecate):
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
            v_remove="\n   Will be removed in {}.".format(self.v_remove) if self.v_remove else "",
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


def split_version(s):
    """Split a version string into a tuple of integers to facilitate comparison

    Parameters
    ----------
    s: String
        Version string containing integers separated by periods

    Returns
    -------
    Tuple
        The integer values from `s`, separated by periods"""
    return tuple([int(_) for _ in s.split(".")])


if __name__ == "__main__":
    pass
