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
from datetime import datetime
from functools import wraps
from inspect import Traceback
import re
import string
from textwrap import dedent
from typing import Any, Callable, Iterable, List, Tuple, Union
from warnings import warn, simplefilter
import wrapt


##################################################
# Iteration Utilities
##################################################
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
    if not default_vals:
        return default_vals

    def _visit(path, key, value):
        """If (`path` + `key`) is a key in `new_vals`, return its value. Else, default return"""
        for _current_key, _current_val in new_vals.items():
            if path + (key,) == _current_key:
                return (key, _current_val)
        return (key, value)

    return remap(default_vals, visit=_visit, enter=extra_enter_attrs(iter_attrs))


BooleanEnterFunc = Callable[[Tuple[str, ...], Union[str, tuple], Any], bool]
EnterFunc = Callable[[Tuple[str, ...], Union[str, tuple], Any], Tuple[Any, Iterable]]


def extra_enter_attrs(iter_attrs: Union[BooleanEnterFunc, List[BooleanEnterFunc]]) -> EnterFunc:
    """Build an `enter` function intended for use with `boltons_utils.remap` that enables entrance
    into non-standard objects defined by `iter_attrs` and iteration over their attributes as dicts

    Parameters
    ----------
    iter_attrs: Callable, list of callables, or None
        If callable, must evaluate to True or False when given three inputs: (path, key, value).
        Callable should return True if the current value should be entered by `remap`. If callable
        returns False, `default_enter` will be called. If `iter_attrs` is a list of callables, the
        value will be entered if any evaluates to True. If None, `default_enter` will be called

    Returns
    -------
    _enter: Callable
        Function to enter non-standard objects according to `iter_attrs` (via `remap`)"""
    iter_attrs = iter_attrs or [lambda *_args: False]
    iter_attrs = [iter_attrs] if not isinstance(iter_attrs, list) else iter_attrs

    def _enter(path, key, value):
        """If any in `iter_attrs` is True, enter `value` as a dict, iterating over non-magic
        attributes. Else, `default_enter`"""
        if any([_(path, key, value) for _ in iter_attrs]):
            included_attrs = [_ for _ in dir(value) if not _.endswith("__")]  # type: List[str]
            # Skips "dunder" methods, but keeps "__hh" attributes
            included_attrs = sorted(included_attrs, reverse=True)
            # Reverse to put public attributes before private - Without reversal, `remap` ignores
            #   `FeatureEngineer.steps` property because the identical `_steps` was already visited
            return dict(), [(_, getattr(value, _)) for _ in included_attrs]
        # TODO: Find better way to avoid entering "__hh_previous_frame" to avoid Traceback added by `tracers.LocationTracer`
        if isinstance(value, Traceback):
            return dict(), []
        # TODO: Find better way to avoid entering "__hh_previous_frame" to avoid Traceback added by `tracers.LocationTracer`

        return default_enter(path, key, value)

    return _enter


def flatten(l):
    return [item for sublist in l for item in sublist]


##################################################
# Miscellaneous Utilities
##################################################
def to_snake_case(s):
    """Convert a string to snake-case format

    Parameters
    ----------
    s: String
        String to convert to snake-case

    Returns
    -------
    String
        Snake-case formatted string

    Notes
    -----
    Adapted from https://gist.github.com/jaytaylor/3660565

    Examples
    --------
    >>> to_snake_case("snakesOnAPlane") == "snakes_on_a_plane"
    True
    >>> to_snake_case("SnakesOnAPlane") == "snakes_on_a_plane"
    True
    >>> to_snake_case("snakes_on_a_plane") == "snakes_on_a_plane"
    True
    >>> to_snake_case("IPhoneHysteria") == "i_phone_hysteria"
    True
    >>> to_snake_case("iPhoneHysteria") == "i_phone_hysteria"
    True
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def now_time():
    return datetime.now().time().strftime("%H:%M:%S")


def sec_to_hms(seconds, ms_places=5, as_str=False):
    """Convert `seconds` to hours, minutes, and seconds

    Parameters
    ----------
    seconds: Integer
        Number of total seconds to be converted to hours, minutes, seconds format
    ms_places: Integer, default=5
        Rounding precision for calculating number of seconds
    as_str: Boolean, default=False
        If True, return string "{hours} h, {minutes} m, {seconds} s". Else, return a triple

    Returns
    -------
    String or tuple
        If `as_str=True`, return a formatted string containing the hours, minutes, and seconds.
        Else, return a 3-item tuple of (hours, minutes, seconds)

    Examples
    --------
    >>> assert sec_to_hms(55, as_str=True) == '55 s'
    >>> assert sec_to_hms(86400) == (24, 0, 0)
    >>> assert sec_to_hms(86400, as_str=True) == '24 h'
    >>> assert sec_to_hms(86370) == (23, 59, 30)
    >>> assert sec_to_hms(86370, as_str=True) == '23 h, 59 m, 30 s'
    """
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
        return result.strip(", ")
    else:
        return (t_hour, t_min, t_sec)


def expand_mins_secs(mins, secs):
    """Format string expansion of `mins`, `secs` to the appropriate units up to (days, hours)

    Parameters
    ----------
    mins: Integer
        Number of minutes to be expanded to the appropriate string format
    secs: Integer
        Number of seconds to be expanded to the appropriate string format

    Returns
    -------
    String
        Formatted pair of one of the following: (minutes, seconds); (hours, minutes); or
        (days, hours) depending on the appropriate units given `mins`

    Examples
    --------
    >>> assert expand_mins_secs(34, 57) == "34m57s"
    >>> assert expand_mins_secs(72, 57) == "01h12m"
    >>> assert expand_mins_secs(1501, 57) == "01d01h"
    >>> assert expand_mins_secs(2880, 57) == "02d00h"
    """
    if mins < 60:
        return "{:>02d}m{:>02d}s".format(int(mins), int(secs))
    else:
        hours, mins = divmod(mins, 60)
        if hours < 24:
            return "{:>02d}h{:>02d}m".format(int(hours), int(mins))
        else:
            days, hours = divmod(hours, 24)
            return "{:>02d}d{:>02d}h".format(int(days), int(hours))


def to_standard_string(a_string):
    for to_replace in string.punctuation + " ":
        a_string = a_string.replace(to_replace, "")

    return a_string.lower()


def standard_equality(string_1, string_2):
    # assert (isinstance(string_1, str) and isinstance(string_2, str))
    return to_standard_string(string_1) == to_standard_string(string_2)


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


def split_version(s: str):
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


class Alias:
    def __init__(self, primary_name, aliases):
        """Convert uses of `aliases` to `primary_name` upon calling the decorated function/method

        Parameters
        ----------
        primary_name: String
            Preferred name for the parameter, the value of which will be set to the value of the
            used alias. If `primary_name` is already explicitly used on call in addition to any
            aliases, the value of `primary_name` will remain unchanged. It only assumes the value of
            an alias if the `primary_name` is not used
        aliases: List, string
            One or multiple string aliases for `primary_name`. If `primary_name` is not used on
            call, its value will be set to that of a random alias in `aliases`. Before calling the
            decorated callable, all `aliases` are removed from its kwargs

        Examples
        --------
        >>> class Foo():
        ...     @Alias("a", ["a2"])
        ...     def __init__(self, a, b=None):
        ...         print(a, b)
        >>> @Alias("a", ["a2"])
        ... @Alias("b", ["b2"])
        ... def bar(a, b=None):
        ...    print(a, b)
        >>> foo = Foo(a2="x", b="y")
        x y
        >>> bar(a2="x", b2="y")
        x y"""
        self.primary_name = primary_name
        self.aliases = aliases if isinstance(aliases, list) else [aliases]

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        for alias in set(self.aliases).intersection(kwargs):
            # Only set if no `primary_name` already. Remove `aliases`, leaving only `primary_name`
            kwargs.setdefault(self.primary_name, kwargs.pop(alias))
            # Record aliases used in `instance.__hh_aliases_used` or `wrapped.__hh_aliases_used`
            if instance:
                set_default_attr(instance, "__hh_aliases_used", {})[self.primary_name] = alias
            else:
                set_default_attr(wrapped, "__hh_aliases_used", {})[self.primary_name] = alias
        return wrapped(*args, **kwargs)


def set_default_attr(obj, name, value):
    """Set the `name` attribute of `obj` to `value` if the attribute does not already exist

    Parameters
    ----------
    obj: Object
        Object whose `name` attribute will be returned (after setting it to `value`, if necessary)
    name: String
        Name of the attribute to set to `value`, or to return
    value: Object
        Default value to give to `obj.name` if the attribute does not already exist

    Returns
    -------
    Object
        `obj.name` if it exists. Else, `value`

    Examples
    --------
    >>> foo = type("Foo", tuple(), {"my_attr": 32})
    >>> set_default_attr(foo, "my_attr", 99)
    32
    >>> set_default_attr(foo, "other_attr", 9000)
    9000
    >>> assert foo.my_attr == 32
    >>> assert foo.other_attr == 9000
    """
    try:
        return getattr(obj, name)
    except AttributeError:
        setattr(obj, name, value)
    return value


##################################################
# Boltons Utilities
##################################################
# Below utilities adapted from the `boltons` library: https://github.com/mahmoud/boltons
# Thank you to the creator and contributors of `boltons` for their excellent work
def subdict(d, keep=None, drop=None, key=None, value=None):
    """Compute the "subdictionary" of a dict, `d`

    Parameters
    ----------
    d: Dict
        Dict whose keys will be filtered according to `keep` and `drop`
    keep: List, or callable, default=`d.keys()`
        Keys to retain in the returned subdict. If callable, return boolean given key input. `keep`
        may contain keys not in `d` without raising errors. `keep` may be better described as the
        keys allowed to be in the returned dict, whether or not they are in `d`. This means that if
        `keep` consists solely of a key not in `d`, an empty dict will be returned
    drop: List, or callable, default=[]
        Keys to remove from the returned subdict. If callable, return boolean given key input.
        `drop` may contain keys not in `d`, which will simply be ignored
    key: Callable, or None, default=None
        Transformation to apply to the keys included in the returned subdictionary
    value: Callable, or None, default=None
        Transformation to apply to the values included in the returned subdictionary

    Returns
    -------
    Dict
        New dict with any keys in `drop` removed and any keys in `keep` still present, provided they
        were in `d`. Calling `subdict` with neither `keep` nor `drop` is equivalent to `dict(d)`

    Examples
    --------
    >>> subdict({"a": 1, "b": 2})
    {'a': 1, 'b': 2}
    >>> subdict({"a": 1, "b": 2, "c": 3}, drop=["b", "c"])
    {'a': 1}
    >>> subdict({"a": 1, "b": 2, "c": 3}, keep=["a", "c"])
    {'a': 1, 'c': 3}
    >>> subdict({"a": 1, "b": 2, "c": 3}, drop=["b", "c"], key=lambda _: _.upper())
    {'A': 1}
    >>> subdict({"a": 1, "b": 2, "c": 3}, keep=["a", "c"], value=lambda _: _ * 10)
    {'a': 10, 'c': 30}
    >>> subdict({("foo", "a"): 1, ("foo", "b"): 2, ("bar", "c"): 3}, drop=lambda _: _[0] == "foo")
    {('bar', 'c'): 3}
    >>> subdict({("foo", "a"): 1, ("foo", "b"): 2, ("bar", "c"): 3}, keep=lambda _: _[0] == "foo")
    {('foo', 'a'): 1, ('foo', 'b'): 2}
    >>> subdict({(6, "a"): 1, (6, "b"): 2, (7, "c"): 3}, lambda _: _[0] == 6, key=lambda _: _[1])
    {'a': 1, 'b': 2}
    >>> subdict({"a": 1, "b": 2, "c": 3}, drop=["d"])
    {'a': 1, 'b': 2, 'c': 3}
    >>> subdict({"a": 1, "b": 2, "c": 3}, keep=["d"])
    {}
    >>> subdict({"a": 1, "b": 2, "c": 3}, keep=["b", "d"])
    {'b': 2}
    >>> subdict({"a": 1, "b": 2, "c": 3}, drop=["b", "c"], key="foo")
    Traceback (most recent call last):
        File "general_utils.py", line ?, in subdict
    TypeError: Expected callable `key` function
    >>> subdict({"a": 1, "b": 2, "c": 3}, drop=["b", "c"], value="foo")
    Traceback (most recent call last):
        File "general_utils.py", line ?, in subdict
    TypeError: Expected callable `value` function"""
    keep = keep or d.keys()
    drop = drop or []
    key = key or (lambda _: _)
    value = value or (lambda _: _)

    if not callable(key):
        raise TypeError("Expected callable `key` function")
    if not callable(value):
        raise TypeError("Expected callable `value` function")

    if callable(keep):
        keep = [_ for _ in d.keys() if keep(_)]
    if callable(drop):
        drop = [_ for _ in d.keys() if drop(_)]

    keys = set(keep) - set(drop)
    return dict([(key(k), value(v)) for k, v in d.items() if k in keys])


if __name__ == "__main__":
    pass
