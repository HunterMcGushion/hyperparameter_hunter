##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
import base64
from functools import partial
import hashlib
from inspect import getsourcelines
import re
import sys


def make_hash_sha256(obj, **kwargs):
    """Create an sha256 hash of the input `obj`

    Parameters
    ----------
    obj: Object
        Object for which a hash will be created
    **kwargs: Dict
        Extra kwargs to supply to :func:`key_handler.hash_callable`

    Returns
    -------
    Stringified sha256 hash"""
    hasher = hashlib.sha256()
    hasher.update(repr(to_hashable(obj, **kwargs)).encode())
    return base64.urlsafe_b64encode(hasher.digest()).decode()


def to_hashable(obj, **kwargs):
    """Format the input `obj` to be hashable

    Parameters
    ----------
    obj: Object
        Object to convert to a hashable format
    **kwargs: Dict
        Extra kwargs to supply to :func:`key_handler.hash_callable`

    Returns
    -------
    obj: object
        Hashable object"""
    if callable(obj):
        return hash_callable(obj, **kwargs)
    if isinstance(obj, (tuple, list)):
        return tuple((to_hashable(_, **kwargs) for _ in obj))
    if isinstance(obj, dict):
        return tuple(sorted((_k, to_hashable(_v, **kwargs)) for _k, _v in obj.items()))
    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(to_hashable(_, **kwargs) for _ in obj))

    return obj


def hash_callable(
    obj,
    ignore_line_comments=True,
    ignore_first_line=False,
    ignore_module=False,
    ignore_name=False,
    ignore_keywords=False,
    ignore_source_lines=False,
):
    """Prepare callable object for hashing by selecting important characterization properties

    Parameters
    ----------
    obj: Callable
        Callable to convert to a hashable format. Supports: function, class, `functools.partial`
    ignore_line_comments: Boolean, default=True
        If True, any line comments will be stripped from the source code of `obj`, specifically any
        lines that start with zero or more whitespaces, followed by an octothorpe (#). This does not
        apply to comments on the same line as code
    ignore_first_line: Boolean, default=False
        If True, strip the first line from the callable's source code, specifically its name and
        signature. If `ignore_name=True`, this will be treated as True
    ignore_module: Boolean, default=False
        If True, ignore the name of the module containing the source code (:attr:`obj.__module__`)
    ignore_name: Boolean, default=False
        If True, ignore :attr:`obj.__name__`. Note the difference from `ignore_first_line`, which
        strips the entire callable signature from the source code. `ignore_name` does not alter the
        source code. To ensure thorough ignorance, `ignore_first_line=True` is recommended
    ignore_keywords: Boolean, default=False
        If True and `obj` is a :class:`functools.partial`, ignore :attr:`obj.keywords`
    ignore_source_lines: Boolean, default=False
        If True, all source code will be ignored by the hashing function. Ignoring all other kwargs,
        this means that only :attr:`obj.__module__`, and :attr:`obj.__name__`,
        (and :attr:`obj.keywords` if `obj` is partial) will be used for hashing

    Returns
    -------
    Tuple
        Hashable properties of the callable object input"""
    keywords, source_lines = None, None

    #################### Clean Up Partial ####################
    if isinstance(obj, partial):
        keywords = None if ignore_keywords else obj.keywords
        obj = obj.func  # Set partial to "func" attr - Expose same functionality as normal callable

    #################### Get Identifying Data ####################
    module = None if ignore_module else obj.__module__
    try:
        name = None if ignore_name else obj.__name__
    except AttributeError:
        obj = obj.__class__
        name = obj.__name__

    #################### Format Source Code Lines ####################
    if not ignore_source_lines:
        # TODO: Below only works on modified Keras `build_fn` during optimization if temp file still exists
        # FLAG: May need to wrap below in try/except TypeError to handle "built-in class" errors during mirroring
        # FLAG: ... Reference `settings.G.mirror_registry` for approval and its `original_sys_module_entry` attribute
        try:
            source_lines = getsourcelines(obj)[0]
        except TypeError as _ex:
            for a_mirror in G.mirror_registry:
                if obj.__name__ == a_mirror.import_name:
                    # TODO: Also, check `a_mirror.original_full_path` somehow, or object equality
                    source_lines = a_mirror.asset_source_lines[0]
                    break
            else:
                raise _ex.with_traceback(sys.exc_traceback)
        # TODO: Above only works on modified Keras `build_fn` during optimization if temp file still exists

        if ignore_line_comments:
            source_lines = [_ for _ in source_lines if not is_line_comment(_)]
        if (ignore_first_line is True) or (ignore_name is True):
            source_lines = source_lines[1:]

    #################### Select Relevant Data ####################
    relevant_data = [_ for _ in [module, name, keywords, source_lines] if _ is not None]
    # noinspection PyTypeChecker
    return tuple(to_hashable(relevant_data))


def is_line_comment(string):
    """Return True if the given string is a line comment, else False

    Parameters
    ----------
    string: Str
        The str in which to check for a line comment

    Returns
    -------
    Boolean"""
    return bool(re.match(r"^\s*#", string))
