"""This module defines utilities for reading, writing, and modifying different types of files"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
from contextlib import suppress
from inspect import signature
import numpy as np
import os
import os.path
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML
import simplejson as json
from typing import Any, List, Tuple, Union
import wrapt


##################################################
# General File Utilities/Decorators
##################################################
def make_dirs(name, mode=0o0777, exist_ok=False):
    """Permissive version of `os.makedirs` that gives full permissions by default

    Parameters
    ----------
    name: Str
        Path/name of directory to create. Will make intermediate-level directories needed to contain
        the leaf directory
    mode: Number, default=0o0777
        File permission bits for creating the leaf directory
    exist_ok: Boolean, default=False
        If False, an `OSError` is raised if the directory targeted by `name` already exists"""
    old_mask = os.umask(000)
    os.makedirs(name, mode=mode, exist_ok=exist_ok)
    os.umask(old_mask)


def clear_file(file_path):
    """Erase the contents of the file located at `file_path`

    Parameters
    ----------
    file_path: String
        The path of the file whose contents should be cleared out"""
    clear_target = open(file_path, "w")
    clear_target.truncate()
    clear_target.close()


class RetryMakeDirs(object):
    def __init__(self):
        """Execute decorated callable, but if `OSError` is raised, call :func:`make_dirs` on the
        directory specified by the exception, then recall the decorated callable again. This also
        works with operations on files, in which case the file's parent directories are created

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory(dir="") as d:  # doctest: +ELLIPSIS
        ...     def f_0():
        ...         os.mkdir(f"{d}/nonexistent_dir/subdir")
        ...     f_0()
        Traceback (most recent call last):
            File "file_utils.py", line ?, in f_0
        FileNotFoundError: [Errno 2] No such file or directory...
        >>> with TemporaryDirectory(dir="") as d:
        ...     @RetryMakeDirs()
        ...     def f_1():
        ...         os.mkdir(f"{d}/nonexistent_dir/subdir")
        ...     f_1()
        """

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        try:
            return wrapped(*args, **kwargs)
        except OSError as _ex:
            # TODO: Add ability to check `kwargs` for value dictating whether to call `make_dirs`
            #   - Provide name or index (if arg) of value to check in `RetryMakeDirs.__init__`
            if _ex.filename:
                make_dirs(os.path.split(_ex.filename)[0], exist_ok=True)
        return wrapped(*args, **kwargs)


class ParametersFromFile(object):
    def __init__(self, key: Union[str, int] = None, file: str = None, verbose: bool = False):
        """Decorator to specify a .json file that defines default values for the decorated callable.
        The location of the file can either be specified explicitly with `file`, or it can be
        retrieved when the decorated callable is called through an argument key/index given by `key`

        Parameters
        ----------
        key: String, or integer, default=None
            Used only if `file` is not also given. Determines a value for `file` based on the
            parameters passed to the decorated callable. If string, represents a key in `kwargs`
            passed to :meth:`ParametersFromFile.__call__`. In other words, this names a keyword
            argument passed to the decorated callable. If `key` is integer, it represents an index
            in `args` passed to :meth:`ParametersFromFile.__call__`, the value at which specifies a
            filepath containing the default parameters dict to use
        file: String, default=None
            If not None, `key` will be ignored, and `file` will be used as the filepath from which
            to read the dict of default parameters for the decorated callable
        verbose: Boolean, default=False
            If True, will log messages when invalid keys are found in the parameters file, and when
            keys are set to the default values in the parameters file. Else, logging is silenced

        Notes
        -----
        The order of precedence for determining the value of each parameter is as follows, with
        items at the top having the highest priority, and deferring only to the items below if
        their own value is not given:

        * 1)parameters explicitly passed to the callable decorated by `ParametersFromFile`,
        * 2)parameters in the .json file denoted by `key` or `file`,
        * 3)parameter defaults defined in the signature of the decorated callable

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory(dir="") as d:
        ...     write_json(f"{d}/config.json", dict(b="I came from config.json", c="Me too!"))
        ...     @ParametersFromFile(file=f"{d}/config.json")
        ...     def f_0(a="first_a", b="first_b", c="first_c"):
        ...         print(f"{a}   ...   {b}   ...   {c}")
        ...     @ParametersFromFile(key="config_file")
        ...     def f_1(a="second_a", b="second_b", c="second_c", config_file=None):
        ...         print(f"{a}   ...   {b}   ...   {c}")
        ...     f_0(c="Hello, there")
        ...     f_0(b="General Kenobi")
        ...     f_1()
        ...     f_1(a="Generic prequel meme", config_file=f"{d}/config.json")
        ...     f_1(c="This is where the fun begins", config_file=None)
        first_a   ...   I came from config.json   ...   Hello, there
        first_a   ...   General Kenobi   ...   Me too!
        second_a   ...   second_b   ...   second_c
        Generic prequel meme   ...   I came from config.json   ...   Me too!
        second_a   ...   second_b   ...   This is where the fun begins"""
        self.key = key
        self.file = file
        self.verbose = verbose

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        file = self.file
        file_params = {}

        #################### Locate Parameters File ####################
        if not file and self.key is not None:
            with suppress(TypeError):
                file = kwargs.get(self.key, None) or args[self.key]

        if file:  # If `file=None`, continue with empty dict of `file_params`
            file_params = read_json(file)

        if not isinstance(file_params, dict):
            raise TypeError("{} must have dict, not {}".format(file, file_params))

        #################### Check Valid Parameters for `wrapped` ####################
        ok_keys = [k for k, v in signature(wrapped).parameters.items() if v.kind == v.KEYWORD_ONLY]

        for k, v in file_params.items():
            if k not in ok_keys:
                if self.verbose:
                    G.warn(f"Invalid key ({k}) in user parameters file: {file}")
            if k not in kwargs:
                kwargs[k] = v
                if self.verbose:
                    G.debug(f"Parameter `{k}` set to user default in parameters file: '{file}'")

        return wrapped(*args, **kwargs)


##################################################
# JSON File Utilities
##################################################
def default_json_write(obj):
    """Convert values that are not JSON-friendly to a more acceptable type

    Parameters
    ----------
    obj: Object
        Object that is expected to be of a type that is incompatible with JSON files

    Returns
    -------
    Object
        Value of `obj` after being cast to a type accepted by JSON

    Raises
    ------
    TypeError
        If the type of `obj` is unhandled

    Examples
    --------
    >>> assert default_json_write(np.array([1, 2, 3])) == [1, 2, 3]
    >>> assert default_json_write(np.int8(32)) == 32
    >>> assert np.isclose(default_json_write(np.float16(3.14)), 3.14, atol=0.001)
    >>> assert default_json_write(pd.Index(["a", "b", "c"])) == ["a", "b", "c"]
    >>> assert default_json_write((1, 2)) == {"__tuple__": [1, 2]}
    >>> default_json_write(object())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        File "file_utils.py", line ?, in default_json_write
    TypeError: <object object at ...> is not JSON serializable"""
    #################### Builtin Types ####################
    if isinstance(obj, tuple):
        return {"__tuple__": list(obj)}
    #################### NumPy Types ####################
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    #################### Pandas Types ####################
    if isinstance(obj, pd.Index):
        return list(obj)

    raise TypeError(f"{obj!r} is not JSON serializable")


def hook_json_read(obj):
    """Hook function to decode JSON objects during reading

    Parameters
    ----------
    obj: Object
        JSON object to process, or return unchanged

    Returns
    -------
    Object
        If `obj` contains the key "__tuple__", its value is cast to a tuple and returned. Else,
        `obj` is returned unchanged

    Examples
    --------
    >>> assert hook_json_read({"__tuple__": [1, 2]}) == (1, 2)
    >>> assert hook_json_read({"__tuple__": (1, 2)}) == (1, 2)
    >>> assert hook_json_read({"a": "foo", "b": 42}) == {"a": "foo", "b": 42}
    """
    if "__tuple__" in obj:
        return tuple(obj["__tuple__"])
    return obj


def read_json(file_path: str) -> object:
    """Get the contents of the .json file located at `file_path`

    Parameters
    ----------
    file_path: String
        Path to the .json file to be read

    Returns
    -------
    content: Object
        The contents of the .json file located at `file_path`"""
    content = json.loads(open(file_path).read(), object_hook=hook_json_read)
    return content


@RetryMakeDirs()
def write_json(file_path: str, data: Any):
    """Write `data` to the JSON file specified by `file_path`

    Parameters
    ----------
    file_path: String
        Target .json file path to which `data` will be written
    data: Object
        Content to save in the .json file given by `file_path`"""
    with open(file_path, "w") as f:
        json.dump(data, f, default=default_json_write, tuple_as_array=False)


def add_to_json(file_path, data_to_add, key=None, condition=None, default=None, append_value=False):
    """Append `data_to_add` to the contents of the .json file specified by `file_path`

    Parameters
    ----------
    file_path: String
        The target .json file path to which `data_to_add` will be added and saved
    data_to_add: Object
        The data to add to the contents of the .json file given by `file_path`
    key: String, or None, default=None
        If None, the original contents of the file at `file_path` should not be of type dict. If
        string, the original content at `file_path` is expected to be a dict, and `data_to_add` will
        be added to the original dict under the key `key`. Therefore, `key` is expected to be a
        unique key to the original dict contents of `file_path`, unless `append_value` is True
    condition: Callable, or None, default=None
        If callable, will be given the original contents of the .json file at `file_path` as input,
        and should return a boolean value. If `condition(original_data)` is truthy, `data_to_add`
        will be added to the contents of the file at `file_path` as usual. Otherwise, `data_to_add`
        will not be added to the file, and the contents at `file_path` will remain unchanged. If
        `condition` is None, it will be treated as having been truthy, and will proceed to append
        `data_to_add` to the target file
    default: Object, or None, default=None
        If the attempt to read the original content at `file_path` raises a `FileNotFoundError` and
        `default` is not None, `default` will be used as the original data for the file. Otherwise,
        the error will be raised
    append_value: Boolean, default=False
        If True and the original data at `file_path` is a dict, then `data_to_add` will be appended
        as a list to the value of the original data at key `key`"""
    try:
        original_data = read_json(file_path)
    except FileNotFoundError:
        if default is not None:
            original_data = default
        else:
            raise

    if condition is None or original_data is None or condition(original_data):
        if key is None and isinstance(original_data, list):
            original_data.append(data_to_add)
        elif isinstance(key, str) and isinstance(original_data, dict):
            if append_value is True:
                original_data[key] = original_data[key] + [data_to_add]
            else:
                original_data[key] = data_to_add

        write_json(file_path, original_data)


##################################################
# YAML File Utilities
##################################################
# Extra Representers used in the default HH Ruamel YAML instance
_RUAMEL_REPRESENTERS: List[Tuple[type, callable]] = [
    (np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist())),
    (np.float64, lambda dumper, data: dumper.represent_float(float(data))),
    (np.int64, lambda dumper, data: dumper.represent_int(int(data))),
    (tuple, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:python/tuple", data)),
    (str, lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')),
]
# Extra Constructors used in the default HH Ruamel YAML instance
_RUAMEL_CONSTRUCTORS: List[Tuple[str, callable]] = [
    ("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
]


def get_ruamel_instance() -> YAML:
    """Get the default :class:`ruamel.yaml.YAML` instance used for dumping/loading YAML files

    Returns
    -------
    yml: YAML
        :class:`ruamel.yaml.YAML` instance configured for HyperparameterHunter, outfitted with
        additional Ruamel Representers to properly format non-standard data types"""
    #################### Prepare Ruamel YAML Instance ####################
    yml = YAML(typ="safe")
    yml.default_flow_style = None
    yml.sort_base_mapping_type_on_output = False  # False retains original mapping order
    yml.top_level_colon_align = True  # Make it easier to see top-level elements
    yml.width = 100

    #################### Add Auxiliary Ruamel Representers/Constructors ####################
    for (data_type, representer) in _RUAMEL_REPRESENTERS:
        yml.representer.add_representer(data_type, representer)

    for (tag, constructor) in _RUAMEL_CONSTRUCTORS:
        yml.constructor.add_constructor(tag, constructor)

    return yml


def read_yaml(file_path: Union[str, Path], yml: YAML = None) -> object:
    """Get the contents of the .yaml file located at `file_path`

    Parameters
    ----------
    file_path: String, or Path
        Path to the .yaml file to be read
    yml: YAML (optional)
        :class:`ruamel.yaml.YAML` instance used to load data from `file_path`. If not given, the
        result of :func:`get_ruamel_instance` is used

    Returns
    -------
    Object
        Contents of the .yaml file located at `file_path`"""
    file_path = Path(file_path)
    yml = get_ruamel_instance() if yml is None else yml
    return yml.load(file_path)


@RetryMakeDirs()
def write_yaml(file_path: Union[str, Path], data: Any, yml: YAML = None):
    """Write `data` to the YAML file specified by `file_path`

    Parameters
    ----------
    file_path: String, or Path
        Target .yaml file path to which `data` will be written
    data: Object
        Content to save in the .yaml file given by `file_path`
    yml: YAML (optional)
        :class:`ruamel.yaml.YAML` instance used to dump `data` to `file_path`. If not given, the
        result of :func:`get_ruamel_instance` is used"""
    file_path = Path(file_path)
    yml = get_ruamel_instance() if yml is None else yml

    with open(file_path, "w+") as f:
        yml.dump(data, f)


##################################################
# Display Utilities
##################################################
def real_name(path, root=None):
    if root is not None:
        path = os.path.join(root, path)

    result = os.path.basename(path)

    if os.path.islink(path):
        real_path = os.readlink(path)
        result = "{} -> {}".format(os.path.basename(path), real_path)

    return result


def print_tree(start_path, depth=-1, pretty=True):
    """Print directory/file tree structure

    Parameters
    ----------
    start_path: String
        Root directory path, whose children should be traversed and printed
    depth: Integer, default=-1
        Maximum number of subdirectories allowed to be between the root `start_path` and the current
        element. -1 allows all child directories beneath `start_path` to be traversed
    pretty: Boolean, default=True
        If True, directory names will be bolded

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory(dir="") as d:
    ...     os.mkdir(f"{d}/root")
    ...     os.mkdir(f"{d}/root/sub_a")
    ...     os.mkdir(f"{d}/root/sub_a/sub_b")
    ...     _ = open(f"{d}/root/file_0.txt", "w+")
    ...     _ = open(f"{d}/root/file_1.py", "w+")
    ...     _ = open(f"{d}/root/sub_a/file_2.py", "w+")
    ...     _ = open(f"{d}/root/sub_a/sub_b/file_3.txt", "w+")
    ...     _ = open(f"{d}/root/sub_a/sub_b/file_4.py", "w+")
    ...     print_tree(f"{d}/root", pretty=False)
    ...     print("#" * 50)
    ...     print_tree(f"{d}/root", depth=2, pretty=False)
    ...     print("#" * 50)
    ...     print_tree(f"{d}/root/", pretty=False)
    |-- root/
    |   |-- file_0.txt
    |   |-- file_1.py
    |   |-- sub_a/
    |   |   |-- file_2.py
    |   |   |-- sub_b/
    |   |   |   |-- file_3.txt
    |   |   |   |-- file_4.py
    ##################################################
    |-- root/
    |   |-- file_0.txt
    |   |-- file_1.py
    |   |-- sub_a/
    |   |   |-- file_2.py
    ##################################################
    root/
    |-- file_0.txt
    |-- file_1.py
    |-- sub_a/
    |   |-- file_2.py
    |   |-- sub_b/
    |   |   |-- file_3.txt
    |   |   |-- file_4.py"""
    prefix = 0

    if start_path != "/":
        if start_path.endswith("/"):
            # If True, the last dir in start_path will be treated as root, rather than the whole thing
            start_path = start_path[:-1]
            prefix = len(start_path)

    for root, dirs, files in os.walk(start_path):
        level = root[prefix:].count(os.sep)
        if level > depth > -1:
            continue

        indent = ""

        if level > 0:
            indent = "|   " * (level - 1) + "|-- "
        sub_indent = "|   " * (level) + "|-- "

        content = "{}{}/".format(indent, real_name(root))
        if pretty:
            content = "\u001b[;1m" + content + "\u001b[0m"

        print(content)

        for d in sorted(dirs):
            if os.path.islink(os.path.join(root, d)):
                content = "{}{}".format(sub_indent, real_name(d, root=root))
                print(content)

        for f in sorted(files):
            content = "{}{}".format(sub_indent, real_name(f, root=root))
            print(content)


if __name__ == "__main__":
    pass
