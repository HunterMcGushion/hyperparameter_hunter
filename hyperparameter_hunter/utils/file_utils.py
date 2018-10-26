"""This module defines utilities for reading, writing, and modifying different types of files"""
##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import os
import os.path
import pickle
import simplejson as json


##################################################
# JSON File Functions
##################################################
def default_json_write(obj):
    """Convert values that are not JSON-friendly to a more acceptable type

    Parameters
    ----------
    obj: Object
        The object that is expected to be of a type that is incompatible with JSON files

    Returns
    -------
    Object
        The value of `obj` after being cast to a type accepted by JSON

    Raises
    ------
    TypeError
        If the type of `obj` is unhandled"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"{obj!r} is not JSON serializable")


def write_json(file_path, data, do_clear=False):
    """Write `data` to the JSON file specified by `file_path`, optionally clearing the file before
    adding `data`

    Parameters
    ----------
    file_path: String
        The target .json file path to which `data` will be written
    data: Object
        The content to save at the .json file given by `file_path`
    do_clear: Boolean, default=False
        If True, the contents of the file at `file_path` will be cleared before saving `data`"""
    file_path = validate_extension(file_path, "json")

    if do_clear is True:
        clear_file(file_path)

    with open(file_path, "w") as f:
        json.dump(data, f, default=default_json_write)


def read_json(file_path, np_arr=False):
    """Get the contents of the .json file located at `file_path`

    Parameters
    ----------
    file_path: String
        The path of the .json file to be read
    np_arr: Boolean, default=False
        If True, the contents read from `file_path` will be cast to a numpy array before returning

    Returns
    -------
    content: Object
        The contents of the .json file located at `file_path`"""
    try:
        content = json.loads(open(file_path).read())
    except json.JSONDecodeError as _ex:
        raise _ex

    if np_arr is True:
        return np.array(content)

    return content


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
# Pickle File Functions
##################################################
def read_pickle(file_path, default_value=None):
    file_path = validate_extension(file_path, "pkl")

    try:
        target_file = open(file_path, "rb")
    except IOError as err:
        if default_value is not None:
            write_pickle(file_path, default_value)
            target_file = open(file_path, "rb")
        else:
            raise err
    file_content = pickle.load(target_file)
    target_file.close()

    return file_content


def write_pickle(file_path, content, do_create=True):
    file_path = validate_extension(file_path, "pkl")

    try:
        target_file = open(file_path, "wb")
    except IOError as err:
        if do_create is True:
            os.makedirs(os.path.dirname(file_path))
            target_file = open(file_path, "wb")
        else:
            raise IOError(
                "write_pickle() received an invalid file_path and did not create it --- {}".format(
                    err
                )
            )

    pickle.dump(content, target_file)
    target_file.close()


##################################################
# General File Functions
##################################################
def validate_extension(file_path, extension, do_fix=False):
    extension = extension if extension.startswith(".") else ".{}".format(extension)

    if os.path.splitext(file_path)[1] != extension:
        if do_fix is True:
            return "{}{}".format(file_path, extension)
        else:
            raise ValueError(
                'Invalid extension ({}) for file_path="{}"'.format(extension, file_path)
            )
    else:
        return file_path


def write_file(file_path, data, do_clear=False, txt=False):
    # file_path = validate_extension(file_path, '.txt') if txt is True else file_path

    if do_clear is True:
        clear_file(file_path)

    file_w = open(file_path, "a+")
    file_w.write(data)


def read_file(file_path):
    with open(file_path, "r") as f:
        read_target = f.read()
    return read_target


def clear_file(file_path):
    """Erase the contents of the file located at `file_path`

    Parameters
    ----------
    file_path: String
        The path of the file whose contents should be cleared out"""
    clear_target = open(file_path, "w")
    clear_target.truncate()
    clear_target.close()


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


def print_tree(start_path, depth=-1):
    prefix = 0

    if start_path != "/":
        if start_path.endswith(
            "/"
        ):  # If True, the last dir in start_path will be treated as root, rather than the whole thing
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
        content = "\u001b[;1m" + content + "\u001b[0m"
        print(content)

        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                content = "{}{}".format(sub_indent, real_name(d, root=root))
                print(content)

        for f in files:
            content = "{}{}".format(sub_indent, real_name(f, root=root))
            print(content)


if __name__ == "__main__":
    pass
