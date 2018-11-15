"""This module handles the creation of `cross_experiment_key` s and `hyperparameter_key` s for
:class:`hyperparameter_hunter.environment.Environment`, and
:class:`hyperparameter_hunter.experiments.BaseExperiment`, respectively. It also handles the
treatment of complex-typed inputs and their storage in the 'KeyAttributeLookup' subdirectory. The
descendants of :class:`hyperparameter_hunter.key_handler.KeyMaker` defined herein are each
responsible for the generation and saving of their keys, as well as determining whether such a key
already exists

Related
-------
:mod:`hyperparameter_hunter.environment`
    This module uses :class:`hyperparameter_hunter.key_handler.CrossExperimentKeyMaker` to set
    :attr:`hyperparameter_hunter.environment.Environment.cross_experiment_key`
:mod:`hyperparameter_hunter.experiments`
    This module uses :class:`hyperparameter_hunter.key_handler.HyperparameterKeyMaker` to set
    :attr:`hyperparameter_hunter.experiments.BaseExperiment.hyperparameter_key`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import EnvironmentInvalidError, EnvironmentInactiveError
from hyperparameter_hunter.library_helpers.keras_helper import (
    keras_callback_to_dict,
    parameterize_compiled_keras_model,
)
from hyperparameter_hunter.library_helpers.keras_optimization_helper import initialize_dummy_model
from hyperparameter_hunter.metrics import Metric
from hyperparameter_hunter.sentinels import Sentinel
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.file_utils import write_json, read_json, add_to_json
from hyperparameter_hunter.utils.boltons_utils import remap, default_enter

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
import base64
from copy import deepcopy
import dill  # TODO: Figure out if this can be safely removed
from functools import partial
import hashlib
from inspect import getsourcelines, isclass, getsource
from os import listdir
import os.path
import pandas as pd
import re
import shelve

##################################################
# Import Learning Assets
##################################################
try:
    from keras.callbacks import Callback as BaseKerasCallback
except ModuleNotFoundError:

    class BaseKerasCallback:
        placeholder_attribute = """
        Hello, there! I am a `placeholder_attribute` for `BaseKerasCallback` if attempting to import `Keras` raised a 
        `ModuleNotFoundError`. You might be wondering what I'm doing here. I'm special because no normal/sane person would make a
        class, or an attribute just like me! That means that if anyone checks to see if something is an instance of yours truly, 
        hopefully it won't be! :) Nice to meet you! &*%#))(%#(*&@*HIOV0(#*W*Q()UFIJW_Q)_#R*(*(T{_E_QWO_))T+VMS"W)|GO{>A?C<A/woe0
        """


##################################################
# KeyMaker Base Class:
##################################################
class KeyMaker(metaclass=ABCMeta):
    def __init__(self, parameters, **kwargs):
        """Base class to handle making key hashes and checking for their existence. Additionally,
        this class handles saving entries for complex-typed parameters, along with their hashes to
        ensure experiments are reproducible

        Parameters
        ----------
        parameters: Dict
            All the parameters to be included when creating the key hash. Keys should correspond to
            parameter names, and values should be the values of the corresponding keys
        **kwargs: Dict
            Additional arguments

        Attributes
        ----------
        parameters: Dict
            A deep copy of the given `parameters` input
        key: Str, or None
            If a key has been generated for `parameters`, it is saved here. Else, None
        exists: Boolean
            If `key` is not None, and was found to already exist in `tested_keys_dir`,
            `exists` = True. Else, False
        key_attribute_lookup_dir: Str
            The directory in which complex-typed parameter entries will be saved
        tested_keys_dir: Str, or None
            The directory is which `key` will be saved if it does not already contain `key`"""
        self.parameters = deepcopy(parameters)
        self.key = None
        self.exists = False

        self.key_attribute_lookup_dir = G.Env.result_paths["key_attribute_lookup"]
        self.tested_keys_dir = G.Env.result_paths["tested_keys"]

        self.validate_environment()
        self.handle_complex_types()
        self.make_key()

        self.does_key_exist()

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key!r})"

    def __str__(self):
        return f"{self.key!s}"

    def __eq__(self, other):
        return self.key == other

    def __ne__(self, other):
        """Instance will always return True for a non-equality check if `key` is unset (None)"""
        return (self.key is None) or (self.key != other)

    ##################################################
    # Core Methods
    ##################################################
    def validate_environment(self):
        """Check that the currently active Environment is suitable"""
        if G.Env is None:
            raise EnvironmentInactiveError("")
        if not all([hasattr(G.Env, _) for _ in ["result_paths", "cross_experiment_key"]]):
            raise EnvironmentInvalidError("")
        try:
            # Ensure :attr:`tested_keys_dir` exists before calling :meth:`does_key_exist`, so "None" paths won't be checked
            if os.path.exists(self.tested_keys_dir) is False:
                # TypeError may also be raised if :func:`os.path.exists` receives invalid input
                raise TypeError
        except TypeError:  # Key-making blacklisted
            if self.tested_keys_dir is None:
                return
            os.makedirs(self.tested_keys_dir)

    def handle_complex_types(self):
        """Locate complex types in :attr:`parameters`, create hashes for them, add lookup entries
        linking their original values to their hashes, then update their values in
        :attr:`parameters` to their hashes to facilitate Description saving"""
        if self.tested_keys_dir is None:  # Key-making blacklisted
            return

        dataframe_hashes = {}

        def enter(path, key, value):
            """Produce iterable of attributes to remap for instances of :class:`metrics.Metric`"""
            if isinstance(value, Metric):
                return (
                    dict(),
                    [(_, getattr(value, _)) for _ in ["name", "metric_function", "direction"]],
                )
            return default_enter(path, key, value)

        def visit(path, key, value):
            """Check whether a parameter is of a complex type. If not, return it unchanged.
            Otherwise, 1) create a hash for its value; 2) save a complex type lookup entry linking
            `key`, `value`, and the hash for `value`; and 3) return the hashed value with `key`,
            instead of the original complex-typed `value`

            Parameters
            ----------
            path: Tuple
                The path of keys that leads to `key`
            key: Str
                The parameter name
            value: *
                The value of the parameter `key`

            Returns
            -------
            Tuple of (`key`, value), in which value is either unchanged or a hash for the original
            `value`"""
            if isinstance(value, BaseKerasCallback):
                return (key, keras_callback_to_dict(value))
            if isinstance(value, Sentinel):
                return (key, value.sentinel)
            elif callable(value) or isinstance(value, pd.DataFrame):
                hashed_value = make_hash_sha256(value)

                if isinstance(value, pd.DataFrame):
                    dataframe_hashes.setdefault(hashed_value, []).append(key)

                try:
                    self.add_complex_type_lookup_entry(path, key, value, hashed_value)
                except FileNotFoundError:
                    os.makedirs(self.key_attribute_lookup_dir, exist_ok=False)
                    self.add_complex_type_lookup_entry(path, key, value, hashed_value)

                return (key, hashed_value)
            return (key, value)

        self.parameters = remap(self.parameters, visit=visit, enter=enter)

        #################### Check for Identical DataFrames ####################
        for df_hash, df_names in dataframe_hashes.items():
            if len(df_names) > 1:
                G.warn(
                    f"The dataframes: {df_names} have an identical hash: {df_hash!s}. This implies the dataframes are "
                    + "identical, which is probably unintentional. If left alone, scores may be misleading!"
                )

    def add_complex_type_lookup_entry(self, path, key, value, hashed_value):
        """Add lookup entry in `key_attribute_lookup_dir` for a complex-typed parameter, linking
        the parameter `key`, its `value`, and its `hashed_value`

        Parameters
        ----------
        path: Tuple
            The path of keys that leads to `key`
        key: Str
            The parameter name
        value: *
            The value of the parameter `key`
        hashed_value: Str
            The hash produced for `value`"""
        # TODO: Combine `path` and `key` to produce actual filepaths
        shelve_params = ["model_initializer", "cross_validation_type"]

        if isclass(value) or (key in shelve_params):
            with shelve.open(os.path.join(self.key_attribute_lookup_dir, f"{key}"), flag="c") as s:
                # NOTE: When reading from shelve file, DO NOT add the ".db" file extension
                s[hashed_value] = value
        elif isinstance(value, pd.DataFrame):
            os.makedirs(os.path.join(self.key_attribute_lookup_dir, key), exist_ok=True)
            value.to_csv(
                os.path.join(self.key_attribute_lookup_dir, key, f"{hashed_value}.csv"), index=False
            )
        else:  # Possible types: partial, function, *other
            add_to_json(
                file_path=os.path.join(self.key_attribute_lookup_dir, f"{key}.json"),
                data_to_add=getsource(value),
                key=hashed_value,
                condition=lambda _: hashed_value not in _.keys(),
                default={},
            )

    def make_key(self):
        """Set :attr:`key` to an sha256 hash for :attr:`parameters`"""
        self.key = make_hash_sha256(self._filter_parameters_to_hash(deepcopy(self.parameters)))

    @staticmethod
    def _filter_parameters_to_hash(parameters):
        """Produce a filtered version of `parameters` that does not include values that should be
        ignored during hashing

        Parameters
        ----------
        parameters: dict
            The full dictionary of initial parameters to be filtered

        Returns
        -------
        parameters: dict
            The filtered version of the given `parameters`"""
        return parameters

    ##################################################
    # Abstract Methods
    ##################################################
    @property
    @abstractmethod
    def key_type(self) -> str:
        """A string in ['hyperparameter', 'cross_experiment'] denoting which type of key is
        being processed"""
        raise NotImplementedError()

    @abstractmethod
    def does_key_exist(self) -> bool:
        """Check if the key hash already exists among previously saved keys in the contents of
        :attr:`tested_keys_dir`"""
        raise NotImplementedError()

    @abstractmethod
    def save_key(self):
        """Save the key hash and the parameters used to make it to :attr:`tested_keys_dir`"""
        raise NotImplementedError()


class CrossExperimentKeyMaker(KeyMaker):
    key_type = "cross_experiment"

    def __init__(self, parameters, **kwargs):
        """A KeyMaker class dedicated to creating cross-experiment keys, which determine when
        experiments were executed under sufficiently similar conditions to permit proper comparison.
        Two separate instances of :class:`environment.Environment` should produce identical
        `cross_experiment_key` s if their arguments are the same (or close enough)

        Parameters
        ----------
        parameters: Dict
            All the parameters to be included when creating the key hash. Keys should correspond to
            parameter names, and values should be the values of the corresponding keys
        **kwargs: Dict
            Additional arguments supplied to :meth:`key_handler.KeyMaker.__init__`"""
        KeyMaker.__init__(self, parameters, **kwargs)

    def does_key_exist(self):
        """Check if a file corresponding to this cross_experiment_key already exists

        Returns
        -------
        Boolean"""
        tested_keys_dir_contents = [os.path.splitext(_)[0] for _ in listdir(self.tested_keys_dir)]
        self.exists = self.key in tested_keys_dir_contents

        return self.exists

    def save_key(self):
        """Create a new file for this cross_experiment_key if :attr:`exists` is False"""
        if not self.exists:
            write_json(f"{self.tested_keys_dir}/{self.key}.json", {})
            self.exists = True
            G.log(f'Saved {self.key_type}_key: "{self.key}"')
        else:
            G.log(f'{self.key_type}_key "{self.key}" already exists - Skipped saving')


class HyperparameterKeyMaker(KeyMaker):
    key_type = "hyperparameter"

    def __init__(self, parameters, cross_experiment_key, **kwargs):
        """A KeyMaker class dedicated to creating hyperparameter keys, which determine when
        experiments were executed using identical hyperparameters. Two separate instances of
        :class:`experiments.CrossValidationExperiment` should produce identical
        `hyperparameter_key` s if their hyperparameters are the same (or close enough)

        Parameters
        ----------
        parameters: Dict
            All the parameters to be included when creating the key hash. Keys should correspond to
            parameter names, and values should be the values of the corresponding keys
        cross_experiment_key: Str
            The key produced by the active Environment via
            :class:`key_handler.CrossExperimentKeyMaker`, used for determining when a
            hyperparameter key has already been tested under the same cross-experiment parameters
        **kwargs: Dict
            Additional arguments supplied to :meth:`key_handler.KeyMaker.__init__`"""
        self.cross_experiment_key = cross_experiment_key

        if (
            hasattr(G.Env, "current_task")
            and G.Env.current_task
            and G.Env.current_task.module_name == "keras"
        ):
            parameters = deepcopy(parameters)

            #################### Initialize and Parameterize Dummy Model ####################
            temp_model = initialize_dummy_model(
                parameters["model_initializer"],
                parameters["model_init_params"]["build_fn"],
                parameters["model_extra_params"],
            )

            temp_layers, temp_compile_params = parameterize_compiled_keras_model(temp_model)

            #################### Process Parameters ####################
            # noinspection PyUnusedLocal
            def _visit(path, key, value):
                """If `key` not in ('input_shape', 'input_dim'), return True. Else, return False"""
                return key not in ("input_shape", "input_dim")

            temp_layers = remap(temp_layers, visit=_visit)

            parameters["model_init_params"]["layers"] = temp_layers
            parameters["model_init_params"]["compile_params"] = temp_compile_params

            if "params" in parameters["model_extra_params"]:
                parameters["model_extra_params"] = {
                    _k: _v for _k, _v in parameters["model_extra_params"].items() if _k != "params"
                }

        KeyMaker.__init__(self, parameters, **kwargs)

    @staticmethod
    def _filter_parameters_to_hash(parameters):
        """Produce a filtered version of `parameters` that does not include hyperparameters that
        should be ignored during hashing, such as those pertaining to verbosity, seeds, and random
        states, as they have no effect on the results of experiments when within the confines of
        hyperparameter_hunter

        Parameters
        ----------
        parameters: dict
            The full dictionary of initial parameters to be filtered

        Returns
        -------
        parameters: dict
            The filtered version of the given `parameters`"""
        reject_keys = {
            "verbose",
            "verbosity",
            "silent",
            "random_state",
            "random_seed",
            "seed",
            "n_jobs",
            "nthread",
        }

        if (
            hasattr(G.Env, "current_task")
            and G.Env.current_task
            and G.Env.current_task.module_name == "keras"
        ):
            reject_keys.add("build_fn")

        for k in reject_keys:
            if parameters["model_init_params"] and (k in parameters["model_init_params"].keys()):
                del parameters["model_init_params"][k]
            if parameters["model_extra_params"] and (k in parameters["model_extra_params"].keys()):
                del parameters["model_extra_params"][k]

        return parameters

    def does_key_exist(self):
        """Check that 1) there is a file corresponding to :attr:`cross_experiment_key.key`,
        2) the aforementioned file contains the key :attr:`key`, and
        3) the value of the file at :attr:`key` is a non-empty list

        Returns
        -------
        Boolean"""
        if self.cross_experiment_key.exists is True:
            records = read_json(f"{self.tested_keys_dir}/{self.cross_experiment_key.key}.json")

            for a_hyperparameter_key in records.keys():
                if self.key == a_hyperparameter_key:
                    if (
                        isinstance(records[a_hyperparameter_key], list)
                        and len(records[a_hyperparameter_key]) > 0
                    ):
                        self.exists = True
                        return self.exists

        return self.exists

    def save_key(self):
        """Create an entry in the dict corresponding to the file at
        :attr:`cross_experiment_key.key`, whose key is :attr:`key`, and whose value is an empty
        list if :attr:`exists` is False"""
        if not self.exists:
            if self.cross_experiment_key.exists is False:
                raise ValueError(
                    'Cannot save hyperparameter_key: "{}", before cross_experiment_key "{}" has been saved'.format(
                        self.key, self.cross_experiment_key.key
                    )
                )

            key_path = f"{self.tested_keys_dir}/{self.cross_experiment_key.key}.json"
            add_to_json(
                file_path=key_path,
                data_to_add=[],
                key=self.key,
                condition=lambda _: self.key not in _.keys(),
            )

            self.exists = True
            G.log(f'Saved {self.key_type}_key: "{self.key}"')
        else:
            G.log(f'{self.key_type}_key "{self.key}" already exists - Skipped saving')


def make_hash_sha256(obj, **kwargs):
    """Create an sha256 hash of the input obj

    Parameters
    ----------
    obj: Any object
        The object for which a hash will be created
    **kwargs: Dict
        Any extra kwargs will be supplied to :func:`key_handler.hash_callable`

    Returns
    -------
    Stringified sha256 hash"""
    hasher = hashlib.sha256()
    hasher.update(repr(to_hashable(obj, **kwargs)).encode())
    return base64.urlsafe_b64encode(hasher.digest()).decode()


def to_hashable(obj, **kwargs):
    """Format the input obj to be hashable

    Parameters
    ----------
    obj: Any object
        The object to convert to a hashable format
    **kwargs: Dict
        Any extra kwargs will be supplied to :func:`key_handler.hash_callable`

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
    """Prepare callable object for hashing

    Parameters
    ----------
    obj: callable
        The callable to convert to a hashable format. Currently supported types are: function,
        class, :class:`functools.partial`
    ignore_line_comments: boolean, default=True
        If True, any line comments will be stripped from the source code of `obj`, specifically any
        lines that start with zero or more whitespaces, followed by an octothorpe (#). This does not
        apply to comments on the same line as code
    ignore_first_line: boolean, default=False
        If True, the first line will be stripped from the callable's source code, specifically the
        function's name and signature. If ignore_name=True, this will be treated as True
    ignore_module: boolean, default=False
        If True, the name of the module in which the source code is located (:attr:`obj.__module__`)
        will be ignored
    ignore_name: boolean, default=False
        If True, :attr:`obj.__name__` will be ignored. Note the distinction between this and
        `ignore_first_line`, which strips the entire callable signature from the source code.
        `ignore_name` does not alter the source code. To ensure thorough ignorance,
        ignore_first_line=True is recommended
    ignore_keywords: boolean, default=False
        If True and `obj` is a :class:`functools.partial` (not a normal function/method),
        :attr:`obj.keywords` will be ignored
    ignore_source_lines: boolean, default=False
        If True, all source code will be ignored by the hashing function. Ignoring all other kwargs,
        this means that only :attr:`obj.__module__`, and :attr:`obj.__name__`,
        (and :attr:`obj.keywords` if `obj` is partial) will be used for hashing

    Returns
    -------
    Hashable properties of the callable object input"""
    source_lines, module, name, keywords = None, None, None, None

    #################### Get Identifying Data ####################
    if isinstance(obj, partial):
        module = obj.func.__module__
        name = obj.func.__name__
        keywords = obj.keywords
        source_lines = getsourcelines(obj.func)[0]
    elif callable(obj):
        module = obj.__module__
        name = obj.__name__
        # TODO: Below only works on modified Keras `build_fn` during optimization if temp file still exists
        try:
            source_lines = getsourcelines(obj)[0]
        except Exception as _ex:
            print(_ex)
            print()
            raise
        # TODO: Above only works on modified Keras `build_fn` during optimization if temp file still exists
    else:
        raise TypeError("Expected functools.partial, or callable. Received {}".format(type(obj)))

    #################### Format Source Code Lines ####################
    if ignore_line_comments is True:
        source_lines = [_ for _ in source_lines if not is_line_comment(_)]
    if (ignore_first_line is True) or (ignore_name is True):
        source_lines = source_lines[1:]

    #################### Select Relevant Data ####################
    source_lines = None if ignore_source_lines is True else source_lines
    module = None if ignore_module is True else module
    name = None if ignore_name is True else name
    keywords = None if ignore_keywords is True else keywords

    relevant_data = [_ for _ in [module, name, keywords, source_lines] if _ is not None]
    # noinspection PyTypeChecker
    return tuple(to_hashable(relevant_data))


def is_line_comment(string):
    """Return True if the given string is a line comment, else False

    Parameters
    ----------
    string: str
        The str in which to check for a line comment

    Returns
    -------
    Boolean"""
    return bool(re.match(r"^\s*#", string))


if __name__ == "__main__":
    pass
