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
from hyperparameter_hunter.utils.file_utils import write_json, read_json, add_to_json, make_dirs
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
from pickle import PicklingError
import re
import shelve
import sys

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
            True if `key` is not None, and already exists in `tested_keys_dir`. Else, False
        lookup_dir: Str
            The directory in which complex-typed parameter entries will be saved
        tested_keys_dir: Str, or None
            The directory is which `key` will be saved if it does not already contain `key`"""
        self.parameters = deepcopy(parameters)
        self.key = None
        self.exists = False

        self.lookup_dir = G.Env.result_paths["key_attribute_lookup"]
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
            make_dirs(self.tested_keys_dir)

    def handle_complex_types(self):
        """Locate complex types in :attr:`parameters`, create hashes for them, add lookup entries
        linking their original values to their hashes, then update their values in
        :attr:`parameters` to their hashes to facilitate Description saving"""
        dataframe_hashes = {}

        def enter(path, key, value):
            """Produce iterable of attributes to remap for instances of :class:`metrics.Metric`"""
            if isinstance(value, Metric):
                metric_attrs = ["name", "metric_function", "direction"]
                return ({}, [(_, getattr(value, _)) for _ in metric_attrs])
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
                # TODO: Check here if callable, and using a `Trace`d model/model_initializer
                # TODO: If so, pass extra kwargs to below `make_hash_sha256`, which are eventually given to `hash_callable`
                # TODO: Notably, `ignore_source_lines=True` should be included
                # FLAG: Also, look into adding package version number to hashed attributes
                hashed_value = make_hash_sha256(value)

                if isinstance(value, pd.DataFrame):
                    dataframe_hashes.setdefault(hashed_value, []).append(key)

                if self.tested_keys_dir is not None:  # Key-making not blacklisted
                    try:
                        self.add_complex_type_lookup_entry(path, key, value, hashed_value)
                    except (FileNotFoundError, OSError):
                        make_dirs(os.path.join(self.lookup_dir, *path), exist_ok=False)
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
        """Add lookup entry in `lookup_dir` for a complex-typed parameter, linking
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
        shelve_params = ["model_initializer", "cross_validation_type"]
        lookup_path = partial(os.path.join, self.lookup_dir, *path)

        if isclass(value) or (key in shelve_params):
            with shelve.open(lookup_path(f"{key}"), flag="c") as s:
                # NOTE: When reading from shelve file, DO NOT add the ".db" file extension
                try:
                    s[hashed_value] = value
                except PicklingError:
                    # "is not the same object" error can be raised due to `Mirror`/`TranslateTrace`
                    # Instead of saving the object that raised the error, save `getsourcelines`
                    # Source lines of traced object are identical to those of its un-traced original
                    s[hashed_value] = getsourcelines(value)
                except Exception:
                    raise
        elif isinstance(value, pd.DataFrame):
            make_dirs(lookup_path(key), exist_ok=True)
            value.to_csv(lookup_path(key, f"{hashed_value}.csv"), index=False)
        else:  # Possible types: partial, function, *other
            add_to_json(
                file_path=lookup_path(f"{key}.json"),
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
        parameters: Dict
            The full dictionary of initial parameters to be filtered

        Returns
        -------
        parameters: Dict
            The filtered version of the given `parameters`"""
        return parameters

    ##################################################
    # Abstract Methods
    ##################################################
    @property
    @abstractmethod
    def key_type(self) -> str:
        """Str in ["hyperparameter", "cross_experiment"], denoting the key type being processed"""
        raise NotImplementedError()

    @abstractmethod
    def does_key_exist(self) -> bool:
        """Check if key hash exists among saved keys in the contents of :attr:`tested_keys_dir`"""
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
            G.log(f'Saved {self.key_type}_key: "{self.key}"', 4)
        else:
            G.log(f'{self.key_type}_key "{self.key}" already exists - Skipped saving', 4)


class HyperparameterKeyMaker(KeyMaker):
    key_type = "hyperparameter"

    def __init__(self, parameters, cross_experiment_key, **kwargs):
        """A KeyMaker class dedicated to creating hyperparameter keys, which determine when
        experiments were executed using identical hyperparameters. Two separate instances of
        :class:`experiments.CVExperiment` should produce identical `hyperparameter_key` s if their
        hyperparameters are the same (or close enough)

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
        self.is_task_keras = (
            hasattr(G.Env, "current_task")
            and G.Env.current_task
            and G.Env.current_task.module_name == "keras"
        )

        if self.is_task_keras:
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

    def _filter_parameters_to_hash(self, parameters):
        """Produce a filtered version of `parameters` that does not include hyperparameters that
        should be ignored during hashing, such as those pertaining to verbosity, seeds, and random
        states, as they have no effect on HyperparameterHunter experiment results

        Parameters
        ----------
        parameters: Dict
            Full dictionary of initial parameters to be filtered

        Returns
        -------
        parameters: Dict
            Filtered version of the given `parameters`"""
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

        if self.is_task_keras:
            reject_keys.add("build_fn")

        for k in reject_keys:
            if parameters["model_init_params"] and (k in parameters["model_init_params"].keys()):
                del parameters["model_init_params"][k]
            if parameters["model_extra_params"] and (k in parameters["model_extra_params"].keys()):
                del parameters["model_extra_params"][k]

        return parameters

    def does_key_exist(self):
        """Check that 1) there is a file for :attr:`cross_experiment_key.key`, 2) the aforementioned
        file contains the key :attr:`key`, and 3) the value at :attr:`key` is a non-empty list

        Returns
        -------
        Boolean"""
        if self.cross_experiment_key.exists is True:
            records = read_json(f"{self.tested_keys_dir}/{self.cross_experiment_key.key}.json")

            for a_hyperparameter_key in records.keys():
                if self.key == a_hyperparameter_key:
                    experiments_run = records[a_hyperparameter_key]
                    if isinstance(experiments_run, list) and len(experiments_run) > 0:
                        self.exists = True
                        return self.exists

        return self.exists

    def save_key(self):
        """Create an entry in the dict contained in the file at :attr:`cross_experiment_key.key`,
        whose key is :attr:`key`, and whose value is an empty list if :attr:`exists` is False"""
        if not self.exists:
            if self.cross_experiment_key.exists is False:
                _err = "Cannot save hyperparameter_key: '{}', before cross_experiment_key '{}'"
                raise ValueError(_err.format(self.key, self.cross_experiment_key.key))

            key_path = f"{self.tested_keys_dir}/{self.cross_experiment_key.key}.json"
            add_to_json(key_path, [], key=self.key, condition=lambda _: self.key not in _.keys())

            self.exists = True
            G.log(f'Saved {self.key_type}_key: "{self.key}"', 4)
        else:
            G.log(f'{self.key_type}_key "{self.key}" already exists - Skipped saving', 4)


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
    name = None if ignore_name else obj.__name__

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


if __name__ == "__main__":
    pass
