"""This module handles the creation of `cross_experiment_key` s and `hyperparameter_key` s for
:class:`hyperparameter_hunter.environment.Environment`, and
:class:`hyperparameter_hunter.experiments.BaseExperiment`, respectively. It also handles the
treatment of complex-typed inputs and their storage in the 'KeyAttributeLookup' subdirectory. The
descendants of :class:`hyperparameter_hunter.keys.makers.KeyMaker` defined herein are each
responsible for the generation and saving of their keys, as well as determining whether such a key
already exists

Related
-------
:mod:`hyperparameter_hunter.environment`
    This module uses :class:`hyperparameter_hunter.keys.makers.CrossExperimentKeyMaker` to set
    :attr:`hyperparameter_hunter.environment.Environment.cross_experiment_key`
:mod:`hyperparameter_hunter.experiments`
    This module uses :class:`hyperparameter_hunter.keys.makers.HyperparameterKeyMaker` to set
    :attr:`hyperparameter_hunter.experiments.BaseExperiment.hyperparameter_key`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exceptions import EnvironmentInvalidError, EnvironmentInactiveError
from hyperparameter_hunter.feature_engineering import FeatureEngineer, EngineerStep
from hyperparameter_hunter.keys.hashing import make_hash_sha256
from hyperparameter_hunter.library_helpers.keras_helper import (
    keras_callback_to_dict,
    keras_initializer_to_dict,
    parameterize_compiled_keras_model,
)
from hyperparameter_hunter.library_helpers.keras_optimization_helper import initialize_dummy_model
from hyperparameter_hunter.metrics import Metric
from hyperparameter_hunter.sentinels import Sentinel
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.file_utils import write_json, read_json, add_to_json, make_dirs
from hyperparameter_hunter.utils.file_utils import RetryMakeDirs
from hyperparameter_hunter.utils.general_utils import subdict
from hyperparameter_hunter.utils.boltons_utils import remap, default_enter

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import dill  # TODO: Figure out if this can be safely removed
from functools import partial
from inspect import getsourcelines, isclass, getsource
from os import listdir
import os.path
import pandas as pd
from pickle import PicklingError
import shelve

##################################################
# Import Learning Assets
##################################################
try:
    from keras.callbacks import Callback as BaseKerasCallback
    from keras.initializers import Initializer as BaseKerasInitializer
except ModuleNotFoundError:
    BaseKerasCallback = type("BaseKerasCallback", tuple(), {})
    BaseKerasInitializer = type("BaseKerasInitializer", tuple(), {})


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

        self.lookup_dir = None
        self.tested_keys_dir = None

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
            self.lookup_dir = G.Env.result_paths["key_attribute_lookup"]
            self.tested_keys_dir = G.Env.result_paths["tested_keys"]

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

            if isinstance(value, EngineerStep):
                return ({}, list(value.get_key_data().items()))
            if isinstance(value, FeatureEngineer):
                return ({}, list(value.get_key_data().items()))

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
            if isinstance(value, BaseKerasInitializer):
                return (key, keras_initializer_to_dict(value))
            if isinstance(value, Sentinel):
                return (key, value.sentinel)
            # from hyperparameter_hunter.feature_engineering import FeatureEngineer, EngineerStep
            # if isinstance(value, EngineerStep):
            #     return (key, value.get_key_data())
            # if isinstance(value, FeatureEngineer):
            #     return (key, value.get_key_data())
            elif callable(value) or isinstance(value, pd.DataFrame):
                # TODO: Check here if callable, and using a `Trace`d model/model_initializer
                # TODO: If so, pass extra kwargs to below `make_hash_sha256`, which are eventually given to `hash_callable`
                # TODO: Notably, `ignore_source_lines=True` should be included
                # FLAG: Also, look into adding package version number to hashed attributes
                hashed_value = make_hash_sha256(value)

                if isinstance(value, pd.DataFrame):
                    dataframe_hashes.setdefault(hashed_value, []).append(key)

                if self.tested_keys_dir is not None:  # Key-making not blacklisted
                    self.add_complex_type_lookup_entry(path, key, value, hashed_value)
                return (key, hashed_value)
            return (key, value)

        self.parameters = remap(self.parameters, visit=visit, enter=enter)

        #################### Check for Identical DataFrames ####################
        for df_hash, df_names in dataframe_hashes.items():
            if len(df_names) > 1:
                G.warn(f"The dataframes: {df_names} are identical. Scores may be misleading!")

    @RetryMakeDirs()
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
        shelve_params = ["model_initializer", "cv_type"]
        lookup_path = partial(os.path.join, self.lookup_dir, *[f"{_}" for _ in path])

        if isclass(value) or (key in shelve_params):
            make_dirs(lookup_path(), exist_ok=True)

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

    @abstractmethod
    def does_key_exist(self) -> bool:
        """Check if key hash exists among saved keys in the contents of :attr:`tested_keys_dir`"""

    @abstractmethod
    def save_key(self):
        """Save the key hash and the parameters used to make it to :attr:`tested_keys_dir`"""


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
            Additional arguments supplied to :meth:`keys.makers.KeyMaker.__init__`"""
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
            :class:`keys.makers.CrossExperimentKeyMaker`, used for determining when a
            hyperparameter key has already been tested under the same cross-experiment parameters
        **kwargs: Dict
            Additional arguments supplied to :meth:`keys.makers.KeyMaker.__init__`"""
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

            parameters["model_extra_params"] = subdict(
                parameters["model_extra_params"], drop=["params"]
            )

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
        reject = ["verbose", "verbosity", "silent"]
        reject += ["random_state", "random_seed", "seed", "n_jobs", "nthread"]

        if self.is_task_keras:
            reject.append("build_fn")

        parameters["model_init_params"] = subdict(parameters["model_init_params"], drop=reject)
        parameters["model_extra_params"] = subdict(parameters["model_extra_params"], drop=reject)
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
