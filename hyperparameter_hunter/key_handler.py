##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.exception_handler import EnvironmentInvalidError, EnvironmentInactiveError
from hyperparameter_hunter.library_helpers.keras_helper import keras_callback_to_key
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.file_utils import write_json, read_json, add_to_json
from hyperparameter_hunter.utils.boltons_utils import remap

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
import base64
from copy import deepcopy
import dill
from functools import partial
import hashlib
from inspect import getsourcelines, isfunction, isclass, getsource
from os import listdir
import os.path
import pandas as pd
import re
import shelve

##################################################
# Import Learning Assets
##################################################
from keras.callbacks import Callback as base_keras_callback


##################################################
# KeyMaker Base Class:
##################################################
class KeyMaker(metaclass=ABCMeta):
    """
    # TODO: Fix documentation to reflect HyperparameterHunterAssets structure changes
    """
    def __init__(self, parameters, **kwargs):
        self.parameters = deepcopy(parameters)
        self.key = None
        self.exists = False

        self.key_attribute_lookup_dir = G.Env.result_paths['key_attribute_lookup']
        self.tested_keys_dir = G.Env.result_paths['tested_keys']

        self.validate_environment()
        self.handle_complex_types()
        self.make_key()

        self.does_key_exist()

    def __repr__(self):
        return F'{self.__class__.__name__}(key={self.key!r})'

    def __str__(self):
        return F'{self.key!s}'

    def __eq__(self, other):
        return self.key == other

    def __ne__(self, other):
        """A KeyMaker instance will always return True for a non-equality check if its key has not been set (is None)"""
        return (self.key is None) or (self.key != other)

    ##################################################
    # Core Methods
    ##################################################
    def validate_environment(self):
        """Check that the currently active Environment is suitable"""
        if G.Env is None:
            raise EnvironmentInactiveError('')
        if not all([hasattr(G.Env, _) for _ in ['result_paths', 'cross_experiment_key']]):
            raise EnvironmentInvalidError('')
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
        """Locate complex types in :attr:`parameters`, create hashes for them, add lookup entries linking their original values
        to their hashes, then update their values in :attr:`parameters` to their hashes to facilitate Description saving"""
        if self.tested_keys_dir is None:  # Key-making blacklisted
            return

        dataframe_hashes = {}

        def visit(path, key, value):
            if isinstance(value, base_keras_callback):
                return (key, keras_callback_to_key(value))
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

        self.parameters = remap(self.parameters, visit=visit)

        #################### Check for Identical DataFrames ####################
        for df_hash, df_names in dataframe_hashes.items():
            if len(df_names) > 1:
                G.warn(
                    F'The dataframes: {df_names} have an identical hash: {df_hash!s}. This implies the dataframes are ' +
                    'identical, which is probably unintentional. If left alone, scores may be misleading!'
                )

    def add_complex_type_lookup_entry(self, path, key, value, hashed_value):
        """Add lookup entry for a complex-typed parameter, linking the parameter `key`, its `value`, and its `hashed_value`"""
        # TODO: Combine `path` and `key` to produce actual filepaths
        shelve_params = ['model_initializer', 'cross_validation_type']

        if isclass(value) or (key in shelve_params):
            with shelve.open(os.path.join(self.key_attribute_lookup_dir, F'{key}'), flag='c') as shelf:
                # TODO: When reading from shelve file, DO NOT add the ".db" file extension
                shelf[hashed_value] = value
        elif isinstance(value, pd.DataFrame):
            os.makedirs(os.path.join(self.key_attribute_lookup_dir, key), exist_ok=True)
            value.to_csv(os.path.join(self.key_attribute_lookup_dir, key, F'{hashed_value}.csv'), index=False)
        else:  # Possible types: partial, function, *other
            add_to_json(
                file_path=os.path.join(self.key_attribute_lookup_dir, F'{key}.json'),
                data_to_add=getsource(value), key=hashed_value, condition=lambda _: hashed_value not in _.keys(), default={},
            )

    def make_key(self):
        """Set :attr:`key` to an sha256 hash for :attr:`parameters`"""
        self.key = make_hash_sha256(self.filter_parameters_to_hash(deepcopy(self.parameters)))

    @staticmethod
    def filter_parameters_to_hash(parameters):
        """Produce a filtered version of `parameters` that does not include values that should be ignored during hashing

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
        raise NotImplementedError()

    @abstractmethod
    def does_key_exist(self) -> bool:
        """Check if the key hash already exists among previously saved keys in the contents of :attr:`tested_keys_dir`"""
        raise NotImplementedError()

    @abstractmethod
    def save_key(self):
        """Save the key hash and the parameters used to make it to :attr:`tested_keys_dir`"""
        raise NotImplementedError()


class CrossExperimentKeyMaker(KeyMaker):
    key_type = 'cross_experiment'

    def __init__(self, parameters, **kwargs):
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
            write_json(F'{self.tested_keys_dir}/{self.key}.json', {})
            self.exists = True
            G.log(F'Saved {self.key_type}_key: "{self.key}"')
        else:
            G.log(F'{self.key_type}_key "{self.key}" already exists - Skipped saving')


class HyperparameterKeyMaker(KeyMaker):
    key_type = 'hyperparameter'

    def __init__(self, parameters, cross_experiment_key, **kwargs):
        self.cross_experiment_key = cross_experiment_key
        KeyMaker.__init__(self, parameters, **kwargs)

    @staticmethod
    def filter_parameters_to_hash(parameters):
        reject_keys = {
            'verbose', 'verbosity', 'silent',
            'random_state', 'random_seed', 'seed',
            'n_jobs', 'nthread',
        }

        for k in reject_keys:
            if k in parameters['model_init_params'].keys():
                del parameters['model_init_params'][k]

        return parameters

    def does_key_exist(self):
        """Check that 1) there is a file corresponding to :attr:`cross_experiment_key.key`, 2) the aforementioned file contains
        the key :attr:`key`, and 3) the value of the file at :attr:`key` is a non-empty list

        Returns
        -------
        Boolean"""
        if self.cross_experiment_key.exists is True:
            records = read_json(F'{self.tested_keys_dir}/{self.cross_experiment_key.key}.json')

            for a_hyperparameter_key in records.keys():
                if self.key == a_hyperparameter_key:
                    if isinstance(records[a_hyperparameter_key], list) and len(records[a_hyperparameter_key]) > 0:
                        self.exists = True
                        return self.exists

        return self.exists

    def save_key(self):
        """Create an entry in the dict corresponding to the file at :attr:`cross_experiment_key.key`, whose key is :attr:`key`,
        and whose value is an empty list if :attr:`exists` is False"""
        if not self.exists:
            if self.cross_experiment_key.exists is False:
                raise ValueError('Cannot save hyperparameter_key: "{}", before cross_experiment_key "{}" has been saved'.format(
                    self.key, self.cross_experiment_key.key
                ))

            key_path = F'{self.tested_keys_dir}/{self.cross_experiment_key.key}.json'
            add_to_json(file_path=key_path, data_to_add=[], key=self.key, condition=lambda _: self.key not in _.keys())

            self.exists = True
            G.log(F'Saved {self.key_type}_key: "{self.key}"')
        else:
            G.log(F'{self.key_type}_key "{self.key}" already exists - Skipped saving')


def make_hash_sha256(obj, **kwargs):
    """Create an sha256 hash of the input obj

    Parameters
    ----------
    obj: Any object
        The object for which a hash will be created
    kwargs: **
        Any extra kwargs will be supplied to :func:`key_handler.hash_callable`

    Returns
    -------
    Stringified sha256 hash
    """
    hasher = hashlib.sha256()
    hasher.update(repr(to_hashable(obj, **kwargs)).encode())
    return base64.urlsafe_b64encode(hasher.digest()).decode()


def to_hashable(obj, **kwargs):
    """Format the input obj to be hashable

    Parameters
    ----------
    obj: Any object
        The object to convert to a hashable format
    kwargs: **
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
        obj, ignore_line_comments=True, ignore_first_line=False,
        ignore_module=False, ignore_name=False, ignore_keywords=False, ignore_source_lines=False,
):
    """Prepare callable object for hashing

    Parameters
    ----------
    obj: callable
        The callable to convert to a hashable format. Currently supported types are: function, class, :class:`functools.partial`
    ignore_line_comments: boolean, default=True
        If True, any line comments will be stripped from the source code of `obj`, specifically any lines that start with
        zero or more whitespaces, followed by an octothorpe (#). This does not apply to comments on the same line as code
    ignore_first_line: boolean, default=False
        If True, the first line will be stripped from the callable's source code, specifically the function's name and
        signature. If ignore_name=True, this will be treated as True
    ignore_module: boolean, default=False
        If True, the name of the module in which the source code is located (:attr:`obj.__module__`) will be ignored
    ignore_name: boolean, default=False
        If True, :attr:`obj.__name__` will be ignored. Note the distinction between this and `ignore_first_line`, which strips
        the entire callable signature from the source code. `ignore_name` does not alter the source code. To ensure thorough
        ignorance, ignore_first_line=True is recommended
    ignore_keywords: boolean, default=False
        If True and `obj` is a :class:`functools.partial` (not a normal function/method), :attr:`obj.keywords` will be ignored
    ignore_source_lines: boolean, default=False
        If True, all source code will be ignored by the hashing function. Ignoring all other kwargs, this means that only
        :attr:`obj.__module__`, and :attr:`obj.__name__`, (and :attr:`obj.keywords` if `obj` is partial) will be used for hashing

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
    # elif isfunction(obj):
    elif callable(obj):
        module = obj.__module__
        name = obj.__name__
        source_lines = getsourcelines(obj)[0]
    else:
        raise TypeError('Expected obj of type functools.partial, or function. Received {}'.format(type(obj)))

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
    return bool(re.match(r'^\s*#', string))


def execute():
    pass


if __name__ == '__main__':
    execute()
