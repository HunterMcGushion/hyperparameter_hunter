"""This module defines Sentinel objects that are used to represent data that is not yet available.
For example, :class:`hyperparameter_hunter.sentinels.DatasetSentinel` is used in
:class:`hyperparameter_hunter.environment.Environment` to enable a user to pass the fold validation
dataset as an argument on Experiment initialization. At the point that the sentinel is provided, the
training dataset has not yet been split into folds, which is why the Sentinel is necessary

Related
-------
:mod:`hyperparameter_hunter.environment`
    :class:`hyperparameter_hunter.environment.Environment` has the following properties that utilize
    :class:`hyperparameter_hunter.sentinels.DatasetSentinel`: [`train_input`, `train_target`,
    `validation_input`, `validation_target`, `holdout_input`, `holdout_target`]. These properties
    can be passed as arguments to Experiment or OptimizationProtocol initialization in order to
    provide the dataset to a Model's `fit` call, for example
:mod:`hyperparameter_hunter.experiments`
    This is one of the points at which one might want to use the Sentinels exposed by
    :class:`hyperparameter_hunter.environment.Environment`, specifically as values in the
    `model_init_params` and `model_extra_params` arguments to a descendant of
    :class:`hyperparameter_hunter.experiments.BaseExperiment`
:mod:`hyperparameter_hunter.optimization_core`
    This is a second point at which one might use the Sentinels exposed by
    :class:`hyperparameter_hunter.environment.Environment`. In this case, they could be provided as
    values in the `model_init_params` and `model_extra_params` arguments in a call to
    :meth:`hyperparameter_hunter.optimization_core.BaseOptimizationProtocol.set_experiment_guidelines`,
    the structure of which intentionally mirrors that of
    :meth:`hyperparameter_hunter.experiments.BaseExperiment.__init__`
:mod:`hyperparameter_hunter.models`
    This is ultimately where `Sentinel` instances will be converted to the actual values that they
    represent via calls to :func:`hyperparameter_hunter.sentinels.locate_sentinels`"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.boltons_utils import remap

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod


##################################################
# Sentinel Base Class
##################################################
class Sentinel(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        """Base class for Sentinels representing data that is not yet available. Subclasses should
        call `super().__init__()` at the end of their `__init__` methods

        Parameters
        ----------
        *args: List
            Extra arguments for subclasses of :class:`Sentinel`
        **kwargs: Dict
            Extra keyword arguments for subclasses of :class:`Sentinel`"""
        self._sentinel = None
        self._validate_parameters()
        self.sentinel = self._build_sentinel()

    def __eq__(self, other):
        if isinstance(other, str):
            return self.sentinel == other
        return self == other

    @property
    def sentinel(self) -> str:
        """Retrieve :attr:`Sentinel._sentinel`

        Returns
        -------
        Str
            The value of :attr:`Sentinel._sentinel`"""
        return self._sentinel

    @sentinel.setter
    def sentinel(self, value: str):
        """Set :attr:`Sentinel._sentinel` to `value`, and add self to `settings.G.sentinel_registry`

        Parameters
        ----------
        value: Str
            The new value of :attr:`Sentinel._sentinel`"""
        if not isinstance(value, str):
            raise TypeError('`sentinel` must be str, not "{}": {}'.format(type(value), value))

        self._sentinel = value
        G.sentinel_registry.append(self)

    @abstractmethod
    def _build_sentinel(self) -> str:
        """Create a string containing all information necessary to identify the sentinel

        Returns
        -------
        sentinel: Str
            A string identifying the sentinel"""
        raise NotImplementedError

    @abstractmethod
    def retrieve_by_sentinel(self) -> object:
        """Retrieve the actual object represented by the sentinel

        Returns
        -------
        object
            The object for which the sentinel was being used as a placeholder"""
        raise NotImplementedError

    @abstractmethod
    def _validate_parameters(self):
        """Ensure input parameters are valid and properly formatted"""
        raise NotImplementedError


##################################################
# Sentinel Location Utilities
##################################################
# noinspection PyUnusedLocal
def _sentinel_visitor(path, key, value):
    """If `value` is a `Sentinel` return the value it represents. Else `default_visit`"""
    if isinstance(value, Sentinel):
        return (key, value.retrieve_by_sentinel())
    return (key, value)


def locate_sentinels(parameters):
    """Produce a mirrored `parameters` dict, wherein `Sentinel` values are converted to the objects
    they represent

    Parameters
    ----------
    parameters: Dict
        Dict of parameters, which may contain nested `Sentinel` values

    Returns
    -------
    Dict
        Mirror of `parameters`, except where a `Sentinel` was found, the value it represents
        is returned instead"""
    if len(G.sentinel_registry) == 0:
        return parameters
    return remap(parameters, visit=_sentinel_visitor)


##################################################
# Sentinel Classes
##################################################
class DatasetSentinel(Sentinel):
    def __init__(
        self,
        dataset_type,
        dataset_hash,
        cross_validation_type=None,
        global_random_seed=None,
        random_seeds=None,
    ):
        """Class to create sentinels representing dataset input/target values

        Parameters
        ----------
        dataset_type: Str
            Dataset type, suffixed with '_input', or '_target', for which a sentinel should be
            created. Acceptable values are as follows: ['train_input', 'train_target',
            'validation_input', 'validation_target', 'holdout_input', 'holdout_target']
        dataset_hash: Str
            The hash of the dataset for which a sentinel should be created that was generated while
            creating :attr:`hyperparameter_hunter.environment.Environment.cross_experiment_key`
        cross_validation_type: Str, or None, default=None
            If None, `dataset_type` should be one of ['holdout_input', 'holdout_target']. Else,
            should be a string that is one of the following: 1) a string attribute of
            `sklearn.model_selection._split`, or 2) a hash produced while creating
            :attr:`hyperparameter_hunter.environment.Environment.cross_experiment_key`
        global_random_seed: Int, or None, default=None
            If None, `dataset_type` should be one of ['holdout_input', 'holdout_target']. If int,
            should be :attr:`hyperparameter_hunter.environment.Environment.global_random_seed`
        random_seeds: List, or None, default=None
            If None, `dataset_type` should be one of ['holdout_input', 'holdout_target']. If list,
            should be :attr:`hyperparameter_hunter.environment.Environment.random_seeds`"""
        self.dataset_type = dataset_type
        self.dataset_hash = dataset_hash
        self.cross_validation_type = cross_validation_type
        self.global_random_seed = global_random_seed
        self.random_seeds = random_seeds
        super().__init__()

    def _build_sentinel(self):
        """Create a string containing all information necessary to identify the sentinel

        Returns
        -------
        sentinel: Str
            A string identifying the sentinel"""
        sentinel = "SENTINEL***"
        sentinel += self.dataset_type + "***"
        sentinel += self.dataset_hash + "***"
        sentinel += (
            self.cross_validation_type + "***" if self.cross_validation_type is not None else ""
        )
        if self.random_seeds is not None:
            sentinel += "{}".format(self.random_seeds)
        elif self.global_random_seed is not None:
            sentinel += "{}".format(self.global_random_seed)

        return sentinel

    def retrieve_by_sentinel(self):
        """Retrieve the actual dataset represented by the sentinel

        Returns
        -------
        object
            The dataset for which the sentinel was being used as a placeholder"""
        fold_dependent_datasets = (
            "train_input",
            "train_target",
            "validation_input",
            "validation_target",
        )

        if self.dataset_type in fold_dependent_datasets:
            return getattr(G.Env.current_task, "fold_{}".format(self.dataset_type))
        else:
            return getattr(G.Env.current_task, "{}_data".format(self.dataset_type))

    def _validate_parameters(self):
        """Ensure input parameters are valid and properly formatted"""
        #################### dataset_type ####################
        acceptable_values = [
            "train_input",
            "train_target",
            "validation_input",
            "validation_target",
            "holdout_input",
            "holdout_target",
        ]

        if self.dataset_type not in acceptable_values:
            raise ValueError("Received invalid `dataset_type`: '{}'".format(self.dataset_type))

        #################### cross_validation_type ####################
        if self.dataset_type in ("holdout_input", "holdout_target"):
            self.cross_validation_type = None
        elif self.cross_validation_type is None:
            raise ValueError(
                "`cross_validation_type` may only be None if `dataset_type` is from 'holdout'"
            )

        #################### global_random_seed ####################
        if self.dataset_type in ("holdout_input", "holdout_target"):
            self.global_random_seed = None
        elif self.global_random_seed is None:
            raise ValueError("`global_random_seed` may only be None if `dataset_type` is 'holdout'")

        #################### random_seeds ####################
        if self.dataset_type in ("holdout_input", "holdout_target"):
            self.random_seeds = None
        elif self.random_seeds is None and self.global_random_seed is None:
            raise ValueError(
                "`random_seeds` may only be None if `dataset_type` is from 'holdout', or `global_random_seed` given"
            )
