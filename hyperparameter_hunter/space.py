"""Defines hyperparameter search space dimension classes used for declaring hyperparameter choices,
as well as some utility functions for processing dimensions and the hyperparameter space as a whole

Related
-------
:mod:`hyperparameter_hunter.optimization_core`
    Defines optimization protocol classes that expect to receive hyperparameter dimension inputs
:mod:`hyperparameter_hunter.utils.optimization_utils`
    Defines utilities for matching a current hyperparameter space with the hyperparameters of saved
    Experiments. Also defines :class:`utils.optimization_utils.AskingOptimizer`, which determines
    the values in the given choices to search next

Notes
-----
This module heavily relies on the Scikit-Optimize library, so thank you to the creators and
contributors of `scikit-optimize` for their excellent work. Their documentation may also be useful
to help understand this module"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.boltons_utils import get_path

##################################################
# Import Miscellaneous Assets
##################################################
from abc import ABCMeta, abstractmethod
from functools import reduce
from sys import maxsize
from uuid import uuid4 as uuid

##################################################
# Import Learning Assets
##################################################
from sklearn.utils import check_random_state
from skopt.space import space as skopt_space


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RejectedOptional(metaclass=Singleton):
    """Singleton class to symbolize the rejection of an `optional` `Categorical` value

    This is used as a sentinel, when the value in `Categorical.categories` is not used, to be
    inserted into a :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`. If
    :attr:`hyperparameter_hunter.feature_engineering.FeatureEngineer.steps` contains an instance
    of `RejectedOptional`, it is removed from `steps`"""

    def __str__(self):
        return "<NONE>"

    def __format__(self, format_spec):
        return str(self).__format__(format_spec)


##################################################
# Dimensions
##################################################
# noinspection PyAbstractClass
class Dimension(skopt_space.Dimension, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """Base class for hyperparameter search space dimensions

        Attributes
        ----------
        id: String
            A stringified UUID used to link space dimensions to their locations in a model's overall
            hyperparameter structure"""
        self.id = str(uuid())
        super().__init__(**kwargs)

    # noinspection PyMethodOverriding
    @skopt_space.Dimension.name.setter
    def name(self, value):
        """Set :attr:`_name` to `value`

        Parameters
        ----------
        value: String, tuple, or None
            The new value of :attr:`_name`

        Raises
        ------
        ValueError
            If `value` is not one of: string, tuple, or None"""
        if isinstance(value, (str, tuple)) or value is None:
            # noinspection PyAttributeOutsideInit
            self._name = value
        else:
            raise ValueError("Dimension's name must be one of: string, tuple, or None.")

    @abstractmethod
    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Dimension`, or their defaults"""


class Real(Dimension, skopt_space.Real):
    def __init__(self, low, high, prior="uniform", transform="identity", name=None):
        """Search space dimension that can assume any real value in a given range

        Parameters
        ----------
        low: Float
            Lower bound (inclusive)
        high: Float
            Upper bound (inclusive)
        prior: String in ['uniform', 'log-uniform'], default='uniform'
            Distribution to use when sampling random points for this dimension. If 'uniform', points
            are sampled uniformly between the lower and upper bounds. If 'log-uniform', points are
            sampled uniformly between `log10(lower)` and `log10(upper)`
        transform: String in ['identity', 'normalize'], default='identity'
            Transformation to apply to the original space. If 'identity', the transformed space is
            the same as the original space. If 'normalize', the transformed space is scaled
            between 0 and 1
        name: String, tuple, or None, default=None
            A name associated with the dimension"""
        super().__init__(low=low, high=high, prior=prior, transform=transform, name=name)

    def __contains__(self, item):
        try:
            return super().__contains__(item)
        except TypeError:
            return False

    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Real`, or their defaults"""
        return dict(
            low=self.low,
            high=self.high,
            prior=self.prior,
            transform=self.transform_,
            name=self.name,
        )


class Integer(Dimension, skopt_space.Integer):
    def __init__(self, low, high, transform=None, name=None):
        """Search space dimension that can assume any integer value in a given range

        Parameters
        ----------
        low: Float
            Lower bound (inclusive)
        high: Float
            Upper bound (inclusive)
        transform: String in ['identity', 'normalize'], default='identity'
            Transformation to apply to the original space. If 'identity', the transformed space is
            the same as the original space. If 'normalize', the transformed space is scaled
            between 0 and 1
        name: String, tuple, or None, default=None
            A name associated with the dimension"""
        super().__init__(low=low, high=high, transform=transform, name=name)

    def __contains__(self, item):
        try:
            return super().__contains__(item)
        except TypeError:
            return False

    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Integer`, or their defaults"""
        return dict(low=self.low, high=self.high, transform=self.transform_, name=self.name)


class Categorical(Dimension, skopt_space.Categorical):
    def __init__(self, categories, prior=None, transform="onehot", optional=False, name=None):
        """Search space dimension that can assume any categorical value in a given list

        Parameters
        ----------
        categories: List
            Sequence of possible categories of shape (n_categories,)
        prior: List, or None, default=None
            If list, prior probabilities for each category of shape (categories,). By default all
            categories are equally likely
        transform: String in ['onehot', 'identity'], default='onehot'
            Transformation to apply to the original space. If 'identity', the transformed space is
            the same as the original space. If 'onehot', the transformed space is a one-hot encoded
            representation of the original space
        optional: Boolean, default=False
            Intended for use by :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`
            when optimizing an :class:`~hyperparameter_hunter.feature_engineering.EngineerStep`.
            Specifically, this enables searching through a space in which an `EngineerStep` either
            may or may not be used. This is contrary to `Categorical`'s usual function of creating
            a space comprising multiple `categories`. When `optional` = True, the space created will
            represent any of the values in `categories` either being included in the entire
            `FeatureEngineer` process, or being skipped entirely. Internally, a value excluded by
            `optional` is represented by a sentinel value that signals it should be removed from the
            containing list, so `optional` will not work for choosing between a single value and
            None, for example
        name: String, tuple, or None, default=None
            A name associated with the dimension"""
        if optional and RejectedOptional() not in categories:
            categories.append(RejectedOptional())
        self.optional = optional
        # TODO: Test using `optional` with `prior` and `transform`

        super().__init__(categories=categories, prior=prior, transform=transform, name=name)

    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Categorical`, or their defaults"""
        return dict(
            categories=self.categories,
            prior=self.prior,
            transform=self.transform_,
            optional=self.optional,
            name=self.name,
        )


##################################################
# Space
##################################################
class Space(skopt_space.Space):
    def __init__(self, dimensions, random_state=None):
        """Hyperparameter search space

        Parameters
        ----------
        dimensions: List
            List of search space dimensions. Each search dimension can be defined as any of the
            following: 1) a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer` dimensions).
            2) A `(lower_bound, upper_bound, "prior")` tuple (for `Real` dimensions).
            3) A list of categories (for `Categorical` dimensions).
            4) An instance of a `Dimension` object (`Real`, `Integer`, or `Categorical`)
        random_state: None
            ... Experimental..."""
        # self.space_random_state = check_random_state(None)  # FLAG: THIS BREAKS AND REPEATS RESULTS OF `rvs`
        self.space_random_state = check_random_state(32)  # FLAG: THIS WORKS
        super().__init__(dimensions=dimensions)

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples from the search space. The samples are in the original space. They
        need to be transformed before being passed to a model or minimizer by :meth:`transform`

        Parameters
        ----------
        n_samples: Int, default=1
            Number of samples to be drawn from the space

        random_state: Int, RandomState instance, or None, default=None
            Set random state to something other than None for reproducible results

        Returns
        -------
        List of lists
           Points sampled from the space. Of shape (n_points, n_dims)"""
        return super().rvs(n_samples=n_samples, random_state=self.space_random_state)

    def __len__(self):
        """Determine the number of possible search points in :attr:`dimensions`

        Returns
        -------
        search_space_size: Integer, or `sys.maxsize`
            The number of different hyperparameter search points. If the hyperparameter search space
            is infinitely large, `sys.maxsize` is returned to represent `np.inf`, which cannot
            itself be returned because `__len__` is required to produce an int >= 0"""
        if any(isinstance(_, Real) for _ in self.dimensions):
            search_space_size = maxsize
        else:
            search_space_size = reduce(
                lambda x, y: x * y,
                [
                    (_.high - _.low + 1) if isinstance(_, Integer) else len(_.bounds)
                    for _ in self.dimensions
                ],
                1,
            )

        return search_space_size

    def names(self, use_location=True):
        """Retrieve the names, or locations of all dimensions in the hyperparameter search space

        Parameters
        ----------
        use_location: Boolean, default=True
            If True and a dimension has a non-null attribute called 'location', its value will be
            used instead of 'name'

        Returns
        -------
        names: List
            A list of strings or tuples, in which each value is the name or location of the
            dimension at that index"""
        names = []
        for dimension in self.dimensions:
            if use_location and hasattr(dimension, "location") and dimension.location:
                names.append(dimension.location)
            else:
                names.append(dimension.name)
        return names


##################################################
# Space Utilities
##################################################
def normalize_dimensions(dimensions):
    """Create a `Space` where all dimensions are normalized to unit range

    Parameters
    ----------
    dimensions: List
        List of search space dimensions. Each search dimension can be defined as any of the
        following: 1) a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer` dimensions).
        2) A `(lower_bound, upper_bound, "prior")` tuple (for `Real` dimensions).
        3) A list of categories (for `Categorical` dimensions).
        4) An instance of a `Dimension` object (`Real`, `Integer`, or `Categorical`)

    Returns
    -------
    :class:`hyperparameter_hunter.space.Space` instance
        Hyperparameter space class instance, in which dimensions have been normalized to unit range

    Raises
    ------
    RuntimeError
        If a processed element of `dimensions` is not one of: `Real`, `Integer`, `Categorical`

    Notes
    -----
    The upper and lower bounds are inclusive for `Integer` dimensions. Based on
    :func:`skopt.utils.normalize_dimensions`"""
    space = Space(dimensions)
    transformed_dimensions = []

    if space.is_categorical:
        for dim in space:
            transformed_dimensions.append(
                Categorical(dim.categories, dim.prior, transform="identity", name=dim.name)
            )
    else:
        for dim in space.dimensions:
            if isinstance(dim, Categorical):
                transformed_dimensions.append(dim)
            elif isinstance(dim, Real):
                transformed_dimensions.append(
                    Real(dim.low, dim.high, dim.prior, transform="normalize", name=dim.name)
                )
            elif isinstance(dim, Integer):
                transformed_dimensions.append(
                    Integer(dim.low, dim.high, transform="normalize", name=dim.name)
                )
            else:
                raise RuntimeError(f"Unknown dimension type: {type(dim)}")
            #################### Replace Lost Attributes ####################
            if hasattr(dim, "location"):
                transformed_dimensions[-1].location = dim.location

    return Space(transformed_dimensions)


def dimension_subset(hyperparameters, dimensions):
    """Return only the values of `hyperparameters` specified by `dimensions`, in the same order as
    `dimensions`

    Parameters
    ----------
    hyperparameters: Dict
        Dict of hyperparameters containing at least the following keys: ['model_init_params',
        'model_extra_params', 'feature_engineer', 'feature_selector']
    dimensions: List of: (strings, or tuples)
        Locations and order of the values to return from `hyperparameters`. If a value is a string,
        it is assumed to belong to `model_init_params`, and its path will be adjusted accordingly

    Returns
    -------
    List of hyperparameter values"""
    dimensions = [("model_init_params", _) if isinstance(_, str) else _ for _ in dimensions]
    values = [get_path(hyperparameters, _, default=RejectedOptional()) for _ in dimensions]
    return values


if __name__ == "__main__":
    pass
