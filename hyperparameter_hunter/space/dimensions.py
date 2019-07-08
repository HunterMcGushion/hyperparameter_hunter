"""Defines Dimension classes used for defining hyperparameter search spaces. Rather than
:class:`hyperparameter_hunter.space.space_core.Space`, the subclasses of
:class:`hyperparameter_hunter.space.dimensions.Dimension` are the only tools necessary for a user
to define a hyperparameter search space, when used as intended, in conjunction with a concrete
descendant of :class:`hyperparameter_hunter.optimization.protocol_core.BaseOptPro`.

Related
-------
:mod:`hyperparameter_hunter.space.space_core`
    Defines :class:`hyperparameter_hunter.space.space_core.Space`, which is used by
    :class:`hyperparameter_hunter.optimization.protocol_core.SKOptPro` to combine search Dimensions
    into a Space to be sampled and searched

Notes
-----
Many of the tools defined herein (although substantially modified) are based on those provided by
the excellent [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) library. See
:mod:`hyperparameter_hunter.optimization.backends.skopt` for a copy of SKOpt's license"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.general_utils import short_repr

##################################################
# Import Miscellaneous Assets
##################################################
from abc import abstractmethod, ABC
from numbers import Integral, Number
import numpy as np
from typing import Union
from uuid import uuid4 as uuid

##################################################
# Import Learning Assets
##################################################
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats.distributions import randint, rv_discrete, uniform
from sklearn.utils import check_random_state
from skopt.space.transformers import CategoricalEncoder, Normalize, Identity, Log10, Pipeline
from skopt.space.transformers import Transformer


##################################################
# Utilities
##################################################
class Singleton(type):
    _instances = {}

    def __new__(mcs, name, bases, namespace):
        namespace["__copy__"] = lambda self, *args: self
        namespace["__deepcopy__"] = lambda self, *args: self
        return super().__new__(mcs, name, bases, namespace)

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

    def __repr__(self):
        return "RejectedOptional()"

    def __format__(self, format_spec):
        return str(self).__format__(format_spec)


def _uniform_inclusive(loc=0.0, scale=1.0):
    # TODO: Add docstring
    # Like scipy.stats.distributions but inclusive of `high`
    # XXX scale + 1. might not actually be a float after scale if scale is very large
    return uniform(loc=loc, scale=np.nextafter(scale, scale + 1.0))


##################################################
# Abstract Dimensions
##################################################
class Dimension(ABC):
    prior = None  # Prevent plotting from breaking on `Integer`, which has no `prior`

    def __init__(self, **kwargs):
        """Abstract base class for hyperparameter search space dimensions

        Attributes
        ----------
        id: String
            A stringified UUID used to link space dimensions to their locations in a model's overall
            hyperparameter structure
        transform_: String
            Original value passed through the `transform` kwarg - Because :meth:`transform` exists
        distribution: rv_generic
            See documentation of :meth:`_make_distribution` or :meth:`distribution`
        transformer: Transformer
            See documentation of :meth:`_make_transformer` or :meth:`transformer`"""
        self.id = str(uuid())

        super().__init__(**kwargs)

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples. Samples are in the original (untransformed) space. They must be
        transformed before being passed to a model or minimizer via :meth:`transform`

        Parameters
        ----------
        n_samples: Int, default=1
            Number of samples to be drawn
        random_state: Int, RandomState, or None, default=None
            Set random state to something other than None for reproducible results

        Returns
        -------
        List
            Randomly drawn samples from the original space"""
        rng = check_random_state(random_state)
        samples = self.distribution.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(samples)

    def transform(self, data):
        """Transform samples from the original space into a warped space

        Parameters
        ----------
        data: List
            Samples to transform. Should be of shape (<# samples>, :attr:`size`)

        Returns
        -------
        List
            Samples transformed into a warped space. Will be of shape
            (<# samples>, :attr:`transformed_size`)

        Notes
        -----
        Expected to be used to project samples into a suitable space for numerical optimization"""
        return self.transformer.transform(data)

    def inverse_transform(self, data_t):
        """Inverse transform samples from the warped space back to the original space

        Parameters
        ----------
        data_t: List
            Samples to inverse transform. Should be of shape (<# samples>, :attr:`transformed_size`)

        Returns
        -------
        List
            Samples transformed back to original space. Will be shape (<# samples>, :attr:`size`)"""
        return self.transformer.inverse_transform(data_t)

    #################### Functional Properties ####################
    @property
    def distribution(self) -> rv_generic:
        """Class used for random sampling of points within the space

        Returns
        -------
        rv_generic
            :attr:`_distribution`

        Notes
        -----
        "setter" work for this property is performed by :meth:`_make_distribution`. The reason for
        this unconventional behavior is noted in `distribution.setter`"""
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        # noinspection PyAttributeOutsideInit
        self._distribution = value if value else self._make_distribution()
        # This is a weird way to do Python properties. However, abstract property setters are not
        #   enforced, and the alternative was to redefine the same getter in all the subclasses, as
        #   well. So the setters use an abstract helper method, which is dumb, but it does the trick

    @abstractmethod
    def _make_distribution(self) -> rv_generic:
        """Produce a value for :attr:`distribution` if one was not explicitly set

        Returns
        -------
        rv_generic
            Concrete descendant of `scipy.stats._distn_infrastructure.rv_generic` to use as
            :attr:`distribution`"""

    @property
    def transformer(self) -> Transformer:
        """Class used to transform and inverse-transform samples in the space

        Returns
        -------
        Transformer
            :attr:`_transformer`

        Notes
        -----
        "setter" work for this property is performed by :meth:`_make_transformer`. The reason for
        this unconventional behavior is noted in `distribution.setter`, which behaves similarly"""
        return self._transformer

    @transformer.setter
    def transformer(self, value):
        # noinspection PyAttributeOutsideInit
        self._transformer = value if value else self._make_transformer()
        # See comment in :meth:`distribution.setter` for why this setter is so un-Pythonic

    @abstractmethod
    def _make_transformer(self) -> Transformer:
        """Produce a value for :attr:`transformer` if one was not explicitly set

        Returns
        -------
        Transformer
            Concrete descendant of `Transformer` to use as :attr:`transformer`"""

    #################### Descriptive Properties ####################
    @property
    def size(self) -> int:
        """Size of the original (untransformed) space for the dimension"""
        return 1

    @property
    def transformed_size(self) -> int:
        """Size of the transformed space for the dimension"""
        return 1

    @property
    @abstractmethod
    def bounds(self):
        """Dimension bounds in the original space"""

    @property
    @abstractmethod
    def transformed_bounds(self):
        """Dimension bounds in the warped space"""

    @property
    def name(self) -> Union[str, tuple, None]:
        """A name associated with the dimension

        Returns
        -------
        String, tuple, or None
            :attr:`_name`"""
        return self._name

    @name.setter
    def name(self, value: Union[str, tuple, None]):
        if isinstance(value, (str, tuple)) or value is None:
            # noinspection PyAttributeOutsideInit
            self._name = value
        else:
            raise ValueError("Dimension's name must be one of: string, tuple, or None")

    #################### Comparison Methods ####################
    @abstractmethod
    def distance(self, a, b) -> Number:
        """Calculate distance between two points in the dimension's bounds"""

    @abstractmethod
    def __eq__(self, other):
        """Intended to be updated by subclasses, meaning subclasses need to not only override this
        method, but also include the result of `super().__eq__(other)` in their results"""
        return type(self) is type(other)

    @abstractmethod
    def __contains__(self, point) -> bool:
        """Determine whether a point fits within the dimension's untransformed bounds"""

    #################### Helper Methods ####################
    def _check_distance(self, a, b):
        """Check that two points fit within the dimension's bounds

        Raises
        ------
        RuntimeError
            If either `a` or `b` fall outside the dimension's original (untransformed) bounds"""
        if not (a in self and b in self):
            raise RuntimeError(
                f"Distance computation requires values within space. Received {a} and {b}"
            )

    @abstractmethod
    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Dimension`, or their defaults"""


class NumericalDimension(Dimension, ABC):
    def __init__(self, low, high, **kwargs):
        """Abstract base class for strictly numerical :class:`Dimension` subclasses

        Parameters
        ----------
        low: Number
            Lower bound (inclusive)
        high: Number
            Upper bound (inclusive)
        **kwargs: Dict
            Additional kwargs passed through from the concrete class to :class:`Dimension`"""
        super().__init__(**kwargs)

        if high <= low:
            raise ValueError(f"Lower bound ({low}) must be less than the upper bound ({high})")

        self.low = low
        self.high = high

    #################### Descriptive Properties ####################
    @property
    def bounds(self) -> tuple:
        """Dimension bounds in the original space

        Returns
        -------
        Tuple
            Tuple of (:attr:`low`, :attr:`high`). For :class:`Real` dimensions, the values will be
            floats. For :class:`Integer` dimensions, the values will be ints"""
        return (self.low, self.high)

    #################### Comparison Methods ####################
    def distance(self, a, b):
        """Calculate distance between two points in the dimension's bounds

        Returns
        -------
        Number
            Absolute value of the difference between `a` and `b`"""
        self._check_distance(a, b)
        return abs(a - b)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose([self.low], [other.low])
            and np.allclose([self.high], [other.high])
        )

    def __contains__(self, point):
        try:
            return self.low <= point <= self.high
        except TypeError:
            return False


##################################################
# Concrete Dimensions
##################################################
class Real(NumericalDimension):
    def __init__(self, low, high, prior="uniform", transform="identity", name=None):
        """Search space dimension that can assume any real value in a given range

        Parameters
        ----------
        low: Float
            Lower bound (inclusive)
        high: Float
            Upper bound (inclusive)
        prior: {"uniform", "log-uniform"}, default="uniform"
            Distribution to use when sampling random points for this dimension. If "uniform", points
            are sampled uniformly between the lower and upper bounds. If "log-uniform", points are
            sampled uniformly between `log10(lower)` and `log10(upper)`
        transform: {"identity", "normalize"}, default="identity"
            Transformation to apply to the original space. If "identity", the transformed space is
            the same as the original space. If "normalize", the transformed space is scaled
            between 0 and 1
        name: String, tuple, or None, default=None
            A name associated with the dimension

        Attributes
        ----------
        distribution: rv_generic
            See documentation of :meth:`_make_distribution` or :meth:`distribution`
        transform_: String
            Original value passed through the `transform` kwarg - Because :meth:`transform` exists
        transformer: Transformer
            See documentation of :meth:`_make_transformer` or :meth:`transformer`"""
        super().__init__(low, high)

        self.prior = prior
        self.transform_ = transform
        self.name = name

        if self.transform_ not in ["normalize", "identity"]:
            raise ValueError(
                "`transform` must be in ['normalize', 'identity']. Got {}".format(self.transform_)
            )

        # Define distribution and transformer spaces. `distribution` is for sampling in transformed
        #   space. `Dimension.rvs` calls inverse_transform on the points sampled using distribution
        self.distribution = None  # TODO: Add as kwarg?
        self.transformer = None

    def inverse_transform(self, data_t):
        """Inverse transform samples from the warped space back to the original space

        Parameters
        ----------
        data_t: List
            Samples to inverse transform. Should be of shape (<# samples>, :attr:`transformed_size`)

        Returns
        -------
        List
            Samples transformed back to original space. Will be shape (<# samples>, :attr:`size`)"""
        return np.clip(super().inverse_transform(data_t).astype(np.float), self.low, self.high)

    #################### Functional Properties ####################
    def _make_distribution(self) -> _uniform_inclusive:
        """Build a distribution to randomly sample points within the space

        Returns
        -------
        _uniform_inclusive
            Precise parameters based on :attr:`transform_` and :attr:`prior`"""
        if self.transform_ == "normalize":
            # Set upper bound to float after 1 to make the numbers inclusive of upper edge
            return _uniform_inclusive(0.0, 1.0)
        else:
            if self.prior == "uniform":
                return _uniform_inclusive(self.low, self.high - self.low)
            else:
                return _uniform_inclusive(
                    np.log10(self.low), np.log10(self.high) - np.log10(self.low)
                )

    def _make_transformer(self) -> Transformer:
        """Build a `Transformer` to transform and inverse-transform samples in the space

        Returns
        -------
        Transformer
            Precise architecture and parameters based on :attr:`transform_` and :attr:`prior`"""
        if self.transform_ == "normalize":
            if self.prior == "uniform":
                return Pipeline([Identity(), Normalize(self.low, self.high)])
            else:
                return Pipeline([Log10(), Normalize(np.log10(self.low), np.log10(self.high))])
        else:
            if self.prior == "uniform":
                return Identity()
            else:
                return Log10()

    #################### Descriptive Properties ####################
    @property
    def transformed_bounds(self):
        """Dimension bounds in the warped space

        Returns
        -------
        low: Float
            0.0 if :attr:`transform_`="normalize". If :attr:`transform_`="identity" and
            :attr:`prior`="uniform", then :attr:`low`. Else `log10(low)`
        high: Float
            1.0 if :attr:`transform_`="normalize". If :attr:`transform_`="identity" and
            :attr:`prior`="uniform", then :attr:`high`. Else `log10(high)`"""
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def __repr__(self):
        return "Real(low={}, high={}, prior='{}', transform='{}')".format(
            self.low, self.high, self.prior, self.transform_
        )

    #################### Comparison Methods ####################
    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.prior == other.prior
            and self.transform_ == other.transform_
        )

    #################### Helper Methods ####################
    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Real`, or their defaults"""
        return dict(
            low=self.low,
            high=self.high,
            prior=self.prior,
            transform=self.transform_,
            name=self.name,
        )


class Integer(NumericalDimension):
    def __init__(self, low, high, transform="identity", name=None):
        """Search space dimension that can assume any integer value in a given range

        Parameters
        ----------
        low: Int
            Lower bound (inclusive)
        high: Int
            Upper bound (inclusive)
        transform: {"identity", "normalize"}, default="identity"
            Transformation to apply to the original space. If "identity", the transformed space is
            the same as the original space. If "normalize", the transformed space is scaled
            between 0 and 1
        name: String, tuple, or None, default=None
            A name associated with the dimension

        Attributes
        ----------
        distribution: rv_generic
            See documentation of :meth:`_make_distribution` or :meth:`distribution`
        transform_: String
            Original value passed through the `transform` kwarg - Because :meth:`transform` exists
        transformer: Transformer
            See documentation of :meth:`_make_transformer` or :meth:`transformer`"""
        super().__init__(low, high)

        self.transform_ = transform
        self.name = name

        if transform not in ["normalize", "identity"]:
            raise ValueError(f"`transform` must be in ['normalize', 'identity']. Got {transform}")

        self.distribution = None  # TODO: Add as kwarg?
        self.transformer = None

    def inverse_transform(self, data_t):
        """Inverse transform samples from the warped space back to the original space

        Parameters
        ----------
        data_t: List
            Samples to inverse transform. Should be of shape (<# samples>, :attr:`transformed_size`)

        Returns
        -------
        List
            Samples transformed back to original space. Will be shape (<# samples>, :attr:`size`)"""
        # Concatenation of all transformed dimensions makes `data_t` of type float,
        #   hence the required cast back to int
        # TODO: This breaks if `Integer.rvs` called with `n_samples`=None - Raises TypeError
        #   when calling `astype` on result of `inverse_transform`, which is Python int, not NumPy
        return super().inverse_transform(data_t).astype(np.int)

    #################### Functional Properties ####################
    def _make_distribution(self) -> rv_generic:
        """Build a distribution to randomly sample points within the space

        Returns
        -------
        rv_generic
            `uniform` distribution between 0 and 1 if :attr:`transform_` == "normalize". Else, a
            `randint` distribution between :attr:`low` and (:attr:`high` + 1)"""
        if self.transform_ == "normalize":
            return uniform(0, 1)
        else:
            return randint(self.low, self.high + 1)

    def _make_transformer(self) -> Transformer:
        """Build a `Transformer` to transform and inverse-transform samples in the space

        Returns
        -------
        Transformer
            `Normalize` with bounds (:attr:`low`, :attr:`high`) if :attr:`transform_` == "onehot".
            Else, `Identity`"""
        if self.transform_ == "normalize":
            return Normalize(self.low, self.high, is_int=True)
        else:
            return Identity()

    #################### Descriptive Properties ####################
    @property
    def transformed_bounds(self):
        """Dimension bounds in the warped space

        Returns
        -------
        low: Int
            0 if :attr:`transform_`="normalize", else :attr:`low`
        high: Int
            1 if :attr:`transform_`="normalize", else :attr:`high`"""
        if self.transform_ == "normalize":
            return 0, 1
        else:
            return (self.low, self.high)

    def __repr__(self):
        return "Integer(low={}, high={})".format(self.low, self.high)

    #################### Helper Methods ####################
    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Integer`, or their defaults"""
        return dict(low=self.low, high=self.high, transform=self.transform_, name=self.name)


class Categorical(Dimension):
    def __init__(
        self, categories: list, prior: list = None, transform="onehot", optional=False, name=None
    ):
        """Search space dimension that can assume any categorical value in a given list

        Parameters
        ----------
        categories: List
            Sequence of possible categories of shape (n_categories,)
        prior: List, or None, default=None
            If list, prior probabilities for each category of shape (categories,). By default all
            categories are equally likely
        transform: {"onehot", "identity"}, default="onehot"
            Transformation to apply to the original space. If "identity", the transformed space is
            the same as the original space. If "onehot", the transformed space is a one-hot encoded
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
            A name associated with the dimension

        Attributes
        ----------
        categories: Tuple
            Original value passed through the `categories` kwarg, cast to a tuple. If `optional` is
            True, then an instance of :class:`RejectedOptional` will be appended to `categories`
        distribution: rv_generic
            See documentation of :meth:`_make_distribution` or :meth:`distribution`
        optional: Boolean
            Original value passed through the `optional` kwarg
        prior: List, or None
            Original value passed through the `prior` kwarg
        prior_actual: List
            Calculated prior value, initially equivalent to :attr:`prior`, but then set to a default
            array if None
        transform_: String
            Original value passed through the `transform` kwarg - Because :meth:`transform` exists
        transformer: Transformer
            See documentation of :meth:`_make_transformer` or :meth:`transformer`"""
        super().__init__()

        if optional and RejectedOptional() not in categories:
            categories.append(RejectedOptional())

        self.categories = tuple(categories)
        self.prior = prior
        self.prior_actual = prior
        self.transform_ = transform
        self.optional = optional
        self.name = name
        # TODO: Test using `optional` with `prior` and `transform`

        if transform not in ["identity", "onehot"]:
            raise ValueError("transform must be 'identity' or 'onehot'. Got {}".format(transform))

        if self.prior_actual is None:
            self.prior_actual = np.tile(1.0 / len(self.categories), len(self.categories))

        self.distribution = None
        self.transformer = None

    def rvs(self, n_samples=None, random_state=None):  # TODO: Make default `n_samples`=1
        """Draw random samples. Samples are in the original (untransformed) space. They must be
        transformed before being passed to a model or minimizer via :meth:`transform`

        Parameters
        ----------
        n_samples: Int (optional)
            Number of samples to be drawn. If not given, a single sample will be returned
        random_state: Int, RandomState, or None, default=None
            Set random state to something other than None for reproducible results

        Returns
        -------
        List
            Randomly drawn samples from the original space"""
        rng = check_random_state(random_state)
        choices = self.distribution.rvs(size=n_samples, random_state=rng)

        # Index `categories`, instead of using `transformer.inverse_transform` because
        #   `distribution` is of all indices, not actual `categories`
        if isinstance(choices, Integral):
            return self.categories[choices]
        else:
            return [self.categories[c] for c in choices]

    #################### Functional Properties ####################
    def _make_distribution(self) -> rv_generic:
        """Build a distribution to randomly sample points within the space

        Returns
        -------
        rv_discrete
            Discrete random variate distribution over the indices of :attr:`categories`"""
        # XXX check that sum(prior) == 1
        # Values of distribution are just indices of `categories` - Basically LabelEncoded
        return rv_discrete(values=(range(len(self.categories)), self.prior_actual))

    def _make_transformer(self) -> Transformer:
        """Build a `Transformer` to transform and inverse-transform samples in the space

        Returns
        -------
        Transformer
            `CategoricalEncoder` fit to :attr:`categories` if :attr:`transform_` == "onehot". Else,
            `Identity`"""
        if self.transform_ == "onehot":
            t = CategoricalEncoder()
            t.fit(self.categories)
            return t
        else:
            return Identity()

    #################### Descriptive Properties ####################
    @property
    def transformed_size(self):
        """Size of the transformed space for the dimension

        Returns
        -------
        Int
            * 1 if :attr:`transform_` == "identity"
            * 1 if :attr:`transform_` == "onehot" and length of :attr:`categories` is 1 or 2
            * Length of :attr:`categories` in all other cases"""
        if self.transform_ == "onehot" and len(self.categories) > 2:
            # When len(categories) == 2, CategoricalEncoder outputs a single value
            return len(self.categories)
        return 1

    @property
    def bounds(self):
        """Dimension bounds in the original space

        Returns
        -------
        Tuple
            :attr:`categories`"""
        return self.categories

    @property
    def transformed_bounds(self):
        """Dimension bounds in the warped space

        Returns
        -------
        Tuple, or list
            If :attr:`transformed_size` == 1, then a tuple of (0.0, 1.0). Otherwise, returns a list
            containing :attr:`transformed_size`-many tuples of (0.0, 1.0)

        Notes
        -----
        :attr:`transformed_size` == 1 when the length of :attr:`categories` == 2, so if there are
        two items in `categories`, (0.0, 1.0) is returned. If there are three items in `categories`,
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] is returned, and so on.

        Because `transformed_bounds` uses :attr:`transformed_size`, it is affected by
        :attr:`transform_`. Specifically, the returns described above are for :attr:`transform_` ==
        "onehot" (default).

        Examples
        --------
        >>> Categorical(["a", "b"]).transformed_bounds
        (0.0, 1.0)
        >>> Categorical(["a", "b", "c"]).transformed_bounds
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        >>> Categorical(["a", "b", "c", "d"]).transformed_bounds
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        """
        # TODO: Below behavior seems odd. If categories=[1, 3, 5, 7], then `transformed_size`=1
        #   makes sense, but `transformed_bounds`=(0.0, 1.0) is weird
        # TODO: Return `categories` if `transform_` == "identity"?
        # FLAG: Below all return (0.0, 1.0), which seems strange
        #   Categorical([1], transform="identity").transformed_bounds
        #   Categorical([1, 3], transform="identity").transformed_bounds
        #   Categorical([1, 3, 5], transform="identity").transformed_bounds
        #   Categorical([1, 3, 5, 7], transform="identity").transformed_bounds
        if self.transformed_size == 1:
            return (0.0, 1.0)
        else:
            return [(0.0, 1.0) for _ in range(self.transformed_size)]

    def __repr__(self):
        return "Categorical(categories={})".format(short_repr(self.categories))

    #################### Comparison Methods ####################
    def distance(self, a, b) -> int:
        """Calculate distance between two points in the dimension's bounds

        Parameters
        ----------
        a
            First category
        b
            Second category

        Returns
        -------
        Int
            0 if `a` == `b`. Else 1 (because categories have no order)"""
        self._check_distance(a, b)
        return 1 if a != b else 0

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.categories == other.categories
            and np.allclose(self.prior_actual, other.prior_actual)
        )

    def __contains__(self, point):
        return point in self.categories

    #################### Helper Methods ####################
    def get_params(self) -> dict:
        """Get dict of parameters used to initialize the `Categorical`, or their defaults"""
        return dict(
            categories=self.categories,
            prior=self.prior,
            transform=self.transform_,
            optional=self.optional,
            name=self.name,
        )
