"""Defines utilities intended for internal use only, most notably
:class:`hyperparameter_hunter.space.space_core.Space`. These tools are used behind the scenes by
:class:`hyperparameter_hunter.optimization.protocol_core.BaseOptPro` to combine instances of
dimensions defined in :mod:`hyperparameter_hunter.space.dimensions` into a usable hyperparameter
search Space

Related
-------
:mod:`hyperparameter_hunter.space.dimensions`
    Defines concrete descendants of :class:`hyperparameter_hunter.space.dimensions.Dimension`, which
    are intended for direct use. :class:`hyperparameter_hunter.space.space_core.Space` is used
    to combine these Dimension instances

Notes
-----
Many of the tools defined herein (although substantially modified) are based on those provided by
the excellent [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) library. See
:mod:`hyperparameter_hunter.optimization.backends.skopt` for a copy of SKOpt's license"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.space.dimensions import Dimension, Real, Integer, Categorical
from hyperparameter_hunter.utils.general_utils import short_repr

##################################################
# Import Miscellaneous Assets
##################################################
from functools import reduce
import numbers
import numpy as np
from sys import maxsize

##################################################
# Import Learning Assets
##################################################
from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version

NONE = object()


##################################################
# Utilities
##################################################
def check_dimension(dimension, transform=None):
    """Turn a provided dimension description into a dimension object. Checks that the provided
    dimension falls into one of the supported types, listed below in the description of `dimension`

    Parameters
    ----------
    dimension: Tuple, list, or Dimension
        Search space `Dimension`. May be any of the following:
        * `(lower_bound, upper_bound)` tuple (`Real` or `Integer`)
        * `(lower_bound, upper_bound, prior)` tuple (`Real`)
        * List of categories (`Categorical`)
        * `Dimension` instance (`Real`, `Integer` or `Categorical`)
    transform: {"identity", "normalize", "onehot"} (optional)
        * `Categorical` dimensions support "onehot" or "identity". See `Categorical` documentation
          for more information
        * `Real` and `Integer` dimensions support "identity" or "normalize". See `Real` or `Integer`
          documentation for more information

    Returns
    -------
    dimension: Dimension
        Dimension instance created from the provided `dimension` description. If `dimension` is
        already an instance of `Dimension`, it is returned unchanged"""
    if isinstance(dimension, Dimension):
        return dimension
    if not isinstance(dimension, (list, tuple, np.ndarray)):
        raise ValueError("Dimension has to be a list or tuple")

    # `Dimension` subclasses define actual `transform` defaults - Only pass `transform` if not None
    kwargs = dict(transform=transform) if transform else {}

    if len(dimension) == 1:
        return Categorical(dimension, **kwargs)

    if len(dimension) == 2:
        if any([isinstance(d, (str, bool)) or isinstance(d, np.bool_) for d in dimension]):
            return Categorical(dimension, **kwargs)
        elif all([isinstance(dim, numbers.Integral) for dim in dimension]):
            return Integer(*dimension, **kwargs)
        elif any([isinstance(dim, numbers.Real) for dim in dimension]):
            return Real(*dimension, **kwargs)

    if len(dimension) == 3:
        # TODO: Below `any` should prolly be `all`
        if any([isinstance(dim, (float, int)) for dim in dimension[:2]]) and dimension[2] in [
            "uniform",
            "log-uniform",
        ]:
            return Real(*dimension, **kwargs)
        else:
            return Categorical(dimension, **kwargs)

    if len(dimension) > 3:
        return Categorical(dimension, **kwargs)

    raise ValueError(f"Invalid `dimension` {dimension}. See documentation for supported types")


##################################################
# Space
##################################################
class Space:
    def __init__(self, dimensions):
        """Initialize a search space from given specifications

        Parameters
        ----------
        dimensions: List
            List of search space `Dimension` instances or representatives. Each search dimension
            may be any of the following:
            * `(lower_bound, upper_bound)` tuple (`Real` or `Integer`)
            * `(lower_bound, upper_bound, prior)` tuple (`Real`)
            * List of categories (`Categorical`)
            * `Dimension` instance (`Real`, `Integer` or `Categorical`)

        Notes
        -----
        The upper and lower bounds are inclusive for `Integer` dimensions"""
        self.dimensions = [check_dimension(dim) for dim in dimensions]

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.dimensions, other.dimensions)])

    def __repr__(self):
        dims = short_repr(self.dimensions, affix_size=15)
        return "Space([{}])".format(",\n       ".join(map(str, dims)))

    def __iter__(self):
        return iter(self.dimensions)

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

    def __contains__(self, point):
        """Determine whether `point` fits within the bounds of the space

        Parameters
        ----------
        point: List
            Search space point, expected to be of the same length as :attr:`dimensions`

        Returns
        -------
        Boolean
            True if `point` fits within :attr:`dimensions`. Else, False"""
        for component, dim in zip(point, self.dimensions):
            if component not in dim:
                return False
        return True

    ##################################################
    # Core Methods
    ##################################################
    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples. Samples are in the original (untransformed) space. They must be
        transformed before being passed to a model or minimizer via :meth:`transform`

        Parameters
        ----------
        n_samples: Int, default=1
            Number of samples to be drawn from the space
        random_state: Int, RandomState, or None, default=None
            Set random state to something other than None for reproducible results

        Returns
        -------
        List
            Randomly drawn samples from the original space. Will be a list of lists, of shape
            (`n_samples`, :attr:`n_dims`)"""
        rng = check_random_state(random_state)

        #################### Draw ####################
        columns = []

        for dim in self.dimensions:
            if sp_version < (0, 16):
                columns.append(dim.rvs(n_samples=n_samples))
            else:
                columns.append(dim.rvs(n_samples=n_samples, random_state=rng))

        #################### Transpose ####################
        rows = []
        # TODO: Use `np.transpose`? Might that screw up the dimension types (mostly `Categorical`)
        for i in range(n_samples):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    def transform(self, data):
        """Transform samples from the original space into a warped space

        Parameters
        ----------
        data: List
            Samples to transform. Should be of shape (<# samples>, :attr:`n_dims`)

        Returns
        -------
        data_t: List
            Samples transformed into a warped space. Will be of shape
            (<# samples>, :attr:`transformed_n_dims`)

        Notes
        -----
        Expected to be used to project samples into a suitable space for numerical optimization"""
        #################### Pack by Dimension ####################
        columns = [[] for _ in self.dimensions]

        for i in range(len(data)):
            for j in range(self.n_dims):
                columns[j].append(data[i][j])

        #################### Transform ####################
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        #################### Repack as Array ####################
        data_t = np.hstack([np.asarray(c).reshape((len(data), -1)) for c in columns])

        return data_t

    def inverse_transform(self, data_t):
        """Inverse transform samples from the warped space back to the original space

        Parameters
        ----------
        data_t: List
            Samples to inverse transform. Should be of shape
            (<# samples>, :attr:`transformed_n_dims`)

        Returns
        -------
        List
            Samples transformed back to the original space. Will be of shape
            (<# samples>, :attr:`n_dims`)"""
        #################### Inverse Transform ####################
        columns = []
        start = 0

        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(data_t[:, start]))
            else:
                columns.append(dim.inverse_transform(data_t[:, start : start + offset]))

            start += offset

        #################### Transpose ####################
        rows = []
        # TODO: Use `np.transpose`? Might that screw up the dimension types (mostly `Categorical`)
        for i in range(len(data_t)):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    ##################################################
    # Descriptive Properties
    ##################################################
    @property
    def n_dims(self) -> int:
        """Dimensionality of the original space

        Returns
        -------
        Int
            Length of :attr:`dimensions`"""
        return len(self.dimensions)

    @property
    def transformed_n_dims(self) -> int:
        """Dimensionality of the warped space

        Returns
        -------
        Int
            Sum of the `transformed_size` of all dimensions in :attr:`dimensions`"""
        return sum([dim.transformed_size for dim in self.dimensions])

    @property
    def bounds(self):
        """The dimension bounds, in the original space

        Returns
        -------
        List
            Collection of the `bounds` of each dimension in :attr:`dimensions`"""
        b = []

        for dim in self.dimensions:
            if dim.size == 1:
                b.append(dim.bounds)
            else:
                b.extend(dim.bounds)

        return b

    @property
    def transformed_bounds(self):
        """The dimension bounds, in the warped space

        Returns
        -------
        List
            Collection of the `transformed_bounds` of each dimension in :attr:`dimensions`"""
        b = []

        for dim in self.dimensions:
            if dim.transformed_size == 1:
                b.append(dim.transformed_bounds)
            else:
                b.extend(dim.transformed_bounds)

        return b

    @property
    def is_real(self):
        """Whether :attr:`dimensions` contains exclusively `Real` dimensions

        Returns
        -------
        Boolean
            True if all dimensions in :attr:`dimensions` are `Real`. Else, False"""
        return all([isinstance(dim, Real) for dim in self.dimensions])

    @property
    def is_categorical(self) -> bool:
        """Whether :attr:`dimensions` contains exclusively `Categorical` dimensions

        Returns
        -------
        Boolean
            True if all dimensions in :attr:`dimensions` are `Categorical`. Else, False"""
        return all([isinstance(dim, Categorical) for dim in self.dimensions])

    ##################################################
    # Helper Methods
    ##################################################
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

    def get_by_name(self, name, use_location=True, default=NONE):
        """Retrieve a single dimension by its name

        Parameters
        ----------
        name: Tuple, or str
            Name of the dimension in :attr:`dimensions` to return
        use_location: Boolean, default=True
            If True and a dimension has a non-null attribute called "location", its value will be
            used instead of that dimension's "name"
        default: Any (optional)
            If given and `name` is not found, `default` will be returned. Otherwise, `KeyError` will
            be raised when `name` is not found

        Returns
        -------
        Dimension
            Dimension subclass in :attr:`dimensions`, whose "name" attribute is equal to `name`"""
        for dimension in self.dimensions:
            if use_location and getattr(dimension, "location", None) == name:
                return dimension
            elif dimension.name == name:
                return dimension

        if default != NONE:
            return default
        raise KeyError(f"{name} not found in dimensions")

    def distance(self, point_a, point_b):
        """Compute distance between two points in this space. Both `point_a` and `point_b` are
        expected to be of the same length as :attr:`dimensions`, with values corresponding to the
        `Dimension` bounds of :attr:`dimensions`

        Parameters
        ----------
        point_a: List
            First point
        point_b: List
            Second point

        Returns
        -------
        Number
            Distance between `point_a` and `point_b`"""
        distance = 0.0
        for a, b, dim in zip(point_a, point_b, self.dimensions):
            distance += dim.distance(a, b)

        return distance


def normalize_dimensions(dimensions):
    """Create a `Space` where all dimensions are instructed to be normalized to unit range. Note
    that this doesn't *really* return normalized `dimensions`. It just returns the given
    `dimensions`, with each one's `transform` set to the appropriate value, so that when each
    dimension's :meth:`transform` is called, the dimensions are actually normalized

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
    :class:`hyperparameter_hunter.space.Space`
        Hyperparameter space class instance, in which dimensions have been instructed to be
        normalized to unit range upon invocation of the `transform` method

    Raises
    ------
    RuntimeError
        If a processed element of `dimensions` is not one of: `Real`, `Integer`, `Categorical`

    Notes
    -----
    The upper and lower bounds are inclusive for `Integer` dimensions"""
    space = Space(dimensions)
    transformed_dimensions = []

    if space.is_categorical:
        for dim in space:
            # `skopt.utils.normalize_dimensions` makes comment on explicitly setting
            #   `transform="identity"`, so apparently there's a good reason for it...
            # Using original `transform` fixes all-`Categorical`/`BayesianOptPro` bug and proper
            #   saved experiment result matching, but optimizer could be secretly misbehaving...
            transformed_dimensions.append(
                Categorical(dim.categories, dim.prior, transform=dim.transform_, name=dim.name)
                # Categorical(dim.categories, dim.prior, transform="identity", name=dim.name)
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
