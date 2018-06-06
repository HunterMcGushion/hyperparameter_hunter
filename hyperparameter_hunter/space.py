##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.boltons_utils import get_path

##################################################
# Import Miscellaneous Assets
##################################################
from functools import reduce
import numpy as np

##################################################
# Import Learning Assets
##################################################
from sklearn.utils import check_random_state
from skopt.space import space as skopt_space


class Real(skopt_space.Real):
    def __init__(self, low, high, prior='uniform', transform=None, name=None):
        if name is None:
            raise ValueError('Please provide a `name` for the hyperparameter dimensions being set: {}(low={}, high={})'.format(
                self.__class__.__name__, low, high
            ))
        super().__init__(low=low, high=high, prior=prior, transform=transform, name=name)


class Integer(skopt_space.Integer):
    def __init__(self, low, high, transform=None, name=None):
        if name is None:
            raise ValueError('Please provide a `name` for the hyperparameter dimensions being set: {}(low={}, high={})'.format(
                self.__class__.__name__, low, high
            ))
        super().__init__(low=low, high=high, transform=transform, name=name)


class Categorical(skopt_space.Categorical):
    def __init__(self, categories, prior=None, transform=None, name=None):
        if name is None:
            raise ValueError('Please provide a `name` for the hyperparameter dimensions being set: {}(categories={})'.format(
                self.__class__.__name__, categories
            ))
        super().__init__(categories=categories, prior=prior, transform=transform, name=name)


class Space(skopt_space.Space):
    def __init__(self, dimensions, random_state=None):
        # self.space_random_state = check_random_state(None)  # FLAG: THIS BREAKS AND REPEATS RESULTS OF `rvs`
        self.space_random_state = check_random_state(32)  # FLAG: THIS WORKS
        super().__init__(dimensions=dimensions)

    def rvs(self, n_samples=1, random_state=None):
        return super().rvs(n_samples=n_samples, random_state=self.space_random_state)

    def __len__(self):
        """Determine the number of possible search points in :attr:`dimensions`

        Returns
        -------
        search_space_size: Integer, or `numpy.inf`
            The number of different hyperparameter search points"""
        if any(isinstance(_, Real) for _ in self.dimensions):
            search_space_size = np.inf
        else:
            search_space_size = reduce(
                lambda x, y: x * y,
                [1] + [(_.high - _.low + 1) if isinstance(_, Integer) else len(_.bounds) for _ in self.dimensions]
            )

        return search_space_size


def normalize_dimensions(dimensions):
    """Create a `Space` where all dimensions are normalized to unit range. This is a modified version of
    :func:`skopt.utils.normalize_dimensions`

    Parameters
    ----------
    dimensions: List
        List of search space dimensions. Each search dimension can be defined as any of the following: 1) a
        `(lower_bound, upper_bound)` tuple (for `Real` or `Integer` dimensions). 2) a `(lower_bound, upper_bound, "prior")` tuple
        (for `Real` dimensions). 3) a list of categories (for `Categorical` dimensions). 4) an instance of a `Dimension` object
        (`Real`, `Integer` or `Categorical`)

    Notes
    -----
    The upper and lower bounds are inclusive for `Integer` dimensions."""
    space = Space(dimensions)
    transformed_dimensions = []

    if space.is_categorical:
        for dimension in space:
            transformed_dimensions.append(Categorical(
                dimension.categories, dimension.prior, transform='identity', name=dimension.name
            ))
    else:
        for dimension in space.dimensions:
            if isinstance(dimension, Categorical):
                transformed_dimensions.append(dimension)
            elif isinstance(dimension, Real):
                transformed_dimensions.append(Real(
                    dimension.low, dimension.high, dimension.prior, transform='normalize', name=dimension.name
                ))
            elif isinstance(dimension, Integer):
                transformed_dimensions.append(Integer(
                    dimension.low, dimension.high, transform='normalize', name=dimension.name
                ))
            else:
                raise RuntimeError(F'Unknown dimension type: {type(dimension)}')

    return Space(transformed_dimensions)


def dimension_subset(hyperparameters, dimensions):
    """Return only the values of `hyperparameters` specified by `dimensions`, in the same order as `dimensions`

    Parameters
    ----------
    hyperparameters: Dict
        A dictionary of hyperparameters containing at least the following keys: ['model_init_params', 'model_extra_params',
        'preprocessing_pipeline', 'preprocessing_params', 'feature_selector']
    dimensions: List of: (strings, or tuples)
        The locations and order of the values to return from `hyperparameters`. If a value is a string, it is assumed to belong
        to `model_init_params`, and its path will be adjusted accordingly

    Returns
    -------
    List of hyperparameter values"""
    dimensions = [('model_init_params', _) if isinstance(_, str) else _ for _ in dimensions]

    if not all(isinstance(_, tuple) for _ in dimensions):
        raise TypeError(F'All dimensions should be strings or tuples. Received: {dimensions}')

    values = [get_path(hyperparameters, _) for _ in dimensions]
    return values


if __name__ == '__main__':
    pass
