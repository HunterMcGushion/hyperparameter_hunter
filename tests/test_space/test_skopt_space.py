"""Tests the tools defined by :mod:`hyperparameter_hunter.space.dimensions` and
:mod:`hyperparameter_hunter.space.space_core`

Notes
-----
Many of the tests defined herein (although substantially modified) are based on those provided by
the excellent [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) library. See
:mod:`hyperparameter_hunter.optimization.backends.skopt` for a copy of SKOpt's license"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.optimization.backends.skopt.engine import Optimizer
from hyperparameter_hunter.space.dimensions import Real, Integer, Categorical
from hyperparameter_hunter.space.space_core import Space, check_dimension, normalize_dimensions

##################################################
# Import Miscellaneous Assets
##################################################
import numbers
import numpy as np
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.utils.testing import assert_array_almost_equal, assert_array_equal, assert_equal


def check_limits(value, low, high):
    """Check if `low` <= `value` <= `high`"""
    assert low <= value
    assert high >= value


##################################################
# Dimension Smoke Tests
##################################################
@pytest.mark.parametrize("dimension", [Real, Integer])
@pytest.mark.parametrize("bounds", [(1, 4), (1.0, 4.0)])
def test_numerical_dimension_equality(dimension, bounds: tuple):
    dim = dimension(*bounds)
    # Assert equality with identical `Dimension`
    assert dim == dimension(*bounds)
    # Assert inequality with `Dimension`s of differing bounds
    assert dim != dimension(bounds[0], bounds[1] + 1)
    assert dim != dimension(bounds[0] + 1, bounds[1])


@pytest.mark.parametrize("categories", [("a", "b", "c", "d"), (1.0, 2.0, 3.0, 4.0)])
def test_categorical_dimension_equality(categories):
    dim = Categorical(categories)
    # Assert equality with identical `Categorical`
    assert dim == Categorical(categories)
    # Assert inequality with `Categorical`, whose final value differs
    assert dim != Categorical(categories[:-1] + ("zzz",))


@pytest.mark.parametrize(
    ["dimension", "random_val"],
    [
        (Real(1.0, 4.0), 2.251066014107722),
        (Real(1, 4), 2.251066014107722),
        (Integer(1, 4), 2),
        (Integer(1.0, 4.0), 2),
        (Categorical(["a", "b", "c", "d"]), "b"),
        (Categorical([1.0, 2.0, 3.0, 4.0]), 2.0),
    ],
)
def test_dimension_rvs(dimension, random_val):
    """Assert random sample is expected"""
    assert dimension.rvs(random_state=1) == random_val


@pytest.mark.fast_test
@pytest.mark.parametrize(
    ["dim", "expected_repr"],
    [
        (Categorical([1, 2, 3, 4, 5]), "Categorical(categories=(1, 2, 3, 4, 5))"),
        (Categorical([1, 2, 3, 4, 5, 6, 7, 8]), "Categorical(categories=(1, 2, 3, ..., 6, 7, 8))"),
        (Real(0.4, 0.9), "Real(low=0.4, high=0.9, prior='uniform', transform='identity')"),
        (Real(4, 23), "Real(low=4, high=23, prior='uniform', transform='identity')"),
        (Integer(4, 23), "Integer(low=4, high=23)"),
    ],
)
def test_dimension_repr(dim, expected_repr):
    assert dim.__repr__() == expected_repr


@pytest.mark.fast_test
def test_real_log_sampling_in_bounds():
    # TODO: Refactor - Use PyTest
    dim = Real(low=1, high=32, prior="log-uniform", transform="normalize")

    # Round-trip a value that is within the bounds of the space

    # x = dim.inverse_transform(dim.transform(31.999999999999999))
    for n in (32.0, 31.999999999999999):
        round_tripped = dim.inverse_transform(dim.transform([n]))
        assert np.allclose([n], round_tripped)
        assert n in dim
        assert round_tripped in dim


@pytest.mark.fast_test
def test_real():
    # TODO: Refactor - Use PyTest
    a = Real(1, 25)
    for i in range(50):
        r = a.rvs(random_state=i)
        check_limits(r, 1, 25)
        assert r in a

    random_values = a.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    assert_array_equal(a.transform(random_values), random_values)
    assert_array_equal(a.inverse_transform(random_values), random_values)

    log_uniform = Real(10 ** -5, 10 ** 5, prior="log-uniform")
    assert log_uniform != Real(10 ** -5, 10 ** 5)
    for i in range(50):
        random_val = log_uniform.rvs(random_state=i)
        check_limits(random_val, 10 ** -5, 10 ** 5)
    random_values = log_uniform.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    transformed_vals = log_uniform.transform(random_values)
    assert_array_equal(transformed_vals, np.log10(random_values))
    assert_array_equal(log_uniform.inverse_transform(transformed_vals), random_values)


@pytest.mark.fast_test
def test_real_bounds():
    # TODO: Refactor - Use PyTest
    # Should give same answer as using check_limits() but this is easier to read
    a = Real(1.0, 2.1)
    assert 0.99 not in a
    assert 1.0 in a
    assert 2.09 in a
    assert 2.1 in a
    assert np.nextafter(2.1, 3.0) not in a


@pytest.mark.fast_test
def test_integer():
    # TODO: Refactor - Use PyTest
    a = Integer(1, 10)
    for i in range(50):
        r = a.rvs(random_state=i)
        assert 1 <= r
        assert 11 >= r
        assert r in a

    random_values = a.rvs(random_state=0, n_samples=10)
    assert_array_equal(random_values.shape, (10))
    assert_array_equal(a.transform(random_values), random_values)
    assert_array_equal(a.inverse_transform(random_values), random_values)


@pytest.mark.fast_test
def test_categorical_transform():
    # TODO: Refactor - Use PyTest
    categories = ["apple", "orange", "banana", None, True, False, 3]
    cat = Categorical(categories)

    apple = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    orange = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    banana = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    none = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    true = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    false = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    three = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    assert_equal(cat.transformed_size, 7)
    assert_equal(cat.transformed_size, cat.transform(["apple"]).size)
    assert_array_equal(cat.transform(categories), [apple, orange, banana, none, true, false, three])
    assert_array_equal(cat.transform(["apple", "orange"]), [apple, orange])
    assert_array_equal(cat.transform(["apple", "banana"]), [apple, banana])
    assert_array_equal(cat.inverse_transform([apple, orange]), ["apple", "orange"])
    assert_array_equal(cat.inverse_transform([apple, banana]), ["apple", "banana"])
    ent_inverse = cat.inverse_transform([apple, orange, banana, none, true, false, three])
    assert_array_equal(ent_inverse, categories)


@pytest.mark.fast_test
def test_categorical_transform_binary():
    # TODO: Refactor - Use PyTest
    categories = ["apple", "orange"]
    cat = Categorical(categories)

    apple = [0.0]
    orange = [1.0]

    assert_equal(cat.transformed_size, 1)
    assert_equal(cat.transformed_size, cat.transform(["apple"]).size)
    assert_array_equal(cat.transform(categories), [apple, orange])
    assert_array_equal(cat.transform(["apple", "orange"]), [apple, orange])
    assert_array_equal(cat.inverse_transform([apple, orange]), ["apple", "orange"])
    ent_inverse = cat.inverse_transform([apple, orange])
    assert_array_equal(ent_inverse, categories)


@pytest.mark.fast_test
def test_space_consistency():
    # TODO: Refactor - Use PyTest
    # Reals (uniform)

    s1 = Space([Real(0.0, 1.0)])
    s2 = Space([Real(0.0, 1.0)])
    s3 = Space([Real(0, 1)])
    s4 = Space([(0.0, 1.0)])
    s5 = Space([(0.0, 1.0, "uniform")])
    s6 = Space([(0, 1.0)])
    s7 = Space([(np.float64(0.0), 1.0)])
    s8 = Space([(0, np.float64(1.0))])
    a1 = s1.rvs(n_samples=10, random_state=0)
    a2 = s2.rvs(n_samples=10, random_state=0)
    a3 = s3.rvs(n_samples=10, random_state=0)
    a4 = s4.rvs(n_samples=10, random_state=0)
    a5 = s5.rvs(n_samples=10, random_state=0)
    assert_equal(s1, s2)
    assert_equal(s1, s3)
    assert_equal(s1, s4)
    assert_equal(s1, s5)
    assert_equal(s1, s6)
    assert_equal(s1, s7)
    assert_equal(s1, s8)
    assert_array_equal(a1, a2)
    assert_array_equal(a1, a3)
    assert_array_equal(a1, a4)
    assert_array_equal(a1, a5)

    # Reals (log-uniform)
    s1 = Space([Real(10 ** -3.0, 10 ** 3.0, prior="log-uniform")])
    s2 = Space([Real(10 ** -3.0, 10 ** 3.0, prior="log-uniform")])
    s3 = Space([Real(10 ** -3, 10 ** 3, prior="log-uniform")])
    s4 = Space([(10 ** -3.0, 10 ** 3.0, "log-uniform")])
    s5 = Space([(np.float64(10 ** -3.0), 10 ** 3.0, "log-uniform")])
    a1 = s1.rvs(n_samples=10, random_state=0)
    a2 = s2.rvs(n_samples=10, random_state=0)
    a3 = s3.rvs(n_samples=10, random_state=0)
    a4 = s4.rvs(n_samples=10, random_state=0)
    assert_equal(s1, s2)
    assert_equal(s1, s3)
    assert_equal(s1, s4)
    assert_equal(s1, s5)
    assert_array_equal(a1, a2)
    assert_array_equal(a1, a3)
    assert_array_equal(a1, a4)

    # Integers
    s1 = Space([Integer(1, 5)])
    s2 = Space([Integer(1.0, 5.0)])
    s3 = Space([(1, 5)])
    s4 = Space([(np.int64(1.0), 5)])
    s5 = Space([(1, np.int64(5.0))])
    a1 = s1.rvs(n_samples=10, random_state=0)
    a2 = s2.rvs(n_samples=10, random_state=0)
    a3 = s3.rvs(n_samples=10, random_state=0)
    assert_equal(s1, s2)
    assert_equal(s1, s3)
    assert_equal(s1, s4)
    assert_equal(s1, s5)
    assert_array_equal(a1, a2)
    assert_array_equal(a1, a3)

    # Categoricals
    s1 = Space([Categorical(["a", "b", "c"])])
    s2 = Space([Categorical(["a", "b", "c"])])
    s3 = Space([["a", "b", "c"]])
    a1 = s1.rvs(n_samples=10, random_state=0)
    a2 = s2.rvs(n_samples=10, random_state=0)
    a3 = s3.rvs(n_samples=10, random_state=0)
    assert_equal(s1, s2)
    assert_array_equal(a1, a2)
    assert_equal(s1, s3)
    assert_array_equal(a1, a3)

    s1 = Space([(True, False)])
    s2 = Space([Categorical([True, False])])
    s3 = Space([np.array([True, False])])
    assert s1 == s2 == s3


@pytest.mark.fast_test
def test_space_api():
    # TODO: Refactor - Use PyTest - Break this up into multiple tests
    space = Space([(0.0, 1.0), (-5, 5), ("a", "b", "c"), (1.0, 5.0, "log-uniform"), ("e", "f")])

    cat_space = Space([(1, "r"), (1.0, "r")])
    assert isinstance(cat_space.dimensions[0], Categorical)
    assert isinstance(cat_space.dimensions[1], Categorical)

    assert_equal(len(space.dimensions), 5)
    assert isinstance(space.dimensions[0], Real)
    assert isinstance(space.dimensions[1], Integer)
    assert isinstance(space.dimensions[2], Categorical)
    assert isinstance(space.dimensions[3], Real)
    assert isinstance(space.dimensions[4], Categorical)

    samples = space.rvs(n_samples=10, random_state=0)
    assert_equal(len(samples), 10)
    assert_equal(len(samples[0]), 5)

    assert isinstance(samples, list)
    for n in range(4):
        assert isinstance(samples[n], list)

    assert isinstance(samples[0][0], numbers.Real)
    assert isinstance(samples[0][1], numbers.Integral)
    assert isinstance(samples[0][2], str)
    assert isinstance(samples[0][3], numbers.Real)
    assert isinstance(samples[0][4], str)

    samples_transformed = space.transform(samples)
    assert_equal(samples_transformed.shape[0], len(samples))
    assert_equal(samples_transformed.shape[1], 1 + 1 + 3 + 1 + 1)

    # our space contains mixed types, this means we can't use
    # `array_allclose` or similar to check points are close after a round-trip
    # of transformations
    for orig, round_trip in zip(samples, space.inverse_transform(samples_transformed)):
        assert space.distance(orig, round_trip) < 1.0e-8

    samples = space.inverse_transform(samples_transformed)
    assert isinstance(samples[0][0], numbers.Real)
    assert isinstance(samples[0][1], numbers.Integral)
    assert isinstance(samples[0][2], str)
    assert isinstance(samples[0][3], numbers.Real)
    assert isinstance(samples[0][4], str)

    for b1, b2 in zip(
        space.bounds,
        [(0.0, 1.0), (-5, 5), np.asarray(["a", "b", "c"]), (1.0, 5.0), np.asarray(["e", "f"])],
    ):
        assert_array_equal(b1, b2)

    for b1, b2 in zip(
        space.transformed_bounds,
        [
            (0.0, 1.0),
            (-5, 5),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (np.log10(1.0), np.log10(5.0)),
            (0.0, 1.0),
        ],
    ):
        assert_array_equal(b1, b2)


@pytest.mark.fast_test
def test_space_from_space():
    """Test that a `Space` instance can be passed to a `Space` constructor"""
    space_0 = Space([(0.0, 1.0), (-5, 5), ("a", "b", "c"), (1.0, 5.0, "log-uniform"), ("e", "f")])
    space_1 = Space(space_0)
    assert_equal(space_0, space_1)


@pytest.mark.fast_test
def test_normalize():
    # TODO: Refactor - Use PyTest
    a = Real(2.0, 30.0, transform="normalize")
    for i in range(50):
        check_limits(a.rvs(random_state=i), 2, 30)

    rng = np.random.RandomState(0)
    X = rng.randn(100)
    X = 28 * (X - X.min()) / (X.max() - X.min()) + 2

    # Check transformed values are in [0, 1]
    assert np.all(a.transform(X) <= np.ones_like(X))
    assert np.all(np.zeros_like(X) <= a.transform(X))

    # Check inverse transform
    assert_array_almost_equal(a.inverse_transform(a.transform(X)), X)

    # log-uniform prior
    a = Real(10 ** 2.0, 10 ** 4.0, prior="log-uniform", transform="normalize")
    for i in range(50):
        check_limits(a.rvs(random_state=i), 10 ** 2, 10 ** 4)

    rng = np.random.RandomState(0)
    X = np.clip(10 ** 3 * rng.randn(100), 10 ** 2.0, 10 ** 4.0)

    # Check transform
    assert np.all(a.transform(X) <= np.ones_like(X))
    assert np.all(np.zeros_like(X) <= a.transform(X))

    # Check inverse transform
    assert_array_almost_equal(a.inverse_transform(a.transform(X)), X)

    a = Integer(2, 30, transform="normalize")
    for i in range(50):
        check_limits(a.rvs(random_state=i), 2, 30)
    assert_array_equal(a.transformed_bounds, (0, 1))

    X = rng.randint(2, 31)
    # Check transformed values are in [0, 1]
    assert np.all(a.transform(X) <= np.ones_like(X))
    assert np.all(np.zeros_like(X) <= a.transform(X))

    # Check inverse transform
    X_orig = a.inverse_transform(a.transform(X))
    assert_equal(X_orig.dtype, "int64")
    assert_array_equal(X_orig, X)


@pytest.mark.parametrize("dim", [Real, Integer])
@pytest.mark.parametrize("transform", ["normalize", "identity"])
def test_valid_numerical_transformation(dim, transform):
    assert dim(2, 30, transform=transform)


@pytest.mark.parametrize("dim", [Real, Integer])
@pytest.mark.parametrize("transform", ["not a valid transform name"])
def test_invalid_numerical_transformation(dim, transform):
    with pytest.raises(ValueError, match=r"`transform` must be in \['normalize', 'identity'\].*"):
        dim(2, 30, transform=transform)


@pytest.mark.fast_test
def test_categorical_identity():
    # TODO: Refactor - Use PyTest
    categories = ["cat", "dog", "rat"]
    cat = Categorical(categories, transform="identity")
    samples = cat.rvs(100)
    assert all([t in categories for t in cat.rvs(100)])
    transformed = cat.transform(samples)
    assert_array_equal(transformed, samples)
    assert_array_equal(samples, cat.inverse_transform(transformed))


@pytest.mark.fast_test
def test_categorical_distance():
    # TODO: Refactor - Use PyTest
    categories = ["car", "dog", "orange"]
    cat = Categorical(categories)
    for cat1 in categories:
        for cat2 in categories:
            delta = cat.distance(cat1, cat2)
            if cat1 == cat2:
                assert delta == 0
            else:
                assert delta == 1


@pytest.mark.parametrize("dimension", [Real(1, 10), Integer(1, 10)])
@pytest.mark.parametrize("points", [(11, 10)])
def test_numerical_distance_out_of_range(dimension, points: tuple):
    err = "Distance computation requires values within space. Received {} and {}".format(*points)
    with pytest.raises(RuntimeError, match=err):
        dimension.distance(*points)


@pytest.mark.fast_test
def test_integer_distance():
    # TODO: Refactor - Use PyTest
    ints = Integer(1, 10)
    for i in range(1, 10 + 1):
        assert_equal(ints.distance(4, i), abs(4 - i))


@pytest.mark.fast_test
def test_real_distance():
    # TODO: Refactor - Use PyTest
    reals = Real(1, 10)
    for i in range(1, 10 + 1):
        assert_equal(reals.distance(4.1234, i), abs(4.1234 - i))


@pytest.mark.parametrize("dimension", [Real, Integer])
@pytest.mark.parametrize("bounds", [(2, 1), (2, 2)])
def test_dimension_bounds(dimension, bounds: tuple):
    err = r"Lower bound \({}\) must be less than the upper bound \({}\)".format(*bounds)
    with pytest.raises(ValueError, match=err):
        dimension(*bounds)


@pytest.mark.parametrize(
    "dimension, name",
    [
        (Real(1, 2, name="learning rate"), "learning rate"),
        (Integer(1, 100, name="no of trees"), "no of trees"),
        (Categorical(["red, blue"], name="colors"), "colors"),
    ],
)
def test_dimension_name(dimension, name):
    assert dimension.name == name


@pytest.mark.parametrize("dimension", [Real(1, 2), Integer(1, 100), Categorical(["red, blue"])])
def test_dimension_name_none(dimension):
    assert dimension.name is None


@pytest.mark.parametrize("name", [1, 1.0, True])
def test_dimension_with_invalid_names(name):
    with pytest.raises(ValueError, match="Dimension's name must be one of: string, tuple, or None"):
        Real(1, 2, name=name)


@pytest.mark.fast_test
def test_purely_categorical_space():
    # TODO: Refactor - Use PyTest
    # Test reproduces the bug in #908, make sure it doesn't come back
    dims = [Categorical(["a", "b", "c"]), Categorical(["A", "B", "C"])]
    optimizer = Optimizer(dims, n_initial_points=1, random_state=3)

    x = optimizer.ask()
    # Before the fix this call raised an exception
    optimizer.tell(x, 1.0)


##################################################
# `space_core.normalize_dimensions` Tests
##################################################
@pytest.mark.fast_test
@pytest.mark.parametrize("dimensions", [(["a", "b", "c"], ["1", "2", "3"])])
def test_normalize_dimensions_all_categorical(dimensions):
    """Test that :func:`normalize_dimensions` works with exclusively-`Categorical` spaces, and that
    the resulting space's :attr:`is_categorical` is True"""
    space = normalize_dimensions(dimensions)
    assert space.is_categorical


@pytest.mark.fast_test
@pytest.mark.parametrize(
    "dimensions, normalizations",
    [
        (((1, 3), (1.0, 3.0)), ("normalize", "normalize")),
        (((1, 3), ("a", "b", "c")), ("normalize", "onehot")),
    ],
)
def test_normalize_dimensions_transform(dimensions, normalizations):
    """Test that dimensions' :attr:`transform_` have been set to the expected value after invoking
    :func:`normalize_dimensions`"""
    space = normalize_dimensions(dimensions)
    for dimension, normalization in zip(space, normalizations):
        assert dimension.transform_ == normalization


@pytest.mark.fast_test
@pytest.mark.parametrize(
    "dimension, name",
    [
        (Real(1, 2, name="learning rate"), "learning rate"),
        (Integer(1, 100, name="no of trees"), "no of trees"),
        (Categorical(["red, blue"], name="colors"), "colors"),
    ],
)
def test_normalize_dimensions_name(dimension, name):
    """Test that a dimension's :attr:`name` is unchanged after invoking `normalize_dimensions`"""
    space = normalize_dimensions([dimension])
    assert space.dimensions[0].name == name


@pytest.mark.parametrize(
    "dimensions",
    [
        ((1, 3), (1.0, 3.0)),
        ((1, 3), ("a", "b", "c")),
        (["a", "b", "c"], ["1", "2", "3"]),
        (["a", "b", "c"], ["1", "2", "3"], ["foo", "bar"]),
        ((1, 3), (1.0, 3.0), ["a", "b", "c"], ["1", "2", "3"]),
    ],
)
def test_normalize_dimensions_consecutive_calls(dimensions):
    """Test that :func:`normalize_dimensions` can be safely invoked consecutively on the space each
    invocation returns. This doesn't test that the result of :func:`normalize_dimensions` is
    actually correct - Only that the result remains unchanged after multiple invocations"""
    space_0 = normalize_dimensions(dimensions)
    space_1 = normalize_dimensions(space_0)
    space_2 = normalize_dimensions(space_1)
    # Same as above, but starting with a `Space` instance to make sure nothing changes
    space_3 = normalize_dimensions(Space(dimensions))
    space_4 = normalize_dimensions(space_3)
    space_5 = normalize_dimensions(space_4)

    assert space_0 == space_1 == space_2 == space_3 == space_4 == space_5


##################################################
# `space_core.check_dimension` Tests
##################################################
@pytest.mark.fast_test
@pytest.mark.parametrize("dim", ["23"])
def test_invalid_check_dimension(dim):
    with pytest.raises(ValueError, match="Dimension has to be a list or tuple"):
        check_dimension("23")


@pytest.mark.parametrize("dim", [(23,)])
def test_valid_check_dimension(dim):
    # Single value fixes dimension of space
    check_dimension(dim)
