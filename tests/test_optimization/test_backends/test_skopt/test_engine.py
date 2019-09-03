"""Tests the tools defined by :mod:`hyperparameter_hunter.optimization.backends.skopt.engine`

Notes
-----
Many of the tests defined herein (although substantially modified) are based on those provided by
the excellent [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) library. See
:mod:`hyperparameter_hunter.optimization.backends.skopt` for a copy of SKOpt's license"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.optimization.backends.skopt.engine import Optimizer

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
import pytest

##################################################
# Import Learning Assets
##################################################
from scipy.optimize import OptimizeResult
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.testing import assert_array_equal
from skopt.benchmarks import bench1, bench1_with_time
from skopt.learning import ExtraTreesRegressor, RandomForestRegressor
from skopt.learning import GradientBoostingQuantileRegressor

TREE_REGRESSORS = (
    ExtraTreesRegressor(random_state=2),
    RandomForestRegressor(random_state=2),
    GradientBoostingQuantileRegressor(random_state=2),
)
ACQ_FUNCS_PS = ["EIps", "PIps"]
ACQ_FUNCS_MIXED = ["EI", "EIps"]
ESTIMATOR_STRINGS = ["GP", "RF", "ET", "GBRT", "DUMMY", "gp", "rf", "et", "gbrt", "dummy"]


##################################################
# Fixtures
##################################################
@pytest.fixture()
def et_optimizer(request):
    dimensions = request.param
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(dimensions, base_estimator, n_initial_points=1, acq_optimizer="sampling")
    return opt


##################################################
# Smoke Tests
##################################################
@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", ESTIMATOR_STRINGS)
def test_optimizer_base_estimator_string_smoke(base_estimator):
    opt = Optimizer([(-2.0, 2.0)], base_estimator=base_estimator, n_initial_points=1, acq_func="EI")
    opt.run(func=lambda x: x[0] ** 2, n_iter=3)


@pytest.mark.fast_test
@pytest.mark.parametrize("et_optimizer", [[(-2.0, 2.0)]], indirect=True)
def test_multiple_asks(et_optimizer):
    """Test that calling `ask` multiple times without a `tell` in-between does nothing"""
    et_optimizer.run(bench1, n_iter=3)
    # `tell` computes the next points for next call to `ask`, hence there are 3 after 3 iterations
    assert len(et_optimizer.models) == 3
    assert len(et_optimizer.Xi) == 3
    et_optimizer.ask()
    assert len(et_optimizer.models) == 3
    assert len(et_optimizer.Xi) == 3
    assert et_optimizer.ask() == et_optimizer.ask()


@pytest.mark.fast_test
@pytest.mark.parametrize("et_optimizer", [[(-2.0, 2.0)]], indirect=True)
@pytest.mark.parametrize(
    ["x", "y"],
    [
        pytest.param([1.0], [1.0, 1.0], id="single_x_with_multiple_y"),
        pytest.param([[1.0], [2.0]], [1.0, None], id="invalid_y_type"),
    ],
)
def test_invalid_tell_arguments(et_optimizer, x, y):
    """Test that `tell`-ing raises ValueError when given invalid `x` and `y` values"""
    with pytest.raises(ValueError):
        et_optimizer.tell(x, y)


@pytest.mark.fast_test
def test_returns_result_object():
    # TODO: Refactor - Use PyTest
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer([(-2.0, 2.0)], base_estimator, n_initial_points=1, acq_optimizer="sampling")
    result = opt.tell([1.5], 2.0)

    assert isinstance(result, OptimizeResult)
    assert len(result.x_iters) == len(result.func_vals)
    assert np.min(result.func_vals) == result.fun


@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", TREE_REGRESSORS)
def test_acq_optimizer(base_estimator):
    err = f"Regressor {type(base_estimator)} requires `acq_optimizer`='sampling'"
    with pytest.raises(ValueError, match=err):
        Optimizer(
            [(-2.0, 2.0)], base_estimator=base_estimator, n_initial_points=1, acq_optimizer="lbfgs"
        )


@pytest.mark.parametrize("base_estimator", TREE_REGRESSORS)
@pytest.mark.parametrize("acq_func", ACQ_FUNCS_PS)
def test_acq_optimizer_with_time_api(base_estimator, acq_func):
    # TODO: Refactor - Use PyTest
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator=base_estimator,
        acq_func=acq_func,
        acq_optimizer="sampling",
        n_initial_points=2,
    )
    x1 = opt.ask()
    opt.tell(x1, (bench1(x1), 1.0))
    x2 = opt.ask()
    res = opt.tell(x2, (bench1(x2), 2.0))

    # `x1` and `x2` are random.
    assert x1 != x2

    assert len(res.models) == 1
    assert_array_equal(res.func_vals.shape, (2,))
    assert_array_equal(res.log_time.shape, (2,))

    # x3 = opt.ask()
    # TODO: Refactor - Split into separate error test
    with pytest.raises(TypeError):
        opt.tell(x2, bench1(x2))


@pytest.mark.fast_test
@pytest.mark.parametrize("acq_func", ACQ_FUNCS_MIXED)
def test_optimizer_copy(acq_func):
    """Check that base estimator, objective and target values are copied correctly"""
    # TODO: Refactor - Use PyTest

    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        acq_func=acq_func,
        n_initial_points=1,
        acq_optimizer="sampling",
    )

    # Run three iterations so that we have some points and objective values
    if "ps" in acq_func:
        opt.run(bench1_with_time, n_iter=3)
    else:
        opt.run(bench1, n_iter=3)

    opt_copy = opt.copy()
    copied_estimator = opt_copy.base_estimator

    if "ps" in acq_func:
        assert isinstance(copied_estimator, MultiOutputRegressor)
        # Check that `base_estimator` is not wrapped multiple times
        assert not isinstance(copied_estimator.estimator, MultiOutputRegressor)
    else:
        assert not isinstance(copied_estimator, MultiOutputRegressor)

    assert_array_equal(opt_copy.Xi, opt.Xi)
    assert_array_equal(opt_copy.yi, opt.yi)


@pytest.mark.parametrize("base_estimator", ESTIMATOR_STRINGS)
def test_exhaust_initial_calls(base_estimator):
    """Check that a model is fitted and used to make suggestions after adding at least
    `n_initial_points` via `tell`"""
    # TODO: Refactor - Use PyTest

    opt = Optimizer(
        [(-2.0, 2.0)], base_estimator, n_initial_points=2, acq_optimizer="sampling", random_state=1
    )

    # Until surrogate model in `Optimizer` has been fitted (after `tell`-ing `n_initial_points`),
    #   `ask` returns random points, which is why `x0` and `x1` must be different
    x0 = opt.ask()  # Random point
    x1 = opt.ask()  # Random point
    assert x0 != x1

    #################### First `tell` Call ####################
    # `tell` with a dummy objective value
    r1 = opt.tell(x1, 3.0)
    assert len(r1.models) == 0
    # Surrogate model still not fitted because only 1 / `n_initial_points` has been `tell`-ed
    x2 = opt.ask()  # Random point
    assert x1 != x2

    #################### Second `tell` Call ####################
    r2 = opt.tell(x2, 4.0)
    # After `tell`-ing a second point, a surrogate model is fitted - Unless using "dummy" estimator
    if base_estimator.lower() == "dummy":
        assert len(r2.models) == 0
    else:
        assert len(r2.models) == 1

    #################### First Non-Random Point ####################
    x3 = opt.ask()
    assert x2 != x3
    x4 = opt.ask()
    r3 = opt.tell(x3, 1.0)

    # No new information was added, so should be the same, unless we are using the dummy estimator,
    #   which will forever return random points and never fits any models
    if base_estimator.lower() == "dummy":
        assert x3 != x4
        assert len(r3.models) == 0
    else:
        assert x3 == x4
        assert len(r3.models) == 2


# def test_defaults_are_equivalent():
#     """Check that the defaults of `Optimizer` reproduce the defaults of `gp_minimize`"""
#     # MARK: BROKEN - `gp_minimize` is from SKOpt, so it uses non-HH dimensions
#
#     space = [(-5.0, 10.0), (0.0, 15.0)]
#     opt = Optimizer(space, random_state=1)
#
#     for n in range(12):
#         x = opt.ask()
#         res_opt = opt.tell(x, branin(x))
#
#     res_min = gp_minimize(branin, space, n_calls=12, random_state=1)
#
#     assert res_min.space == res_opt.space
#     # Tolerate small differences in the points sampled
#     assert np.allclose(res_min.x_iters, res_opt.x_iters)  # , atol=1e-5)
#     assert np.allclose(res_min.x, res_opt.x)  # , atol=1e-5)


@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", ["rtr"])
def test_optimizer_base_estimator_string_invalid(base_estimator):
    """Check that error is raised when `Optimizer` is given an invalid `base_estimator` string"""
    err = r"Expected `base_estimator` in \['GP', 'ET', 'RF', 'GBRT', 'DUMMY'\]\. Got {}".format(
        base_estimator
    )
    with pytest.raises(ValueError, match=err):
        Optimizer([(-2.0, 2.0)], base_estimator=base_estimator, n_initial_points=1)


##################################################
# `optimization.backends.skopt.engine.check_x_in_space` Tests
##################################################
@pytest.mark.fast_test
@pytest.mark.parametrize("et_optimizer", [[(-2.0, 2.0)]], indirect=True)
@pytest.mark.parametrize(
    ["tell_x", "tell_y"],
    [
        pytest.param([2.5], 2.0, id="1d_single (high)"),
        pytest.param([-2.5], 2.0, id="1d_single (low)"),
        pytest.param([2.5, 2.0], (2.0, 3.0), id="1d_multi (high)"),
        pytest.param([-2.5, 2.0], (2.0, 3.0), id="1d_multi (low)"),
    ],
)
def test_bounds_checking_1D(et_optimizer, tell_x, tell_y):
    # NOTE: See `test_bounds_checking_2D` docstring for details on `id` naming scheme
    with pytest.raises(ValueError):
        et_optimizer.tell(tell_x, tell_y)


@pytest.mark.fast_test
@pytest.mark.parametrize("et_optimizer", [[(-2.0, 2.0), (2.0, 6.0)]], indirect=True)
@pytest.mark.parametrize(
    ["tell_x", "tell_y"],
    [
        pytest.param((2.5, 6.5), 2.0, id="2d_single (high, high)"),
        pytest.param((-2.5, -6.5), 2.0, id="2d_single (low, low)"),
        pytest.param((2.5, 2.5), 2.0, id="2d_single (high, ok)"),
        pytest.param((-2.5, 2.5), 2.0, id="2d_single (low, ok)"),
        pytest.param([(2.5, 2.5), (2.5, 2.5)], [2.0, 3.0], id="2d_multi (high, ok)"),
        pytest.param([(-2.5, 2.5), (-2.5, 2.5)], [2.0, 3.0], id="2d_multi (low, ok)"),
    ],
)
def test_bounds_checking_2D(et_optimizer, tell_x, tell_y):
    """Test that `tell`-ing points outside of an `Optimizer`'s bounds raises ValueError. The
    parenthesized "high", "low", "ok" values in the `pytest.param` `id` fields refer to which
    dimensions fall outside of the expected bounds and in which direction. "(low, low)" means both
    the first and second dimensions fall below their lower bounds. "(high, ok)" means the first
    dimension exceeds its upper bound, while the second dimension is within its expected bounds"""
    with pytest.raises(ValueError):
        et_optimizer.tell(tell_x, tell_y)


@pytest.mark.fast_test
@pytest.mark.parametrize(
    ["et_optimizer", "tell_x", "tell_y"],
    [
        pytest.param([(-2, 2)], [-1, -1], 2.0, id="1d_extra_dim"),
        pytest.param([(-2, 2), (-2, 2)], [-1], 2.0, id="2d_missing_dim"),
        pytest.param([(-2, 2), (-2, 2)], [-1, -1, -1], 2.0, id="2d_extra_dim"),
    ],
    indirect=["et_optimizer"],
)
def test_dimension_count_checking_single_point(et_optimizer, tell_x, tell_y):
    with pytest.raises(ValueError, match="Dimensions of point .*"):
        et_optimizer.tell(tell_x, tell_y)


@pytest.mark.fast_test
@pytest.mark.parametrize("et_optimizer", [[(-2.0, 2.0), (-2.0, 2.0)]], indirect=True)
@pytest.mark.parametrize(
    ["tell_x", "tell_y"],
    [
        pytest.param([[-1], [-1, 0], [-1, 1]], 2.0, id="2d_missing_dim"),
        pytest.param([[-1, -1, -1], [-1, 0], [-1, 1]], 2.0, id="2d_extra_dim"),
    ],
)
def test_dimension_count_checking_multiple_points(et_optimizer, tell_x, tell_y):
    with pytest.raises(ValueError, match="Not all points have the same dimensions as the space"):
        et_optimizer.tell(tell_x, tell_y)


@pytest.mark.parametrize("base_estimator", ["GP", "RF", "ET", "GBRT", "gp", "rf", "et", "gbrt"])
@pytest.mark.parametrize("next_x", [[-1.0]])
def test_warn_on_re_ask(base_estimator, next_x):
    """Test that `Optimizer.warn_on_re_ask` logs warning when `Optimizer._ask` suggests a point
    that has already been `tell`-ed to `Optimizer`

    Notes
    -----
    "DUMMY"/"dummy" is invalid for `base_estimator` here because it always suggests random points"""
    # Initialize `Optimizer` and `tell` it about `next_x`
    opt = Optimizer(
        [(-2.0, 2.0)], base_estimator, n_initial_points=1, random_state=1, warn_on_re_ask=True
    )
    opt.tell(next_x, 1.0)

    # Force `Optimizer._next_x` (set by `Optimizer._tell`) to the point we just told it
    opt._next_x = next_x

    with pytest.warns(UserWarning, match="Repeated suggestion: .*"):
        opt.ask()
