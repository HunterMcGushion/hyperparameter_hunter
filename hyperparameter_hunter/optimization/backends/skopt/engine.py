"""This module contains various modified SKOpt assets that are used to support the other
:mod:`hyperparameter_hunter.optimization.backends.skopt` modules

Related
-------
...

Notes
-----
Many of the tools defined herein (although substantially modified) are based on those provided by
the excellent [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) library. See
:mod:`hyperparameter_hunter.optimization.backends.skopt` for a copy of SKOpt's license.

What follows is a record of the first few commits to this file in order to clearly define what code
was taken from the original Scikit-Optimize source, and how it was modified thereafter.

* 81a70ddfa0270495f0ed39127adbac4eb1f4fa59:
  The content of this module (less module docstring) is identical to SKOpt's module
  `skopt.optimizer.optimizer` at the time of SKOpt commit 6740876a6f9ad92c732d394e8534a5236a8d3f84
* 744043d09f11cf90609cbef6ca8ab43515958feb:
  Add SKOpt's `skopt.utils.cook_estimator` at the time of the above SKOpt commit, as well as the
  original import statements required by the function
* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:
  [Diverging from SKOpt] Fix broken imports, and (substantially) refactor code and documentation to
  follow HH conventions or for readability - Changes on and after this point are originally authored
  by the contributors of HyperparameterHunter and are, therefore, subject to the
  HyperparameterHunter license"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.space.space_core import Space, normalize_dimensions

##################################################
# Import Miscellaneous Assets
##################################################
from math import log
from numbers import Number
import numpy as np
import sys
import warnings

##################################################
# Import Learning Assets
##################################################
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import clone, is_regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals.joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state

# noinspection PyProtectedMember
from skopt.acquisition import _gaussian_acquisition, gaussian_acquisition_1D
from skopt.learning import (
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    GradientBoostingQuantileRegressor,
    RandomForestRegressor,
)
from skopt.learning.gaussian_process.kernels import ConstantKernel, HammingKernel, Matern

# noinspection PyProtectedMember
from skopt.utils import create_result, has_gradients


class Optimizer(object):
    """Run bayesian optimisation loop

    An `Optimizer` represents the steps of a bayesian optimisation loop. To use it you need to
    provide your own loop mechanism. The various optimisers provided by `skopt` use this class
    under the hood. Use this class directly if you want to control the iterations of your bayesian
    optimisation loop

    Parameters
    ----------
    dimensions: List
        List of shape (n_dims,) containing search space dimensions. Each search dimension can be
        defined as any of the following:

        * Instance of a `Dimension` object (`Real`, `Integer` or `Categorical`)
        * (<lower_bound>, <upper_bound>) tuple (for `Real` or `Integer` dimensions)
        * (<lower_bound>, <upper_bound>, <prior>) tuple (for `Real` dimensions)
        * List of categories (for `Categorical` dimensions)
    base_estimator: {SKLearn Regressor, "GP", "RF", "ET", "GBRT", "DUMMY"}, default="GP"
        If not string, should inherit from `sklearn.base.RegressorMixin`. In addition, the `predict`
        method should have an optional `return_std` argument, which returns `std(Y | x)`,
        along with `E[Y | x]`.

        If `base_estimator` is a string in {"GP", "RF", "ET", "GBRT", "DUMMY"}, a surrogate model
        corresponding to the relevant `X_minimize` function is created
    n_initial_points: Int, default=10
        Number of evaluations of `func` with initialization points before approximating it with
        `base_estimator`. Points provided as `x0` count as initialization points.
        If len(`x0`) < `n_initial_points`, additional points are sampled at random
    acq_func: {"LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"}, default="gp_hedge"
        Function to minimize over the posterior distribution. Can be any of the following:

        * "LCB": Lower confidence bound
        * "EI": Negative expected improvement
        * "PI": Negative probability of improvement
        * "gp_hedge": Probabilistically choose one of the above three acquisition functions at
          every iteration

            * The gains `g_i` are initialized to zero
            * At every iteration,

                * Each acquisition function is optimised independently to propose a candidate point
                  `X_i`
                * Out of all these candidate points, the next point `X_best` is chosen by
                  `softmax(eta g_i)`
                * After fitting the surrogate model with `(X_best, y_best)`, the gains are updated
                  such that `g_i -= mu(X_i)`

        * "EIps": Negated expected improvement per second to take into account the function compute
          time. Then, the objective function is assumed to return two values, the first being the
          objective value and the second being the time taken in seconds
        * "PIps": Negated probability of improvement per second. The return type of the objective
          function is identical to that of "EIps"
    acq_optimizer: {"sampling", "lbfgs", "auto"}, default="auto"
        Method to minimize the acquisition function. The fit model is updated with the optimal
        value obtained by optimizing `acq_func` with `acq_optimizer`

        * "sampling": `acq_func` is optimized by computing `acq_func` at `n_initial_points`
          randomly sampled points.
        * "lbfgs": `acq_func` is optimized by

              * Randomly sampling `n_restarts_optimizer` (from `acq_optimizer_kwargs`) points
              * "lbfgs" is run for 20 iterations with these initial points to find local minima
              * The optimal of these local minima is used to update the prior

        * "auto": `acq_optimizer` is configured on the basis of the `base_estimator` and the search
          space. If the space is `Categorical` or if the provided estimator is based on tree-models,
          then this is set to "sampling"
    random_state: Int, or RandomState instance (optional)
        Set random state to something other than None for reproducible results
    acq_func_kwargs: Dict (optional)
        Additional arguments to be passed to the acquisition function.
    acq_optimizer_kwargs: Dict (optional)
        Additional arguments to be passed to the acquisition optimizer
    warn_on_re_ask: Boolean, default=False
        If True, and the internal `optimizer` recommends a point that has already been evaluated
        on invocation of `ask`, a warning is logged before recommending a random point. Either
        way, a random point is used instead of already-evaluated recommendations. However,
        logging the fact that this has taken place can be useful to indicate that the optimizer
        may be stalling, especially if it repeatedly recommends the same point. In these cases,
        if the suggested point is not optimal, it can be helpful to switch a different OptPro
        (especially `DummyOptPro`), which will suggest points using different criteria

    Attributes
    ----------
    Xi: List
        Points at which objective has been evaluated
    yi: List
        Values of objective at corresponding points in `Xi`
    models: List
        Regression models used to fit observations and compute acquisition function
    space: `hyperparameter_hunter.space.space_core.Space`
        Stores parameter search space used to sample points, bounds, and type of parameters
    n_initial_points_: Int
        Original value passed through the `n_initial_points` kwarg. The value of this attribute
        remains unchanged along the lifespan of `Optimizer`, unlike :attr:`_n_initial_points`
    _n_initial_points: Int
        Number of remaining points that must be evaluated before fitting a surrogate estimator and
        using it to recommend incumbent search points. Initially, :attr:`_n_initial_points` is set
        to the value of the `n_initial_points` kwarg, like :attr:`n_initial_points_`. However,
        :attr:`_n_initial_points` is decremented for each point `tell`-ed to `Optimizer`

    """

    def __init__(
        self,
        dimensions,
        base_estimator="gp",
        n_initial_points=10,
        acq_func="gp_hedge",
        acq_optimizer="auto",
        random_state=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        warn_on_re_ask=False,
    ):
        self.rng = check_random_state(random_state)
        self.space = Space(dimensions)

        #################### Configure Acquisition Function ####################
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ["gp_hedge", "EI", "LCB", "PI", "EIps", "PIps"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError(f"Expected `acq_func` in {allowed_acq_funcs}. Got {self.acq_func}")

        # Treat hedging method separately
        if self.acq_func == "gp_hedge":
            self.cand_acq_funcs_ = ["EI", "LCB", "PI"]
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        #################### Configure Point Counters ####################
        if n_initial_points < 0:
            raise ValueError(f"Expected `n_initial_points` >= 0. Got {n_initial_points}")
        self._n_initial_points = n_initial_points  # TODO: Rename to `remaining_n_points`
        self.n_initial_points_ = n_initial_points

        #################### Configure Estimator ####################
        self.base_estimator = base_estimator

        #################### Configure Optimizer ####################
        self.acq_optimizer = acq_optimizer

        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get("n_restarts_optimizer", 5)
        n_jobs = acq_optimizer_kwargs.get("n_jobs", 1)
        self.n_jobs = n_jobs
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        self.warn_on_re_ask = warn_on_re_ask

        #################### Configure Search Space ####################
        if isinstance(self.base_estimator, GaussianProcessRegressor):
            self.space = normalize_dimensions(self.space)

        #################### Initialize Optimization Storage ####################
        self.models = []
        self.Xi = []
        self.yi = []

        # Initialize cache for `ask` method responses. Ensures that multiple calls to `ask` with
        #   n_points set return same sets of points. Reset to {} at every call to `tell`
        self.cache_ = {}

    ##################################################
    # Properties
    ##################################################
    @property
    def base_estimator(self):
        return self._base_estimator

    @base_estimator.setter
    def base_estimator(self, value):
        # Build `base_estimator` if string given
        if isinstance(value, str):
            value = cook_estimator(
                value, space=self.space, random_state=self.rng.randint(0, np.iinfo(np.int32).max)
            )

        # Check if regressor
        if not is_regressor(value) and value is not None:
            raise ValueError(f"`base_estimator` must be a regressor. Got {value}")

        # Treat per second acquisition function specially
        is_multi_regressor = isinstance(value, MultiOutputRegressor)
        if self.acq_func.endswith("ps") and not is_multi_regressor:
            value = MultiOutputRegressor(value)

        self._base_estimator = value

    @property
    def acq_optimizer(self) -> str:
        """Method to minimize the acquisition function. See documentation for the `acq_optimizer`
        kwarg in :meth:`Optimizer.__init__` for additional information

        Returns
        -------
        {"lbfgs", "sampling"}
            String in {"lbfgs", "sampling"}. If originally "auto", one of the two aforementioned
            strings is selected based on :attr:`base_estimator`"""
        return self._acq_optimizer

    @acq_optimizer.setter
    def acq_optimizer(self, value):
        # Decide optimizer based on gradient information
        if value == "auto":
            if has_gradients(self.base_estimator):
                value = "lbfgs"
            else:
                value = "sampling"

        if value not in ["lbfgs", "sampling"]:
            raise ValueError(f"`acq_optimizer` must be 'lbfgs' or 'sampling'. Got {value}")

        if not has_gradients(self.base_estimator) and value != "sampling":
            raise ValueError(
                f"Regressor {type(self.base_estimator)} requires `acq_optimizer`='sampling'"
            )
        self._acq_optimizer = value

    ##################################################
    # Ask
    ##################################################
    def ask(self, n_points=None, strategy="cl_min"):  # TODO: Try `n_points` default=1
        """Request point (or points) at which objective should be evaluated next

        Parameters
        ----------
        n_points: Int (optional)
            Number of points returned by the ask method. If `n_points` not given, a single point
            to evaluate is returned. Otherwise, a list of points to evaluate is returned of size
            `n_points`. This is useful if you can evaluate your objective in parallel, and thus
            obtain more objective function evaluations per unit of time
        strategy: {"cl_min", "cl_mean", "cl_max"}, default="cl_min"
            Method used to sample multiple points if `n_points` is an integer. If `n_points` is not
            given, `strategy` is ignored.

            If set to "cl_min", then "Constant Liar" strategy (see reference) is used with lie
            objective value being minimum of observed objective values. "cl_mean" and "cl_max"
            correspond to the mean and max of values, respectively.

            With this strategy, a copy of optimizer is created, which is then asked for a point,
            and the point is told to the copy of optimizer with some fake objective (lie), the
            next point is asked from copy, it is also told to the copy with fake objective and so
            on. The type of lie defines different flavours of "cl..." strategies

        Returns
        -------
        List
            Point (or points) recommended to be evaluated next

        References
        ----------
        .. [1] Chevalier, C.; Ginsbourger, D.: "Fast Computation of the Multi-points Expected
            Improvement with Applications in Batch Selection".
            https://hal.archives-ouvertes.fr/hal-00732512/document"""
        if n_points is None:
            return self._ask()

        #################### Validate Parameters ####################
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(f"`n_points` must be int > 0. Got {n_points}")

        supported_strategies = ["cl_min", "cl_mean", "cl_max"]
        if strategy not in supported_strategies:
            raise ValueError(f"Expected `strategy` in {supported_strategies}. Got {strategy}")

        #################### Check Cache ####################
        # If repeated parameters given to `ask`, return cached entry
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        #################### Constant Liar ####################
        # Copy of the optimizer is made in order to manage the deletion of points with "lie"
        #   objective (the copy of optimizer is simply discarded)
        opt = self.copy(random_state=self.rng.randint(0, np.iinfo(np.int32).max))

        points = []
        for i in range(n_points):
            x = opt.ask()
            points.append(x)

            # TODO: Put below section into `how_to_lie` helper function for easier testing
            ti_available = self.acq_func.endswith("ps") and len(opt.yi) > 0
            ti = [t for (_, t) in opt.yi] if ti_available else None

            # TODO: Do below `y_lie` lines directly calculate min/max/mean on `opt.yi` when it could also contain times?
            if strategy == "cl_min":
                y_lie = np.min(opt.yi) if opt.yi else 0.0  # CL-min lie
                t_lie = np.min(ti) if ti is not None else log(sys.float_info.max)
            elif strategy == "cl_mean":
                y_lie = np.mean(opt.yi) if opt.yi else 0.0  # CL-mean lie
                t_lie = np.mean(ti) if ti is not None else log(sys.float_info.max)
            else:
                y_lie = np.max(opt.yi) if opt.yi else 0.0  # CL-max lie
                t_lie = np.max(ti) if ti is not None else log(sys.float_info.max)

            #################### Lie to Optimizer ####################
            # Use `_tell` (not `tell`) to prevent repeated log transformations of computation times
            if self.acq_func.endswith("ps"):
                opt._tell(x, (y_lie, t_lie))
            else:
                opt._tell(x, y_lie)

        #################### Cache and Return Result ####################
        self.cache_ = {(n_points, strategy): points}
        return points

    def _ask(self):
        """Suggest next point at which to evaluate the objective

        Returns
        -------
        Some point in :attr:`space`, which is random while less than `n_initial_points` observations
        have been `tell`-ed. After that, `base_estimator` is used to determine the next point

        Notes
        -----
        If the suggested point has already been evaluated, a random point will be returned instead,
        optionally accompanied by a warning message (depending on :attr:`warn_on_re_ask`)"""
        if self._n_initial_points > 0 or self.base_estimator is None:
            # Does not copy `self.rng` in order to keep advancing random state
            return self.space.rvs(random_state=self.rng)[0]
        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no model has been fit")

            #################### Check for Repeated Suggestion ####################
            next_x = self._next_x
            # Check distances between `next_x` and all evaluated points
            min_delta_x = min([self.space.distance(next_x, xi) for xi in self.Xi])

            if abs(min_delta_x) <= 1e-8:  # `next_x` has already been evaluated
                if self.warn_on_re_ask:
                    G.warn_("Repeated suggestion: {}".format(next_x))

                # Set `_next_x` to random point, then re-invoke `_ask` to validate new point
                self._next_x = self.space.rvs(random_state=self.rng)[0]
                return self._ask()

            # Return point computed from last call to `tell`
            return next_x

    ##################################################
    # Tell
    ##################################################
    def tell(self, x, y, fit=True):
        """Record an observation (or several) of the objective function

        Provide values of the objective function at points suggested by :meth:`ask`, or arbitrary
        points. By default, a new model will be fit to all observations. The new model is used to
        suggest the next point at which to evaluate the objective. This point can be retrieved by
        calling :meth:`ask`.

        To add multiple observations in a batch, pass a list-of-lists for `x`, and a list of
        scalars for `y`

        Parameters
        ----------
        x: List, or list-of-lists
            Point(s) at which objective was evaluated
        y: Scalar, or list
            Value(s) of objective at `x`
        fit: Boolean, default=True
            Whether to fit a model to observed evaluations of the objective. A model will only be
            fitted after `n_initial_points` points have been `tell`-ed to the optimizer,
            irrespective of the value of `fit`. To add observations without fitting a new model,
            set `fit` to False"""
        check_x_in_space(x, self.space)
        self._check_y_is_valid(x, y)

        # Take logarithm of the computation times
        if self.acq_func.endswith("ps"):
            if is_2d_list_like(x):
                y = [[val, log(t)] for (val, t) in y]
            elif is_list_like(x):
                y = list(y)
                y[1] = log(y[1])

        return self._tell(x, y, fit=fit)

    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points. See :meth:`tell` for
        the full description. This method exists to give access to the internals of adding points
        by side-stepping all input validation and transformation"""
        #################### Collect Search Points and Evaluations ####################
        # TODO: Clean up below - Looks like the 4 extend/append blocks may be duplicated
        if "ps" in self.acq_func:
            if is_2d_list_like(x):
                self.Xi.extend(x)
                self.yi.extend(y)
                self._n_initial_points -= len(y)
            elif is_list_like(x):
                self.Xi.append(x)
                self.yi.append(y)
                self._n_initial_points -= 1
        # If `y` isn't a scalar, we have been handed a batch of points
        elif is_list_like(y) and is_2d_list_like(x):
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len(y)
        elif is_list_like(x):
            self.Xi.append(x)
            self.yi.append(y)
            self._n_initial_points -= 1
        else:
            raise ValueError(f"Incompatible argument types: `x` ({type(x)}) and `y` ({type(y)})")

        # Optimizer learned something new. Discard `cache_`
        self.cache_ = {}

        #################### Fit Surrogate Model ####################
        # After being `tell`-ed `n_initial_points`, use surrogate model instead of random sampling
        # TODO: Clean up and separate below. Pretty hard to follow the whole thing
        if fit and self._n_initial_points <= 0 and self.base_estimator is not None:
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(self.space.transform(self.Xi), self.yi)

            if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
                self.gains_ -= est.predict(np.vstack(self.next_xs_))
            self.models.append(est)

            # Even with BFGS optimizer, we want to sample a large number of points, and
            #   pick the best ones as starting points
            X = self.space.transform(self.space.rvs(n_samples=self.n_points, random_state=self.rng))

            self.next_xs_ = []
            for cand_acq_func in self.cand_acq_funcs_:
                # TODO: Rename `values` - Maybe `utilities`?
                values = _gaussian_acquisition(
                    X=X,
                    model=est,
                    y_opt=np.min(self.yi),
                    acq_func=cand_acq_func,
                    acq_func_kwargs=self.acq_func_kwargs,
                )

                #################### Find Acquisition Function Minimum ####################
                # Find acquisition function minimum by randomly sampling points from the space
                if self.acq_optimizer == "sampling":
                    next_x = X[np.argmin(values)]

                # Use BFGS to find the minimum of the acquisition function, the minimization starts
                #   from `n_restarts_optimizer` different points and the best minimum is used
                elif self.acq_optimizer == "lbfgs":
                    x0 = X[np.argsort(values)[: self.n_restarts_optimizer]]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = Parallel(n_jobs=self.n_jobs)(
                            delayed(fmin_l_bfgs_b)(
                                gaussian_acquisition_1D,
                                x,
                                args=(est, np.min(self.yi), cand_acq_func, self.acq_func_kwargs),
                                bounds=self.space.transformed_bounds,
                                approx_grad=False,
                                maxiter=20,
                            )
                            for x in x0
                        )

                    cand_xs = np.array([r[0] for r in results])
                    cand_acqs = np.array([r[1] for r in results])
                    next_x = cand_xs[np.argmin(cand_acqs)]
                else:
                    # `acq_optimizer` should have already been checked, so this shouldn't be hit,
                    #   but, it's here anyways to prevent complaints about `next_x` not existing in
                    #   the absence of this `else` clause
                    raise RuntimeError(f"Invalid `acq_optimizer` value: {self.acq_optimizer}")

                # L-BFGS-B should handle this, but just in case of precision errors...
                if not self.space.is_categorical:
                    next_x = np.clip(next_x, transformed_bounds[:, 0], transformed_bounds[:, 1])
                self.next_xs_.append(next_x)

            if self.acq_func == "gp_hedge":
                logits = np.array(self.gains_)
                logits -= np.max(logits)
                exp_logits = np.exp(self.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = self.next_xs_[np.argmax(self.rng.multinomial(1, probs))]
            else:
                next_x = self.next_xs_[0]

            # Note the need for [0] at the end
            self._next_x = self.space.inverse_transform(next_x.reshape((1, -1)))[0]

        # Pack results
        return create_result(self.Xi, self.yi, self.space, self.rng, models=self.models)

    ##################################################
    # Helper Methods
    ##################################################
    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer

        Parameters
        ----------
        random_state: Int, or RandomState instance (optional)
            Set random state of the copy

        Returns
        -------
        Optimizer
            Shallow copy of self"""
        optimizer = Optimizer(
            dimensions=self.space.dimensions,
            base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points_,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state,
        )

        if hasattr(self, "gains_"):
            optimizer.gains_ = np.copy(self.gains_)

        if self.Xi:
            optimizer._tell(self.Xi, self.yi)

        return optimizer

    def _check_y_is_valid(self, x, y):
        """Check if the shapes and types of `x` and `y` are consistent. Complains if anything
        is weird about `y`"""
        #################### Per-Second Acquisition Function ####################
        if self.acq_func.endswith("ps"):
            if is_2d_list_like(x):
                if not (np.ndim(y) == 2 and np.shape(y)[1] == 2):
                    raise TypeError("Expected `y` to be a list of (func_val, t)")
            elif is_list_like(x):
                if not (np.ndim(y) == 1 and len(y) == 2):
                    raise TypeError("Expected `y` to be (func_val, t)")

        #################### Standard Acquisition Function ####################
        # If `y` isn't a scalar, we have been handed a batch of points
        elif is_list_like(y) and is_2d_list_like(x):
            for y_value in y:
                if not isinstance(y_value, Number):
                    raise ValueError("Expected `y` to be a list of scalars")
        elif is_list_like(x):
            if not isinstance(y, Number):
                raise ValueError("`func` should return a scalar")
        else:
            raise ValueError(f"Incompatible argument types: `x` ({type(x)}) and `y` ({type(y)})")

    def run(self, func, n_iter=1):
        """Execute :meth:`ask` + :meth:`tell` loop for `n_iter` iterations

        Parameters
        ----------
        func: Callable
            Function that returns the objective value `y`, when given a search point `x`
        n_iter: Int, default=1
            Number of `ask`/`tell` sequences to execute

        Returns
        -------
        OptimizeResult
            `scipy.optimize.OptimizeResult` instance"""
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))

        return create_result(self.Xi, self.yi, self.space, self.rng, models=self.models)


##################################################
# Utilities
##################################################
def is_list_like(x):
    """Determine whether a point is list-like

    Parameters
    ----------
    x: List
        Some point to check for list-likeness

    Returns
    -------
    Boolean
        True if `x` is list-like. Else False"""
    return isinstance(x, (list, tuple))


def is_2d_list_like(x):
    """Determine whether a point is 2-dimensional list-like

    Parameters
    ----------
    x: List
        Some point to check for 2D list-likeness

    Returns
    -------
    Boolean
        True if `x` is 2D list-like. Else False"""
    return np.all([is_list_like(_) for _ in x])


def check_x_in_space(x, space):
    """Check that an arbitrary point, or list of points, fits within the bounds of `space`

    Parameters
    ----------
    x: List
        Some point (or list of points), whose compatibility with `space` will be checked. If `x` is
        a collection of multiple points, it should be a list of lists
    space: Space
        Instance of :class:`hyperparameter_hunter.space.space_core.Space` that defines the
        dimensions and bounds within which `x` should fit

    Raises
    ------
    ValueError
        If `x` is incompatible with `space` for any reason"""
    if is_2d_list_like(x):
        if not np.all([p in space for p in x]):
            raise ValueError("Not all points are within the bounds of the space")
        if any([len(p) != len(space.dimensions) for p in x]):
            raise ValueError("Not all points have the same dimensions as the space")
    elif is_list_like(x):
        if x not in space:
            raise ValueError(f"Point {x} is not within the bounds of the space ({space.bounds})")
        if len(x) != len(space.dimensions):
            raise ValueError(f"Dimensions of point {x} and space ({space.bounds}) do not match")


def cook_estimator(base_estimator, space=None, **kwargs):
    """Cook a default estimator

    For the special `base_estimator` called "DUMMY", the return value is None. This corresponds to
    sampling points at random, hence there is no need for an estimator

    Parameters
    ----------
    base_estimator: {SKLearn Regressor, "GP", "RF", "ET", "GBRT", "DUMMY"}, default="GP"
        If not string, should inherit from `sklearn.base.RegressorMixin`. In addition, the `predict`
        method should have an optional `return_std` argument, which returns `std(Y | x)`,
        along with `E[Y | x]`.

        If `base_estimator` is a string in {"GP", "RF", "ET", "GBRT", "DUMMY"}, a surrogate model
        corresponding to the relevant `X_minimize` function is created
    space: `hyperparameter_hunter.space.space_core.Space`
        Required only if the `base_estimator` is a Gaussian Process. Ignored otherwise
    **kwargs: Dict
        Extra parameters provided to the `base_estimator` at initialization time

    Returns
    -------
    SKLearn Regressor
        Regressor instance cooked up according to `base_estimator` and `kwargs`"""
    #################### Validate `base_estimator` ####################
    str_estimators = ["GP", "ET", "RF", "GBRT", "DUMMY"]
    if isinstance(base_estimator, str):
        if base_estimator.upper() not in str_estimators:
            raise ValueError(f"Expected `base_estimator` in {str_estimators}. Got {base_estimator}")
        # Convert to upper after error check, so above error shows actual given `base_estimator`
        base_estimator = base_estimator.upper()
    elif not is_regressor(base_estimator):
        raise ValueError("`base_estimator` must be a regressor")

    #################### Get Cooking ####################
    if base_estimator == "GP":
        if space is not None:
            space = Space(space)
            # NOTE: Below `normalize_dimensions` is NOT an unnecessary duplicate of the call in
            #   `Optimizer` - `Optimizer` calls `cook_estimator` before its `dimensions` have been
            #   normalized, so `normalize_dimensions` must also be called here
            space = Space(normalize_dimensions(space.dimensions))
            n_dims = space.transformed_n_dims
            is_cat = space.is_categorical
        else:
            raise ValueError("Expected a `Space` instance, not None")

        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        # Only special if *all* dimensions are `Categorical`
        if is_cat:
            other_kernel = HammingKernel(length_scale=np.ones(n_dims))
        else:
            other_kernel = Matern(
                length_scale=np.ones(n_dims), length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5
            )

        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True,
            noise="gaussian",
            n_restarts_optimizer=2,
        )
    elif base_estimator == "RF":
        base_estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
    elif base_estimator == "ET":
        base_estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3)
    elif base_estimator == "GBRT":
        gbrt = GradientBoostingRegressor(n_estimators=30, loss="quantile")
        base_estimator = GradientBoostingQuantileRegressor(base_estimator=gbrt)
    elif base_estimator == "DUMMY":
        return None

    base_estimator.set_params(**kwargs)
    return base_estimator
