"""This module defines the OptPro (Optimization Protocol) classes that are intended for direct use.
All classes defined herein should be descendants of one of the base classes defined in
:mod:`hyperparameter_hunter.optimization.protocol_core`

Related
-------
:mod:`hyperparameter_hunter.optimization.protocol_core`
    Defines the base Optimization Protocol classes from which the classes in
    :mod:`hyperparameter_hunter.optimization.backends.skopt.protocols` are descendants"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import __version__
from hyperparameter_hunter.optimization.protocol_core import SKOptPro
from hyperparameter_hunter.utils.version_utils import Deprecated

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
from typing import Optional, Type, Union

##################################################
# Import Learning Assets
##################################################
from sklearn.base import BaseEstimator
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor


EstTypes = Union[str, Type[BaseEstimator]]


def _validate_estimator(estimator: EstTypes, *valid_values: EstTypes) -> Optional[bool]:
    # TODO: Add docstring
    for valid_value in valid_values:
        if isinstance(valid_value, str) and estimator == valid_value.upper():
            return True
        if isinstance(valid_value, type) and isinstance(estimator, valid_value):
            return True

    raise TypeError(f"Expected `base_estimator` in {valid_values}, not {estimator}")


##################################################
# SKOpt-Based Optimization Protocols
##################################################
class BayesianOptPro(SKOptPro):
    """Bayesian optimization with Gaussian Processes"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
        base_estimator="GP",
        n_initial_points=10,
        acquisition_function="gp_hedge",
        acquisition_optimizer="auto",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts="DEPRECATED",
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        _validate_estimator(base_estimator, "GP", GaussianProcessRegressor)
        base_estimator_kwargs = base_estimator_kwargs or {}
        base_estimator_kwargs.setdefault("noise", "gaussian")

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
            warn_on_re_ask=warn_on_re_ask,
            base_estimator=base_estimator,
            n_initial_points=n_initial_points,
            acquisition_function=acquisition_function,
            acquisition_optimizer=acquisition_optimizer,
            random_state=random_state,
            acquisition_function_kwargs=acquisition_function_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            n_random_starts=n_random_starts,
            callbacks=callbacks,
            base_estimator_kwargs=base_estimator_kwargs,
        )


class GradientBoostedRegressionTreeOptPro(SKOptPro):
    """Sequential optimization with gradient boosted regression trees"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
        base_estimator="GBRT",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts="DEPRECATED",
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        _validate_estimator(base_estimator, "GBRT", GradientBoostingQuantileRegressor)
        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
            warn_on_re_ask=warn_on_re_ask,
            base_estimator=base_estimator,
            n_initial_points=n_initial_points,
            acquisition_function=acquisition_function,
            acquisition_optimizer=acquisition_optimizer,
            random_state=random_state,
            acquisition_function_kwargs=acquisition_function_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            n_random_starts=n_random_starts,
            callbacks=callbacks,
            base_estimator_kwargs=base_estimator_kwargs,
        )


class RandomForestOptPro(SKOptPro):
    """Sequential optimization with random forest regressor decision trees"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
        base_estimator="RF",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts="DEPRECATED",
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        _validate_estimator(base_estimator, "RF", RandomForestRegressor)
        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
            warn_on_re_ask=warn_on_re_ask,
            base_estimator=base_estimator,
            n_initial_points=n_initial_points,
            acquisition_function=acquisition_function,
            acquisition_optimizer=acquisition_optimizer,
            random_state=random_state,
            acquisition_function_kwargs=acquisition_function_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            n_random_starts=n_random_starts,
            callbacks=callbacks,
            base_estimator_kwargs=base_estimator_kwargs,
        )


class ExtraTreesOptPro(SKOptPro):
    """Sequential optimization with extra trees regressor decision trees"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
        base_estimator="ET",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts="DEPRECATED",
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        _validate_estimator(base_estimator, "ET", ExtraTreesRegressor)
        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
            warn_on_re_ask=warn_on_re_ask,
            base_estimator=base_estimator,
            n_initial_points=n_initial_points,
            acquisition_function=acquisition_function,
            acquisition_optimizer=acquisition_optimizer,
            random_state=random_state,
            acquisition_function_kwargs=acquisition_function_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            n_random_starts=n_random_starts,
            callbacks=callbacks,
            base_estimator_kwargs=base_estimator_kwargs,
        )


class DummyOptPro(SKOptPro):
    """Random search by uniform sampling"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        warn_on_re_ask=False,
        base_estimator="DUMMY",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts="DEPRECATED",
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        _validate_estimator(base_estimator, "DUMMY")
        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
            warn_on_re_ask=warn_on_re_ask,
            base_estimator=base_estimator,
            n_initial_points=n_initial_points,
            acquisition_function=acquisition_function,
            acquisition_optimizer=acquisition_optimizer,
            random_state=random_state,
            acquisition_function_kwargs=acquisition_function_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            n_random_starts=n_random_starts,
            callbacks=callbacks,
            base_estimator_kwargs=base_estimator_kwargs,
        )


##################################################
# Optimization Protocol Aliases
##################################################
GBRT = GradientBoostedRegressionTreeOptPro
RF = RandomForestOptPro
ET = ExtraTreesOptPro

##################################################
# Deprecated Optimization Protocols
##################################################
# Below classes have been renamed to the primary classes above. Scheduled to be removed in 3.2.0
_DeprecatedOptPro = partial(
    Deprecated, v_deprecate="3.0.0a2", v_remove="3.2.0", v_current=__version__
)


@_DeprecatedOptPro(details="Renamed to `BayesianOptPro`")
class BayesianOptimization(BayesianOptPro):
    ...


@_DeprecatedOptPro(details="Renamed to `GradientBoostedRegressionTreeOptPro`")
class GradientBoostedRegressionTreeOptimization(GradientBoostedRegressionTreeOptPro):
    ...


@_DeprecatedOptPro(details="Renamed to `RandomForestOptPro`")
class RandomForestOptimization(RandomForestOptPro):
    ...


@_DeprecatedOptPro(details="Renamed to `ExtraTreesOptPro`")
class ExtraTreesOptimization(ExtraTreesOptPro):
    ...


@_DeprecatedOptPro(details="Renamed to `DummyOptPro`")
class DummySearch(DummyOptPro):
    ...


##################################################
# Unimplemented Optimization Protocols
##################################################
# class TreeStructuredParzenEstimatorsOptPro(SKOptPro):
#     # FLAG: http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html#id24
#     pass


# class EvolutionaryOptPro(SKOptPro):
#     # FLAG: See TPOT's Genetic Programming approach
#     pass


# class ParticleSwarmOptPro(SKOptPro):
#     # FLAG: ...
#     pass


if __name__ == "__main__":
    pass
