"""This module defines the Optimization Protocol classes that are intended for direct use. All
classes defined herein should be descendants of one of the base classes defined in
:mod:`hyperparameter_hunter.optimization_core`

Related
-------
:mod:`hyperparameter_hunter.optimization_core`
    Defines the base Optimization Protocol classes from which the classes in
    :mod:`hyperparameter_hunter.optimization` are descendants"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.optimization_core import SKOptimizationProtocol
from hyperparameter_hunter.space import normalize_dimensions

##################################################
# Import Learning Assets
##################################################
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor


##################################################
# SKOpt-Based Optimization Protocols
##################################################
class BayesianOptimization(SKOptimizationProtocol):
    """Bayesian optimization with Gaussian Processes"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        base_estimator="GP",
        n_initial_points=10,
        acquisition_function="gp_hedge",
        acquisition_optimizer="auto",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts=10,
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        if base_estimator.upper() != "GP" and not isinstance(
            base_estimator, GaussianProcessRegressor
        ):
            raise TypeError(
                f'Expected `base_estimator`="GP", or `GaussianProcessRegressor`, not {base_estimator}'
            )

        base_estimator_kwargs = base_estimator_kwargs or {}
        base_estimator_kwargs.setdefault("noise", "gaussian")

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
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

    def go(self):
        self.dimensions = normalize_dimensions(self.dimensions)
        super().go()


class GradientBoostedRegressionTreeOptimization(SKOptimizationProtocol):
    """Sequential optimization with gradient boosted regression trees"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        base_estimator="GBRT",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts=10,
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        if base_estimator.upper() != "GBRT" and not isinstance(
            base_estimator, GradientBoostingQuantileRegressor
        ):
            raise TypeError(
                f'Expected `base_estimator`="GBRT", or `GradientBoostingQuantileRegressor`, not {base_estimator}'
            )

        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
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


class RandomForestOptimization(SKOptimizationProtocol):
    """Sequential optimization with random forest regressor decision trees"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        base_estimator="RF",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts=10,
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        if base_estimator.upper() != "RF" and not isinstance(base_estimator, RandomForestRegressor):
            raise TypeError(
                f'Expected `base_estimator`="RF", or `RandomForestRegressor`, not {base_estimator}'
            )

        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
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


class ExtraTreesOptimization(SKOptimizationProtocol):
    """Sequential optimization with extra trees regressor decision trees"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        base_estimator="ET",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts=10,
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        if base_estimator.upper() != "ET" and not isinstance(base_estimator, ExtraTreesRegressor):
            raise TypeError(
                f'Expected `base_estimator`="ET", or `ExtraTreesRegressor`, not {base_estimator}'
            )

        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
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


class DummySearch(SKOptimizationProtocol):
    """Random search by uniform sampling"""

    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
        base_estimator="DUMMY",
        n_initial_points=10,
        acquisition_function="EI",
        acquisition_optimizer="sampling",
        random_state=32,
        acquisition_function_kwargs=None,
        acquisition_optimizer_kwargs=None,
        n_random_starts=10,
        callbacks=None,
        base_estimator_kwargs=None,
    ):
        if base_estimator.upper() != "DUMMY":
            raise TypeError(f'Expected `base_estimator`="DUMMY", not {base_estimator}')

        base_estimator_kwargs = base_estimator_kwargs or {}

        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
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
GBRT = GradientBoostedRegressionTreeOptimization
RF = RandomForestOptimization
ET = ExtraTreesOptimization


##################################################
# Unimplemented Optimization Protocols
##################################################
class TreeStructuredParzenEstimatorsOptimization(SKOptimizationProtocol):
    # FLAG: http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html#id24
    pass


class EvolutionaryOptimization(SKOptimizationProtocol):
    # FLAG: See TPOT's Genetic Programming approach
    pass


class ParticleSwarmOptimization(SKOptimizationProtocol):
    # FLAG: ...
    pass


if __name__ == "__main__":
    pass
