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
from hyperparameter_hunter.optimization_core import (
    InformedOptimizationProtocol,
    UninformedOptimizationProtocol,
)
from hyperparameter_hunter.space import normalize_dimensions

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np

##################################################
# Import Learning Assets
##################################################
from sklearn.model_selection import ParameterGrid, ParameterSampler
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.learning.forest import RandomForestRegressor, ExtraTreesRegressor


##################################################
# Informed Optimization Protocols
##################################################
class BayesianOptimization(InformedOptimizationProtocol):
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


class GradientBoostedRegressionTreeOptimization(InformedOptimizationProtocol):
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


class RandomForestOptimization(InformedOptimizationProtocol):
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


class ExtraTreesOptimization(InformedOptimizationProtocol):
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


class DummySearch(InformedOptimizationProtocol):
    """Random search by uniform sampling. Technically this is not "Informed", but it fits better as
    an Informed subclass due to its reliance on `Scikit-Optimize`"""

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


class TreeStructuredParzenEstimatorsOptimization(InformedOptimizationProtocol):
    # FLAG: http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html#id24
    pass


class EvolutionaryOptimization(InformedOptimizationProtocol):
    # FLAG: See TPOT's Genetic Programming approach
    pass


class ParticleSwarmOptimization(InformedOptimizationProtocol):
    # FLAG: ...
    pass


##################################################
# Uninformed Optimization Protocols
##################################################
class GridSearch(UninformedOptimizationProtocol):
    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
    ):
        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
        )

    def _set_hyperparameter_space(self):
        self.hyperparameter_space = ParameterGrid(self.search_bounds).__iter__()

    @property
    def search_space_size(self):
        if self._search_space_size is None:
            self._search_space_size = len(self.hyperparameter_space)
        return self._search_space_size


class RandomizedGridSearch(UninformedOptimizationProtocol):
    def __init__(
        self,
        target_metric=None,
        iterations=1,
        verbose=1,
        read_experiments=True,
        reporter_parameters=None,
    ):
        super().__init__(
            target_metric=target_metric,
            iterations=iterations,
            verbose=verbose,
            read_experiments=read_experiments,
            reporter_parameters=reporter_parameters,
        )

    def _set_hyperparameter_space(self):
        # FLAG: Might be more efficient to use ParameterGrid with __getitem__ because ParameterSampler repeats keys
        self.hyperparameter_space = ParameterSampler(
            self.search_bounds, n_iter=self.iterations
        ).__iter__()

    @property
    def search_space_size(self):
        if self._search_space_size is None:
            if np.any([hasattr(_, "rvs") for _ in self.search_bounds.values()]):
                self._search_space_size = np.inf
            else:
                self._search_space_size = len(ParameterGrid(self.search_bounds))
        return self._search_space_size


if __name__ == "__main__":
    pass
