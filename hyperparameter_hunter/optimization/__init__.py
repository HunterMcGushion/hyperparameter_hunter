##################################################
# Set __all__
##################################################
from .backends.skopt.protocols import BayesianOptPro
from .backends.skopt.protocols import GradientBoostedRegressionTreeOptPro, GBRT
from .backends.skopt.protocols import RandomForestOptPro, RF
from .backends.skopt.protocols import ExtraTreesOptPro, ET
from .backends.skopt.protocols import DummyOptPro

#################### Deprecated OptPros - Remove in 3.2.0 ####################
from .backends.skopt.protocols import BayesianOptimization
from .backends.skopt.protocols import GradientBoostedRegressionTreeOptimization
from .backends.skopt.protocols import RandomForestOptimization
from .backends.skopt.protocols import ExtraTreesOptimization
from .backends.skopt.protocols import DummySearch

__all__ = [
    #################### Optimization Protocols ####################
    "BayesianOptPro",
    "GradientBoostedRegressionTreeOptPro",
    "GBRT",
    "RandomForestOptPro",
    "RF",
    "ExtraTreesOptPro",
    "ET",
    "DummyOptPro",
    #################### Deprecated OptPros - Remove in 3.2.0 ####################
    "BayesianOptimization",
    "GradientBoostedRegressionTreeOptimization",
    "RandomForestOptimization",
    "ExtraTreesOptimization",
    "DummySearch",
]
