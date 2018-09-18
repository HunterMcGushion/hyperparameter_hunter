##################################################
# Execute Import Interceptors
##################################################
from pkg_resources import DistributionNotFound
from .importer import hook_keras_layer

try:
    hook_keras_layer()
except DistributionNotFound:
    pass
except Exception as _ex:
    raise


##################################################
# Set __all__
##################################################
from .environment import Environment
from .experiments import CrossValidationExperiment
from .optimization import BayesianOptimization
from .optimization import GradientBoostedRegressionTreeOptimization
from .optimization import RandomForestOptimization
from .optimization import ExtraTreesOptimization
from .optimization import DummySearch
from .space import Real
from .space import Integer
from .space import Categorical
from .callbacks.bases import lambda_callback

__all__ = [
    #################### Environment ####################
    "Environment",
    #################### Experimentation ####################
    "CrossValidationExperiment",
    #################### Hyperparameter Optimization ####################
    "BayesianOptimization",
    "GradientBoostedRegressionTreeOptimization",
    "RandomForestOptimization",
    "ExtraTreesOptimization",
    "DummySearch",
    #################### Search Space ####################
    "Real",
    "Integer",
    "Categorical",
    #################### Callbacks ####################
    "lambda_callback",
]
