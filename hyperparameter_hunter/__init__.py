##################################################
# Store Library Version
##################################################
import os.path

try:
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        __version__ = f.read().strip()
except Exception:
    raise

##################################################
# Execute Import Interceptors
##################################################
from pkg_resources import DistributionNotFound
from .importer import hook_keras_layer, hook_keras_initializers

try:
    hook_keras_layer()
    hook_keras_initializers()
except DistributionNotFound:
    pass
except Exception:
    raise

##################################################
# Set __all__
##################################################
from .environment import Environment
from .experiments import CVExperiment
from .optimization.backends.skopt.protocols import BayesianOptPro
from .optimization.backends.skopt.protocols import GradientBoostedRegressionTreeOptPro, GBRT
from .optimization.backends.skopt.protocols import RandomForestOptPro, RF
from .optimization.backends.skopt.protocols import ExtraTreesOptPro, ET
from .optimization.backends.skopt.protocols import DummyOptPro
from .space.dimensions import Real
from .space.dimensions import Integer
from .space.dimensions import Categorical
from .callbacks.bases import lambda_callback
from .feature_engineering import FeatureEngineer
from .feature_engineering import EngineerStep

#################### Deprecated OptPros - Remove in 3.2.0 ####################
from .optimization.backends.skopt.protocols import BayesianOptimization
from .optimization.backends.skopt.protocols import GradientBoostedRegressionTreeOptimization
from .optimization.backends.skopt.protocols import RandomForestOptimization
from .optimization.backends.skopt.protocols import ExtraTreesOptimization
from .optimization.backends.skopt.protocols import DummySearch

__all__ = [
    #################### Environment ####################
    "Environment",
    #################### Experimentation ####################
    "CVExperiment",
    #################### Hyperparameter Optimization ####################
    "BayesianOptPro",
    "GradientBoostedRegressionTreeOptPro",
    "GBRT",
    "RandomForestOptPro",
    "RF",
    "ExtraTreesOptPro",
    "ET",
    "DummyOptPro",
    #################### Search Space ####################
    "Real",
    "Integer",
    "Categorical",
    #################### Callbacks ####################
    "lambda_callback",
    #################### Feature Engineering ####################
    "FeatureEngineer",
    "EngineerStep",
    #################### Miscellaneous ####################
    "__version__",
    #################### Deprecated OptPros - Remove in 3.2.0 ####################
    "BayesianOptimization",
    "GradientBoostedRegressionTreeOptimization",
    "RandomForestOptimization",
    "ExtraTreesOptimization",
    "DummySearch",
]
