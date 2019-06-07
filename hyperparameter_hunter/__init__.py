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
# Store Library Version
##################################################
import os.path

try:
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        __version__ = f.read().strip()
except Exception:
    raise

##################################################
# Set __all__
##################################################
from .environment import Environment
from .experiments import CVExperiment
from .optimization import BayesianOptimization
from .optimization import GradientBoostedRegressionTreeOptimization, GBRT
from .optimization import RandomForestOptimization, RF
from .optimization import ExtraTreesOptimization, ET
from .optimization import DummySearch
from .space import Real
from .space import Integer
from .space import Categorical
from .callbacks.bases import lambda_callback
from .feature_engineering import FeatureEngineer
from .feature_engineering import EngineerStep

__all__ = [
    #################### Environment ####################
    "Environment",
    #################### Experimentation ####################
    "CVExperiment",
    #################### Hyperparameter Optimization ####################
    "BayesianOptimization",
    "GradientBoostedRegressionTreeOptimization",
    "GBRT",
    "RandomForestOptimization",
    "RF",
    "ExtraTreesOptimization",
    "ET",
    "DummySearch",
    #################### Search Space ####################
    "Real",
    "Integer",
    "Categorical",
    #################### Callbacks ####################
    "lambda_callback",
    #################### Feature Engineering ####################
    "FeatureEngineer",
    "EngineerStep",
]
