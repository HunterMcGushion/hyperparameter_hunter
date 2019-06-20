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
from .optimization import BayesianOptPro
from .optimization import GradientBoostedRegressionTreeOptPro, GBRT
from .optimization import RandomForestOptPro, RF
from .optimization import ExtraTreesOptPro, ET
from .optimization import DummyOptPro
from .space import Real
from .space import Integer
from .space import Categorical
from .callbacks.bases import lambda_callback
from .feature_engineering import FeatureEngineer
from .feature_engineering import EngineerStep

#################### Deprecated OptPros - Remove in 3.2.0 ####################
from .optimization import BayesianOptimization
from .optimization import GradientBoostedRegressionTreeOptimization
from .optimization import RandomForestOptimization
from .optimization import ExtraTreesOptimization
from .optimization import DummySearch

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
