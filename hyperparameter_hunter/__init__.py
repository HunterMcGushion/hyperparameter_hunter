##################################################
# Nullify Excess Documentation
##################################################
from .importer import nullify_module_docstrings

# try:
#     importer.nullify_module_docstrings('hyperparameter_hunter.utils.boltons_utils')
# except Exception:
#     pass 

##################################################
# Execute Import Interceptors
##################################################
# TODO: importer.hook_keras_layer


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

__all__ = [
    'Environment',

    'CrossValidationExperiment',

    'BayesianOptimization',
    'GradientBoostedRegressionTreeOptimization',
    'RandomForestOptimization',
    'ExtraTreesOptimization',
    'DummySearch',

    'Real',
    'Integer',
    'Categorical',
]
