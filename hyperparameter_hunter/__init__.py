##################################################
# Nullify Excess Documentation
##################################################
from .importer import nullify_module_docstrings, hook_keras_layer

# try:
#     importer.nullify_module_docstrings('hyperparameter_hunter.utils.boltons_utils')
# except Exception:
#     pass

##################################################
# Execute Import Interceptors
##################################################
try:
    hook_keras_layer()
except Exception as _ex:
    # TODO: Probably need to raise only certain exceptions - If keras is even available to import
    # TODO: If keras isn't available at all, `pass` - Wasn't installed by user
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
