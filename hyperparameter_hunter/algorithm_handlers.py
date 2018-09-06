##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
from inspect import signature
import numpy as np


def identify_algorithm(model_initializer):
    """Determine the name, and module of the algorithm provided by `model_initializer`

    Parameters
    ----------
    model_initializer: functools.partial, or class, or class instance
        The algorithm class being used to initialize a model

    Returns
    -------
    algorithm_name: str
        The name of the algorithm provided by `model_initializer`
    module_name: str
        The name of the module housing the algorithm provided by `model_initializer`

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from sklearn.cluster import DBSCAN, SpectralClustering
    >>> from functools import partial
    >>> identify_algorithm(XGBClassifier)
    ('XGBClassifier', 'xgboost')
    >>> identify_algorithm(DBSCAN())
    ('DBSCAN', 'sklearn')
    >>> identify_algorithm(partial(SpectralClustering))
    ('SpectralClustering', 'sklearn')
    """
    # FLAG: Will need different way to handle neural network libraries (keras, pytorch, skorch)

    try:
        if isinstance(model_initializer, partial):
            algorithm_name = model_initializer.func.__name__
        else:
            algorithm_name = model_initializer.__name__
    except AttributeError:
        algorithm_name = type(model_initializer).__name__

    try:
        module_name = model_initializer.__module__.split(".")[0]
    except AttributeError:
        module_name = model_initializer.func.__module__.split(".")[0]

    return algorithm_name, module_name


def identify_algorithm_hyperparameters(model_initializer):  # FLAG: Play nice with Keras
    """Determine keyword-arguments accepted by `model_initializer`, along with their default values

    Parameters
    ----------
    model_initializer: functools.partial, or class, or class instance
        The algorithm class being used to initialize a model

    Returns
    -------
    hyperparameter_defaults: dict
        The dict of kwargs accepted by `model_initializer` and their default values"""

    hyperparameter_defaults = dict()

    # FLAG: Play nice with Keras
    try:
        signature_parameters = signature(model_initializer).parameters
    except TypeError:
        signature_parameters = signature(model_initializer.__class__).parameters

    for k, v in signature_parameters.items():
        if (v.kind == v.KEYWORD_ONLY) or (v.kind == v.POSITIONAL_OR_KEYWORD):
            hyperparameter_defaults[k] = v.default

    return hyperparameter_defaults


def identify_hyperparameter_choices(algorithm_name, module_name, hyperparameter_defaults):
    choices = dict()
    for hyperparameter, default_value in hyperparameter_defaults.items():
        if isinstance(default_value, (int, float)):
            # Likely Continuous Feature
            choices[hyperparameter] = dict(
                lower_bound=-np.inf,  # FLAG: Might be safer to set to 0
                # FLAG: If sticking with -np.inf, wrap optimization rounds in try/except. If fail, try raising lower_bound to 0
                upper_bound=np.inf,
            )
        elif isinstance(default_value, bool):
            # Likely Binary Feature
            choices[hyperparameter] = dict(
                select=[True, False]
                # FLAG: May be other possible values mentioned in docstring, though...
            )
        elif isinstance(default_value, str):
            # Likely Categorical Feature
            choices[hyperparameter] = dict(
                select=[default_value]
                # FLAG: Other types may be possible (like callable)
                # FLAG: Will need to manually define options for all parameters, or rely on user to reveal different options
            )
        else:
            # Likely Categorical Feature
            # FLAG: Other values that should be expected are: dict, None, callable, other object from library
            choices[hyperparameter] = dict(
                select=[default_value]
                # FLAG: Will need to handle this like other categorical features
            )


if __name__ == "__main__":
    pass
