##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
from inspect import signature


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
    >>> from sklearn.cluster import DBSCAN, SpectralClustering
    >>> from functools import partial
    >>> identify_algorithm(DBSCAN)
    ('DBSCAN', 'sklearn')
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


if __name__ == "__main__":
    pass
