"""This module defines utilities for assisting in processing Keras Experiments"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
# noinspection PyProtectedMember
from inspect import signature, _empty
from types import MethodType


def keras_callback_to_key(callback):
    """Convert a `Keras` callback instance to a string that identifies it, along with the parameters
     used to create it

    Parameters
    ----------
    callback: Child instance of `keras.callbacks.Callback`
        A Keras callback for which a key string describing it will be created

    Returns
    -------
    callback_key: String
        A string identifying and describing `callback`"""
    signature_args = sorted(signature(callback.__class__).parameters.items())
    string_args = []

    for arg_name, arg_val in signature_args:
        if arg_name not in ["verbose"]:
            try:
                string_args.append(f"{arg_name}={getattr(callback, arg_name)!r}")
            except AttributeError:
                string_args.append(f"{arg_name}={arg_val.default!r}")

    callback_key = f"{callback.__class__.__name__}(" + ", ".join(string_args) + ")"
    return callback_key


def keras_callback_to_dict(callback):
    """Convert a `Keras` callback instance to a dict that identifies it, along with the parameters
    used to create it

    Parameters
    ----------
    callback: Child instance of `keras.callbacks.Callback`
        A Keras callback for which a dict describing it will be created

    Returns
    -------
    callback_dict: Dict
        A dict identifying and describing `callback`"""
    signature_args = sorted(signature(callback.__class__).parameters.items())
    callback_dict = dict(class_name=callback.__class__.__name__)

    for arg_name, arg_val in signature_args:
        if arg_name not in ["verbose"]:
            try:
                temp_val = getattr(callback, arg_name)
                callback_dict[arg_name] = temp_val if temp_val is not _empty else None
            except AttributeError:
                callback_dict[arg_name] = arg_val.default if arg_val.default is not _empty else None

    return callback_dict


def reinitialize_callbacks(callbacks):
    """Ensures the contents of `callbacks` are valid Keras callbacks

    Parameters
    ----------
    callbacks: List
        Expected to contain Keras callbacks, or dicts describing callbacks

    Returns
    -------
    callbacks: List
        A validated list of Keras callbacks"""

    if len(callbacks) > 0:
        for i in range(len(callbacks)):
            current_callback = callbacks[i]

            if not isinstance(current_callback, dict):
                continue

            callback_initializer = next(
                _ for _ in current_callback.values() if isinstance(_, MethodType)
            ).__self__.__class__
            callback_parameters = list(signature(callback_initializer).parameters)
            callbacks[i] = callback_initializer(
                **{_: current_callback.get(_, None) for _ in callback_parameters}
            )
    return callbacks


sentinel_default_value = object()


def get_keras_attr(model, attr, max_depth=3, default=sentinel_default_value):
    """Retrieve specific Keras model attributes safely across different versions of Keras

    Parameters
    ----------
    model: Instance of :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        A compiled instance of a Keras model, made using the Keras `wrappers.scikit_learn` module
    attr: String
        Name of the attribute to retrieve from `model`
    max_depth: Integer, default=3
        Maximum number of times to check the "model" attribute of `model` for the target `attr` if
        `attr` itself is not in `model` before returning `default` or raising AttributeError
    default: Object, default=object()
        If given, `default` will be returned once `max_depth` attempts have been made to find `attr`
        in `model`. If not given and total attempts exceed `max_depth`, AttributeError is raised

    Returns
    -------
    Object
        Value of `attr` for `model` (or a nested `model` if necessary), or None"""
    try:
        return getattr(model, attr)
    except AttributeError:  # Keras<2.2.0 has these attributes deeper in `model`
        if max_depth > 0 and hasattr(model, "model"):
            return get_keras_attr(model.model, attr, max_depth=max_depth - 1, default=default)
        elif default is not sentinel_default_value:
            return default
        raise


def parameterize_compiled_keras_model(model):
    """Traverse a compiled Keras model to gather critical information about the layers used to
    construct its architecture, and the parameters used to compile it

    Parameters
    ----------
    model: Instance of :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        A compiled instance of a Keras model, made using the Keras `wrappers.scikit_learn` module.
        This must be a completely valid Keras model, which means that it often must be the result
        of :func:`library_helpers.keras_optimization_helper.initialize_dummy_model`. Using the
        resulting dummy model ensures the model will pass Keras checks that would otherwise reject
        instances of `space.Space` descendants used to provide hyperparameter choices

    Returns
    -------
    layers: List
        A list containing a dict for each layer found in the architecture of `model`. A layer dict
        should contain the following keys: ['class_name', '__hh_default_args',
        '__hh_default_kwargs', '__hh_used_args', '__hh_used_kwargs']
    compile_params: Dict
        The parameters used on the call to :meth:`model.compile`. If a value for a certain parameter
        was not explicitly provided, its default value will be included in `compile_params`"""
    # NOTE: Tested optimizer and loss with both callable and string inputs - Converted to callables automatically
    ##################################################
    # Model Compile Parameters
    ##################################################
    compile_params = dict()

    compile_params["optimizer"] = get_keras_attr(model, "optimizer").__class__.__name__.lower()
    compile_params["optimizer_params"] = get_keras_attr(model, "optimizer").get_config()

    compile_params["metrics"] = get_keras_attr(model, "metrics")
    compile_params["metrics_names"] = get_keras_attr(model, "metrics_names")

    compile_params["loss_functions"] = get_keras_attr(model, "loss_functions")
    compile_params["loss_function_names"] = [_.__name__ for _ in compile_params["loss_functions"]]

    # FLAG: BELOW PARAMETERS SHOULD ONLY BE DISPLAYED IF EXPLICITLY GIVEN (probably have to be in key by default, though):
    compile_params["loss_weights"] = get_keras_attr(model, "loss_weights")
    compile_params["sample_weight_mode"] = get_keras_attr(model, "sample_weight_mode")
    compile_params["weighted_metrics"] = get_keras_attr(model, "weighted_metrics")

    compile_params["target_tensors"] = get_keras_attr(model, "target_tensors", default=None)
    compile_params["compile_kwargs"] = get_keras_attr(model, "_function_kwargs")

    ##################################################
    # Model Architecture
    ##################################################
    hh_attributes = [
        "__hh_default_args",
        "__hh_default_kwargs",
        "__hh_used_args",
        "__hh_used_kwargs",
    ]
    layers = []

    for layer in get_keras_attr(model, "layers"):
        layer_obj = dict(class_name=layer.__class__.__name__)

        for hh_attr in hh_attributes:
            layer_obj[hh_attr] = getattr(layer, hh_attr, None)

        layers.append(layer_obj)

    ##################################################
    # Handle Custom Losses/Optimizers
    ##################################################
    if any([_.__module__ != "keras.losses" for _ in compile_params["loss_functions"]]):
        G.warn(
            "Custom loss functions will not be hashed and saved, meaning they are identified only by their names."
            + "\nIf you plan on tuning loss functions at all, please ensure custom functions are not given the same names as any"
            + " of Keras's loss functions. Otherwise, naming conflicts may occur and make results very confusing."
        )
    if get_keras_attr(model, "optimizer").__module__ != "keras.optimizers":
        G.warn(
            "Custom optimizers will not be hashed and saved, meaning they are identified only by their names."
            + "\nIf you plan on tuning optimizers at all, please ensure custom optimizers are not given the same names as any"
            + " of Keras's optimizers. Otherwise, naming conflicts may occur and make results very confusing."
        )

    return layers, compile_params


if __name__ == "__main__":
    pass
