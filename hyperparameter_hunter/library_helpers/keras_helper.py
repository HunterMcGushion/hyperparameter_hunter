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


def parameterize_compiled_keras_model(model):
    """Traverse a compiled Keras model to gather critical information about the layers used to
    construct its architecture, and the parameters used to compile it

    Parameters
    ----------
    model: Instance of :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        A compiled instance of a Keras model, made using the Keras `wrappers.scikit_learn` module

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

    # TODO: MIGHT NEED TO CHECK KERAS VERSION...
    # TODO: If the "TEST" lines below don't work for older Keras versions, add check here to set `model = model.model`...
    # TODO: ... For newer Keras versions, but leave it alone for older versions

    ##################################################
    # Model Compile Parameters
    ##################################################
    compile_params = dict()
    # compile_params['optimizer'] = model.optimizer.__class__.__name__  # -> 'Adam'  # FLAG: ORIGINAL
    compile_params["optimizer"] = model.model.optimizer.__class__.__name__.lower()  # FLAG: TEST
    # compile_params['optimizer_params'] = model.optimizer.get_config()  # -> {**kwargs}  # FLAG: ORIGINAL
    compile_params["optimizer_params"] = model.model.optimizer.get_config()  # FLAG: TEST

    # compile_params['metrics'] = model.metrics  # -> ['accuracy']  # FLAG: ORIGINAL
    compile_params["metrics"] = model.model.metrics  # FLAG: TEST
    # compile_params['metrics_names'] = model.metrics_names  # -> ['loss', 'acc']  # FLAG: ORIGINAL
    compile_params["metrics_names"] = model.model.metrics_names  # FLAG: TEST

    compile_params["loss_functions"] = model.model.loss_functions
    compile_params["loss_function_names"] = [_.__name__ for _ in compile_params["loss_functions"]]

    # FLAG: BELOW PARAMETERS SHOULD ONLY BE DISPLAYED IF EXPLICITLY GIVEN (probably have to be in key by default, though):
    # compile_params['loss_weights'] = model.loss_weights  # -> None, [], or {}  # FLAG: ORIGINAL
    compile_params["loss_weights"] = model.model.loss_weights  # FLAG: TEST
    # compile_params['sample_weight_mode'] = model.sample_weight_mode  # -> None, or ''  # FLAG: ORIGINAL
    compile_params["sample_weight_mode"] = model.model.sample_weight_mode  # FLAG: TEST
    # compile_params['weighted_metrics'] = model.weighted_metrics  # -> None, or []  # FLAG: ORIGINAL
    compile_params["weighted_metrics"] = model.model.weighted_metrics  # FLAG: TEST

    try:
        # compile_params['target_tensors'] = model.target_tensors  # FLAG: ORIGINAL
        compile_params["target_tensors"] = model.model.target_tensors  # FLAG: TEST
    except AttributeError:
        compile_params["target_tensors"] = None

    # noinspection PyProtectedMember
    compile_params["compile_kwargs"] = model.model._function_kwargs  # -> {}

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

    # for layer in model.layers:  # FLAG: ORIGINAL
    for layer in model.model.layers:  # FLAG: TEST
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
    # if model.optimizer.__module__ != 'keras.optimizers':  # FLAG: ORIGINAL
    if model.model.optimizer.__module__ != "keras.optimizers":  # FLAG: TEST
        G.warn(
            "Custom optimizers will not be hashed and saved, meaning they are identified only by their names."
            + "\nIf you plan on tuning optimizers at all, please ensure custom optimizers are not given the same names as any"
            + " of Keras's optimizers. Otherwise, naming conflicts may occur and make results very confusing."
        )

    return layers, compile_params


if __name__ == "__main__":
    pass
