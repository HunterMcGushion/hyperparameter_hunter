##################################################
# Import Miscellaneous Assets
##################################################
from inspect import signature


def keras_callback_to_key(callback):
    signature_args = sorted(signature(callback.__class__).parameters.items())
    string_args = []

    for arg_name, arg_val in signature_args:
        if arg_name not in ['verbose']:
            try:
                string_args.append(F'{arg_name}={getattr(callback, arg_name)!r}')
            except AttributeError:
                string_args.append(F'{arg_name}={arg_val.default!r}')

    callback_key = F'{callback.__class__.__name__}(' + ', '.join(string_args) + ')'
    return callback_key


def parameterize_compiled_keras_model(model):
    # Expect compiled architecture result of "build_fn", like "define_architecture"
    # NOTE: Tested optimizer and loss with both callable and string inputs - Converted to callables automatically

    ##################################################
    # Model Compile Parameters
    ##################################################
    compile_params = dict()
    compile_params['optimizer'] = model.optimizer.__class__.__name__  # -> 'Adam'
    compile_params['optimizer_params'] = model.optimizer.get_config()  # -> {**kwargs}

    compile_params['metrics'] = model.metrics  # -> ['accuracy']
    compile_params['metrics_names'] = model.metrics_names  # -> ['loss', 'acc']

    compile_params['loss_functions'] = model.model.loss_functions  # -> [<function binary_crossentropy at 0x118832268>]
    compile_params['loss_function_names'] = [_.__name__ for _ in compile_params['loss_functions']]  # -> ['binary_crossentropy']

    # FLAG: BELOW PARAMETERS SHOULD ONLY BE DISPLAYED IF EXPLICITLY GIVEN (probably have to be in key by default, though):
    compile_params['loss_weights'] = model.loss_weights  # -> None, [], or {}
    compile_params['sample_weight_mode'] = model.sample_weight_mode  # -> None, or ''
    compile_params['weighted_metrics'] = model.weighted_metrics  # -> None, or []

    try:
        compile_params['target_tensors'] = model.target_tensors
    except AttributeError:
        compile_params['target_tensors'] = None

    # noinspection PyProtectedMember
    compile_params['compile_kwargs'] = model.model._function_kwargs  # -> {}

    ##################################################
    # Model Architecture
    ##################################################
    hh_attributes = ['__hh_default_args', '__hh_default_kwargs', '__hh_used_args', '__hh_used_kwargs']
    layers = []

    for layer in model.layers:
        layer_obj = dict(
            # name=layer.name,
            class_name=layer.__class__.__name__,
        )

        for hh_attr in hh_attributes:
            layer_obj[hh_attr] = getattr(layer, hh_attr, None)

        layers.append(layer_obj)

    ##################################################
    # Handle Custom Losses/Optimizers
    ##################################################
    if any([_.__module__ != 'keras.losses' for _ in compile_params['loss_functions']]):
        G.warn(
            'Custom loss functions will not be hashed and saved, meaning they are identified only by their names.' +
            '\nIf you plan on tuning loss functions at all, please ensure custom functions are not given the same names as any' +
            ' of Keras\'s loss functions. Otherwise, naming conflicts may occur and make results very confusing.'
        )
    if model.optimizer.__module__ != 'keras.optimizers':
        G.warn(
            'Custom optimizers will not be hashed and saved, meaning they are identified only by their names.' +
            '\nIf you plan on tuning optimizers at all, please ensure custom optimizers are not given the same names as any' +
            ' of Keras\'s optimizers. Otherwise, naming conflicts may occur and make results very confusing.'
        )

    return layers, compile_params


if __name__ == '__main__':
    pass
