##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.tracers import KerasTracer

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial, wraps
from importlib.machinery import PathFinder, ModuleSpec, SourceFileLoader
from inspect import ismodule, isclass, ismethod, isfunction
from pkg_resources import get_distribution
import sys


class Interceptor(PathFinder):
    def __init__(self, module_name, custom_loader):
        # TODO: Add documentation
        self.module_name = module_name
        self.custom_loader = custom_loader

    def find_spec(self, full_name, path=None, target=None):
        # TODO: Add documentation
        if full_name == self.module_name:
            spec = super().find_spec(full_name, path, target)
            loader = self.custom_loader(full_name, spec.origin)

            return ModuleSpec(full_name, loader)


class KerasLayerLoader(SourceFileLoader):
    # TODO: Add documentation

    def exec_module(self, module):
        # TODO: Add documentation
        super().exec_module(module)
        module.Layer = KerasTracer(module.Layer.__name__, module.Layer.__bases__, module.Layer.__dict__)
        return module


def hook_keras_layer():
    """If Keras has yet to be imported, modify the inheritance structure of its base `Layer` class to inject attributes that
    keep track of the parameters provided to each layer"""
    if 'keras' in sys.modules:
        raise ImportError('{} must be executed before importing Keras or other hyperparameter_hunter assets'.format(
            'hyperparameter_hunter.importer.hook_keras_layer()'
        ))

    if get_distribution('keras').version >= '2.2.0':
        sys.meta_path.insert(0, Interceptor('keras.engine.base_layer', KerasLayerLoader))  # Keras == 2.2.0
    else:
        sys.meta_path.insert(0, Interceptor('keras.engine.topology', KerasLayerLoader))  # Keras == 2.1.3
        # Determine version number at which this becomes untrue (Minimum Keras version requirement)

    G.import_hooks.append('keras_layer')


# class KerasOptimizerGetLoader(SourceFileLoader):
#     def exec_module(self, module):
#         super().exec_module(module)
#
#         def safe_get_builder(f):
#             @wraps(f)
#             def safe_get(identifier):
#                 # def safe_get(*args, **kwargs):
#                 if isinstance(identifier, (Real, Integer, Categorical)):
#                     safe_identifier = identifier.bounds[0]
#                     get_result = f(safe_identifier)
#                 else:
#                     get_result = f(identifier)
#
#                 setattr(get_result, '__original_hh_identifier', identifier)
#                 return get_result
#
#             return safe_get
#
#         setattr(module, 'get', safe_get_builder(module.get))
#
#
# def hook_keras_optimizer_get():
#     if 'keras' in sys.modules:
#         raise ImportError('{} must be executed before importing Keras or other hyperparameter_hunter assets'.format(
#             'hyperparameter_hunter.importer.hook_keras_optimizer_get()'
#         ))
#     sys.meta_path.insert(0, Interceptor('keras.optimizers', KerasOptimizerGetLoader))
#     G.import_hooks.append('keras_optimizer_get')


##################################################
# Docstring Modification
##################################################
class ModuleMultiWrapper(SourceFileLoader):
    def __init__(self, fullname, path, wrappers=None, checks=None):
        # TODO: Add documentation
        self.wrappers = wrappers or []
        self.checks = checks or [lambda _: True]
        super().__init__(fullname, path)

    # noinspection PyMethodOverriding
    def exec_module(self, module):
        # TODO: Add documentation
        super().exec_module(module)

        for attr in module.__dict__:
            obj = getattr(module, attr)

            if any([_(obj) for _ in self.checks]):
                for wrapper in self.wrappers:
                    obj = wrapper(obj)

                setattr(module, attr, obj)


def nullify_docstring(f):
    # TODO: Add documentation
    @wraps(f)
    def new_func(*args, **kwargs):
        return f(*args, **kwargs)

    new_func.__doc__ = None
    return new_func


def nullify_module_docstrings(module_name):
    # TODO: Add documentation
    sys.meta_path.insert(0, Interceptor(
        module_name, partial(
            ModuleMultiWrapper, wrappers=[nullify_docstring], checks=[ismodule, isclass, ismethod, isfunction]
        )
    ))


if __name__ == '__main__':
    pass
