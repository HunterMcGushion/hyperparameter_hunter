"""This module provides utilities to intercept external imports and load them using custom logic

Related
-------
:mod:`hyperparameter_hunter.__init__`
    Executes the import hooks to ensure assets are properly imported prior to starting any real work
:mod:`hyperparameter_hunter.tracers`
    Defines tracing metaclasses applied by :mod:`hyperparameter_hunter.importer` to imports"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.tracers import KerasTracer

##################################################
# Import Miscellaneous Assets
##################################################
from importlib.machinery import PathFinder, ModuleSpec, SourceFileLoader
from pkg_resources import get_distribution
import sys


class Interceptor(PathFinder):
    def __init__(self, module_name, custom_loader):
        """Class to intercept loading of an external module in order to provide custom loading logic

        Parameters
        ----------
        module_name: String
            The path of the module, for which loading should be handled by `custom_loader`
        custom_loader: Descendant of `importlib.machinery.SourceFileLoader`
            Should implement :meth:`exec_module`, which should call its superclass's
            :meth:`exec_module`, then perform the custom loading logic, and return `module`"""
        self.module_name = module_name
        self.custom_loader = custom_loader

    def find_spec(self, full_name, path=None, target=None):
        """Perform custom loading logic if `full_name` == :attr:`module_name`"""
        if full_name == self.module_name:
            spec = super().find_spec(full_name, path, target)
            loader = self.custom_loader(full_name, spec.origin)

            return ModuleSpec(full_name, loader)


##################################################
# Keras Layer Interception
##################################################
class KerasLayerLoader(SourceFileLoader):
    def exec_module(self, module):
        """Set `module.Layer` a traced version of itself via
        :class:`hyperparameter_hunter.tracers.KerasTracer`"""
        super().exec_module(module)
        module.Layer = KerasTracer(
            module.Layer.__name__, module.Layer.__bases__, module.Layer.__dict__
        )
        return module


def hook_keras_layer():
    """If Keras has yet to be imported, modify the inheritance structure of its base `Layer` class
    to inject attributes that keep track of the parameters provided to each layer"""
    if "keras" in sys.modules:
        raise ImportError(
            "{} must be executed before importing Keras or other hyperparameter_hunter assets".format(
                "hyperparameter_hunter.importer.hook_keras_layer()"
            )
        )

    if get_distribution("keras").version >= "2.2.0":
        sys.meta_path.insert(
            0, Interceptor("keras.engine.base_layer", KerasLayerLoader)
        )  # Keras == 2.2.0
    else:
        sys.meta_path.insert(
            0, Interceptor("keras.engine.topology", KerasLayerLoader)
        )  # Keras == 2.1.3
        # Determine version number at which this becomes untrue (Minimum Keras version requirement)

    G.import_hooks.append("keras_layer")


##################################################
# Keras Optimizer Interception
##################################################
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
# class ModuleMultiWrapper(SourceFileLoader):
#     def __init__(self, fullname, path, wrappers=None, checks=None):
#         self.wrappers = wrappers or []
#         self.checks = checks or [lambda _: True]
#         super().__init__(fullname, path)
#
#     # noinspection PyMethodOverriding
#     def exec_module(self, module):
#         super().exec_module(module)
#
#         for attr in module.__dict__:
#             obj = getattr(module, attr)
#
#             if any([_(obj) for _ in self.checks]):
#                 for wrapper in self.wrappers:
#                     obj = wrapper(obj)
#
#                 setattr(module, attr, obj)


# def nullify_docstring(f):
#     @wraps(f)
#     def new_func(*args, **kwargs):
#         return f(*args, **kwargs)
#
#     new_func.__doc__ = None
#     return new_func


# def nullify_module_docstrings(module_name):
#     sys.meta_path.insert(0, Interceptor(
#         module_name, partial(
#             ModuleMultiWrapper, wrappers=[nullify_docstring], checks=[ismodule, isclass, ismethod, isfunction]
#         )
#     ))
