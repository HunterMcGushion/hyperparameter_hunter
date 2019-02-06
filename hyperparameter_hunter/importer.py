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
from hyperparameter_hunter.tracers import ArgumentTracer, LocationTracer

##################################################
# Import Miscellaneous Assets
##################################################
from importlib.machinery import PathFinder, ModuleSpec, SourceFileLoader
from pkg_resources import get_distribution
import sys


class Interceptor(PathFinder):
    def __init__(self, module_name, custom_loader, asset_name=None):
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
        self.asset_name = asset_name

    def find_spec(self, full_name, path=None, target=None):
        """Perform custom loading logic if `full_name` == :attr:`module_name`"""
        if full_name == self.module_name:
            spec = super().find_spec(full_name, path, target)

            if self.asset_name:  # Some loaders need `asset_name` (like `TraceLoader`). Others don't
                loader = self.custom_loader(full_name, spec.origin, self.asset_name)
            else:
                loader = self.custom_loader(full_name, spec.origin)

            return ModuleSpec(full_name, loader)


##################################################
# Keras Layer Interception
##################################################
class KerasLayerLoader(SourceFileLoader):
    def exec_module(self, module):
        """Set `module.Layer` to a traced version of itself via :class:`tracers.ArgumentTracer`"""
        super().exec_module(module)
        module.Layer = ArgumentTracer(
            module.Layer.__name__, module.Layer.__bases__, module.Layer.__dict__
        )
        return module


def hook_keras_layer():
    """If Keras has yet to be imported, modify the inheritance structure of its base `Layer` class
    to inject attributes that keep track of the parameters provided to each layer"""
    if "keras" in sys.modules:
        _name = "hyperparameter_hunter.importer.hook_keras_layer"
        raise ImportError(f"Call {_name} before importing Keras/other hyperparameter_hunter assets")

    if get_distribution("keras").version >= "2.2.0":  # Keras == 2.2.0
        sys.meta_path.insert(0, Interceptor("keras.engine.base_layer", KerasLayerLoader))
    else:  # Keras == 2.1.3
        sys.meta_path.insert(0, Interceptor("keras.engine.topology", KerasLayerLoader))
        # Determine version number at which this becomes untrue (Minimum Keras version requirement)

    G.import_hooks.append("keras_layer")


##################################################
# Keras Initializer Interception
##################################################
class KerasMultiInitializerLoader(SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)

        #################### Trace `Initializer` Descendants/Aliases ####################
        requirements = [
            lambda _: isinstance(_, type),
            lambda _: issubclass(_, module.Initializer),
            lambda _: not issubclass(_, module.VarianceScaling),
            lambda _: not ((_ is module.Initializer) or (_ is module.VarianceScaling)),
        ]

        child_classes = {k: v for k, v in vars(module).items() if all(_(v) for _ in requirements)}
        # Need keys and values to handle class aliases (like `Zeros` = `zeros` = `zero`)
        # Otherwise, we just trace `Zeros` 3x, which may not sound like a problem, but it is
        for child_class_name, child_class in child_classes.items():
            setattr(
                module,
                child_class_name,
                ArgumentTracer(child_class_name, child_class.__bases__, child_class.__dict__),
            )

        #################### Trace `VarianceScaling` ####################
        module.VarianceScaling = LocationTracer(
            module.VarianceScaling.__name__,
            module.VarianceScaling.__bases__,
            module.VarianceScaling.__dict__,
        )

        return module


def hook_keras_initializers():
    if "keras" in sys.modules:
        _name = "hyperparameter_hunter.importer.hook_keras_initializers"
        raise ImportError(f"Call {_name} before importing Keras/other hyperparameter_hunter assets")

    sys.meta_path.insert(0, Interceptor("keras.initializers", KerasMultiInitializerLoader))
    G.import_hooks.append("keras_initializer")
    G.import_hooks.append("keras_variance_scaling")


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
