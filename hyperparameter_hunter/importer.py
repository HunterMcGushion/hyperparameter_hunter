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
import sys


# def patcher(a_function):
#     def wrapper(*args, **kwargs):
#         print(F'Patcher called with args: {args}...   kwargs: {kwargs}')
#         return a_function(*args, **kwargs)
#     return wrapper


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
        # module.function = patcher(module.function)
        module.Layer = KerasTracer(module.Layer.__name__, module.Layer.__bases__, module.Layer.__dict__)
        return module


def hook_keras_layer():
    # TODO: Add documentation
    if 'keras' in sys.modules:
        raise ImportError('{} must be executed before importing Keras or other hyperparameter_hunter assets'.format(
            'hyperparameter_hunter.importer.hook_keras_layer()'
        ))
    # sys.meta_path.insert(0, Interceptor('keras.layers.core'))
    sys.meta_path.insert(0, Interceptor('keras.engine.topology', KerasLayerLoader))
    G.import_hooks.append('keras_layer')


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
