"""This module defines metaclasses used to trace the parameters passed through operation-critical
classes that are members of other libraries. These are only used in cases where it is impractical
or impossible to effectively retrieve the arguments explicitly provided by a user, as well as the
default arguments for the classes being traced. Generally, tracer metaclasses will aim to add some
attributes to the class, that will collect default values, and provided arguments on the class's
creation, and an instance's call

Related
-------
:mod:`hyperparameter_hunter.importer`
    This module handles the interception of certain imports in order to inject the tracer
    metaclasses defined in :mod:`hyperparameter_hunter.tracers` into the inheritance structure of
    objects that need to be traced"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.space.dimensions import Real, Integer, Categorical

##################################################
# Import Miscellaneous Assets
##################################################
# noinspection PyProtectedMember
from inspect import signature, _empty, currentframe, getframeinfo


class ArgumentTracer(type):
    """Metaclass to trace the default arguments and explicitly provided arguments of its
    descendants. It also has special provisions for instantiating dummy models if directed to"""

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace = dict(
            __hh_default_args=[], __hh_default_kwargs={}, __hh_used_args=[], __hh_used_kwargs={}
        )
        return namespace

    def __new__(mcs, name, bases, namespace, **kwargs):
        class_obj = super().__new__(mcs, name, bases, dict(namespace))
        all_args, all_kwargs = [], {}

        signature_parameters = signature(class_obj.__init__).parameters
        for k, v in signature_parameters.items():
            if k not in ["self", "args", "kwargs"]:  # FLAG: Might need kwargs to ok "input_dim"
                if (v.kind in [v.KEYWORD_ONLY, v.POSITIONAL_OR_KEYWORD]) and v.default != _empty:
                    all_kwargs[k] = v.default
                else:
                    all_args.append(k)

        setattr(class_obj, "__hh_default_args", all_args)
        setattr(class_obj, "__hh_default_kwargs", all_kwargs)

        return class_obj

    def __call__(cls, *args, **kwargs):
        if getattr(G, "use_dummy_tracer", False) is True:
            space = (Real, Integer, Categorical)
            _args = [_ if not isinstance(_, space) else _.bounds[0] for _ in args]
            _kwargs = {k: v if not isinstance(v, space) else v.bounds[0] for k, v in kwargs.items()}
            instance = super().__call__(*_args, **_kwargs)
        else:
            instance = super().__call__(*args, **kwargs)

        setattr(instance, "__hh_used_args", args)
        setattr(instance, "__hh_used_kwargs", kwargs)

        return instance


class LocationTracer(ArgumentTracer):
    """Metaclass to trace the origin of the call to initialize the descending class"""

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace = dict(__hh_previous_frame=None)
        return namespace

    def __new__(mcs, name, bases, namespace, **kwargs):
        class_obj = super().__new__(mcs, name, bases, dict(namespace))
        return class_obj

    def __call__(cls, *args, **kwargs):
        previous_frame_info = getframeinfo(currentframe().f_back)
        # FLAG: This has `filename` and `function` attributes
        # FLAG: Calling `he_normal` function, gave `function="he_normal"` here and `filename` correctly pointed to `keras.initializers`

        # TODO: Fetch default args/kwargs for the VarianceScaling class function?

        instance = super().__call__(*args, **kwargs)

        setattr(instance, "__hh_previous_frame", previous_frame_info)

        return instance
