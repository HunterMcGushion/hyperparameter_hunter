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
from hyperparameter_hunter.space import Real, Integer, Categorical

##################################################
# Import Miscellaneous Assets
##################################################
# noinspection PyProtectedMember
from inspect import signature, _empty, currentframe, getframeinfo
from functools import wraps


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


class TranslateTrace:
    def __init__(self, traced_parameter_name, translated_names=None):
        """Decorator to convert a class instance passed through one parameter into the class and
        the parameters used to initialize the instance. The translated values are then passed to the
        decorated callable through two other parameters named by `translated_names`, while the
        original value passed through `traced_parameter_name` is erased

        Parameters
        ----------
        traced_parameter_name: String
            Name of the parameter that is expected to receive a traced object instance as input
        translated_names: Tuple, None, default=None
            Names of the parameters to which the translations of the traced object should be passed

        Notes
        -----
        For a `traced_parameter_name` value of a class instance, the values passed through the
        `translated_names` parameters will be (respectively) 1) the class of which the value of
        `traced_parameter_name` is an instance, and 2) the dict of parameters used to initialize
        the instance of the class"""
        self.traced_parameter_name = traced_parameter_name
        self.translated_names = translated_names

    def __call__(self, obj):
        """
        # TODO: Documentation description

        Parameters
        ----------
        obj: Object
            Decorated callable, which is expected to receive a traced object as input, but will be
            passed its translation via the parameters in :attr:`translated_names`, instead

        Returns
        -------
        Object
            Callable  # TODO: Documentation
        """

        @wraps(obj)
        def wrapped(*args, **kwargs):
            binding = signature(obj).bind_partial(*args, **kwargs)

            #################### Traced Parameter Unused ####################
            if not binding.arguments[self.traced_parameter_name]:
                # Traced parameter not given. Do nothing, and proceed normally
                return obj(*args, **kwargs)

            #################### Traced Parameter Used ####################
            if any(binding.arguments[_] for _ in self.translated_names):
                # Traced parameter and target translations both given. Only expected one
                raise ValueError(
                    "Received both `{0}` and at least one of: {1}. Expected one of following:\n"
                    + " - 1) Both of [{1}] are given, and `{0}` is not given; or\n"
                    + " - 2) `{0}` is given, and neither of [{1}] are given".format(
                        self.traced_parameter_name, self.translated_names
                    )
                )

            # Traced parameter was given, and target translations were not given
            # TODO: Generalize below to use `traced_parameter_name` and `translated_names`
            binding.arguments["model_initializer"] = binding.arguments["model"].__class__

            # FLAG: ATTEMPT TO RESTORE ASSET FOR HASHING, BUT IT'S PROBABLY SCREWING SHIT UP
            #################### Replace Traced Asset with Original ####################
            # target_module = binding.arguments["model_initializer"].__module__
            # target_asset = binding.arguments["model_initializer"].__name__
            # for a_mirror in G.mirror_registry:
            #     if a_mirror.module_path == target_module and a_mirror.import_name == target_asset:
            #         # Found mirrored asset
            #         sys.modules[a_mirror.module_path] = a_mirror.original_sys_module_entry
            #         binding.arguments["model_initializer"] = getattr(
            #             a_mirror.original_module, target_asset
            #         )
            #         break
            # FLAG: ATTEMPT TO RESTORE ASSET FOR HASHING, BUT IT'S PROBABLY SCREWING SHIT UP

            translation_binding = signature(binding.arguments["model_initializer"]).bind_partial(
                *getattr(binding.arguments["model"], "__hh_used_args"),
                **getattr(binding.arguments["model"], "__hh_used_kwargs"),
            )
            binding.arguments["model_init_params"] = translation_binding.arguments
            binding.arguments["model"] = None
            # TODO: Generalize above to use `traced_parameter_name` and `translated_names`

            return obj(*binding.args, **binding.kwargs)

        return wrapped
