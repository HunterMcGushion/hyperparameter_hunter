from hyperparameter_hunter.space import Real, Integer, Categorical
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
# noinspection PyProtectedMember
from inspect import signature, _empty


class KerasTracer(type):
    # TODO: Add documentation
    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace = dict(
            __hh_default_args=[],
            __hh_default_kwargs={},
            __hh_used_args=[],
            __hh_used_kwargs={},
        )
        return namespace

    def __new__(mcs, name, bases, namespace, **kwargs):
        class_obj = super().__new__(mcs, name, bases, dict(namespace))
        all_args, all_kwargs = [], {}

        signature_parameters = signature(class_obj.__init__).parameters
        for k, v in signature_parameters.items():
            if k not in ['self', 'args', 'kwargs']:  # FLAG: Might want to remove kwargs - Could be necessary to ok "input_dim"
                if ((v.kind in [v.KEYWORD_ONLY, v.POSITIONAL_OR_KEYWORD]) and v.default != _empty):
                    all_kwargs[k] = v.default
                else:
                    all_args.append(k)

        setattr(class_obj, '__hh_default_args', all_args)
        setattr(class_obj, '__hh_default_kwargs', all_kwargs)

        return class_obj

    def __call__(cls, *args, **kwargs):
        if getattr(G, 'use_dummy_keras_tracer', False) is True:
            _args = [_ if not isinstance(_, (Real, Integer, Categorical)) else _.bounds[0] for _ in args]
            _kwargs = {_k: _v if not isinstance(_v, (Real, Integer, Categorical)) else _v.bounds[0] for _k, _v in kwargs.items()}
            instance = super().__call__(*_args, **_kwargs)
        else:
            instance = super().__call__(*args, **kwargs)

        setattr(instance, '__hh_used_args', args)
        setattr(instance, '__hh_used_kwargs', kwargs)

        return instance
