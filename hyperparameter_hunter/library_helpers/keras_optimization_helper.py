"""This module performs additional processing necessary when optimizing hyperparameters in the
`Keras` library. Its purpose is twofold: 1) to enable the construction of Keras models while
requiring minimal syntactic changes on the user's end when defining hyperparameter space choices;
and 2) to enable thorough collection of all hyperparameters used to define a Keras model - not only
those being optimized - in order to ensure the continued usefulness of an Experiment's result files
even under different hyperparameter search constraints

Related
-------
:mod:`hyperparameter_hunter.importer`
    Performs interception of `Keras` import to inject the hyperparameter-recording attributes
:mod:`hyperparameter_hunter.tracers`
    Defines the new metaclass used by :mod:`hyperparameter_hunter.importer` to apply to key Keras
    classes (like `Layer`)
:mod:`hyperparameter_hunter.utils.parsing_utils`
    Defines utilities to assist in parsing source code provided by users to declare Keras
    model-building functions
:mod:`hyperparameter_hunter.library_helpers.keras_helper`
    Defines utilities to assist in characterizing Keras models"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.library_helpers.keras_helper import parameterize_compiled_keras_model
from hyperparameter_hunter.space import Real, Integer, Categorical
from hyperparameter_hunter.utils.boltons_utils import remap, default_enter
from hyperparameter_hunter.utils.general_utils import deep_restricted_update
from hyperparameter_hunter.utils.parsing_utils import (
    stringify_model_builder,
    write_python_source,
    build_temp_model_file,
)

##################################################
# Import Miscellaneous Assets
##################################################
from collections import OrderedDict
from copy import deepcopy
import os
import re
import sys
from types import FunctionType, MethodType

##################################################
# Import Learning Assets
##################################################
try:
    from keras.callbacks import Callback as base_keras_callback
except ModuleNotFoundError:
    base_keras_callback = type("PlaceholderBaseKerasCallback", (), {})


def keras_prep_workflow(model_initializer, build_fn, extra_params, source_script):
    """Conduct preparation steps necessary before hyperparameter optimization on a `Keras` model.
    Such steps include parsing and modifying `build_fn` to be of the form used by
    :class:`hyperparameter_hunter.optimization_core.BaseOptimizationProtocol`, compiling a dummy
    model to identify universal locations of given hyperparameter choices, and creating a simplified
    characterization of the models to be built during optimization in order to enable collection of
    similar Experiments

    Parameters
    ----------
    model_initializer: :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        A descendant of :class:`keras.wrappers.scikit_learn.BaseWrapper` used to build a Keras model
    build_fn: Callable
        The `build_fn` value provided to :meth:`keras.wrappers.scikit_learn.BaseWrapper.__init__`.
        Expected to return a compiled Keras model. May contain hyperparameter space choices
    extra_params: Dict
        The parameters expected to be passed to the extra methods of the compiled Keras model. Such
        methods include (but are not limited to) `fit`, `predict`, and `predict_proba`. Some of the
        common parameters given here include `epochs`, `batch_size`, and `callbacks`
    source_script: Str
        Absolute path to a Python file. Expected to end with one of the following extensions:
        '.py', '.ipynb'

    Returns
    -------
    reusable_build_fn: Callable
        Modified `build_fn` in which hyperparameter space choices are replaced by dict lookups, and
        the signature is given a standard name, and additional input parameters necessary for reuse
    reusable_wrapper_params: Dict
        The parameters expected to be passed to the extra methods of the compiled Keras model. Such
        methods include (but are not limited to) `fit`, `predict`, and `predict_proba`. Some of the
        common parameters given here include `epochs`, `batch_size`, and `callbacks`
    dummy_layers: List
        The layers of a compiled dummy Keras model constructed according to the given
        hyperparameters, in which each layer is a dict containing at least the following: the name
        of the layer class, allowed and used args, and default and used kwargs
    dummy_compile_params: Dict
        The parameters used on the `compile` call for the dummy model. If a parameter is accepted
        by the `compile` method, but is not explicitly given, its default value is included in
        `dummy_compile_params`"""
    #################### Prepare Model-Builder String ####################
    temp_builder_name = "__temp_model_builder"
    reusable_build_fn, expected_params = rewrite_model_builder(stringify_model_builder(build_fn))
    temp_model_file_str = build_temp_model_file(reusable_build_fn, source_script)

    #################### Save and Import Temporary Model Builder ####################
    write_python_source(
        temp_model_file_str, "{}/{}.py".format(os.path.split(__file__)[0], temp_builder_name)
    )

    if temp_builder_name in sys.modules:
        del sys.modules[temp_builder_name]

    try:
        from .__temp_model_builder import build_fn as temp_build_fn
    except:
        raise

    #################### Translate Hyperparameter Names to Universal Paths ####################
    wrapper_params = dict(
        params={_k: eval(_v) for _k, _v in expected_params.items()}, **extra_params
    )
    wrapper_params, dummified_params = check_dummy_params(wrapper_params)

    if ("optimizer_params" in dummified_params) and ("optimizer" in dummified_params):
        raise ValueError(
            "Unable to optimize both `optimizer` and `optimizer_params`. Please try optimizing them separately"
        )

    compiled_dummy = initialize_dummy_model(model_initializer, temp_build_fn, wrapper_params)
    dummy_layers, dummy_compile_params = parameterize_compiled_keras_model(compiled_dummy)
    merged_compile_params = merge_compile_params(dummy_compile_params, dummified_params)
    # FLAG: Will need to deal with capitalization conflicts when comparing similar experiments: `optimizer`='Adam' vs 'adam'

    consolidated_layers = consolidate_layers(
        dummy_layers, class_name_key=False, separate_args=False
    )
    wrapper_params = deep_restricted_update(wrapper_params, dummified_params)

    return (temp_build_fn, wrapper_params, consolidated_layers, merged_compile_params)


def consolidate_layers(layers, class_name_key=True, separate_args=True):
    """For each of the layer dicts in `layers`, merge the dict's keys to reflect the end value of
    the key, rather than its default value, and whether a value was explicitly given

    Parameters
    ----------
    layers: List
        A list of dicts, wherein each dict represents a layer in a Keras model, and contains
        information about its arguments
    class_name_key: Boolean, default=True
        If True, 'class_name' will be added as a key to the dict describing each layer. Else, it
        will be used as a key to create an outer dict containing the rest of the keys describing
        each layer
    separate_args: Boolean, default=True
        If True, each layer dict will be given two keys: 'arg_vals', and 'kwarg_vals', which are
        both dicts containing their respective values. Else, each layer dict will directly contain
        all the keys of 'arg_vals', and 'kwarg_vals', removing any indication of whether the
        parameter was a positional or keyword argument, aside from order

    Returns
    -------
    consolidated_layers: List
        A list of the same length as `layers`, except each element has fewer keys than it did in
        `layers`. The new keys are as follows: 'class_name', 'arg_vals', 'kwarg_vals'"""
    consolidated_layers = []

    for layer in layers:
        arg_vals = {}

        #################### Gather Args ####################
        for i, expected_arg in enumerate(layer["__hh_default_args"]):
            try:
                arg_vals[expected_arg] = layer["__hh_used_args"][i]
            except IndexError:
                if expected_arg in layer["__hh_used_kwargs"]:
                    arg_vals[expected_arg] = layer["__hh_used_kwargs"][expected_arg]
                else:
                    raise

        #################### Gather Kwargs ####################
        # Merge default and used kwargs with constraints: only include if k in default, and give priority to used values
        # This means that kwargs like `input_shape` won't make it through because they have no default values, also
        # nonsensical kwargs won't make it through because the defaults are the point of reference
        kwarg_vals = {
            _k: layer["__hh_used_kwargs"].get(_k, _v)
            for _k, _v in layer["__hh_default_kwargs"].items()
        }

        #################### Consolidate ####################
        new_layer_dict = (
            dict(arg_vals=arg_vals, kwarg_vals=kwarg_vals)
            if separate_args
            else {**arg_vals, **kwarg_vals}
        )

        if class_name_key:
            new_layer_dict["class_name"] = layer["class_name"]
        else:
            new_layer_dict = {layer["class_name"]: new_layer_dict}

        consolidated_layers.append(new_layer_dict)
    return consolidated_layers


def merge_compile_params(compile_params, dummified_params):
    """Update `compile_params` to reflect those values that were given hyperparameter space choices,
    as specified by `dummified_params`

    Parameters
    ----------
    compile_params: Dict
        All the compile parameters provided to a dummy model's `compile` method, or their default
        values if they were not explicitly given. If the original value of one of the keys in
        `compile_params` was a hyperparameter space choice, its current value will be the dummy
        chosen for it, and this change will be reflected by the contents of `dummified_params`
    dummified_params: Dict
        A mapping of keys in `compile_params` (possibly nested keys) to a tuple pair of
        (<original hyperparameter space choice>, <tuple path to key>)

    Returns
    -------
    merged_params: Dict
        A dictionary that mirrors `compile_params`, except where an element of `dummified_params`
        has the same path/key, in which case the hyperparameter space choice value in
        `dummified_params` is used"""
    # FLAG: Deal with capitalization conflicts when comparing similar experiments: `optimizer`='Adam' vs 'adam'
    _dummified_params = {
        (_k[1:] if _k[0] == "params" else _k): _v for _k, _v in dummified_params.copy().items()
    }

    def _visit(path, key, value):
        """If (`path` + `key`) in `_dummified_params`, return its value instead. Else, default"""
        location = path + (key,)
        if len(_dummified_params) and location in _dummified_params:
            return (key, _dummified_params.pop(location))
        return (key, value)

    merged_params = remap(compile_params, visit=_visit)
    return merged_params


def check_dummy_params(params):
    """Locate and dummify hyperparameter space choices in `params`, if the hyperparameter is used
    for model compilation

    Parameters
    ----------
    params: Dict
        A dictionary of hyperparameters, in which values may be hyperparameter space choices

    Returns
    -------
    checked_params: Dict
        A replica of `params`, in which instances of hyperparameter space choices are replaced with
        dummy values
    dummified_params: Dict
        A record of keys that were found whose values were hyperparameter space choices, mapped to
        tuple pairs of (<original value>, <path to key>)"""
    compile_keys = [
        "optimizer",
        "loss",
        "metrics",
        "loss_weights",
        "sample_weight_mode",
        "weighted_metrics",
        "target_tensors",
    ]

    dummified_params = dict()

    # noinspection PyUnusedLocal
    def _visit(path, key, value):
        """If `value` is a descendant of :class:`space.Dimension`, return its lower bound and
        collect it. Else, default return"""
        if key in compile_keys:
            if isinstance(value, (Real, Integer, Categorical)):
                dummified_params[path + (key,)] = value
                return (key, value.bounds[0])
        return (key, value)

    checked_params = remap(params, visit=_visit)
    return checked_params, dummified_params


def link_choice_ids(layers, compile_params, extra_params, dimensions):
    """Update `extra_params` to include a 'location' attribute on any descendants of
    :class:`space.Dimension`, specifying its position among all hyperparameters

    Parameters
    ----------
    layers: List
        A list of dicts, in which each dict describes a network layer
    compile_params: Dict
        A dict containing the hyperparameters supplied to the model's `compile` call
    extra_params: Dict
        A dict containing the hyperparameters for the model's extra methods, such as `fit`,
        `predict`, and `predict_proba`
    dimensions: List
        A list containing descendants of :class:`space.Dimension`, representing the entire
        hyperparameter search space

    Returns
    -------
    extra_params: Dict
        Mirrors the given `extra_params`, except any descendants of :class:`space.Dimension` now
        have a 'location' attribute"""

    def visit_builder(param_type):
        """Make visit func that prepends `param_type` to the 'location' tuple added in `_visit`"""
        param_type = (param_type,) if not isinstance(param_type, tuple) else param_type

        def _visit(path, key, value):
            """If `value` is a descendant of :class:`space.Dimension`, add 'location' to itself and
            its copy in `dimensions`"""
            if isinstance(value, (Real, Integer, Categorical)):
                for i in range(len(dimensions)):
                    if dimensions[i].id == value.id:
                        setattr(dimensions[i], "location", (param_type + path + (key,)))
                        setattr(value, "location", (param_type + path + (key,)))
            return (key, value)

        return _visit

    def _enter(path, key, value):
        """If `value` is in `keras.callbacks`, enter as a dict, iterating over non-magic attributes.
        Else, `default_enter`"""
        if isinstance(value, base_keras_callback):
            return dict(), [(_, getattr(value, _)) for _ in dir(value) if not _.startswith("__")]
        return default_enter(path, key, value)

    # noinspection PyUnusedLocal
    _new_layers = remap(layers.copy(), visit=visit_builder(("model_init_params", "layers")))
    # noinspection PyUnusedLocal
    _new_compile_params = remap(
        compile_params.copy(), visit=visit_builder(("model_init_params", "compile_params"))
    )
    # noinspection PyUnusedLocal
    _new_extra_params = remap(
        {_k: _v for _k, _v in extra_params.items() if _k != "params"},
        visit=visit_builder("model_extra_params"),
        enter=_enter,
    )

    # `extra_params` has locations for `layers`, `compile_params`, `extra_params` - Of form expected by `build_fn` (less choices)
    return extra_params


##################################################
# Keras Dummy Model Tracing Utilities
##################################################
def initialize_dummy_model(model_initializer, build_fn, wrapper_params):
    """Creates a dummy model with placeholder values wherever hyperparameter options are provided
    via `hyperparameter_hunter.space` classes in order to produce a valid Keras model, albeit one
    with semi-useless values, which also contains attributes injected by
    :mod:`hyperparameter_hunter.importer`, and :mod:`hyperparameter_hunter.tracers` in order to
    keep a record of given hyperparameter choices

    Parameters
    ----------
    model_initializer: :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        A descendant of :class:`keras.wrappers.scikit_learn.BaseWrapper` used to build a Keras model
    build_fn: Callable
        The `build_fn` value provided to :meth:`keras.wrappers.scikit_learn.BaseWrapper.__init__`
    wrapper_params: Dict
        Additional parameters given to :meth:`keras.wrappers.scikit_learn.BaseWrapper.__init__`, as
        `sk_params`. Some acceptable values include arguments of `build_fn`; and arguments for the
        `fit`, `predict`, `predict_proba`, and `score` methods. For further information on
        acceptable values see the Keras documentation

    Returns
    -------
    dummy: Instance of :class:`keras.wrappers.scikit_learn.<KerasClassifier; KerasRegressor>`
        An initialized, compiled descendant of :class:`keras.wrappers.scikit_learn.BaseWrapper`"""
    setattr(G, "use_dummy_keras_tracer", True)  # Handles dummifying params via `KerasTracer`

    wrapper_params = deepcopy(wrapper_params)

    if "input_dim" in wrapper_params:
        wrapper_params["input_shape"] = (wrapper_params["input_dim"],)
        del wrapper_params["input_dim"]
    if ("input_shape" not in wrapper_params) or (wrapper_params["input_shape"][0] <= 0):
        wrapper_params["input_shape"] = (1,)

    dummy = model_initializer(build_fn=build_fn, **wrapper_params)

    # NOTE: Below if/else might be unnecessary since `build_fn` should always be a function
    if dummy.build_fn is None:
        dummy.model = dummy.__call__(**dummy.filter_sk_params(dummy.__call__))
    elif not isinstance(dummy.build_fn, (FunctionType, MethodType)):
        dummy.model = dummy.build_fn(**dummy.filter_sk_params(dummy.build_fn.__call__))
    else:
        dummy.model = dummy.build_fn(**dummy.filter_sk_params(dummy.build_fn))

    setattr(G, "use_dummy_keras_tracer", False)
    return dummy


##################################################
# Keras Model-Builder Parsing Utilities
##################################################
def rewrite_model_builder(build_fn_source):
    """Convert the build function used to construct a Keras model to a reusable format by replacing
    usages of `hyperparameter_hunter.space` classes (`Real`, `Integer`, `Categorical`) with key
    lookups to a new build_fn input dict containing keys for each of the hyperparameter search
    space choices found in the original source code

    Parameters
    ----------
    build_fn_source: String
        The stringified source code of a callable (assumed to be Keras `build_fn`)

    Returns
    -------
    reusable_build_fn: String
        The given `build_fn_source`, in which any usages of `hyperparameter_hunter.space` classes
        (`Real`, `Integer`, `Categorical`) are replaced with key lookups to a new build_fn input
        dict containing keys for each of the hyperparameter search space choices found in the
        original `build_fn_source`,
    expected_params: `collections.OrderedDict` instance
        A mapping of the names of the located hyperparameter choices to their given ranges
        (as described by `hyperparameter_hunter.space` classes)"""
    clipped_choices, names, start_indexes = find_space_fragments(build_fn_source)
    expected_params = OrderedDict(zip(names, clipped_choices))

    for i, name in enumerate(names):
        lookup_val = "params[{!r}]".format(name)
        build_fn_source = build_fn_source.replace(clipped_choices[i], lookup_val, 1)

    new_first_line = "def build_fn(input_shape=(10, ), params=None):"
    reusable_build_fn = build_fn_source.replace(build_fn_source.split("\n")[0], new_first_line)

    return reusable_build_fn, expected_params


def find_space_fragments(string):
    """Locate and name all hyperparameter choice declaration fragments in `string`

    Parameters
    ----------
    string: String
        A string assumed to be the source code of a Keras model-building function, in which
        hyperparameter choice declaration strings may be found

    Returns
    -------
    clipped_choices: List
        All hyperparameter choice declaration strings found in `string` - in order of appearance
    names: List
        The names of all hyperparameter choice declarations in `string` - in order of appearance
    start_indexes: List
        The indexes at which each hyperparameter choice declaration string was found in `string` -
        in order of appearance"""
    unclipped_choices, start_indexes = zip(*iter_fragments(string, is_match=is_space_match))
    clipped_choices = []
    names = []

    for choice in unclipped_choices:
        name = re.findall(r"(\w+(?=\s*[=(]\s*" + re.escape(choice) + r"))", string)
        # FLAG: Might need to prepend name with '_' to prevent possible duplicate extra params
        names.append(name[0] if (len(name) > 0) else names[-1])
        clipped_choices.append(clean_parenthesized_string(choice))

    #################### Fix Duplicated Names ####################
    for i in list(range(len(names)))[::-1]:
        duplicates = [_ for _ in names[0:i] if _ == names[i]]
        names[i] += "_{}".format(len(duplicates)) if len(duplicates) > 0 else ""

    return clipped_choices, names, list(start_indexes)


def is_space_match(string):
    """Determine whether `string` consists of a hyperparameter space declaration

    Parameters
    ----------
    string: String
        Str assumed to be source code fragment, which may contain a hyperparameter space declaration

    Returns
    -------
    Boolean
        True if `string` begins with a valid hyperparameter space declaration. Else, False"""
    starting_sequences = ["Real(", "Integer(", "Categorical("]
    # prefix_regex = r"[_\.A-Za-z0-9]"  # TODO: Support prefixes - To cover import aliases or importing modules to call classes
    # r"((?=([_\.A-Za-z0-9]+\.)?(?:(Real|Integer|Categorical)\()))"
    return any(string.startswith(_) for _ in starting_sequences)


##################################################
# General-Use Utilities
##################################################
def iter_fragments(string, is_match=None):
    """Yield fragments of `string` that are of a desired form as dictated by `is_match`

    Parameters
    ----------
    string: String
        A string containing fragments, which, when passed to `is_match` return True
    is_match: Callable, or None, default=lambda _: False
        Callable given a single string input that is a fragment of `string`, starting at any index.
        Expected to return boolean, which is truthy when the given fragment is of the desired form

    Yields
    ------
    String
        Fragment of `string` starting at an index and continuing to the end, for which `is_match`
        returned a truthy value
    Int
        The index at which the aforementioned string fragment begins"""
    is_match = is_match or (lambda _: False)

    for i in range(len(string)):
        if is_match(string[i:]):
            yield string[i:], i


def clean_parenthesized_string(string):
    """Produce a clipped substring of `string` comprising all characters from the beginning of
    `string` through the closing paren that matches the first opening paren in `string`

    Parameters
    ----------
    string: String
        A string that contains a parenthesized statement in its entirety, along with extra content
        to be removed. The target parenthesized statement may contain additional parentheses

    Returns
    -------
    clean_string: String
        A substring of `string`, extending from the beginning of `string`, through the closing paren
        that matches the first opening paren found, producing a valid parenthesized statement"""
    extra_closing_parens = 0

    for i in range(len(string)):
        if string[i] == "(":
            extra_closing_parens += 1
        elif string[i] == ")":
            if extra_closing_parens > 1:
                extra_closing_parens -= 1
            else:
                return string[: i + 1]

    raise ValueError(
        'No closing paren for """\n{}\n"""\nRemaining extra_closing_parens: {}'.format(
            string, extra_closing_parens
        )
    )
