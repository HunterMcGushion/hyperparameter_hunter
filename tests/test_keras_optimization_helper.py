##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.compat.keras_optimization_helper import (
    clean_parenthesized_string,
    consolidate_layers,
    find_space_fragments,
    merge_compile_params,
    rewrite_model_builder,
)
from hyperparameter_hunter import Real, Categorical

##################################################
# Import Miscellaneous Assets
##################################################
from collections import OrderedDict
import pytest

##################################################
# `consolidate_layers` Scenarios
##################################################
#################### Parametrization Helper Dicts ####################
default_dense = {
    "activation": None,
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
}
default_dropout = {"noise_shape": None, "seed": None}

simple_mlp_layers = [
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": [100],
        "__hh_used_kwargs": dict(kernel_initializer="uniform", input_shape=[30], activation="relu"),
    },
    {
        "class_name": "Dropout",
        "__hh_default_args": ["rate"],
        "__hh_default_kwargs": default_dropout,
        "__hh_used_args": [0.5],
        "__hh_used_kwargs": {},
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": [],
        "__hh_used_kwargs": dict(units=1, kernel_initializer="uniform", activation="sigmoid"),
    },
]

dense_0_kwargs = dict(default_dense, **dict(activation="relu", kernel_initializer="uniform"))
dense_1_kwargs = dict(default_dense, **dict(activation="sigmoid", kernel_initializer="uniform"))

#################### Expected Layers ####################
expected_layers_both_true = [
    dict(class_name="Dense", arg_vals={"units": 100}, kwarg_vals=dense_0_kwargs),
    dict(class_name="Dropout", arg_vals={"rate": 0.5}, kwarg_vals=default_dropout),
    dict(class_name="Dense", arg_vals={"units": 1}, kwarg_vals=dense_1_kwargs),
]
expected_layers_class_name_key_false = [
    {"Dense": dict(arg_vals={"units": 100}, kwarg_vals=dense_0_kwargs)},
    {"Dropout": dict(arg_vals={"rate": 0.5}, kwarg_vals=default_dropout)},
    {"Dense": dict(arg_vals={"units": 1}, kwarg_vals=dense_1_kwargs)},
]
expected_layers_split_args_false = [
    dict(**dict(class_name="Dense", units=100), **dense_0_kwargs),
    dict(**dict(class_name="Dropout", rate=0.5), **default_dropout),
    dict(**dict(class_name="Dense", units=1), **dense_1_kwargs),
]
expected_layers_both_false = [
    {"Dense": dict(**dict(units=100), **dense_0_kwargs)},
    {"Dropout": dict(**dict(rate=0.5), **default_dropout)},
    {"Dense": dict(**dict(units=1), **dense_1_kwargs)},
]


#################### Test `consolidate_layers` Equality ####################
@pytest.mark.parametrize(
    ["expected", "class_name_key", "split_args"],
    [
        pytest.param(expected_layers_both_true, True, True, id="both=true"),
        pytest.param(expected_layers_class_name_key_false, False, True, id="class_name_key=False"),
        pytest.param(expected_layers_split_args_false, True, False, id="split_args=False"),
        pytest.param(expected_layers_both_false, False, False, id="both=false"),
    ],
)
def test_consolidate_layers(expected, class_name_key, split_args):
    assert consolidate_layers(simple_mlp_layers, class_name_key, split_args) == expected


##################################################
# `merge_compile_params` Scenarios
##################################################
@pytest.mark.parametrize(
    "compile_params",
    [
        {
            "compile_kwargs": {},
            "loss_function_names": ["binary_crossentropy"],
            "loss_functions": ["<binary_crossentropy function>"],
            "loss_weights": None,
            "metrics": ["accuracy"],
            "metrics_names": ["loss", "acc"],
            "optimizer": "Adam",
            "optimizer_params": dict(
                amsgrad=False, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=1e-07, lr=0.001
            ),
            "sample_weight_mode": None,
            "target_tensors": None,
            "weighted_metrics": None,
        }
    ],
)
@pytest.mark.parametrize(
    ["dummified_params", "expected"],
    [
        (
            {("params", "optimizer"): Categorical(categories=("adam", "rmsprop"))},
            {
                "compile_kwargs": {},
                "loss_function_names": ["binary_crossentropy"],
                "loss_functions": ["<binary_crossentropy function>"],
                "loss_weights": None,
                "metrics": ["accuracy"],
                "metrics_names": ["loss", "acc"],
                "optimizer": Categorical(categories=("adam", "rmsprop")),
                "optimizer_params": dict(
                    amsgrad=False, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=1e-07, lr=0.001
                ),
                "sample_weight_mode": None,
                "target_tensors": None,
                "weighted_metrics": None,
            },
        ),
        (
            {("params", "optimizer_params", "lr"): Real(0.0001, 0.1)},
            {
                "compile_kwargs": {},
                "loss_function_names": ["binary_crossentropy"],
                "loss_functions": ["<binary_crossentropy function>"],
                "loss_weights": None,
                "metrics": ["accuracy"],
                "metrics_names": ["loss", "acc"],
                "optimizer": "Adam",
                "optimizer_params": dict(
                    amsgrad=False,
                    beta_1=0.9,
                    beta_2=0.999,
                    decay=0.0,
                    epsilon=1e-07,
                    lr=Real(0.0001, 0.1),
                ),
                "sample_weight_mode": None,
                "target_tensors": None,
                "weighted_metrics": None,
            },
        ),
    ],
)
def test_merge_compile_params(compile_params, dummified_params, expected):
    assert merge_compile_params(compile_params, dummified_params) == expected


##################################################
# `clean_parenthesized_string` Scenarios
##################################################
@pytest.mark.parametrize(
    "expected",
    ["Categorical([Dropout(0.5), Activation('linear')])", "Dense(Integer(256, 1024))"],
    # Notice, the beginning remains un-trimmed, despite not starting with `space` class
)
def test_clean_parenthesized_string(expected):
    assert clean_parenthesized_string(expected + " ... I am some extra text") == expected


@pytest.mark.parametrize(
    "string", ["Categorical([Dropout(0.5), Activation('linear')]", "Dense(Integer(256, (1024))"]
)
def test_clean_parenthesized_string_value_error(string):
    with pytest.raises(ValueError):
        clean_parenthesized_string(string)


##################################################
# Build Function/Space Fragment-Finding Objects
##################################################
# The objects defined in this section may be used in a number of tests of different functionality
_expected_params_0 = OrderedDict(
    [
        ("rate", "Real(0.0, 1.0)"),
        ("units", "Integer(256, 1024)"),
        ("kernel_initializer", "Categorical(['glorot_uniform', 'lecun_normal'])"),
        ("Activation", "Categorical(['relu', 'sigmoid'], transform='onehot')"),
        ("Dropout", "Real(low=0.0, high=1.0)"),
        ("loss", "Categorical(['categorical_crossentropy', 'binary_crossentropy'])"),
        ("optimizer", "Categorical(['rmsprop', 'adam', 'sgd'], transform='onehot')"),
    ]
)
_expected_params_1 = OrderedDict(
    [
        ("Dropout", "Real(0.0, 1.0)"),
        ("Dense", "Integer(256, 1024)"),
        ("Activation", "Categorical(['relu', 'sigmoid'], transform='onehot')"),
        ("Dropout_1", "Real(low=0.0, high=1.0)"),
        ("Dropout_2", "Categorical(categories=['three', 'four'])"),
        # FLAG: Above is weird, but probably ok, maybe
        ("add", "Categorical([Dropout(0.5), Activation('linear')])"),
        # TODO: Above layer selection not fully supported
        ("loss", "Categorical(['categorical_crossentropy', 'binary_crossentropy'])"),
        ("optimizer", "Categorical(['rmsprop', 'adam', 'sgd'], transform='identity')"),
    ]
)


def _build_fn_source_0(stage):
    stage_vals = [(_v, "params[{!r}]".format(_k)) for _k, _v in _expected_params_0.items()]
    src = """def {7}:
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(rate={0}, seed=32))
    model.add(Dense(units={1}, kernel_initializer={2}))
    model.add(Activation({3}))
    model.add(Dropout({4}))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
        loss={5},
        metrics=['accuracy'],
        optimizer={6}
    )

    return model"""

    if stage == "original":
        return src.format(*[_[0] for _ in stage_vals], "create_model(input_shape)")
    elif stage == "reusable":
        return src.format(*[_[1] for _ in stage_vals], "build_fn(input_shape=(10, ), params=None)")


def _build_fn_source_1(stage):
    stage_vals = [(_v, "params[{!r}]".format(_k)) for _k, _v in _expected_params_1.items()]
    src = '''def {8}:
    """Keras model-building function that contains hyperparameter space declarations
    Parameters
    ----------
    input_shape: Int
        The shape of the input provided to the first layer of the model

    Returns
    -------
    model: Instance of :class:`keras.Sequential`
        A compiled Keras model"""
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({0}))
    model.add(Dense({1}))
    model.add(Activation({2}))
    model.add(Dropout({3}))

    if {4} == 'four':
        model.add(Dense(100))
        model.add({5})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(
        loss={6}, 
        metrics=['accuracy'], 
        optimizer={7}
    )

    return model'''

    if stage == "original":
        return src.format(*[_[0] for _ in stage_vals], "create_model(input_shape)")
    elif stage == "reusable":
        return src.format(*[_[1] for _ in stage_vals], "build_fn(input_shape=(10, ), params=None)")


##################################################
# `rewrite_model_builder` Scenarios
##################################################
@pytest.mark.parametrize(
    ["src_str", "expected_src_str", "expected_params"],
    [
        [_build_fn_source_0("original"), _build_fn_source_0("reusable"), _expected_params_0],
        [_build_fn_source_1("original"), _build_fn_source_1("reusable"), _expected_params_1],
    ],
    ids=["0", "1"],
)
def test_rewrite_model_builder(src_str, expected_src_str, expected_params):
    assert rewrite_model_builder(src_str) == (expected_src_str, expected_params)


##################################################
# `find_space_fragments` Scenarios
##################################################
@pytest.mark.parametrize(
    ["string", "expected_choices", "expected_names", "expected_indexes"],
    [
        (
            _build_fn_source_0("original"),
            list(_expected_params_0.values()),
            list(_expected_params_0.keys()),
            [168, 220, 259, 334, 411, 533, 647],
        ),
        (
            _build_fn_source_1("original"),
            list(_expected_params_1.values()),
            list(_expected_params_1.keys()),
            [474, 511, 557, 634, 668, 769, 954, 1070],
        ),
    ],
    ids=["0", "1"],
)
def test_find_space_fragments(string, expected_choices, expected_names, expected_indexes):
    assert find_space_fragments(string) == (expected_choices, expected_names, expected_indexes)
