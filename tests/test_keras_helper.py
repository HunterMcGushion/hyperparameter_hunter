##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.compat.keras_helper import get_concise_params_dict
from hyperparameter_hunter.compat.keras_helper import get_keras_attr
from hyperparameter_hunter.compat.keras_helper import keras_callback_to_key
from hyperparameter_hunter.compat.keras_helper import keras_initializer_to_dict
from hyperparameter_hunter.compat.keras_helper import parameterize_compiled_keras_model
from hyperparameter_hunter.compat.keras_helper import parameters_by_signature
from hyperparameter_hunter import Integer

##################################################
# Import Miscellaneous Assets
##################################################
from pkg_resources import get_distribution
import pytest

try:
    keras = pytest.importorskip("keras")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from keras import initializers
from keras import callbacks
from keras.layers import Dense, Dropout, Embedding, Flatten, SpatialDropout1D
from keras.losses import binary_crossentropy, mean_absolute_error
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

##################################################
# Parametrization Helper Dicts
##################################################
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


##################################################
# Dummy Model #0
##################################################
def dummy_0_build_fn(input_shape=(30,)):
    model = Sequential(
        [
            Dense(50, kernel_initializer="uniform", input_shape=input_shape, activation="relu"),
            Dropout(0.5),
            Dense(1, kernel_initializer="uniform", activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


dummy_0_layers = [
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": [50],
        "__hh_used_kwargs": dict(
            kernel_initializer="uniform", input_shape=(30,), activation="relu"
        ),
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
        "__hh_used_args": [1],
        "__hh_used_kwargs": dict(kernel_initializer="uniform", activation="sigmoid"),
    },
]
dummy_0_compile_params = {
    "optimizer": "adam",
    "optimizer_params": Adam().get_config(),
    "metrics": ["accuracy"],
    "metrics_names": ["loss", "acc"],
    "loss_functions": [binary_crossentropy],
    "loss_function_names": ["binary_crossentropy"],
    "loss_weights": None,
    "sample_weight_mode": None,
    "weighted_metrics": None,
    "target_tensors": None,
    "compile_kwargs": {},
}


##################################################
# Dummy Model #1
##################################################
# noinspection PyUnusedLocal
def dummy_1_build_fn(input_shape=(1,)):
    model = Sequential(
        [
            Embedding(input_dim=9999, output_dim=200, input_length=100, trainable=True),
            SpatialDropout1D(rate=0.5),
            Flatten(),
            Dense(100, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=RMSprop(lr=0.02, decay=0.001),
        loss=mean_absolute_error,
        metrics=["mean_absolute_error"],
    )
    return model


dummy_1_layers = [
    {
        "class_name": "Embedding",
        "__hh_default_args": ["input_dim", "output_dim"],
        "__hh_default_kwargs": dict(
            embeddings_initializer="uniform",
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
        ),
        "__hh_used_args": [],
        "__hh_used_kwargs": dict(input_dim=9999, output_dim=200, input_length=100, trainable=True),
    },
    {
        "class_name": "SpatialDropout1D",
        "__hh_default_args": ["rate"],
        "__hh_default_kwargs": dict(),
        "__hh_used_args": [],
        "__hh_used_kwargs": dict(rate=0.5),
    },
    {
        "class_name": "Flatten",
        "__hh_default_args": [],
        "__hh_default_kwargs": (
            dict(data_format=None) if get_distribution("keras").version >= "2.2.0" else {}
        ),
        "__hh_used_args": [],
        "__hh_used_kwargs": dict(),
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": [100],
        "__hh_used_kwargs": dict(activation="relu"),
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": [1],
        "__hh_used_kwargs": dict(activation="sigmoid"),
    },
]
dummy_1_compile_params = {
    "optimizer": "rmsprop",
    "optimizer_params": dict(
        RMSprop().get_config(), **dict(lr=pytest.approx(0.02), decay=pytest.approx(0.001))
    ),
    "metrics": ["mean_absolute_error"],
    "metrics_names": ["loss", "mean_absolute_error"],
    "loss_functions": [mean_absolute_error],
    "loss_function_names": ["mean_absolute_error"],
    "loss_weights": None,
    "sample_weight_mode": None,
    "weighted_metrics": None,
    "target_tensors": None,
    "compile_kwargs": {},
}


##################################################
# `parameterize_compiled_keras_model` Scenarios
##################################################
@pytest.mark.parametrize(
    ["model", "layers", "compile_params"],
    [
        # TODO: Might need to wrap dummy build_fns in `KerasClassifier/Regressor` - That is what actually happens
        (dummy_0_build_fn, dummy_0_layers, dummy_0_compile_params),
        (dummy_1_build_fn, dummy_1_layers, dummy_1_compile_params),
    ],
    ids=["dummy_model_0", "dummy_model_1"],
)
def test_parameterize_compiled_keras_model(model, layers, compile_params):
    assert parameterize_compiled_keras_model(model()) == (layers, compile_params)


##################################################
# Mock Model Class
##################################################
class MockModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def mock_0():
    return MockModel(a="0.a", b="0.b", c="0.c")


def mock_1():
    return MockModel(b="1.b", d="1.d")


def mock_2():
    return MockModel(c="2.c", e="2.e")


def mock_3():
    return MockModel(f="3.f")


def mock_4():
    return MockModel()


##################################################
# `get_keras_attr` Scenarios
##################################################
@pytest.mark.parametrize(
    ["models", "expected", "get_keras_attr_kwargs"],
    [
        ([mock_0, mock_1, mock_2], dict(a="0.a", b="0.b", c="0.c", d="1.d", e="2.e"), {}),
        ([mock_2, mock_1, mock_0], dict(a="0.a", b="1.b", c="2.c", d="1.d", e="2.e"), {}),
        ([mock_2, mock_1, mock_0], dict(b="1.b", c="2.c", d="1.d", e="2.e"), dict(max_depth=2)),
        ([mock_0, mock_4, mock_4], dict(a="0.a", b="0.b", c="0.c"), {}),
        ([mock_4, mock_4, mock_4, mock_0], dict(), {}),
        ([mock_4], dict(a="foo", b="foo", c="foo", d="foo", e="foo", f="foo"), dict(default="foo")),
        ([mock_3, mock_4, mock_2], dict(c="2.c", e="2.e", f="3.f"), {}),
    ],
)
def test_get_keras_attr(models, expected, get_keras_attr_kwargs):
    #################### Build Mock Nested Model ####################
    nested_model = None
    for model in models[::-1]:
        model = model()
        if nested_model:
            setattr(model, "model", nested_model)
        nested_model = model

    #################### Check Expected Attributes ####################
    for attr in list("abcdef"):
        if attr in expected:
            assert get_keras_attr(nested_model, attr, **get_keras_attr_kwargs) == expected[attr]
        else:
            with pytest.raises(AttributeError):
                get_keras_attr(nested_model, attr, **get_keras_attr_kwargs)


##################################################
# `parameters_by_signature` Scenarios
##################################################
@pytest.mark.parametrize(
    ["instance", "signature_filter", "params"],
    [
        (
            callbacks.ReduceLROnPlateau(patience=Integer(5, 10)),
            lambda arg_name, arg_val: arg_name not in ["verbose"],
            dict(
                monitor="val_loss",
                factor=0.1,
                patience=Integer(5, 10),
                mode="auto",
                min_delta=1e-4,
                cooldown=0,
                min_lr=0,
                kwargs=None,
            ),
        ),
        (
            callbacks.ReduceLROnPlateau(patience=Integer(5, 10)),
            None,
            dict(
                monitor="val_loss",
                factor=0.1,
                patience=Integer(5, 10),
                verbose=0,
                mode="auto",
                min_delta=1e-4,
                cooldown=0,
                min_lr=0,
                kwargs=None,
            ),
        ),
        (
            callbacks.EarlyStopping(patience=5, min_delta=0.5),
            lambda arg_name, arg_val: arg_name not in ["verbose"],
            dict(
                monitor="val_loss",
                min_delta=-0.5,
                patience=5,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            ),
        ),
    ],
)
def test_parameters_by_signature(instance, signature_filter, params):
    assert parameters_by_signature(instance, signature_filter) == params


##################################################
# `keras_initializer_to_dict` Scenarios
##################################################
@pytest.mark.parametrize(
    ["initializer", "initializer_dict"],
    [
        #################### Normal Initializers ####################
        pytest.param(initializers.zeros(), dict(class_name="zeros"), id="zero_0"),
        pytest.param(initializers.Zeros(), dict(class_name="zeros"), id="zero_1"),
        pytest.param(initializers.ones(), dict(class_name="ones"), id="one_0"),
        pytest.param(initializers.Ones(), dict(class_name="ones"), id="one_1"),
        pytest.param(initializers.constant(), dict(class_name="constant", value=0), id="c_0"),
        pytest.param(initializers.Constant(5), dict(class_name="constant", value=5), id="c_1"),
        pytest.param(
            initializers.RandomNormal(0.1),
            dict(class_name="random_normal", mean=0.1, stddev=0.05, seed=None),
            id="rn_0",
        ),
        pytest.param(
            initializers.random_normal(mean=0.2, stddev=0.003, seed=42),
            dict(class_name="random_normal", mean=0.2, stddev=0.003, seed=42),
            id="rn_1",
        ),
        pytest.param(
            initializers.RandomUniform(maxval=0.1),
            dict(class_name="random_uniform", minval=-0.05, maxval=0.1, seed=None),
            id="ru_0",
        ),
        pytest.param(
            initializers.random_uniform(minval=-0.2, seed=42),
            dict(class_name="random_uniform", minval=-0.2, maxval=0.05, seed=42),
            id="ru_1",
        ),
        pytest.param(
            initializers.TruncatedNormal(0.1),
            dict(class_name="truncated_normal", mean=0.1, stddev=0.05, seed=None),
            id="tn_0",
        ),
        pytest.param(
            initializers.truncated_normal(mean=0.2, stddev=0.003, seed=42),
            dict(class_name="truncated_normal", mean=0.2, stddev=0.003, seed=42),
            id="tn_1",
        ),
        pytest.param(
            initializers.Orthogonal(1.1),
            dict(class_name="orthogonal", gain=1.1, seed=None),
            id="o_0",
        ),
        pytest.param(
            initializers.orthogonal(gain=1.2, seed=42),
            dict(class_name="orthogonal", gain=1.2, seed=42),
            id="o_1",
        ),
        pytest.param(initializers.Identity(1.1), dict(class_name="identity", gain=1.1), id="i_0"),
        pytest.param(initializers.identity(), dict(class_name="identity", gain=1.0), id="i_1"),
        #################### VarianceScaling ####################
        pytest.param(
            initializers.glorot_normal(), dict(class_name="glorot_normal", seed=None), id="gn_0"
        ),
        pytest.param(
            initializers.glorot_uniform(42), dict(class_name="glorot_uniform", seed=42), id="gu_0"
        ),
        pytest.param(initializers.he_normal(), dict(class_name="he_normal", seed=None), id="hn_0"),
        pytest.param(
            initializers.he_uniform(42), dict(class_name="he_uniform", seed=42), id="hu_0"
        ),
        pytest.param(
            initializers.lecun_normal(), dict(class_name="lecun_normal", seed=None), id="ln_0"
        ),
        pytest.param(
            initializers.lecun_uniform(42), dict(class_name="lecun_uniform", seed=42), id="lu_0"
        ),
    ],
)
def test_keras_initializer_to_dict(initializer, initializer_dict):
    assert get_concise_params_dict(keras_initializer_to_dict(initializer)) == initializer_dict


##################################################
# `get_concise_params_dict` Scenarios
##################################################
hh_arg_attrs = ["__hh_default_args", "__hh_default_kwargs", "__hh_used_args", "__hh_used_kwargs"]
empty_hh_args = [[], {}, [], {}]
_arg_dict = lambda _: dict(zip(hh_arg_attrs, _))


@pytest.mark.parametrize(
    ["params", "expected_params"],
    [
        (dict(), dict()),
        (_arg_dict(empty_hh_args), dict()),
        (dict(name="foo"), dict(name="foo")),
        (dict(name="foo", **_arg_dict(empty_hh_args)), dict(name="foo")),
        (dict(name="foo", **_arg_dict([["a"], {}, [1], {}])), dict(name="foo", a=1)),
        (dict(name="foo", **_arg_dict([["a"], {}, [], dict(a=1)])), dict(name="foo", a=1)),
        (_arg_dict([["a"], dict(b=2, c=3), [1], dict(c=42)]), dict(a=1, b=2, c=42)),
        (_arg_dict([["a"], dict(b=2, c=3), [42], dict()]), dict(a=42, b=2, c=3)),
        (dict(x="F", **_arg_dict([[], dict(b=2, c=3), [], dict()])), dict(x="F", b=2, c=3)),
    ],
)
def test_get_concise_params_dict(params, expected_params):
    assert get_concise_params_dict(params) == expected_params


def test_get_concise_params_dict_index_error():
    with pytest.raises(IndexError):
        get_concise_params_dict(_arg_dict([["a"], {}, [], {}]))


##################################################
# `keras_callback_to_key` Scenarios
##################################################
def test_keras_callback_to_key():
    expected_key = "ReduceLROnPlateau(cooldown=0, factor=0.1, kwargs=None, min_delta=0.0001, min_lr=0, mode='auto', monitor='val_loss', patience=32)"
    assert keras_callback_to_key(callbacks.ReduceLROnPlateau(patience=32)) == expected_key
