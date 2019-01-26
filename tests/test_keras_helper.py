##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.library_helpers.keras_helper import get_keras_attr
from hyperparameter_hunter.library_helpers.keras_helper import parameterize_compiled_keras_model

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
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, Embedding, Flatten, SpatialDropout1D
from keras.losses import binary_crossentropy, mean_absolute_error
from keras.models import Sequential

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
        "__hh_used_args": (50,),
        "__hh_used_kwargs": dict(
            kernel_initializer="uniform", input_shape=(30,), activation="relu"
        ),
    },
    {
        "class_name": "Dropout",
        "__hh_default_args": ["rate"],
        "__hh_default_kwargs": default_dropout,
        "__hh_used_args": (0.5,),
        "__hh_used_kwargs": {},
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": (1,),
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
        "__hh_used_args": tuple(),
        "__hh_used_kwargs": dict(input_dim=9999, output_dim=200, input_length=100, trainable=True),
    },
    {
        "class_name": "SpatialDropout1D",
        "__hh_default_args": ["rate"],
        "__hh_default_kwargs": dict(),
        "__hh_used_args": tuple(),
        "__hh_used_kwargs": dict(rate=0.5),
    },
    {
        "class_name": "Flatten",
        "__hh_default_args": [],
        "__hh_default_kwargs": (
            dict(data_format=None) if get_distribution("keras").version >= "2.2.0" else {}
        ),
        "__hh_used_args": tuple(),
        "__hh_used_kwargs": dict(),
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": (100,),
        "__hh_used_kwargs": dict(activation="relu"),
    },
    {
        "class_name": "Dense",
        "__hh_default_args": ["units"],
        "__hh_default_kwargs": default_dense,
        "__hh_used_args": (1,),
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
