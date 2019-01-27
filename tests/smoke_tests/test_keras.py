##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, Real, Integer, Categorical, DummySearch
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data, get_diabetes_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

try:
    keras = pytest.importorskip("keras")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.initializers import glorot_normal, orthogonal

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# Environment Fixtures
##################################################
@pytest.fixture(scope="function", autouse=False)
def env_0():
    return Environment(
        train_dataset=get_breast_cancer_data(target="target"),
        root_results_path=assets_dir,
        metrics_map=["roc_auc_score"],
        cross_validation_type="StratifiedKFold",
        cross_validation_params=dict(n_splits=2, shuffle=True, random_state=32),
    )


@pytest.fixture(scope="function", autouse=False)
def env_2():
    return Environment(
        train_dataset=get_diabetes_data(target="target"),
        root_results_path=assets_dir,
        metrics_map=["mean_absolute_error"],
        cross_validation_type="KFold",
        cross_validation_params=dict(n_splits=2, shuffle=True, random_state=32),
    )


##################################################
# Optimization Protocol Fixtures
##################################################
def _build_fn_optimization(input_shape):
    model = Sequential(
        [
            Dense(
                Integer(50, 100),
                kernel_initializer="uniform",
                input_shape=input_shape,
                activation="relu",
            ),
            Dropout(Real(0.2, 0.7)),
            Dense(1, kernel_initializer="uniform", activation=Categorical(["sigmoid", "relu"])),
        ]
    )
    model.compile(
        optimizer=Categorical(["adam", "rmsprop"]), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


@pytest.fixture(scope="function", autouse=False)
def opt_keras_0():
    optimizer = DummySearch(iterations=1)
    optimizer.set_experiment_guidelines(
        model_initializer=KerasClassifier,
        model_init_params=dict(build_fn=_build_fn_optimization),
        model_extra_params=dict(
            callbacks=[ReduceLROnPlateau(patience=Integer(5, 10))],
            batch_size=Categorical([32, 64], transform="onehot"),
            epochs=5,
            verbose=0,
        ),
    )
    optimizer.go()


##################################################
# Test Scenarios
##################################################
def test_classification_optimization(env_0, opt_keras_0):
    ...


##################################################
# `kernel_initializer` Callables (Issue #111)
##################################################
def _build_fn_kernel_initializer_0(input_shape):
    model = Sequential(
        [
            Dense(
                100, activation="relu", input_shape=input_shape, kernel_initializer="glorot_uniform"
            ),
            # TODO: 1-26-19 - Actually is hashing below instance, but doesn't save source anywhere - Probably because un-traced objects
            Dense(50, activation="relu", kernel_initializer=orthogonal(gain=0.9)),
            Dense(1, activation="relu", kernel_initializer=glorot_normal(seed=32)),
        ]
    )
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mean_absolute_error"])
    return model


@pytest.fixture(scope="function", autouse=False)
def exp_kernel_initializer_0():
    return CVExperiment(
        model_initializer=KerasRegressor,
        model_init_params=_build_fn_kernel_initializer_0,
        model_extra_params=dict(batch_size=32, epochs=10, verbose=0),
    )


def test_kernel_initializer_callable(env_2, exp_kernel_initializer_0):
    ...
