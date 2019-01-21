##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, Real, Integer, Categorical, DummySearch
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data

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
from keras.wrappers.scikit_learn import KerasClassifier

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
        cross_validation_params=dict(n_splits=3, shuffle=True, random_state=32),
    )


##################################################
# Optimization Protocol Fixtures
##################################################
def _build_fn_optimization(input_shape):
    model = Sequential(
        [
            Dense(
                Integer(50, 150),
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
    optimizer = DummySearch(iterations=2)
    optimizer.set_experiment_guidelines(
        model_initializer=KerasClassifier,
        model_init_params=dict(build_fn=_build_fn_optimization),
        model_extra_params=dict(
            callbacks=[ReduceLROnPlateau(patience=Integer(5, 10))],
            batch_size=Categorical([32, 64], transform="onehot"),
            epochs=10,
            verbose=0,
        ),
    )
    optimizer.go()


##################################################
# Test Scenarios
##################################################
def test_classification_optimization(env_0, opt_keras_0):
    ...
