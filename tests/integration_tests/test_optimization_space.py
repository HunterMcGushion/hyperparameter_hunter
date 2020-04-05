"""This module contains tests for specific combinations of dimensions and `OptimizationProtocol` s
that have not integrated smoothly in the past. These tests are mostly concerned with
`BayesianOptPro` and `Categorical` dimensions

Related
-------
:mod:`tests.integration_tests.feature_engineering.test_feature_optimization`
    This module defines regression tests for `BayesianOptPro` when given exclusively `Categorical`
    spaces that include `FeatureEngineer`. The tests therein are closely related to these tests,
    except `test_feature_optimization` is focused on `FeatureEngineer` in particular"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import __version__, Environment, Real, Integer, Categorical
from hyperparameter_hunter import BayesianOptPro, GBRT, RF, ET, DummyOptPro
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from hyperparameter_hunter.utils.version_utils import HHVersion

##################################################
# Import Miscellaneous Assets
##################################################
from os import makedirs
import pytest
from shutil import rmtree

try:
    keras = pytest.importorskip("keras")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


@pytest.fixture(scope="function", autouse=False)
def hh_assets():
    """Construct a temporary HyperparameterHunterAssets directory that exists only for the duration
    of the tests contained in each function, before it and its contents are deleted"""
    temp_assets_path = assets_dir
    try:
        makedirs(temp_assets_path)
    except FileExistsError:
        rmtree(temp_assets_path)
        makedirs(temp_assets_path)
    yield


@pytest.fixture()
def env_breast_cancer():
    env = Environment(
        train_dataset=get_breast_cancer_data(target="target"),
        results_path=assets_dir,
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )
    return env


##################################################
# Keras Optimization Tests
##################################################
def _build_tri_cat(input_shape):
    model = Sequential(
        [
            Dense(90, input_shape=input_shape, activation=Categorical(["elu", "selu", "relu"])),
            Dropout(0.5),
            Dense(1, activation=Categorical(["elu", "softsign", "relu", "tanh", "sigmoid"])),
        ]
    )
    model.compile(
        optimizer=Categorical(["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_tri_cat_real(input_shape):
    model = Sequential(
        [
            Dense(90, input_shape=input_shape, activation=Categorical(["elu", "selu", "relu"])),
            Dropout(Real(0.2, 0.7)),
            Dense(1, activation=Categorical(["selu", "softsign", "relu", "tanh", "sigmoid"])),
        ]
    )
    model.compile(
        optimizer=Categorical(["adam", "rmsprop"]), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def _build_penta_cat(input_shape):
    model = Sequential(
        [
            Dense(
                90,
                kernel_initializer=Categorical(["lecun_uniform", "lecun_normal", "glorot_normal"]),
                input_shape=input_shape,
                activation=Categorical(["elu", "selu", "softsign", "relu", "tanh", "sigmoid"]),
            ),
            Dropout(0.5),
            Dense(
                1,
                kernel_initializer=Categorical(["lecun_uniform", "lecun_normal", "glorot_normal"]),
                activation=Categorical(["elu", "selu", "softsign", "relu", "tanh", "sigmoid"]),
            ),
        ]
    )
    model.compile(
        optimizer=Categorical(["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_penta_cat_int(input_shape):
    model = Sequential(
        [
            Dense(
                Integer(50, 100),
                kernel_initializer=Categorical(["lecun_uniform", "lecun_normal", "glorot_normal"]),
                input_shape=input_shape,
                activation=Categorical(["elu", "selu", "softsign", "relu", "tanh", "sigmoid"]),
            ),
            Dropout(0.5),
            Dense(
                1,
                kernel_initializer=Categorical(["lecun_uniform", "lecun_normal", "glorot_normal"]),
                activation=Categorical(["elu", "selu", "softsign", "relu", "tanh", "sigmoid"]),
            ),
        ]
    )
    model.compile(
        optimizer=Categorical(["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def do_optimization(opt_pro, build_fn, ready_only=False):
    opt = opt_pro(iterations=2, random_state=32, n_initial_points=1)
    opt.forge_experiment(
        KerasClassifier, build_fn, model_extra_params=dict(batch_size=32, epochs=1, verbose=0)
    )

    if ready_only:
        opt.get_ready()
    else:
        opt.go()

    return opt


@pytest.mark.parametrize("opt_pro", [DummyOptPro, ET, GBRT, RF])
@pytest.mark.parametrize("build_fn", [_build_tri_cat, _build_tri_cat_real])
# @pytest.mark.parametrize(
#     "build_fn", [_build_tri_cat, _build_tri_cat_real, _build_penta_cat, _build_penta_cat_int],
# )
def test_multi_cat_keras_non_bayes(opt_pro, build_fn, hh_assets, env_breast_cancer):
    opt_0 = do_optimization(opt_pro, build_fn)
    opt_1 = do_optimization(opt_pro, build_fn, ready_only=True)
    assert len(opt_1.similar_experiments) == 2


@pytest.mark.parametrize(
    "build_fn",
    [
        pytest.param(
            _build_tri_cat,
            marks=pytest.mark.xfail(
                condition="HHVersion(__version__) <= '3.0.0alpha2'",
                reason="BayesianOptPro hates exclusively-Categorical spaces",
            ),
        ),
        _build_tri_cat_real,
    ],
)
# @pytest.mark.parametrize(
#     "build_fn", [_build_tri_cat, _build_tri_cat_real, _build_penta_cat, _build_penta_cat_int],
# )
def test_multi_cat_keras_bayes(build_fn, hh_assets, env_breast_cancer):
    opt_0 = do_optimization(BayesianOptPro, build_fn)
    opt_1 = do_optimization(BayesianOptPro, build_fn)
    # Note that above does not use `ready_only=True` like :func:`test_multi_cat_keras_non_bayes`.
    # Let it run through the whole optimization process to make sure there aren't any other
    #   problems further along as a result of Experiment-matching
    assert len(opt_1.similar_experiments) == 2
