##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, Real, Integer, Categorical
from hyperparameter_hunter import DummyOptPro, BayesianOptPro
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data, get_diabetes_data

##################################################
# Import Miscellaneous Assets
##################################################
import pandas as pd
import pytest

try:
    keras = pytest.importorskip("keras")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.initializers import glorot_normal, orthogonal, Orthogonal
from sklearn.datasets import load_digits

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


def get_digits_data(n_class=2):
    input_data, target_data = load_digits(n_class=n_class, return_X_y=True)
    train_df = pd.DataFrame(data=input_data)
    train_df["target"] = target_data
    return train_df


##################################################
# Environment Fixtures
##################################################
@pytest.fixture(scope="function", autouse=False)
def env_0():
    return Environment(
        train_dataset=get_diabetes_data(target="target"),
        results_path=assets_dir,
        metrics=["mean_absolute_error"],
        cv_type="KFold",
        cv_params=dict(n_splits=2, shuffle=True, random_state=32),
    )


def initialization_matching_env():
    return Environment(
        train_dataset=get_breast_cancer_data(target="target"),
        results_path=assets_dir,
        metrics=["roc_auc_score"],
        cv_type="KFold",
        cv_params=dict(n_splits=2, shuffle=True, random_state=32),
    )


@pytest.fixture(scope="function", autouse=False)
def env_digits():
    return Environment(
        train_dataset=get_digits_data(),
        results_path=assets_dir,
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=3, shuffle=True, random_state=32),
    )


##################################################
# Regressor Optimization
##################################################
def _build_fn_regressor(input_shape):
    model = Sequential(
        [
            Dense(100, activation="relu", input_shape=input_shape),
            Dense(Integer(40, 60), activation="relu", kernel_initializer="glorot_normal"),
            Dropout(Real(0.2, 0.7)),
            Dense(1, activation=Categorical(["relu", "sigmoid"]), kernel_initializer="orthogonal"),
        ]
    )
    model.compile(
        optimizer=Categorical(["adam", "rmsprop"]),
        loss="mean_absolute_error",
        metrics=["mean_absolute_error"],
    )
    return model


@pytest.fixture(scope="function", autouse=False)
def opt_regressor():
    optimizer = DummyOptPro(iterations=1)
    optimizer.forge_experiment(
        model_initializer=KerasRegressor,
        model_init_params=_build_fn_regressor,
        model_extra_params=dict(
            callbacks=[ReduceLROnPlateau(patience=Integer(5, 10))],
            batch_size=Categorical([32, 64], transform="onehot"),
            epochs=10,
            verbose=0,
        ),
    )
    optimizer.go()


def test_regressor_opt(env_0, opt_regressor):
    ...


##################################################
# Categorical Tuple Optimization
##################################################
def build_fn_digits_exp(input_shape=-1):
    model = Sequential(
        [
            Reshape((8, 8, -1), input_shape=(64,)),
            Conv2D(filters=32, kernel_size=(5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_fn_digits_opt(input_shape=-1):
    model = Sequential(
        [
            Reshape((8, 8, -1), input_shape=(64,)),
            Conv2D(32, kernel_size=Categorical([(3, 3), (5, 5)]), activation="relu"),
            MaxPooling2D(pool_size=Categorical([(2, 2), (3, 3)])),
            Dropout(0.5),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def test_categorical_tuple_match(env_digits):
    """Test that optimization of a `Categorical` space, whose values are tuples can be performed
    and that saved results from such a space are correctly identified as similar Experiments"""
    model_extra_params = dict(batch_size=32, epochs=3, verbose=0, shuffle=True)
    exp_0 = CVExperiment(KerasClassifier, build_fn_digits_exp, model_extra_params)

    #################### First OptPro ####################
    opt_0 = BayesianOptPro(iterations=1, random_state=32, n_initial_points=1)
    opt_0.forge_experiment(KerasClassifier, build_fn_digits_opt, model_extra_params)
    opt_0.go()
    assert len(opt_0.similar_experiments) == 1  # Should match `exp_0`

    #################### Second OptPro ####################
    opt_1 = BayesianOptPro(iterations=1, random_state=32, n_initial_points=1)
    opt_1.forge_experiment(KerasClassifier, build_fn_digits_opt, model_extra_params)
    opt_1.go()
    assert len(opt_1.similar_experiments) == 2  # Should match `exp_0` and `opt_0`


##################################################
# Keras Initialization Matching Tests
##################################################
def in_similar_experiment_ids(opt_0, opt_1):
    """Determine whether the `experiment_id` of `opt_0`'s `current_experiment` is included in
    `opt_1`'s list of `similar_experiments`"""
    return opt_0.current_experiment.experiment_id in [_[-1] for _ in opt_1.similar_experiments]


def run_initialization_matching_optimization_0(build_fn):
    optimizer = DummyOptPro(iterations=1)
    optimizer.forge_experiment(
        model_initializer=KerasClassifier,
        model_init_params=dict(build_fn=build_fn),
        model_extra_params=dict(epochs=1, batch_size=128, verbose=0),
    )
    optimizer.go()
    return optimizer


#################### `glorot_normal` (`VarianceScaling`) ####################
def _build_fn_glorot_normal_0(input_shape):  # `glorot_normal()`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=glorot_normal()),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_glorot_normal_1(input_shape):  # `"glorot_normal"`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer="glorot_normal"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


#################### `orthogonal` - Excluding default (`Initializer`) ####################
def _build_fn_orthogonal_e_0(input_shape):  # `orthogonal(gain=Real(0.3, 0.9))`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=orthogonal(gain=Real(0.3, 0.9))),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_e_1(input_shape):  # `Orthogonal(gain=Real(0.3, 0.9))`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Orthogonal(gain=Real(0.3, 0.9))),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_e_2(input_shape):  # `orthogonal(gain=0.5)`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=orthogonal(gain=0.5)),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_e_3(input_shape):  # `Orthogonal(gain=0.5)`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Orthogonal(gain=0.5)),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


#################### `orthogonal` - Including default (`Initializer`) ####################
def _build_fn_orthogonal_i_0(input_shape):  # `"orthogonal"`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer="orthogonal"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_i_1(input_shape):  # `orthogonal()`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=orthogonal()),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_i_2(input_shape):  # `Orthogonal()`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Orthogonal()),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_i_3(input_shape):  # `orthogonal(gain=1.0)`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=orthogonal(gain=1.0)),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_i_4(input_shape):  # `Orthogonal(gain=1.0)`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Orthogonal(gain=1.0)),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_i_5(input_shape):  # `orthogonal(gain=Real(0.6, 1.6))`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=orthogonal(gain=Real(0.6, 1.6))),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_orthogonal_i_6(input_shape):  # `Orthogonal(gain=Real(0.6, 1.6))`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Orthogonal(gain=Real(0.6, 1.6))),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


#################### Categorical Initializers ####################
def _build_fn_categorical_0(input_shape):  # `Categorical(["glorot_normal", "orthogonal"])`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Categorical(["glorot_normal", "orthogonal"])),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_categorical_1(input_shape):  # `Categorical([glorot_normal(), orthogonal()])`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Categorical([glorot_normal(), orthogonal()])),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_categorical_2(input_shape):  # `Categorical([glorot_normal(), Orthogonal()])`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Categorical([glorot_normal(), Orthogonal()])),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_categorical_3(input_shape):  # `Categorical(["glorot_normal", orthogonal(gain=1)])`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Categorical(["glorot_normal", orthogonal(gain=1)])),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_categorical_4(input_shape):  # `Categorical(["glorot_normal", Orthogonal(gain=1)])`
    model = Sequential(
        [
            Dense(Integer(50, 100), input_shape=input_shape),
            Dense(1, kernel_initializer=Categorical(["glorot_normal", Orthogonal(gain=1)])),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


xfail_string_callable = pytest.mark.xfail(reason="Strings vs. callables")
xfail_explicit_default = pytest.mark.xfail(reason="Explicit default args vs. arg-less callables")


class TestInitializerMatching(object):
    """Test proper matching of experiments when optimizing various values of Keras Initializers.

    Tests in this class are organized in several sections dedicated to the following: 1) Testing
    standard Keras `Initializer`s with kwarg values outside the default range; 2) Testing standard
    Keras `Initializer`s with kwarg values inside (or equal to) the default range; 3) Testing Keras
    initializers that return `VarianceScaling` instances; and 4) Testing `Categorical` optimization
    on the initializer values, themselves, rather than their parameters.

    In all but the final section, optimization will be performed twice using the same `build_fn` and
    saved to class attributes prefixed with "opt". The first execution is to create the experiment
    records, which are then located by the second execution. Tests check that certain experiments
    conducted by the first execution are identified by the second execution as being "similar"
    experiments"""

    env = initialization_matching_env()

    ##################################################
    # `orthogonal` - Excluding default (`Initializer`)
    ##################################################
    opt_a_0 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_0)
    opt_a_1 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_1)
    opt_a_2 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_2)
    opt_a_3 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_3)

    opt_b_0 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_0)
    opt_b_1 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_1)
    opt_b_2 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_2)
    opt_b_3 = run_initialization_matching_optimization_0(_build_fn_orthogonal_e_3)

    @pytest.mark.parametrize(
        "old_opt",
        [opt_a_0, opt_a_1, opt_a_2, opt_a_3],
        ids=["o(gain=Real(0.3, 0.9))", "O(gain=Real(0.3, 0.9))", "o(gain=0.5)", "O(gain=0.5)"],
    )
    @pytest.mark.parametrize(
        "new_opt", [opt_b_0, opt_b_1], ids=["o(gain=Real(0.3, 0.9))", "O(gain=Real(0.3, 0.9))"]
    )
    def test_in_space_exclusive_callable(self, old_opt, new_opt):
        assert in_similar_experiment_ids(old_opt, new_opt)

    @pytest.mark.parametrize("old_opt", [opt_a_2, opt_a_3], ids=["o(gain=0.5)", "O(gain=0.5)"])
    @pytest.mark.parametrize("new_opt", [opt_b_2, opt_b_3], ids=["o(gain=0.5)", "O(gain=0.5)"])
    def test_in_custom_arg_callable(self, old_opt, new_opt):
        assert in_similar_experiment_ids(old_opt, new_opt)

    ##################################################
    # `orthogonal` - Including default (`Initializer`)
    ##################################################
    opt_c_0 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_0)
    opt_c_1 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_1)
    opt_c_2 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_2)
    opt_c_3 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_3)
    opt_c_4 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_4)
    opt_c_5 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_5)
    opt_c_6 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_6)

    opt_d_0 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_0)
    opt_d_1 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_1)
    opt_d_2 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_2)
    opt_d_3 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_3)
    opt_d_4 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_4)
    opt_d_5 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_5)
    opt_d_6 = run_initialization_matching_optimization_0(_build_fn_orthogonal_i_6)

    #################### test_in_orthogonal_string ####################
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'"),
            pytest.param(opt_c_1, id="o()", marks=xfail_string_callable),
            pytest.param(opt_c_2, id="O()", marks=xfail_string_callable),
            pytest.param(opt_c_3, id="o(gain=1.0)", marks=xfail_string_callable),
            pytest.param(opt_c_4, id="O(gain=1.0)", marks=xfail_string_callable),
        ],
    )
    def test_in_orthogonal_string(self, old_opt):
        assert in_similar_experiment_ids(old_opt, self.opt_d_0)

    #################### test_in_empty_orthogonal_callable ####################
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'", marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()"),
            pytest.param(opt_c_2, id="O()"),
            pytest.param(opt_c_3, id="o(gain=1.0)", marks=xfail_explicit_default),
            pytest.param(opt_c_4, id="O(gain=1.0)", marks=xfail_explicit_default),
        ],
    )
    @pytest.mark.parametrize("new_opt", [opt_d_1, opt_d_2], ids=["o()", "O()"])
    def test_in_empty_orthogonal_callable(self, old_opt, new_opt):
        assert in_similar_experiment_ids(old_opt, new_opt)

    #################### test_in_default_arg_callable ####################
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'", marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()", marks=xfail_explicit_default),
            pytest.param(opt_c_2, id="O()", marks=xfail_explicit_default),
            pytest.param(opt_c_3, id="o(gain=1.0)"),
            pytest.param(opt_c_4, id="O(gain=1.0)"),
        ],
    )
    @pytest.mark.parametrize("new_opt", [opt_d_3, opt_d_4], ids=["o(gain=1.0)", "O(gain=1.0)"])
    def test_in_default_arg_callable(self, old_opt, new_opt):
        assert in_similar_experiment_ids(old_opt, new_opt)

    #################### test_in_space_inclusive_callable ####################
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'", marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()", marks=xfail_explicit_default),
            pytest.param(opt_c_2, id="O()", marks=xfail_explicit_default),
            pytest.param(opt_c_3, id="o(gain=1.0)"),
            pytest.param(opt_c_4, id="O(gain=1.0)"),
            pytest.param(opt_c_5, id="o(gain=Real(0.6, 1.6))"),
            pytest.param(opt_c_6, id="O(gain=Real(0.6, 1.6))"),
        ],
    )
    @pytest.mark.parametrize(
        "new_opt", [opt_d_5, opt_d_6], ids=["o(gain=Real(0.6, 1.6))", "O(gain=Real(0.6, 1.6))"]
    )
    def test_in_space_inclusive_callable(self, old_opt, new_opt):
        assert in_similar_experiment_ids(old_opt, new_opt)

    ##################################################
    # `glorot_normal` (`VarianceScaling`)
    ##################################################
    opt_e_0 = run_initialization_matching_optimization_0(_build_fn_glorot_normal_0)
    opt_e_1 = run_initialization_matching_optimization_0(_build_fn_glorot_normal_1)

    opt_f_0 = run_initialization_matching_optimization_0(_build_fn_glorot_normal_0)
    opt_f_1 = run_initialization_matching_optimization_0(_build_fn_glorot_normal_1)

    #################### test_in_empty_glorot_normal_callable ####################
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_e_0, id="glorot_normal()"),
            pytest.param(opt_e_1, id="'glorot_normal'", marks=xfail_string_callable),
        ],
    )
    def test_in_empty_glorot_normal_callable(self, old_opt):
        assert in_similar_experiment_ids(old_opt, self.opt_f_0)

    #################### test_in_glorot_normal_string ####################
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_e_0, id="glorot_normal()", marks=xfail_string_callable),
            pytest.param(opt_e_1, id="'glorot_normal'"),
        ],
    )
    def test_in_glorot_normal_string(self, old_opt):
        assert in_similar_experiment_ids(old_opt, self.opt_f_1)

    ##################################################
    # Categorical Initializers
    ##################################################
    opt_g_0 = run_initialization_matching_optimization_0(_build_fn_categorical_0)

    # opt_g_1 = run_initialization_matching_optimization_0(_build_fn_categorical_1)
    # opt_g_2 = run_initialization_matching_optimization_0(_build_fn_categorical_2)
    # opt_g_3 = run_initialization_matching_optimization_0(_build_fn_categorical_3)
    # opt_g_4 = run_initialization_matching_optimization_0(_build_fn_categorical_4)

    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'"),
            pytest.param(opt_c_1, id="o()", marks=xfail_string_callable),
            pytest.param(opt_c_2, id="O()", marks=xfail_string_callable),
            pytest.param(opt_c_3, id="o(gain=1.0)", marks=xfail_string_callable),
            pytest.param(opt_c_4, id="O(gain=1.0)", marks=xfail_string_callable),
            pytest.param(opt_e_0, id="glorot_normal()", marks=xfail_string_callable),
            pytest.param(opt_e_1, id="'glorot_normal'"),
        ],
    )
    def test_in_categorical_0(self, old_opt):  # `Categorical(["glorot_normal", "o"])`
        assert in_similar_experiment_ids(old_opt, self.opt_g_0)

    @pytest.mark.skip(reason="Optimization of selection between callable initializers unavailable")
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'"),  # , marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()"),
            pytest.param(opt_c_2, id="O()"),
            pytest.param(opt_c_3, id="o(gain=1.0)"),  # , marks=xfail_explicit_default),
            pytest.param(opt_c_4, id="O(gain=1.0)"),  # , marks=xfail_explicit_default),
            pytest.param(opt_e_0, id="glorot_normal()"),
            pytest.param(opt_e_1, id="'glorot_normal'"),  # , marks=xfail_string_callable),
        ],
    )
    def test_in_categorical_1(self, old_opt):  # `Categorical([glorot_normal(), o()])`
        assert in_similar_experiment_ids(old_opt, self.opt_g_1)

    @pytest.mark.skip(reason="Optimization of selection between callable initializers unavailable")
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'"),  # , marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()"),
            pytest.param(opt_c_2, id="O()"),
            pytest.param(opt_c_3, id="o(gain=1.0)"),  # , marks=xfail_explicit_default),
            pytest.param(opt_c_4, id="O(gain=1.0)"),  # , marks=xfail_explicit_default),
            pytest.param(opt_e_0, id="glorot_normal()"),
            pytest.param(opt_e_1, id="'glorot_normal'"),  # , marks=xfail_string_callable),
        ],
    )
    def test_in_categorical_2(self, old_opt):  # `Categorical([glorot_normal(), O()])`
        assert in_similar_experiment_ids(old_opt, self.opt_g_2)

    @pytest.mark.skip(reason="Optimization of selection between callable initializers unavailable")
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'"),  # , marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()"),  # , marks=xfail_explicit_default),
            pytest.param(opt_c_2, id="O()"),  # , marks=xfail_explicit_default),
            pytest.param(opt_c_3, id="o(gain=1.0)"),
            pytest.param(opt_c_4, id="O(gain=1.0)"),
            pytest.param(opt_e_0, id="glorot_normal()"),  # , marks=xfail_string_callable),
            pytest.param(opt_e_1, id="'glorot_normal'"),
        ],
    )
    def test_in_categorical_3(self, old_opt):  # `Categorical(["glorot_normal", o(gain=1)])`
        assert in_similar_experiment_ids(old_opt, self.opt_g_3)

    @pytest.mark.skip(reason="Optimization of selection between callable initializers unavailable")
    @pytest.mark.parametrize(
        "old_opt",
        [
            pytest.param(opt_c_0, id="'o'"),  # , marks=xfail_string_callable),
            pytest.param(opt_c_1, id="o()"),  # , marks=xfail_explicit_default),
            pytest.param(opt_c_2, id="O()"),  # , marks=xfail_explicit_default),
            pytest.param(opt_c_3, id="o(gain=1.0)"),
            pytest.param(opt_c_4, id="O(gain=1.0)"),
            pytest.param(opt_e_0, id="glorot_normal()"),  # , marks=xfail_string_callable),
            pytest.param(opt_e_1, id="'glorot_normal'"),
        ],
    )
    def test_in_categorical_4(self, old_opt):  # `Categorical(["glorot_normal", O(gain=1)])`
        assert in_similar_experiment_ids(old_opt, self.opt_g_4)
