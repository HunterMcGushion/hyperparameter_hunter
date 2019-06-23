"""This module tests performance of hyperparameter optimization on feature engineering steps"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer, EngineerStep
from hyperparameter_hunter import Categorical, Integer, BayesianOptPro, GBRT, RF, ET, DummyOptPro
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data, get_boston_data
from hyperparameter_hunter.utils.optimization_utils import get_choice_dimensions

##################################################
# Import Miscellaneous Assets
##################################################
import numpy as np
from os import makedirs
import pandas as pd
import pytest
from shutil import rmtree
import sys

##################################################
# Import Learning Assets
##################################################
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, QuantileTransformer, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    pass

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# Feature Engineering Steps
##################################################
#################### Target Transformers ####################
def quantile_transform(train_targets, non_train_targets):
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=100)
    train_targets[train_targets.columns] = transformer.fit_transform(train_targets.values)
    non_train_targets[train_targets.columns] = transformer.transform(non_train_targets.values)
    return train_targets, non_train_targets, transformer


def quantile_transform_no_invert(train_targets, non_train_targets):
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=100)
    train_targets[train_targets.columns] = transformer.fit_transform(train_targets.values)
    non_train_targets[train_targets.columns] = transformer.transform(non_train_targets.values)
    return train_targets, non_train_targets


def nothing_transform(train_targets, non_train_targets):
    return train_targets, non_train_targets, lambda _: _


#################### Input Transformers ####################
def min_max_scale(train_inputs, non_train_inputs):
    scaler = MinMaxScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    return train_inputs, non_train_inputs


def normalize(train_inputs, non_train_inputs):
    normalizer = Normalizer()
    train_inputs[train_inputs.columns] = normalizer.fit_transform(train_inputs.values)
    non_train_inputs[train_inputs.columns] = normalizer.transform(non_train_inputs.values)
    return train_inputs, non_train_inputs


def standard_scale(train_inputs, non_train_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    return train_inputs, non_train_inputs


#################### Feature Builders ####################
def sqr_sum_feature(all_inputs):
    """De Gua's theorem (Pythagorean theorem in n-dimensional space)"""
    all_inputs["square_sum_pre"] = all_inputs.agg(
        lambda row: np.sqrt(np.sum([np.square(_) for _ in row])), axis="columns"
    )
    return all_inputs


#################### Re-Samplers ####################
def pos_upsample(train_inputs, train_targets):
    # Find and add positive examples
    sample_mask = pd.Series(train_targets["target"] == 1)
    train_inputs = pd.concat([train_inputs, train_inputs.loc[sample_mask]], axis=0)
    train_targets = pd.concat([train_targets, train_targets.loc[sample_mask]], axis=0)
    # Shuffle data
    idx = np.arange(len(train_inputs))
    np.random.shuffle(idx)
    train_inputs = train_inputs.iloc[idx]
    train_targets = train_targets.iloc[idx]
    return train_inputs, train_targets


def neg_upsample(train_inputs, train_targets):
    # Find and add negative examples
    sample_mask = pd.Series(train_targets["target"] == 0)
    train_inputs = pd.concat([train_inputs, train_inputs.loc[sample_mask]], axis=0)
    train_targets = pd.concat([train_targets, train_targets.loc[sample_mask]], axis=0)
    # Shuffle data
    idx = np.arange(len(train_inputs))
    np.random.shuffle(idx)
    train_inputs = train_inputs.iloc[idx]
    train_targets = train_targets.iloc[idx]
    return train_inputs, train_targets


##################################################
# `get_choice_dimensions` Scenarios
##################################################
@pytest.fixture()
def feature_engineer(request):
    return FeatureEngineer(steps=request.param)


class ChoiceUpsample:
    functions = Categorical([pos_upsample, neg_upsample])
    engineers = Categorical([EngineerStep(pos_upsample), EngineerStep(neg_upsample)])


class ChoiceNormalizeSS:
    functions = Categorical([normalize, standard_scale])
    engineers = Categorical([EngineerStep(normalize), EngineerStep(standard_scale)])


class ChoiceMMNormalizeSS:
    functions = Categorical([min_max_scale, normalize, standard_scale])
    engineers = Categorical(
        [EngineerStep(min_max_scale), EngineerStep(normalize), EngineerStep(standard_scale)]
    )
    o_functions = Categorical([min_max_scale, normalize, standard_scale], optional=True)
    o_engineers = Categorical(
        [EngineerStep(min_max_scale), EngineerStep(normalize), EngineerStep(standard_scale)],
        optional=True,
    )


class ChoiceTarget:
    functions = Categorical([quantile_transform, nothing_transform])
    engineers = Categorical([EngineerStep(quantile_transform), EngineerStep(nothing_transform)])


@pytest.mark.parametrize(
    ["feature_engineer", "expected_choices"],
    [
        ([pos_upsample, min_max_scale, quantile_transform], []),
        (
            [ChoiceUpsample.functions, ChoiceNormalizeSS.functions, quantile_transform],
            [(("steps", 0), ChoiceUpsample.engineers), (("steps", 1), ChoiceNormalizeSS.engineers)],
        ),
        (
            [ChoiceUpsample.engineers, ChoiceNormalizeSS.engineers, quantile_transform],
            [(("steps", 0), ChoiceUpsample.engineers), (("steps", 1), ChoiceNormalizeSS.engineers)],
        ),
        (
            [ChoiceMMNormalizeSS.functions, nothing_transform, ChoiceUpsample.functions],
            [
                (("steps", 0), ChoiceMMNormalizeSS.engineers),
                (("steps", 2), ChoiceUpsample.engineers),
            ],
        ),
        (
            [ChoiceMMNormalizeSS.engineers, nothing_transform, ChoiceUpsample.engineers],
            [
                (("steps", 0), ChoiceMMNormalizeSS.engineers),
                (("steps", 2), ChoiceUpsample.engineers),
            ],
        ),
    ],
    indirect=["feature_engineer"],
)
def test_is_choice_dimension(feature_engineer, expected_choices):
    choices = get_choice_dimensions(
        feature_engineer, iter_attrs=lambda p, k, v: isinstance(v, FeatureEngineer)
    )
    assert choices == expected_choices


#################### ChoiceUpsample Contains ####################
@pytest.mark.parametrize(
    "space_item",
    [
        pytest.param(EngineerStep(pos_upsample), id="E(pos_upsample)"),
        pytest.param(EngineerStep(neg_upsample), id="E(neg_upsample)"),
        pytest.param(EngineerStep(pos_upsample, stage="pre_cv"), id="E(pos_upsample, stage)"),
        pytest.param(
            EngineerStep(neg_upsample, params=("train_inputs", "train_targets")),
            id="E(neg_upsample, params)",
        ),
    ],
)
def test_in_upsample_space(space_item):
    assert space_item in ChoiceUpsample.engineers


@pytest.mark.parametrize(
    "space_item",
    [
        pytest.param(EngineerStep(pos_upsample, stage="intra_cv"), id="E(pos_upsample, bad_stage"),
        pytest.param(EngineerStep(neg_upsample, name="Carl"), id="E(neg_upsample, bad_name"),
        pytest.param(
            EngineerStep(neg_upsample, params=("train_targets", "train_inputs")),
            id="E(neg_upsample, bad_params",
        ),
        pytest.param(pos_upsample, id="pos_upsample"),
        pytest.param(neg_upsample, id="neg_upsample"),
        pytest.param(EngineerStep(nothing_transform), id="E(nothing_transform)"),
        pytest.param(nothing_transform, id="nothing_transform"),
    ],
)
def test_not_in_upsample_space(space_item):
    assert space_item not in ChoiceUpsample.engineers


#################### ChoiceTarget Contains ####################
@pytest.mark.parametrize(
    "space_item",
    [
        pytest.param(EngineerStep(quantile_transform), id="E(quantile_transform)"),
        pytest.param(EngineerStep(nothing_transform), id="E(nothing_transform)"),
        pytest.param(
            EngineerStep(quantile_transform, stage="intra_cv"), id="E(quantile_transform, stage)"
        ),
        pytest.param(
            EngineerStep(nothing_transform, params=("train_targets", "non_train_targets")),
            id="E(nothing_transform, params)",
        ),
    ],
)
def test_in_target_space(space_item):
    assert space_item in ChoiceTarget.engineers


@pytest.mark.parametrize(
    "space_item",
    [
        pytest.param(
            EngineerStep(quantile_transform, stage="pre_cv"), id="E(quantile_transform, bad_stage)"
        ),
        pytest.param(
            EngineerStep(nothing_transform, params=("non_train_targets", "train_targets")),
            id="E(nothing_transform, bad_params)",
        ),
        pytest.param(quantile_transform, id="quantile_transform"),
        pytest.param(nothing_transform, id="nothing_transform"),
        pytest.param(
            EngineerStep(quantile_transform_no_invert), id="E(quantile_transform_no_invert)"
        ),
        pytest.param(
            EngineerStep(quantile_transform_no_invert, name="quantile_transform"),
            id="E(quantile_transform_no_invert, sneaky_name)",
        ),
    ],
)
def test_not_in_target_space(space_item):
    assert space_item not in ChoiceTarget.engineers


##################################################
# Similar Experiment Description Scenarios
##################################################
# noinspection PyUnusedLocal
def get_holdout_data(train, target_column):
    train_data, holdout_data = train_test_split(train, random_state=1)
    return train_data, holdout_data


@pytest.fixture()
def env_boston():
    return Environment(
        train_dataset=get_boston_data(),
        results_path=assets_dir,
        holdout_dataset=get_holdout_data,
        target_column="DIS",
        metrics=["r2_score", "median_absolute_error"],
        cv_type="KFold",
        cv_params=dict(n_splits=3, random_state=1),
        runs=1,
        verbose=1,
    )


@pytest.fixture()
def fe_experiment(request):
    if request.param is not None:
        request.param = FeatureEngineer(request.param)
    return CVExperiment(
        model_initializer=Ridge, model_init_params={}, feature_engineer=request.param
    )


@pytest.fixture()
def fe_optimizer(request):
    if request.param is not None:
        request.param = FeatureEngineer(request.param)
    opt = BayesianOptPro()
    opt.set_experiment_guidelines(
        model_initializer=Ridge, model_init_params={}, feature_engineer=request.param
    )
    opt.go()
    return opt


optional_quantile_transform = [Categorical([quantile_transform], optional=True)]


# TODO: Expand below tests to multiple steps, multiple `optional`s, multiple categories for options
@pytest.mark.parametrize(
    ["fe_experiment", "fe_optimizer"],
    [
        ([normalize], [ChoiceNormalizeSS.engineers]),
        ([EngineerStep(normalize)], [ChoiceNormalizeSS.engineers]),
        ([standard_scale], [ChoiceNormalizeSS.engineers]),
        ([EngineerStep(standard_scale)], [ChoiceNormalizeSS.engineers]),
        ([normalize], [ChoiceNormalizeSS.functions]),
        ([EngineerStep(normalize)], [ChoiceNormalizeSS.functions]),
        ([standard_scale], [ChoiceNormalizeSS.functions]),
        ([EngineerStep(standard_scale)], [ChoiceNormalizeSS.functions]),
    ],
    indirect=["fe_experiment", "fe_optimizer"],
)
def test_similar_experiments(env_boston, fe_experiment, fe_optimizer):
    """Test that OptimizationProtocols given `Categorical` choices for `FeatureEngineer` steps
    correctly identify past `similar_experiments` when those Experiments' `FeatureEngineer` steps
    executed one of the `EngineerStep` functions in the OptimizationProtocol's choices

    Parameters
    ----------
    env_boston: Environment
        Active `Environment` for `fe_experiment` and `fe_optimizer`
    fe_experiment: CVExperiment
        Indirectly parametrized `CVExperiment` that expects as input either None, or a list of
        `steps` given to its :class:`hyperparameter_hunter.feature_engineering.FeatureEngineer`
    fe_optimizer: BaseOptPro
        Indirectly parametrized :class:`~hyperparameter_hunter.optimization_core.BaseOptPro`
        subclass that expects as input either None, or a list of `steps` (some of which should be
        `Categorical`) given to its :class:`hyperparameter_hunter.FeatureEngineer`. `fe_optimizer`
        is expected to contain the `experiment_id` of `fe_experiment` in its `similar_experiments`

    Notes
    -----
    This does not test the `optional` kwarg of :class:`hyperparameter_hunter.space.Categorical`. See
    `test_similar_experiments_optional` for such tests"""
    assert fe_experiment.experiment_id in [_[2] for _ in fe_optimizer.similar_experiments]


@pytest.mark.parametrize(
    ["fe_experiment", "fe_optimizer"],
    [
        (None, optional_quantile_transform),
        ([quantile_transform], optional_quantile_transform),
        (None, [ChoiceMMNormalizeSS.o_functions]),
        (None, [ChoiceMMNormalizeSS.o_engineers]),
        ([min_max_scale], [ChoiceMMNormalizeSS.o_functions]),
        ([EngineerStep(min_max_scale)], [ChoiceMMNormalizeSS.o_functions]),
        ([min_max_scale], [ChoiceMMNormalizeSS.o_engineers]),
        ([EngineerStep(min_max_scale)], [ChoiceMMNormalizeSS.o_engineers]),
        ([normalize], [ChoiceMMNormalizeSS.o_functions]),
        ([EngineerStep(normalize)], [ChoiceMMNormalizeSS.o_functions]),
        ([normalize], [ChoiceMMNormalizeSS.o_engineers]),
        ([EngineerStep(normalize)], [ChoiceMMNormalizeSS.o_engineers]),
        ([standard_scale], [ChoiceMMNormalizeSS.o_functions]),
        ([EngineerStep(standard_scale)], [ChoiceMMNormalizeSS.o_functions]),
        ([standard_scale], [ChoiceMMNormalizeSS.o_engineers]),
        ([EngineerStep(standard_scale)], [ChoiceMMNormalizeSS.o_engineers]),
    ],
    indirect=["fe_experiment", "fe_optimizer"],
)
def test_similar_experiments_optional(env_boston, fe_experiment, fe_optimizer):
    """Very much like `test_similar_experiments`, except the indirect parameters to `fe_optimizer`
    make use of the `optional` kwarg of :class:`hyperparameter_hunter.space.Categorical`"""
    assert fe_experiment.experiment_id in [_[2] for _ in fe_optimizer.similar_experiments]


# noinspection PyUnusedLocal
# TODO: Add `condition="__version__ < '...'"` to `xfail` when supported
@pytest.mark.xfail(reason="`EngineerStep` matching is index-sensitive")
@pytest.mark.skipif("xgboost" not in sys.modules, reason="Requires `XGBoost` library")
def test_similar_experiments_unordered():
    """Check that an experiment with a single `EngineerStep` is considered "similar" by an
    Optimization Protocol, with two `optional` `EngineerStep`s, where the second step is identical
    to the single step used by the standalone experiment. As of v3.0.0alpha2, this is expected to
    fail because the otherwise identical engineer steps occur at different indexes in
    `FeatureEngineer.steps` for the experiment and the OptPro. The experiment has `sqr_sum_feature`
    at index=0, while the same step in the OptPro is at index=1. Note that the step index in OptPro
    is still 1 despite the fact that the other step immediately preceding it is `optional`"""
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        results_path=assets_dir,
        target_column="diagnosis",
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    exp = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective="reg:linear", subsample=0.5, max_depth=3),
        feature_engineer=FeatureEngineer([EngineerStep(sqr_sum_feature)]),
    )

    opt = BayesianOptPro(iterations=1)
    opt.set_experiment_guidelines(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective="reg:linear", subsample=0.5, max_depth=3),
        feature_engineer=FeatureEngineer(
            [
                Categorical([standard_scale, normalize, min_max_scale], optional=True),
                Categorical([sqr_sum_feature], optional=True),
            ]
        ),
    )
    opt.go()

    assert exp.experiment_id in [_[2] for _ in opt.similar_experiments]


##################################################
# Regression Tests: `Categorical`/`FeatureEngineer`-Only `BayesianOptPro`
##################################################
#################### Fixtures ####################
@pytest.fixture()
def env_boston_regression():
    env = Environment(
        train_dataset=get_boston_data(),
        results_path=assets_dir,
        target_column="DIS",
        metrics=["median_absolute_error"],
        cv_type="KFold",
        cv_params=dict(n_splits=3, random_state=1),
    )
    return env


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


#################### Layout of Tests in Section ####################
# All tests herein alternate between including/excluding three different dimensions in search:
#   1. `FeatureEngineer([Categorical([standard_scale, min_max_scale, normalize])])`: `feature_engineer`
#   2. `loss=Categorical(["linear", "square", "exponential"])` in `model_init_params`
#   3. `n_estimators=Integer(10, 40)` in `model_init_params`
# Above dimensions will be abbreviated as "D1", "D2", and "D3", respectively in illustration below
#
# | function                                 | D1? | D2? | D3? | Result                  |
# |------------------------------------------|-----|-----|-----|-------------------------|
# | test_reg_engineer                        |  Y  |     |     | `BayesianOptPro` breaks |
# | test_reg_engineer_integer_ok             |  Y  |     |  Y  | All good                |
# | test_reg_engineer_categorical            |  Y  |  Y  |     | `BayesianOptPro` breaks |
# | test_reg_engineer_categorical_integer_ok |  Y  |  Y  |  Y  | All good                |
# | test_reg_categorical_ok                  |     |  Y  |     | All good (sanity check) |
# | test_reg_categorical_integer_ok          |     |  Y  |  Y  | All good (sanity check) |
# | test_reg_integer_ok                      |     |     |  Y  | All good (sanity check) |
#
# Conditions to summon bug (assume a blood offering has already been made):
#   1. Use `BayesianOptPro`
#   2. Use exclusively `Categorical` dimensions
#   3. At least one of the `Categorical` dimensions must be in `FeatureEngineer`


#################### Actual Tests ####################
@pytest.mark.parametrize(
    "opt_pro",
    [
        DummyOptPro,
        ET,
        GBRT,
        RF,
        pytest.param(
            BayesianOptPro,
            marks=pytest.mark.xfail(reason="BayesianOptPro hates Engineer/Categorical-only space"),
        ),
    ],
)
def test_reg_engineer(env_boston_regression, hh_assets, opt_pro):
    """Demonstrate problem with `BayesianOptPro` specifically - same configuration is fine with all
    other `OptPro`s"""
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor,
        model_init_params=dict(),
        feature_engineer=FeatureEngineer([Categorical([standard_scale, min_max_scale, normalize])]),
    )
    opt.go()


@pytest.mark.parametrize("opt_pro", [DummyOptPro, ET, GBRT, RF, BayesianOptPro])
def test_reg_engineer_integer_ok(env_boston_regression, hh_assets, opt_pro):
    """Identical to `test_reg_engineer`, except `Integer` dimension added to show that everything is
    fine now. Problem limited to not only `BayesianOptPro`, but also exclusively `Categorical`
    search spaces"""
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor,
        model_init_params=dict(n_estimators=Integer(10, 40)),
        feature_engineer=FeatureEngineer([Categorical([standard_scale, min_max_scale, normalize])]),
    )
    opt.go()


@pytest.mark.parametrize(
    "opt_pro",
    [
        DummyOptPro,
        ET,
        GBRT,
        RF,
        pytest.param(
            BayesianOptPro,
            marks=pytest.mark.xfail(reason="BayesianOptPro hates Engineer/Categorical-only space"),
        ),
    ],
)
def test_reg_engineer_categorical(env_boston_regression, hh_assets, opt_pro):
    """Demonstrate that `BayesianOptPro` breaks with multiple `Categorical`s when `FeatureEngineer`
    is included in the dimensions"""
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor,
        model_init_params=dict(loss=Categorical(["linear", "square", "exponential"])),
        feature_engineer=FeatureEngineer([Categorical([standard_scale, min_max_scale, normalize])]),
    )
    opt.go()


@pytest.mark.parametrize("opt_pro", [DummyOptPro, ET, GBRT, RF, BayesianOptPro])
def test_reg_engineer_categorical_integer_ok(env_boston_regression, hh_assets, opt_pro):
    """Identical to `test_reg_engineer_categorical`, except `Integer` added to demonstrate that all
    `OptPro`s can optimize with `FeatureEngineer` if space is not exclusively `Categorical`"""
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor,
        model_init_params=dict(
            loss=Categorical(["linear", "square", "exponential"]), n_estimators=Integer(10, 40)
        ),
        feature_engineer=FeatureEngineer([Categorical([standard_scale, min_max_scale, normalize])]),
    )
    opt.go()


@pytest.mark.parametrize("opt_pro", [DummyOptPro, ET, GBRT, RF, BayesianOptPro])
def test_reg_categorical_ok(env_boston_regression, hh_assets, opt_pro):
    """Demonstrate that all `OptPro`s are fine with exclusively-`Categorical` space that doesn't
    include `FeatureEngineer`"""
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor,
        model_init_params=dict(loss=Categorical(["linear", "square", "exponential"])),
    )
    opt.go()


@pytest.mark.parametrize("opt_pro", [DummyOptPro, ET, GBRT, RF, BayesianOptPro])
def test_reg_categorical_integer_ok(env_boston_regression, hh_assets, opt_pro):
    """Identical to `test_reg_categorical_ok`, except `Integer` added to show cooperation"""
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor,
        model_init_params=dict(
            loss=Categorical(["linear", "square", "exponential"]), n_estimators=Integer(10, 40)
        ),
    )
    opt.go()


@pytest.mark.parametrize("opt_pro", [DummyOptPro, ET, GBRT, RF, BayesianOptPro])
def test_reg_integer_ok(env_boston_regression, hh_assets, opt_pro):
    opt = opt_pro(iterations=3, random_state=32, n_initial_points=1)
    opt.set_experiment_guidelines(
        model_initializer=AdaBoostRegressor, model_init_params=dict(n_estimators=Integer(10, 40))
    )
    opt.go()
