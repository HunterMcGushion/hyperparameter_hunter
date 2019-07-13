"""This module tests proper functioning of miscellaneous supporting activities performed by
:class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer` and
:class:`~hyperparameter_hunter.feature_engineering.EngineerStep` when used by Experiments"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer, EngineerStep
from hyperparameter_hunter import Categorical, GBRT
from hyperparameter_hunter.utils.learning_utils import get_boston_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


@pytest.fixture(
    params=[None, lambda train, _: train_test_split(train, test_size=0.25, random_state=1)],
    ids=["no_holdout", "yes_holdout"],
)
def env_boston(request):
    """Environment fixture using the Boston regression dataset. Parametrizes `holdout_dataset`, so
    all tests using this fixture will be run twice: once with no `holdout_dataset`, and once with a
    `holdout_dataset` constructed using SKLearn's `train_test_split`"""
    return Environment(
        train_dataset=get_boston_data(),
        results_path=assets_dir,
        target_column="DIS",
        metrics=["r2_score"],
        holdout_dataset=request.param,
        cv_type="KFold",
        cv_params=dict(n_splits=3, random_state=1),
    )


##################################################
# Engineer Step Functions
##################################################
def standard_scale(train_inputs, non_train_inputs):
    scaler = StandardScaler()
    train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    return train_inputs, non_train_inputs


def bad_quantile_transform(train_targets, non_train_targets):
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=100)
    train_targets[train_targets.columns] = transformer.fit_transform(train_targets.values)
    non_train_targets[train_targets.columns] = transformer.transform(non_train_targets.values)
    return train_targets, non_train_targets, "i am the wrong type for an inversion result"


def nothing_transform(train_targets, non_train_targets):
    return train_targets, non_train_targets, lambda _: _


##################################################
# `FeatureEngineer.do_validate` Tests
##################################################
def test_do_validate(env_boston):
    exp = CVExperiment(
        model_initializer=Ridge,
        model_init_params={},
        feature_engineer=FeatureEngineer([standard_scale], do_validate=True),
    )

    for step in exp.feature_engineer.steps:
        assert step.original_hashes != {}
        assert step.updated_hashes != {}
        assert step.original_hashes != step.updated_hashes


def test_do_not_validate(env_boston):
    exp = CVExperiment(
        model_initializer=Ridge,
        model_init_params={},
        feature_engineer=FeatureEngineer([standard_scale], do_validate=False),
    )

    for step in exp.feature_engineer.steps:
        assert step.original_hashes == {}
        assert step.updated_hashes == {}


##################################################
# `FeatureEngineer.inverse_transform` TypeError Tests
##################################################
# noinspection PyUnusedLocal
def test_inverse_type_error(env_boston):
    """Test that an error is raised if an `EngineerStep` function returns an extra value that is
    not a function or class instance. Extra return values are used for inverse transformations"""
    with pytest.raises(TypeError, match="`inversion` must be callable, or class with .*"):
        exp = CVExperiment(
            model_initializer=Ridge,
            model_init_params={},
            feature_engineer=FeatureEngineer([bad_quantile_transform]),
        )


##################################################
# `CVExperiment`: `FeatureEngineer` as List
##################################################
#################### Equality ####################
@pytest.mark.parametrize(
    ["steps_0", "steps_1"],
    [
        ([standard_scale], [standard_scale]),
        ([standard_scale, standard_scale], [standard_scale, standard_scale]),
        ([standard_scale], [EngineerStep(standard_scale, stage="intra_cv")]),
        ([nothing_transform, standard_scale], [nothing_transform, standard_scale]),
        ([nothing_transform, standard_scale], [EngineerStep(nothing_transform), standard_scale]),
        (
            [EngineerStep(nothing_transform, name="nothing_transform"), standard_scale],
            [nothing_transform, standard_scale],
        ),
    ],
)
def test_feature_engineer_list_experiment_equality(env_boston, steps_0, steps_1):
    """Test that the `feature_engineer` attribute constructed by
    :class:`~hyperparameter_hunter.experiments.CVExperiment` is the same whether it was given a
    list as input, or a :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`"""
    exp_0 = CVExperiment(Ridge, feature_engineer=steps_0)
    exp_1 = CVExperiment(Ridge, feature_engineer=FeatureEngineer(steps_1))
    assert exp_0.feature_engineer == exp_1.feature_engineer

    # Repeat above, but switch which steps are wrapped in `FeatureEngineer`
    exp_2 = CVExperiment(Ridge, feature_engineer=steps_1)
    exp_3 = CVExperiment(Ridge, feature_engineer=FeatureEngineer(steps_0))
    assert exp_2.feature_engineer == exp_3.feature_engineer


#################### Inequality ####################
@pytest.mark.parametrize(
    ["steps_0", "steps_1"],
    [
        ([standard_scale], [standard_scale, standard_scale]),
        ([standard_scale], [EngineerStep(standard_scale, name="foo")]),
        ([nothing_transform, standard_scale], [standard_scale, nothing_transform]),
        ([nothing_transform, standard_scale], [standard_scale, EngineerStep(nothing_transform)]),
        (
            [EngineerStep(nothing_transform, name="foo"), standard_scale],
            [nothing_transform, standard_scale],
        ),
    ],
)
def test_feature_engineer_list_experiment_inequality(env_boston, steps_0, steps_1):
    """Test that the `feature_engineer` attribute constructed by
    :class:`~hyperparameter_hunter.experiments.CVExperiment` is NOT the same when given a list as
    input vs. a :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer` when the two are
    actually different. This is an insanity test to make sure that the related test in this module,
    :func:`test_feature_engineer_list_experiment_equality`, is not simply equating everything"""
    exp_0 = CVExperiment(Ridge, feature_engineer=steps_0)
    exp_1 = CVExperiment(Ridge, feature_engineer=FeatureEngineer(steps_1))
    assert exp_0.feature_engineer != exp_1.feature_engineer

    # Repeat above, but switch which steps are wrapped in `FeatureEngineer`
    exp_2 = CVExperiment(Ridge, feature_engineer=steps_1)
    exp_3 = CVExperiment(Ridge, feature_engineer=FeatureEngineer(steps_0))
    assert exp_2.feature_engineer != exp_3.feature_engineer


##################################################
# OptPros: `FeatureEngineer` as List
##################################################
#################### Equality ####################
@pytest.mark.parametrize(
    ["steps_0", "steps_1"],
    [
        ([standard_scale], [standard_scale]),
        ([standard_scale, standard_scale], [standard_scale, standard_scale]),
        ([standard_scale], [EngineerStep(standard_scale, stage="intra_cv")]),
        ([nothing_transform, standard_scale], [nothing_transform, standard_scale]),
        ([nothing_transform, standard_scale], [EngineerStep(nothing_transform), standard_scale]),
        (
            [EngineerStep(nothing_transform, name="nothing_transform"), standard_scale],
            [nothing_transform, standard_scale],
        ),
        ([Categorical([standard_scale])], [Categorical([standard_scale])]),
        (
            [Categorical([standard_scale], optional=True), standard_scale],
            [Categorical([standard_scale], optional=True), standard_scale],
        ),
        (
            [Categorical([standard_scale], optional=True)],
            [Categorical([EngineerStep(standard_scale, stage="intra_cv")], optional=True)],
        ),
        (
            [nothing_transform, Categorical([standard_scale], optional=True)],
            [nothing_transform, Categorical([EngineerStep(standard_scale)], optional=True)],
        ),
        (
            [nothing_transform, Categorical([standard_scale, EngineerStep(nothing_transform)])],
            [
                EngineerStep(nothing_transform),
                Categorical([EngineerStep(standard_scale), nothing_transform]),
            ],
        ),
        (
            [
                Categorical([nothing_transform, EngineerStep(standard_scale)]),
                Categorical([standard_scale, EngineerStep(nothing_transform)]),
            ],
            [
                Categorical([EngineerStep(nothing_transform), standard_scale]),
                Categorical([EngineerStep(standard_scale), nothing_transform]),
            ],
        ),
    ],
)
def test_feature_engineer_list_optimization_equality(env_boston, steps_0, steps_1):
    """Test that the `feature_engineer` attribute constructed by an OptPro is the same whether given
    a list as input, or a :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`"""
    opt_0, opt_1, opt_2, opt_3 = GBRT(), GBRT(), GBRT(), GBRT()
    opt_0.forge_experiment(Ridge, feature_engineer=steps_0)
    opt_1.forge_experiment(Ridge, feature_engineer=FeatureEngineer(steps_1))
    assert opt_0.feature_engineer == opt_1.feature_engineer

    # Repeat above, but switch which steps are wrapped in `FeatureEngineer`
    opt_2.forge_experiment(Ridge, feature_engineer=steps_1)
    opt_3.forge_experiment(Ridge, feature_engineer=FeatureEngineer(steps_0))
    assert opt_2.feature_engineer == opt_3.feature_engineer


#################### Inequality ####################
@pytest.mark.parametrize(
    ["steps_0", "steps_1"],
    [
        ([standard_scale], [standard_scale, standard_scale]),
        ([standard_scale], [EngineerStep(standard_scale, name="foo")]),
        ([nothing_transform, standard_scale], [standard_scale, nothing_transform]),
        ([nothing_transform, standard_scale], [standard_scale, EngineerStep(nothing_transform)]),
        (
            [EngineerStep(nothing_transform, name="foo"), standard_scale],
            [nothing_transform, standard_scale],
        ),
        ([Categorical([standard_scale])], [Categorical([standard_scale], optional=True)]),
        (
            [Categorical([standard_scale]), standard_scale],
            [Categorical([standard_scale], optional=True), standard_scale],
        ),
        (
            [Categorical([standard_scale, nothing_transform])],
            [Categorical([EngineerStep(standard_scale, stage="intra_cv")], optional=True)],
        ),
        (
            [nothing_transform, Categorical([standard_scale, nothing_transform], optional=True)],
            [nothing_transform, Categorical([EngineerStep(standard_scale)], optional=True)],
        ),
        (
            [
                Categorical([nothing_transform, EngineerStep(standard_scale)]),
                Categorical([standard_scale, EngineerStep(nothing_transform)]),
            ],
            [
                Categorical([EngineerStep(nothing_transform), standard_scale]),
                Categorical([EngineerStep(standard_scale), nothing_transform, standard_scale]),
            ],
        ),
    ],
)
def test_feature_engineer_list_optimization_inequality(env_boston, steps_0, steps_1):
    """Test that the `feature_engineer` attribute constructed by an OptPro is NOT the same when
    given a list as input vs. a :class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer`
    when the two are actually different. This is an insanity test to make sure that the related
    test in this module, :func:`test_feature_engineer_list_optimization_equality`, is not simply
    equating everything"""
    opt_0, opt_1, opt_2, opt_3 = GBRT(), GBRT(), GBRT(), GBRT()
    opt_0.forge_experiment(Ridge, feature_engineer=steps_0)
    opt_1.forge_experiment(Ridge, feature_engineer=FeatureEngineer(steps_1))
    assert opt_0.feature_engineer != opt_1.feature_engineer

    # Repeat above, but switch which steps are wrapped in `FeatureEngineer`
    opt_2.forge_experiment(Ridge, feature_engineer=steps_1)
    opt_3.forge_experiment(Ridge, feature_engineer=FeatureEngineer(steps_0))
    assert opt_2.feature_engineer != opt_3.feature_engineer
