"""This module tests proper functioning of miscellaneous supporting activities performed by
:class:`~hyperparameter_hunter.feature_engineering.FeatureEngineer` and
:class:`~hyperparameter_hunter.feature_engineering.EngineerStep` when used by Experiments"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer
from hyperparameter_hunter.utils.learning_utils import get_boston_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Import Learning Assets
##################################################
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


@pytest.fixture
def env_boston():
    return Environment(
        train_dataset=get_boston_data(),
        results_path=assets_dir,
        target_column="DIS",
        metrics=["r2_score"],
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
    with pytest.raises(TypeError, match="`inversion` must be callable, or class with .*"):
        exp = CVExperiment(
            model_initializer=Ridge,
            model_init_params={},
            feature_engineer=FeatureEngineer([bad_quantile_transform]),
        )
