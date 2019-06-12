"""This module tests that the `optimizer` of informed Optimization Protocols is `tell`-ed about
hyperparameters in their proper format.

The tests herein are regression tests for a bug that would cause Optimization Protocols to break at
some point on (or after) the tenth optimization round when attempting to invoke
`self.optimizer.tell` with `EngineerStep`s still formatted as dicts, rather than proper instances.

This bug was a bit tricky to track down for a few reasons:
1. Limited to `Categorical` optimization of `EngineerStep`/functions within `FeatureEngineer`
2. Limited to results being read in from saved experiment description files
3. Following #2, required that the optimization protocol in question be preceded by either:
    * Another optimization protocol whose search space was compatible with the current space, or
    * An experiment, whose result would fit in the current search space
4. Only came up on (or after) the 10th optimization round, so limited to protocols with 10 or more
   samples
5. Despite #4, this was not necessarily the optimization round that actually caused the error -
   in fact, it usually wasn't

See PR #139 (https://github.com/HunterMcGushion/hyperparameter_hunter/pull/139)"""
##################################################
# Import Own Assets
##################################################
# noinspection PyProtectedMember
from hyperparameter_hunter import Environment, FeatureEngineer, EngineerStep, __version__
from hyperparameter_hunter import Real, Integer, Categorical, BayesianOptimization, DummySearch
from hyperparameter_hunter.utils.learning_utils import get_boston_data

##################################################
# Import Miscellaneous Assets
##################################################
from os import makedirs
import pytest
from shutil import rmtree

try:
    xgboost = pytest.importorskip("xgboost")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from xgboost import XGBRegressor

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"


# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


@pytest.fixture(scope="function", autouse=True)
def hh_assets():
    """Construct a temporary HyperparameterHunterAssets directory that exists only for the duration
    of the tests contained in each function, before it and its contents are deleted"""
    temp_assets_path = assets_dir
    try:
        makedirs(temp_assets_path)
    except FileExistsError:  # Can happen if tests stopped before deleting directory - Must empty it
        rmtree(temp_assets_path)
        makedirs(temp_assets_path)
    yield


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


def opt_pro(optimization_protocol):
    opt = optimization_protocol(iterations=11)
    opt.set_experiment_guidelines(
        model_initializer=XGBRegressor,
        model_init_params=dict(
            max_depth=Integer(2, 20),
            n_estimators=Integer(50, 900),
            learning_rate=Real(0.0001, 0.9),
            subsample=0.5,
            booster=Categorical(["gbtree", "gblinear"]),
        ),
        model_extra_params=dict(fit=dict(eval_metric=Categorical(["rmse", "mae"]))),
        feature_engineer=FeatureEngineer([Categorical([nothing_transform], optional=True)]),
    )
    opt.go()
    return opt


##################################################
# Feature Engineering Steps
##################################################
def nothing_transform(train_targets, non_train_targets):
    return train_targets, non_train_targets, lambda _: _


##################################################
# Regression Tests: Saved Engineer Step Optimization
##################################################
@pytest.mark.parametrize(
    ["protocol_0", "protocol_1"], [(DummySearch, DummySearch), (BayesianOptimization, DummySearch)]
)
def test_saved_engineer_step_update_0(env_boston, protocol_0, protocol_1):
    """This test would not trigger the bug tested by this module, but it is an interesting
    exception, and it should never be a problem, so it makes sense to test it"""
    opt_0 = opt_pro(protocol_0)  # First Optimization Execution
    opt_1 = opt_pro(protocol_1)  # Second (Uninformed) Execution
    assert len(opt_1.similar_experiments) == 11


@pytest.mark.xfail(condition="__version__ < '3.0.0alpha2'")
@pytest.mark.parametrize(
    ["protocol_0", "protocol_1"],
    [(DummySearch, BayesianOptimization), (BayesianOptimization, BayesianOptimization)],
)
def test_saved_engineer_step_update_1(env_boston, protocol_0, protocol_1):
    """This test is exactly the same as the previous tests, except it uses an informed protocol,
    instead of `DummySearch` as the second protocol - This should trigger the bug"""
    opt_0 = opt_pro(protocol_0)  # First Optimization Execution
    opt_1 = opt_pro(protocol_1)  # Second (Informed) Execution


##################################################
# `EngineerStep.honorary_step_from_dict` Tests
##################################################
@pytest.mark.parametrize(
    ["step_dict", "dimension", "expected"],
    [
        (
            dict(
                name="nothing_transform",
                f="2jDrngAKAWUo9OtZOL7VNfoJBj7XXy340dsgNjVj7AE=",
                params=["train_targets", "non_train_targets"],
                stage="intra_cv",
                do_validate=False,
            ),
            Categorical([EngineerStep(nothing_transform)], optional=True),
            EngineerStep(nothing_transform),
        ),
        (
            dict(
                name="nothing_transform",
                f="2jDrngAKAWUo9OtZOL7VNfoJBj7XXy340dsgNjVj7AE=",
                params=["train_targets", "non_train_targets"],
                stage="pre_cv",
                do_validate=False,
            ),
            Categorical(
                [EngineerStep(nothing_transform), EngineerStep(nothing_transform, stage="pre_cv")]
            ),
            EngineerStep(nothing_transform, stage="pre_cv"),
        ),
    ],
)
def test_honorary_step_from_dict(step_dict, dimension, expected):
    actual = EngineerStep.honorary_step_from_dict(step_dict, dimension)
    assert isinstance(actual, EngineerStep)
    assert actual == expected


@pytest.mark.parametrize(
    ["step_dict", "dimension"],
    [
        (
            dict(
                name="nothing_transform",
                f="2jDrngAKAWUo9OtZOL7VNfoJBj7XXy340dsgNjVj7AE=",
                params=["train_targets", "non_train_targets"],
                stage="pre_cv",
                do_validate=False,
            ),
            Categorical([EngineerStep(nothing_transform)], optional=True),
        ),
        (
            dict(
                name="foo",
                f="2jDrngAKAWUo9OtZOL7VNfoJBj7XXy340dsgNjVj7AE=",
                params=["train_targets", "non_train_targets"],
                stage="pre_cv",
                do_validate=False,
            ),
            Categorical(
                [EngineerStep(nothing_transform), EngineerStep(nothing_transform, stage="pre_cv")]
            ),
        ),
    ],
)
def test_honorary_step_from_dict_value_error(step_dict, dimension):
    with pytest.raises(ValueError, match="`step_dict` could not be found in `dimension`"):
        actual = EngineerStep.honorary_step_from_dict(step_dict, dimension)
