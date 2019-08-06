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
from hyperparameter_hunter import CVExperiment, Real, Integer, Categorical
from hyperparameter_hunter import BayesianOptPro, DummyOptPro, ExtraTreesOptPro
from hyperparameter_hunter.utils.learning_utils import get_boston_data
from hyperparameter_hunter.utils.version_utils import HHVersion

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
    opt = optimization_protocol(iterations=3, random_state=32, n_initial_points=1)
    opt.forge_experiment(
        model_initializer=XGBRegressor,
        model_init_params=dict(
            max_depth=Integer(2, 10),
            n_estimators=Integer(50, 300),
            learning_rate=Real(0.1, 0.9),
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
    ["protocol_0", "protocol_1"], [(DummyOptPro, DummyOptPro), (BayesianOptPro, DummyOptPro)]
)
def test_saved_engineer_step_update_0(env_boston, protocol_0, protocol_1):
    """This test would not trigger the bug tested by this module, but it is an interesting
    exception, and it should never be a problem, so it makes sense to test it"""
    opt_0 = opt_pro(protocol_0)  # First Optimization Execution
    opt_1 = opt_pro(protocol_1)  # Second (Uninformed) Execution
    assert len(opt_1.similar_experiments) == 3  # From `opt_pro`'s `iterations`


@pytest.mark.xfail(condition="HHVersion(__version__) < '3.0.0alpha2'")
@pytest.mark.parametrize(
    ["protocol_0", "protocol_1"], [(DummyOptPro, BayesianOptPro), (BayesianOptPro, BayesianOptPro)]
)
def test_saved_engineer_step_update_1(env_boston, protocol_0, protocol_1):
    """This test is exactly the same as the previous tests, except it uses an informed protocol,
    instead of `DummyOptPro` as the second protocol - This should trigger the bug"""
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


##################################################
# Regression Tests: Optional `EngineerStep`s
##################################################
#################### Dummy EngineerStep Functions ####################
def es_a(all_inputs):
    return all_inputs


def es_b(all_inputs):
    return all_inputs


def es_c(all_inputs):
    return all_inputs


def es_d(all_inputs):
    return all_inputs


def es_e(all_inputs):
    return all_inputs


#################### Result Matching Tests ####################
@pytest.mark.parametrize(
    "feature_engineer",
    [
        [Categorical([es_b, es_c], optional=True), Categorical([es_d, es_e], optional=True)],
        [Categorical([es_b, es_c], optional=True), Categorical([es_d, es_e])],
        [Categorical([es_a]), Categorical([es_b, es_c], optional=True), Categorical([es_d, es_e])],
        [
            Categorical([es_a]),
            Categorical([es_b, es_c], optional=True),
            Categorical([es_d, es_e], optional=True),
        ],
        [
            Categorical([es_a], optional=True),
            Categorical([es_b, es_c], optional=True),
            Categorical([es_d, es_e], optional=True),
        ],
    ],
)
def test_optional_step_matching(env_boston, feature_engineer):
    """Tests that a Space containing `optional` `Categorical` Feature Engineering steps matches with
    the expected saved Experiments. This regression test is focused on issues that arise when
    `EngineerStep`s other than the last one in the `FeatureEngineer` are `optional`. The simplified
    version of this test below, :func:`test_limited_optional_step_matching`, demonstrates that
    result matching works properly when only the final `EngineerStep` is `optional`"""
    opt_0 = DummyOptPro(iterations=20, random_state=32)
    opt_0.forge_experiment(XGBRegressor, feature_engineer=feature_engineer)
    opt_0.go()

    opt_1 = ExtraTreesOptPro(iterations=20, random_state=32)
    opt_1.forge_experiment(XGBRegressor, feature_engineer=feature_engineer)
    opt_1.get_ready()

    # Assert `opt_1` matched with all Experiments executed by `opt_0`
    assert len(opt_1.similar_experiments) == opt_0.successful_iterations


@pytest.mark.parametrize(
    "feature_engineer",
    [
        [Categorical([es_b, es_c])],
        [Categorical([es_b, es_c], optional=True)],
        [Categorical([es_b, es_c]), Categorical([es_d, es_e], optional=True)],
        [Categorical([es_a]), Categorical([es_b, es_c]), Categorical([es_d, es_e])],
        [Categorical([es_a]), Categorical([es_b, es_c]), Categorical([es_d, es_e], optional=True)],
    ],
)
def test_limited_optional_step_matching(env_boston, feature_engineer):
    """Simplified counterpart to above :func:`test_optional_step_matching`. Tests that a Space
    containing `Categorical` Feature Engineering steps -- of which only the last ones may be
    `optional` -- matches with the expected saved Experiments. These test cases do not demonstrate
    the same bug being regression-tested by `test_optional_step_matching`. Instead, this test
    function exists to ensure that the areas close to the above bug are behaving properly and to
    help define the bug being tested by `test_optional_step_matching`. This function demonstrates
    that `optional` is not problematic when used only in the final `EngineerStep`"""
    opt_0 = DummyOptPro(iterations=20, random_state=32)
    opt_0.forge_experiment(XGBRegressor, feature_engineer=feature_engineer)
    opt_0.go()

    opt_1 = ExtraTreesOptPro(iterations=20, random_state=32)
    opt_1.forge_experiment(XGBRegressor, feature_engineer=feature_engineer)
    opt_1.get_ready()

    # Assert `opt_1` matched with all Experiments executed by `opt_0`
    assert len(opt_1.similar_experiments) == opt_0.successful_iterations


##################################################
# Exhaustive Experiment Matching Tests
##################################################
# The tests in this section are still related to the regression tests above, but these are
#   conducted using a group of one-off Experiments, comprising all `FeatureEngineer` permutations
#   that should fit within the `feature_engineer` space of `opt_0`:
#   ```
#   [
#       Categorical([es_a], optional=True),
#       Categorical([es_b, es_c], optional=True),
#       Categorical([es_d, es_e], optional=True),
#   ]
#   ```
@pytest.mark.parametrize("es_0", [es_a, None])
@pytest.mark.parametrize("es_1", [es_b, es_c, None])
@pytest.mark.parametrize("es_2", [es_d, es_e, None])
def test_optional_step_matching_by_exp(env_boston, es_0, es_1, es_2):
    """Test that the result of an Experiment is correctly matched by an OptPro with all-`optional`
    `EngineerStep` dimensions"""
    feature_engineer = [_ for _ in [es_0, es_1, es_2] if _ is not None]
    exp_0 = CVExperiment(XGBRegressor, feature_engineer=feature_engineer)

    opt_0 = ExtraTreesOptPro(iterations=1, random_state=32)
    opt_0.forge_experiment(
        XGBRegressor,
        feature_engineer=[
            Categorical([es_a], optional=True),
            Categorical([es_b, es_c], optional=True),
            Categorical([es_d, es_e], optional=True),
        ],
    )
    opt_0.get_ready()

    # Assert `opt_0` matched with `exp_0`
    assert len(opt_0.similar_experiments) == 1
