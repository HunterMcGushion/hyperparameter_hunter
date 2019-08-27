##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Environment, CVExperiment, Real, Integer, Categorical
from hyperparameter_hunter import RandomForestOptPro
from hyperparameter_hunter.io.result_reader import has_experiment_result_file
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

try:
    xgboost = pytest.importorskip("xgboost")
except Exception:
    raise

##################################################
# Import Learning Assets
##################################################
from xgboost import XGBClassifier

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
        train_dataset=get_toy_classification_data(),
        results_path=assets_dir,
        metrics=["roc_auc_score"],
        cv_type="RepeatedStratifiedKFold",
        cv_params=dict(n_splits=3, n_repeats=2, random_state=32),
    )


##################################################
# Experiment Fixtures
##################################################
@pytest.fixture(scope="function", autouse=False)
def exp_xgb_0():
    return CVExperiment(
        XGBClassifier, dict(subsample=0.01), model_extra_params=dict(fit=dict(verbose=False))
    )


##################################################
# Optimization Protocol Fixtures
##################################################
@pytest.fixture(scope="function", autouse=False)
def opt_xgb_0():
    optimizer = RandomForestOptPro(iterations=2, random_state=1337)
    optimizer.forge_experiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            objective="reg:linear",
            max_depth=Integer(2, 20),
            learning_rate=Real(0.0001, 0.5),
            subsample=0.5,
            booster=Categorical(["gbtree", "dart"]),
        ),
    )
    optimizer.go()
    yield optimizer


##################################################
# Test Scenarios
##################################################
def test_classification_experiment(env_0, exp_xgb_0):
    assert has_experiment_result_file(assets_dir, exp_xgb_0)


def test_classification_optimization(env_0, opt_xgb_0):
    ...
