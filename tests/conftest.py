##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import settings
from hyperparameter_hunter.environment import Environment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data

##################################################
# Import Miscellaneous Assets
##################################################
import pytest


# noinspection PyUnusedLocal
@pytest.fixture(scope="module", autouse=True)
def new_G(request):
    settings.G.reset_attributes()
    yield


@pytest.fixture(scope="function")
def env_fixture_0():
    return Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path="HyperparameterHunterAssets",
        metrics_map=["roc_auc_score"],
        cross_validation_type="StratifiedKFold",
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
    )


@pytest.fixture(scope="function")
def env_fixture_1():
    return Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path=None,
        metrics_map=["roc_auc_score"],
        cross_validation_type="StratifiedKFold",
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
    )
