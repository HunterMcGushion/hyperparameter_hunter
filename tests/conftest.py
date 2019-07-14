##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import settings
from hyperparameter_hunter.environment import Environment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data

##################################################
# Import Miscellaneous Assets
##################################################
from os import makedirs
import pytest
from shutil import rmtree

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


# noinspection PyUnusedLocal
@pytest.fixture(scope="module", autouse=True)
def new_G(request):
    settings.G.reset_attributes()
    yield


# noinspection PyUnusedLocal
@pytest.fixture(scope="module", autouse=True)
def hh_assets(request):
    """Construct a temporary HyperparameterHunterAssets directory that exists only for the duration
    of the tests contained in each module, before it and its contents are deleted"""
    temp_assets_path = assets_dir
    try:
        makedirs(temp_assets_path)
    except FileExistsError:  # Can happen if tests stopped before deleting directory - Must empty it
        rmtree(temp_assets_path)
        makedirs(temp_assets_path)
    yield
    rmtree(temp_assets_path)


@pytest.fixture(scope="function")
def env_fixture_0():
    return Environment(
        train_dataset=get_toy_classification_data(),
        results_path=assets_dir,
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )


@pytest.fixture(scope="function")
def env_fixture_1():
    return Environment(
        train_dataset=get_toy_classification_data(),
        results_path=None,
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )
