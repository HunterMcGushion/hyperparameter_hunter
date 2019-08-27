##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import Real, Integer, Categorical
from hyperparameter_hunter.io.reporting import get_param_column_sizes
from tests.integration_tests.feature_engineering.test_feature_optimization import (
    ChoiceMMNormalizeSS,
    ChoiceUpsample,
)

##################################################
# Import Miscellaneous Assets
##################################################
import pytest

##################################################
# Global Settings
##################################################
assets_dir = "hyperparameter_hunter/__TEST__HyperparameterHunterAssets__"
# assets_dir = "hyperparameter_hunter/HyperparameterHunterAssets"


##################################################
# `get_param_column_sizes` Tests
##################################################
@pytest.mark.parametrize(
    ["space", "names", "sizes"],
    [
        ([ChoiceMMNormalizeSS.engineers], ["(steps, 0)"], [14]),
        (
            [
                Categorical(["auc", "mae"]),
                Categorical(["gbtree", "dart"]),
                Real(0.0001, 0.5),
                Integer(2, 20),
            ],
            ["(fit, eval_metric)", "booster", "learning_rate", "max_depth"],
            [18, 7, 13, 9],
        ),
        (
            [
                Categorical(["auc", "mae"]),
                Categorical(["gbtree", "dart"]),
                Real(0.0001, 0.5),
                Integer(2, 20),
                ChoiceMMNormalizeSS.engineers,
            ],
            ["(fit, eval_metric)", "booster", "learning_rate", "max_depth", "(steps, 0)"],
            [18, 7, 13, 9, 14],
        ),
        (
            [ChoiceMMNormalizeSS.engineers, ChoiceUpsample.engineers],
            ["(steps, 0)", "(steps, 1)"],
            [14, 12],
        ),
    ],
)
def test_get_param_column_sizes(space, names, sizes):
    assert sizes == get_param_column_sizes(space, names)
