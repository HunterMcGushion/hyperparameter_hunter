##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.models import model_selector, Model, KerasModel

##################################################
# Import Miscellaneous Assets
##################################################
import pytest
import sys

##################################################
# Import Learning Assets
##################################################
try:
    from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
except (ModuleNotFoundError, ImportError):
    KerasClassifier, KerasRegressor = None, None

from sklearn.svm import SVC


@pytest.mark.parametrize(
    ["initializer", "model_cls"],
    [
        pytest.param(
            KerasClassifier, KerasModel, marks=pytest.mark.skipif("'keras' not in sys.modules")
        ),
        pytest.param(
            KerasRegressor, KerasModel, marks=pytest.mark.skipif("'keras' not in sys.modules")
        ),
        (SVC, Model),
        (None, Model),
    ],
)
def test_model_selector(initializer, model_cls):
    assert model_selector(initializer) == model_cls
