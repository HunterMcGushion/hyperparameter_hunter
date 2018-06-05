import sys
import os.path

try:
    sys.path.append(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
except Exception as _ex:
    raise _ex

from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path='HyperparameterHunterAssets',
        metrics_map=['roc_auc_score'],
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    experiment = CrossValidationExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective='reg:linear', max_depth=3, n_estimators=100, subsample=0.5)
    )


if __name__ == '__main__':
    execute()
