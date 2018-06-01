import sys
import os.path

try:
    sys.path.append(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])
except Exception as _ex:
    raise _ex

from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier


def execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        root_results_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../HyperparameterHunterAssets')),
        target_column='diagnosis',
        metrics_map=['roc_auc_score', 'f1_score'],
        cross_validation_type=KFold,
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
        runs=3,
    )

    experiment = CrossValidationExperiment(
        model_initializer=LGBMClassifier,
        model_init_params=dict(boosting_type='gbdt', num_leaves=31, max_depth=-1, min_child_samples=5, subsample=0.5)
    )


if __name__ == '__main__':
    execute()
