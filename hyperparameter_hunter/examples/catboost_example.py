import sys
import os.path

try:
    sys.path.append(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])
except Exception as _ex:
    raise _ex

from hyperparameter_hunter import Environment, CrossValidationExperiment, BayesianOptimization, Real, Integer, Categorical
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(target='diagnosis'),
        root_results_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../HyperparameterHunterAssets')),
        target_column='diagnosis',
        metrics_map=['roc_auc_score'],
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
        runs=2,
        # file_blacklist='ALL',
    )

    experiment = CrossValidationExperiment(
        model_initializer=CatBoostClassifier,
        model_init_params=dict(iterations=500, learning_rate=0.01, depth=7, save_snapshot=False),
        # NOTE: Inside `model_init_params` can be any of the many kwargs accepted by :meth:`CatBoostClassifier.__init__`
    )


if __name__ == '__main__':
    execute()
