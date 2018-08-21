from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import RepeatedStratifiedKFold
from rgf import RGFClassifier


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(target='diagnosis'),
        root_results_path='HyperparameterHunterAssets',
        target_column='diagnosis',
        metrics_map=['roc_auc_score'],
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=5, n_repeats=2, random_state=32),
    )

    experiment = CrossValidationExperiment(
        model_initializer=RGFClassifier,
        model_init_params=dict(max_leaf=1000, algorithm='RGF', min_samples_leaf=10),
    )


if __name__ == '__main__':
    execute()
