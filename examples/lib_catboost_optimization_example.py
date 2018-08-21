from hyperparameter_hunter import Environment, Real, Integer, Categorical, GradientBoostedRegressionTreeOptimization
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from catboost import CatBoostClassifier


def _execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path='HyperparameterHunterAssets',
        metrics_map=['roc_auc_score'],
        cross_validation_type='StratifiedKFold',
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
        runs=1,
    )

    optimizer = GradientBoostedRegressionTreeOptimization(
        iterations=10, read_experiments=True, random_state=None,
    )

    optimizer.set_experiment_guidelines(
        model_initializer=CatBoostClassifier,
        model_init_params=dict(
            iterations=100,
            eval_metric=Categorical(['Logloss', 'Accuracy', 'AUC'], transform='onehot'),
            learning_rate=Real(low=0.0001, high=0.5),
            depth=Integer(4, 7),
            save_snapshot=False
        ),
    )

    optimizer.go()

    print('')


if __name__ == '__main__':
    _execute()
