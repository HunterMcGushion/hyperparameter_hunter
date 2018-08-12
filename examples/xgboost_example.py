from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from xgboost import XGBClassifier


def execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(target='target'),
        root_results_path='HyperparameterHunterAssets',
        metrics_map=['roc_auc_score'],
        cross_validation_type='StratifiedKFold',
        cross_validation_params=dict(n_splits=10, shuffle=True, random_state=32),
    )

    experiment = CrossValidationExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective='reg:linear', max_depth=3, n_estimators=100, subsample=0.5),
        model_extra_params=dict(
            fit=dict(eval_set=[(env.validation_input, env.validation_target)]),
        )
    )


if __name__ == '__main__':
    execute()
