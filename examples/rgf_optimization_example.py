from hyperparameter_hunter import Environment, Real, Integer, Categorical, ExtraTreesOptimization
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import RepeatedStratifiedKFold
from rgf import RGFClassifier


def _execute():
    env = Environment(
        train_dataset=get_toy_classification_data(target='diagnosis'),
        root_results_path='HyperparameterHunterAssets',
        target_column='diagnosis',
        metrics_map=['roc_auc_score'],
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=5, n_repeats=2, random_state=32),
    )

    optimizer = ExtraTreesOptimization(
        iterations=10, read_experiments=True, random_state=None,
    )

    optimizer.set_experiment_guidelines(
        model_initializer=RGFClassifier,
        model_init_params=dict(
            max_leaf=1000,
            algorithm=Categorical(['RGF', 'RGF_Opt', 'RGF_Sib']),
            l2=Real(0.01, 0.3),
            normalize=Categorical([True, False]),
            learning_rate=Real(0.3, 0.7),
            loss=Categorical(['LS', 'Expo', 'Log', 'Abs'])
        ),
    )

    optimizer.go()


if __name__ == '__main__':
    _execute()
