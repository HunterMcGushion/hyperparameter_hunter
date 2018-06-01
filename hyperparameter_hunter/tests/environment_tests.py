##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.environment import Environment
from hyperparameter_hunter.utils.test_utils import equals_suite, format_suites, get_module

##################################################
# Import Miscellaneous Assets
##################################################
from functools import partial
import pandas as pd
from unittest import TestCase, TestSuite, TextTestRunner

##################################################
# Import Learning Assets
##################################################
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score


def get_breast_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    return df


def get_holdout_set(train, target_column):
    # Hello, I am a test comment
    return train, train.copy()


class TestCrossExperimentKeyMaker(TestCase):
    train_dataset = get_breast_cancer_data()
    test_dataset = train_dataset.copy()
    experiment_params = dict(
        # global_random_seed=31,
        random_seeds=None,
        # runs=1,
        runs=3,
        cross_validation_type='StratifiedKFold'
        # cross_validation_type='KFold'
    )
    cross_validation_params = dict(
        # n_splits=10,
        n_splits=5,
        shuffle=True,
        random_state=32
    )
    repeated_cross_validation_params = dict(
        # n_splits=10,
        n_splits=5,
        n_repeats=2,
        random_state=32,
    )

    # metrics_parameters = dict(metrics_map=['roc_auc_score'], in_fold='all', oof='all', holdout='all')
    # metrics_parameters = dict(metrics_map=dict(roc_auc='roc_auc_score'), in_fold='all', oof='all', holdout='all')
    metrics_parameters = dict(metrics_map=dict(
        roc='roc_auc_score',
        # f1=lambda _t, _p: f1_score(_t, np.clip(np.round(_p), 0, 1)),
        f1='f1_score',
        acc='accuracy_score',
        # acc=lambda _t, _p: accuracy_score(_t, np.clip(np.round(_p), 0, 1)),
        # f1=lambda _t, _p: f1_score(_t, np.clip(np.round(_p), 0, 2))
    ), in_fold='all', oof='all', holdout='all')

    # FLAG: Below may not be accurate after adding "id_column" kwarg and changing "target_variable" to "target_column"

    simple_tests = {
        'cross_validation_params': [
            [dict(
                train_dataset=train_dataset,
                environment_params_path=None,
                root_results_path='hyperparameter_hunter/HyperparameterHunterAssets',
                holdout_dataset=get_holdout_set,
                test_dataset=test_dataset,
                target_column='diagnosis',
                do_predict_proba=False,
                prediction_formatter=None,
                metrics_params=metrics_parameters,
                cross_experiment_params=experiment_params,
                cross_validation_params=repeated_cross_validation_params,
                verbose=True,
                file_blacklist=None,
                reporting_handler_params=dict(add_frame=False)
            ), 'kJKV0kCGPUM4HuVlZVVcQAYGy+U1ChZnyzftCLw2n3o='],
            [dict(
                train_dataset=train_dataset,
                environment_params_path=None,
                root_results_path='hyperparameter_hunter/HyperparameterHunterAssets',
                holdout_dataset=get_holdout_set,
                test_dataset=test_dataset,
                target_column='diagnosis',
                do_predict_proba=False,
                prediction_formatter=None,
                metrics_params=metrics_parameters,
                cross_experiment_params=experiment_params,
                cross_validation_params=cross_validation_params,
                verbose=True,
                file_blacklist=None,
                reporting_handler_params=dict(add_frame=False)
            ), 'xbfyOmtuDqGxr98zQjWCzOMle1m7Zr5tI+3CYvkO6gs=']
        ],
        'metrics_params': [
            [dict(
                train_dataset=train_dataset,
                environment_params_path=None,
                root_results_path='hyperparameter_hunter/HyperparameterHunterAssets',
                holdout_dataset=get_holdout_set,
                test_dataset=test_dataset,
                target_column='diagnosis',
                do_predict_proba=False,
                prediction_formatter=None,
                metrics_params=dict(metrics_map=dict(
                    roc='roc_auc_score',
                    acc='accuracy_score',
                ), in_fold='all', oof='all', holdout='all'),
                cross_experiment_params=experiment_params,
                cross_validation_params=repeated_cross_validation_params,
                verbose=True,
                file_blacklist=None,
                reporting_handler_params=dict(add_frame=False)
            ), 'oy7EE1FOZb4D6Mz1x4l96Qlr0pv2ZB8PGYnYEbR2G38='],
            [dict(
                train_dataset=train_dataset,
                environment_params_path=None,
                root_results_path='hyperparameter_hunter/HyperparameterHunterAssets',
                holdout_dataset=get_holdout_set,
                test_dataset=test_dataset,
                target_column='diagnosis',
                do_predict_proba=False,
                prediction_formatter=None,
                metrics_params=dict(metrics_map=dict(
                    roc='roc_auc_score',
                    f1='f1_score',
                    acc=lambda _t, _p: accuracy_score(_t, np.clip(np.round(_p), 0, 1)),
                ), in_fold='all', oof='all', holdout='all'),
                cross_experiment_params=experiment_params,
                cross_validation_params=repeated_cross_validation_params,
                verbose=True,
                file_blacklist=None,
                reporting_handler_params=dict(add_frame=False)
            ), 'XEnUqrPDZ3RTwV3W/bowhKLls5rx0S1WoeMFKvcKKeY='],
        ],
        'cross_experiment_params': [
            [dict(
                train_dataset=train_dataset,
                environment_params_path=None,
                root_results_path='hyperparameter_hunter/HyperparameterHunterAssets',
                holdout_dataset=get_holdout_set,
                test_dataset=test_dataset,
                target_column='diagnosis',
                do_predict_proba=False,
                prediction_formatter=None,
                metrics_params=metrics_parameters,
                cross_experiment_params=dict(
                    random_seeds=None,
                    runs=10,
                    cross_validation_type='StratifiedKFold'
                ),
                cross_validation_params=repeated_cross_validation_params,
                verbose=True,
                file_blacklist=None,
                reporting_handler_params=dict(add_frame=False)
            ), 'XDdtCQnpc2imFany7ZGMVCwkwOKSGKeN9jwx9R9LWrU='],
            [dict(
                train_dataset=train_dataset,
                environment_params_path=None,
                root_results_path='hyperparameter_hunter/HyperparameterHunterAssets',
                holdout_dataset=get_holdout_set,
                test_dataset=test_dataset,
                target_column='diagnosis',
                do_predict_proba=False,
                prediction_formatter=None,
                metrics_params=metrics_parameters,
                cross_experiment_params=dict(
                    random_seeds=None,
                    runs=3,
                    cross_validation_type='KFold',
                ),
                cross_validation_params=cross_validation_params,
                verbose=True,
                file_blacklist=None,
                reporting_handler_params=dict(add_frame=False)
            ), 'mkhKRknVEEpPCHgYHcl49KaF2Root8OaCGD5BK1aDs0=']
        ],
    }

    def setUp(self):
        self.test_runner = partial(TextTestRunner, verbosity=0)  # verbosity=2  # verbosity=0
        self.test_function = Environment

    ##################################################
    # Build Suites for Test Cases
    ##################################################
    def do_simple_tests(self):
        cases, keys = format_suites(self.simple_tests, group_format='simple_test_{}_')
        targets = [_[1] for _ in cases]
        cases = [dict(**_[0]) for _ in cases]

        self.test_runner().run(equals_suite(self.test_function, cases, targets, keys, get_module(__name__, self)))
