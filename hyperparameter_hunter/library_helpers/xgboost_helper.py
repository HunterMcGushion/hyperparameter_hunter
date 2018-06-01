##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter import environment
# from hyperparameter_hunter import result_handler
from hyperparameter_hunter.utils.learning_utils import upsample
from hyperparameter_hunter.utils.metrics_utils import gini_xgb, gini_normalized_c

##################################################
# Import Miscellaneous Assets
##################################################
from datetime import datetime
import inspect
from inspect import getsourcefile
import numpy as np
from os.path import abspath, dirname, sep
import pandas as pd
import sys

###############################################
# Import Learning Assets
###############################################
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

from xgboost import XGBClassifier

##################################################
# Add parent directory to sys.path
##################################################
# current_dir = dirname(abspath(getsourcefile(lambda: 0)))
# sys.path.insert(0, current_dir[:current_dir.rfind(sep)])
#
# try:
#     from . import environment
#     from .utils.learning_utils import upsample
#     from .utils.metrics_utils import gini_xgb, gini_normalized_c
# except ImportError:
#     import environment
#     from utils.learning_utils import upsample
#     from utils.metrics_utils import gini_xgb, gini_normalized_c
# finally:
#     sys.path.pop(0)


##################################################
# Declare Constants
##################################################
np.random.seed(32)


def xgboost_test_function():
    return 'I am a test function for hyperparameter_hunter. I am in the xgboost_helper.py script'


class XGBoostNode():
    def __init__(
            self, environment, xgb_parameters, train_input, train_target, test_input=None,
            fit_parameters=None, k_fold_parameters=None, stratified_k_fold_parameters=None,
            sampling_parameters=None, encoding_parameters=None, preprocessing_parameters=None,
            oof_evaluator=None, model_description='', validation_metric_name='gini', do_print=True,
            full_save_trigger_key=None, full_save_trigger_lower_threshold=None, full_save_trigger_upper_threshold=None
    ):
        self.environment = environment

        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_ids = []

        if test_input is not None and 'id' in test_input.columns.values:
            self.test_ids = test_input['id']
            self.test_input = test_input.drop(['id'], axis=1)

        self.xgb_parameters = xgb_parameters
        self.fit_parameters = fit_parameters
        self.sampling_parameters = sampling_parameters
        self.encoding_parameters = encoding_parameters
        self.preprocessing_parameters = preprocessing_parameters

        self.feature_importance = None
        self.xgb_evaluations = None

        self.oof_train_predictions = np.empty(len(train_input))
        self.oof_ids = np.empty(len(train_input))
        self.oof_targets = np.empty(len(train_input))
        self.test_predictions = np.zeros(len(test_input)) if test_input is not None else None

        self.oof_evaluator = oof_evaluator
        self.model_description = model_description
        self.validation_metric_name = validation_metric_name
        self.do_print = do_print

        self.full_save_trigger_key = full_save_trigger_key
        self.full_save_trigger_lower_threshold = full_save_trigger_lower_threshold
        self.full_save_trigger_upper_threshold = full_save_trigger_upper_threshold

        self.results = dict()

        ##################################################
        # Set Up K-Fold Cross Validation
        ##################################################
        if k_fold_parameters is not None and stratified_k_fold_parameters is None:
            self.fold_parameters = k_fold_parameters
            self.folds = KFold(**k_fold_parameters)
        elif stratified_k_fold_parameters is not None and k_fold_parameters is None:
            self.fold_parameters = stratified_k_fold_parameters
            self.folds = StratifiedKFold(**stratified_k_fold_parameters)
        elif k_fold_parameters is not None and stratified_k_fold_parameters is not None:
            raise Exception('XGBoostNode can be initialized with at most one of: k_fold_parameters, stratified_k_fold_parameters')

        ##################################################
        # Set Up Feature Importance Recording
        ##################################################
        if 'n_splits' in self.fold_parameters.keys():
            self.feature_importance = np.zeros((len(train_input.columns), self.fold_parameters['n_splits']))

        ##################################################
        # Set Up XGBoost Best Round/Fold Recording
        ##################################################
        assert (('n_estimators' in self.xgb_parameters.keys()) and ('n_splits' in self.fold_parameters.keys()) and (
            'eval_metric' in self.fit_parameters.keys()))
        self.xgb_evaluations = np.zeros((self.xgb_parameters['n_estimators'], self.fold_parameters['n_splits']))

    def cv_train(self):
        start_time = datetime.now()

        for fold, (train_indexes, validation_indexes) in enumerate(self.folds.split(self.train_input, self.train_target)):
            train_input, train_target = self.train_input.iloc[train_indexes], self.train_target.iloc[train_indexes]
            validation_input, validation_target = self.train_input.iloc[validation_indexes], self.train_target.iloc[
                validation_indexes]

            model = XGBClassifier(**self.xgb_parameters)

            ##################################################
            # Perform Upsampling During Cross Validation
            ##################################################
            if self.sampling_parameters is not None:
                print('Shapes before upsampling: {} --- {}'.format(train_input.shape, train_target.shape))
                train_input, train_target = upsample(train_input, train_target, **self.sampling_parameters)
                print('Shapes after upsampling: {} --- {}'.format(train_input.shape, train_target.shape))

            model.fit(
                train_input, train_target,
                eval_set=[(validation_input, validation_target)],
                **self.fit_parameters if self.fit_parameters is not None else {}
            )

            ##################################################
            # Get Feature Importance Values
            ##################################################
            if self.feature_importance is not None:
                self.feature_importance[:, fold] = model.feature_importances_

            ##################################################
            # Get Out-of-Fold Training Predictions
            ##################################################
            self.oof_train_predictions[validation_indexes] = model.predict_proba(validation_input)[:, 1]
            self.oof_ids[validation_indexes] = validation_indexes
            self.oof_targets[validation_indexes] = validation_target

            ##################################################
            # Find Best Round for Validation Set
            ##################################################
            # self.xgb_evaluations[:, fold] = model.evals_result_['validation_1'][self.validation_metric_name]
            self.xgb_evaluations[:, fold] = model.evals_result_['validation_0'][self.validation_metric_name]
            best_round = np.argsort(self.xgb_evaluations[:, fold])[::-1][0]

            ##################################################
            # Display Results
            ##################################################
            if self.do_print is True:
                print("... Fold #{}    Gini_N:{}    Best_Score:{}    Best_Round:{}    Time:{}".format(
                    fold + 1,
                    self.oof_evaluator(validation_target, self.oof_train_predictions[validation_indexes]),
                    self.xgb_evaluations[best_round, fold],
                    best_round,
                    datetime.now().time().__str__()
                ))

            ##################################################
            # Update Test Predictions
            ##################################################
            if self.test_predictions is not None:
                self.test_predictions += model.predict_proba(self.test_input)[:, 1]

        ##################################################
        # Average Test Predictions by Folds
        ##################################################
        if (self.test_predictions is not None) and ('n_splits' in self.fold_parameters.keys()):
            self.test_predictions /= self.fold_parameters['n_splits']

        ##################################################
        # Evaluate Model
        ##################################################
        # full_oof_score = self.oof_evaluator(self.train_target, self.oof_train_predictions)

        mean_eval = np.mean(self.xgb_evaluations, axis=1)
        std_eval = np.std(self.xgb_evaluations, axis=1)
        best_mean_round = np.argsort(mean_eval)[::-1][0]

        # if self.do_print is True:
        #     print('Full OOF Score: {}'.format(full_oof_score))
        #     print('Best Mean Score:{} === StdDev:{} === Best_Round:{}'.format(
        #         mean_eval[best_mean_round], std_eval[best_mean_round], best_mean_round
        #     ))

        ##################################################
        # Record Feature Importances
        ##################################################
        if self.feature_importance is not None:
            self.feature_importance = self.feature_importance.mean(axis=1)

            # if self.do_print is True:
            #     for i, importance in enumerate(self.feature_importance):
            #         print('%-20s : %10.4f' % (self.train_input.columns[i], importance))

        ##################################################
        # Update Predictions
        ##################################################
        if self.test_predictions is not None:
            self.test_input['target'] = self.test_predictions
            self.test_input['id'] = self.test_ids

        train_predictions = pd.DataFrame(
            data=np.c_[self.oof_ids, self.oof_targets, self.oof_train_predictions],
            columns=['id', 'target', 'prediction']
        )
        train_predictions[['id']] = train_predictions[['id']].astype(int)

        ##################################################
        # Save Results
        ##################################################
        # FLAG: Must do this because function PSUtils.gini_xgb() is not JSON-serializable - SEE BELOW PROBLEM
        del self.fit_parameters['eval_metric']

        self.results = dict(
            source_script=inspect.stack()[0][1],
            model_class='xgboost',
            # specification_group_id=self.environment['specification_group_id'] if 'specification_group_id' in self.environment.keys() else None,
            description=self.model_description,
            oof_predictions=train_predictions,
            test_predictions=self.test_input[['id', 'target']],

            oof_normalized_gini=gini_normalized_c(train_predictions['target'], train_predictions['prediction']),
            oof_roc_auc=roc_auc_score(train_predictions['target'], train_predictions['prediction']),
            oof_accuracy=accuracy_score(train_predictions['target'], train_predictions['prediction'].round()),
            oof_log_loss=log_loss(train_predictions['target'], train_predictions['prediction']),
            confusion_matrix=confusion_matrix(train_predictions['target'], train_predictions['prediction'].round()).tolist(),

            time_elapsed=str(datetime.now() - start_time),
            features=self.train_input.columns.values.tolist(),
            hyperparameters=dict(
                xgb_params=self.xgb_parameters,
                fit_parameters=self.fit_parameters,
                fold_params=self.fold_parameters,
                upsample_params=self.sampling_parameters
            ),

            # Extras:
            feature_importances=zip(self.train_input.columns.values.tolist(), self.feature_importance.tolist()),
            mean_eval=mean_eval.tolist(),
            std_eval=std_eval.tolist(),
            best_round=best_mean_round
        )

        self.results['experiment_id'] = result_handler.record_results(
            self.environment,
            full_save_trigger_key=self.full_save_trigger_key,
            full_save_trigger_lower_threshold=self.full_save_trigger_lower_threshold,
            full_save_trigger_upper_threshold=self.full_save_trigger_upper_threshold,
            **self.results
        )

        return self.results

    def normal_train(self):
        pass

    def start(self):
        pass


def execute():
    # train_data = pd.read_csv('../data/porto_seguro_train.csv')
    # test_data = pd.read_csv('../data/porto_seguro_test.csv')

    sample_environment = None  # environment.Environment()

    sample_xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        objective='binary:logistic',
        learning_rate=.1,
        subsample=.8,
        colsample_bytree=.8,
        gamma=1,
        reg_alpha=0,
        reg_lambda=1,
        nthread=4
    )
    sample_fit_parameters = dict(
        eval_metrics=gini_xgb,
        early_stopping_rounds=None,
        # sample_weight=
        verbose=True
    )
    sample_k_fold_params = dict(
        n_splits=5,
        shuffle=True,
        random_state=32
    )
    sample_upsample_params = dict(
        target_feature='target',
        target_value=1
    )

    sample_node = XGBoostNode(
        sample_environment,
        sample_xgb_params,
        train_data.drop(['id', 'target'], axis=1),
        train_data['target'].values,
        test_input=test_data.drop(['id'], axis=1),
        fit_parameters=sample_fit_parameters,
        k_fold_parameters=sample_k_fold_params,
        sampling_parameters=sample_upsample_params
    )


if __name__ == '__main__':
    execute()
