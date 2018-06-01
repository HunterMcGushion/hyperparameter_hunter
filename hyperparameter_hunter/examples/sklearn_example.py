import sys
import os.path

try:
    sys.path.append(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])
except Exception as _ex:
    raise _ex

from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


def execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        root_results_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../HyperparameterHunterAssets')),
        target_column='diagnosis',
        metrics_map=['roc_auc_score'],
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    experiment_0 = CrossValidationExperiment(model_initializer=KNeighborsClassifier, model_init_params={})
    experiment_1 = CrossValidationExperiment(model_initializer=SVC, model_init_params={})
    experiment_2 = CrossValidationExperiment(model_initializer=LinearSVC, model_init_params={})
    experiment_3 = CrossValidationExperiment(model_initializer=NuSVC, model_init_params={})
    experiment_4 = CrossValidationExperiment(model_initializer=DecisionTreeClassifier, model_init_params={})
    experiment_5 = CrossValidationExperiment(model_initializer=RandomForestClassifier, model_init_params={})
    experiment_6 = CrossValidationExperiment(model_initializer=AdaBoostClassifier, model_init_params={})
    experiment_7 = CrossValidationExperiment(model_initializer=GradientBoostingClassifier, model_init_params={})
    experiment_8 = CrossValidationExperiment(model_initializer=GaussianNB, model_init_params={})
    experiment_9 = CrossValidationExperiment(model_initializer=LinearDiscriminantAnalysis, model_init_params={})
    experiment_10 = CrossValidationExperiment(model_initializer=QuadraticDiscriminantAnalysis, model_init_params={})
    experiment_11 = CrossValidationExperiment(model_initializer=MLPClassifier, model_init_params={})

    # Of course, SKLearn has many more algorithms than those shown here, but I think you get the idea


if __name__ == '__main__':
    execute()
