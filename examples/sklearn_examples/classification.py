from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter import RandomForestOptimization, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

#################### Format DataFrame ####################
data = load_breast_cancer()
train_df = pd.DataFrame(data.data, columns=data.feature_names)
train_df["diagnosis"] = data.target

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    root_results_path="HyperparameterHunterAssets",
    target_column="diagnosis",
    metrics_map=["roc_auc_score"],
    cross_validation_type="StratifiedKFold",
    cross_validation_params=dict(n_splits=5, random_state=32),
    verbose=1,
)

# We're initializing our `Environment` with `verbose=1` to tell our experiments to only log the
# ... essentials because we're about to run lots of experiments.

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
# `CrossValidationExperiment`'s `model_init_params={}` means use the `model_initializer`'s defaults
experiment_0 = CrossValidationExperiment(KNeighborsClassifier, {})
experiment_1 = CrossValidationExperiment(SVC, {})
experiment_2 = CrossValidationExperiment(LinearSVC, {})
experiment_3 = CrossValidationExperiment(NuSVC, {})
experiment_4 = CrossValidationExperiment(DecisionTreeClassifier, {})
experiment_5 = CrossValidationExperiment(RandomForestClassifier, {})
experiment_6 = CrossValidationExperiment(AdaBoostClassifier, {})
experiment_7 = CrossValidationExperiment(GradientBoostingClassifier, {})
experiment_8 = CrossValidationExperiment(GaussianNB, {})
experiment_9 = CrossValidationExperiment(LinearDiscriminantAnalysis, {})
experiment_10 = CrossValidationExperiment(QuadraticDiscriminantAnalysis, {})
experiment_11 = CrossValidationExperiment(MLPClassifier, {})
# Of course, SKLearn has many more algorithms than those shown here, but I think you get the idea

# Notice that in all the above experiments, we gave `CrossValidationExperiment` `model_init_params={}`.
# Passing an empty dict tells it to use the default hyperparameters for the `model_initializer`, which it'll figure out on its own.

#################### 2. Hyperparameter Optimization ####################
# We're just going to do optimization on one of the algorithms used above (`AdaBoostClassifier`);
# ... although, HyperparameterHunter can certainly do consecutive optimization rounds.

# Notice below that `optimizer` correctly identifies `experiment_6` as being the only saved
# ... experiment it can learn from because it's optimizing `AdaBoostClassifier`.

optimizer = RandomForestOptimization(iterations=12, random_state=42)
optimizer.set_experiment_guidelines(
    model_initializer=AdaBoostClassifier,
    model_init_params=dict(
        n_estimators=Integer(25, 100),
        learning_rate=Real(0.5, 1.0),
        algorithm=Categorical(["SAMME", "SAMME.R"]),
    ),
)
optimizer.go()
