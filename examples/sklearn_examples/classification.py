from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import RandomForestOptPro, Real, Integer, Categorical
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
    results_path="HyperparameterHunterAssets",
    target_column="diagnosis",
    metrics=["roc_auc_score"],
    cv_type="StratifiedKFold",
    cv_params=dict(n_splits=5, random_state=32),
    verbose=1,
)

# We're initializing our `Environment` with `verbose=1` to tell our experiments to only log the
# ... essentials because we're about to run lots of experiments.

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
# `CVExperiment`'s `model_init_params={}` means use the `model_initializer`'s defaults
experiment_0 = CVExperiment(KNeighborsClassifier, {})
experiment_1 = CVExperiment(SVC, {})
experiment_2 = CVExperiment(LinearSVC, {})
experiment_3 = CVExperiment(NuSVC, {})
experiment_4 = CVExperiment(DecisionTreeClassifier, {})
experiment_5 = CVExperiment(RandomForestClassifier, {})
experiment_6 = CVExperiment(AdaBoostClassifier, {})
experiment_7 = CVExperiment(GradientBoostingClassifier, {})
experiment_8 = CVExperiment(GaussianNB, {})
experiment_9 = CVExperiment(LinearDiscriminantAnalysis, {})
experiment_10 = CVExperiment(QuadraticDiscriminantAnalysis, {})
experiment_11 = CVExperiment(MLPClassifier, {})
# Of course, SKLearn has many more algorithms than those shown here, but I think you get the idea

# Notice that in all the above experiments, we gave `CVExperiment` `model_init_params={}`.
# Passing an empty dict tells it to use the default hyperparameters for the `model_initializer`, which it'll figure out on its own.

#################### 2. Hyperparameter Optimization ####################
# We're just going to do optimization on one of the algorithms used above (`AdaBoostClassifier`);
# ... although, HyperparameterHunter can certainly do consecutive optimization rounds.

# Notice below that `optimizer` correctly identifies `experiment_6` as being the only saved
# ... experiment it can learn from because it's optimizing `AdaBoostClassifier`.

optimizer = RandomForestOptPro(iterations=12, random_state=42)
optimizer.set_experiment_guidelines(
    model_initializer=AdaBoostClassifier,
    model_init_params=dict(
        n_estimators=Integer(25, 100),
        learning_rate=Real(0.5, 1.0),
        algorithm=Categorical(["SAMME", "SAMME.R"]),
    ),
)
optimizer.go()
