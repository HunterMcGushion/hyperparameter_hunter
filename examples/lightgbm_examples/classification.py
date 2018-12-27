from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import RandomForestOptimization, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

#################### Format DataFrame ####################
# Be advised, this dataset (SKLearn's Forest Cover Types) can take a little while to download...
# This is a multi-class classification task, in which the target is label-encoded.
data = fetch_covtype(shuffle=True, random_state=32)
train_df = pd.DataFrame(data.data, columns=["x_{}".format(_) for _ in range(data.data.shape[1])])
train_df["y"] = data.target

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    root_results_path="HyperparameterHunterAssets",
    target_column="y",
    metrics_map=dict(f1=lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")),
    cross_validation_type="StratifiedKFold",
    cross_validation_params=dict(n_splits=5, random_state=32),
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CVExperiment(
    model_initializer=LGBMClassifier,
    model_init_params=dict(boosting_type="gbdt", num_leaves=31, max_depth=-1, subsample=0.5),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = RandomForestOptimization(iterations=10, random_state=32)
optimizer.set_experiment_guidelines(
    model_initializer=LGBMClassifier,
    model_init_params=dict(
        boosting_type=Categorical(["gbdt", "dart"]),
        num_leaves=Integer(10, 40),
        max_depth=-1,
        subsample=Real(0.3, 0.7),
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
