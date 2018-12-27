from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import GBRT, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier

#################### Format DataFrame ####################
# Be advised, this dataset (SKLearn's Forest Cover Types) can take a little while to download...
# This is a multi-class classification task, in which the target is label-encoded.
# We'll also subtract one from the targets, to make the seven labels fall within the range of 0-6,
# ... rather than the default range of 1-7. This is to keep CatBoost from complaining.
data = fetch_covtype(shuffle=True, random_state=32)
train_df = pd.DataFrame(data.data, columns=["x_{}".format(_) for _ in range(data.data.shape[1])])
train_df["y"] = data.target - 1

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    root_results_path="HyperparameterHunterAssets",
    target_column="y",
    metrics_map=dict(f1=lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")),
    cross_validation_type=KFold,
    cross_validation_params=dict(n_splits=5, random_state=32),
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CVExperiment(
    model_initializer=CatBoostClassifier,
    model_init_params=dict(
        iterations=100,
        learning_rate=0.03,
        depth=6,
        save_snapshot=False,
        allow_writing_files=False,
        loss_function="MultiClass",
        classes_count=7,
    ),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = GBRT(iterations=8, random_state=42)
optimizer.set_experiment_guidelines(
    model_initializer=CatBoostClassifier,
    model_init_params=dict(
        iterations=100,
        learning_rate=Real(low=0.0001, high=0.5),
        depth=Integer(4, 15),
        save_snapshot=False,
        allow_writing_files=False,
        loss_function="MultiClass",
        classes_count=7,
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
